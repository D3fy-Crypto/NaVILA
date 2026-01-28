#!/usr/bin/env python3
"""
Export ground-truth poses for NaVILA training samples from VLN-CE episodes.

This script:
1. Loads NaVILA training annotations (R2R/RxR)
2. Maps each sample to its VLN-CE episode
3. Uses Habitat simulator to extract ground-truth poses
4. Exports poses and deltas in JSONL format for GRU training

Usage:
    python export_gru_poses.py --dataset r2r --split train --frames 8
    python export_gru_poses.py --dataset rxr --split val_unseen --frames 8
"""

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import habitat
import numpy as np
from habitat.config.default import get_config as get_habitat_config
from habitat.datasets.utils import VocabDict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from habitat_extensions.utils import quaternion_to_yaw


def load_navila_annotations(dataset_root: Path, dataset: str) -> List[Dict]:
    """Load NaVILA training annotations."""
    anno_file = dataset_root / dataset.upper() / "annotations.json"
    if not anno_file.exists():
        raise FileNotFoundError(f"Annotations not found: {anno_file}")
    
    with open(anno_file, "r") as f:
        annotations = json.load(f)
    
    print(f"Loaded {len(annotations)} samples from {anno_file}")
    return annotations


def load_vlnce_episodes(data_root: Path, dataset: str, split: str) -> Dict[int, Dict]:
    """Load VLN-CE episodes and index by episode_id."""
    if dataset == "r2r":
        dataset_dir = data_root / "R2R_VLNCE_v1-3_preprocessed" / split
        episode_file = dataset_dir / f"{split}.json.gz"
    elif dataset == "rxr":
        dataset_dir = data_root / "RxR_VLNCE_v0" / split
        episode_file = dataset_dir / f"{split}_guide.json.gz"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if not episode_file.exists():
        raise FileNotFoundError(f"Episode file not found: {episode_file}")
    
    with gzip.open(episode_file, "rt") as f:
        data = json.load(f)
    
    episodes = data["episodes"]
    episode_map = {ep["episode_id"]: ep for ep in episodes}
    
    print(f"Loaded {len(episode_map)} episodes from {episode_file}")
    return episode_map


def parse_video_id(video_id: str) -> Tuple[int, int]:
    """
    Parse NaVILA video_id to extract episode_id and timestep.
    
    Format: "{episode_id}-{timestep}"
    Example: "914-23" -> episode_id=914, timestep=23
    
    NOTE: The suffix is the TIMESTEP number (for sequential action prediction),
    NOT the ending frame index. Multiple samples can exist per episode.
    
    Returns:
        (episode_id, timestep)
    """
    parts = video_id.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid video_id format: {video_id}")
    
    episode_id = int(parts[0])
    timestep = int(parts[1])
    return episode_id, timestep


def interpolate_dense_poses(reference_path: List, num_frames: int) -> List[Tuple[float, float, float]]:
    """
    Interpolate dense (x, y, yaw) poses for all frames in a trajectory.
    
    VLN-CE reference_path is sparse (5-7 waypoints), but frames are dense (25-80+ frames).
    We need to interpolate to create a pose for every frame.
    
    Args:
        reference_path: List of waypoint positions [[x, y_up, z], ...] from VLN-CE
        num_frames: Total number of frames rendered for this trajectory segment
    
    Returns:
        List of (x, y, yaw) tuples, one per frame
    """
    if len(reference_path) == 1:
        # Single waypoint - all frames have same pose
        pos = reference_path[0]
        return [(pos[0], pos[2], 0.0)] * num_frames
    
    # Create interpolation indices
    waypoint_indices = np.arange(len(reference_path))
    frame_indices = np.linspace(0, len(reference_path) - 1, num_frames)
    
    # Extract x, y (Habitat z) from waypoints
    x_waypoints = np.array([wp[0] for wp in reference_path])
    y_waypoints = np.array([wp[2] for wp in reference_path])  # Habitat z → our y
    
    # Interpolate positions
    x_interp = np.interp(frame_indices, waypoint_indices, x_waypoints)
    y_interp = np.interp(frame_indices, waypoint_indices, y_waypoints)
    
    # Compute yaw for each frame from direction of motion
    yaws = []
    for i in range(num_frames):
        if i < num_frames - 1:
            # Look ahead to next frame
            dx = x_interp[i + 1] - x_interp[i]
            dy = y_interp[i + 1] - y_interp[i]
            if dx != 0 or dy != 0:
                yaw = math.atan2(dy, dx)
            else:
                yaw = 0.0 if i == 0 else yaws[-1]
        else:
            # Last frame - use previous yaw
            yaw = yaws[-1] if yaws else 0.0
        yaws.append(yaw)
    
    poses = [(float(x), float(y), float(yaw)) for x, y, yaw in zip(x_interp, y_interp, yaws)]
    return poses


def extract_pose_from_position_rotation(position: List[float], rotation: List[float]) -> Tuple[float, float, float]:
    """
    Convert Habitat position and quaternion rotation to (x, y, yaw).
    
    Habitat uses:
    - position: [x, y_up, z] where y_up is the vertical axis
    - rotation: [w, x, y, z] quaternion
    
    We convert to:
    - x: same as Habitat x
    - y: Habitat z (forward direction)
    - yaw: rotation around y_up axis
    
    Args:
        position: [x, y_up, z] in Habitat coordinates
        rotation: [w, x, y, z] quaternion
    
    Returns:
        (x, y, yaw) in 2D navigation space
    """
    x = position[0]
    y = position[2]  # Habitat z -> our y
    
    # Convert quaternion to yaw
    yaw = quaternion_to_yaw(rotation)
    
    return x, y, yaw


def compute_deltas(poses: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """
    Compute relative pose deltas between consecutive poses.
    
    Args:
        poses: List of (x, y, yaw) tuples
    
    Returns:
        List of (dx, dy, dyaw) tuples with len = len(poses) - 1
    """
    deltas = []
    
    for i in range(len(poses) - 1):
        x1, y1, yaw1 = poses[i]
        x2, y2, yaw2 = poses[i + 1]
        
        # Compute deltas in global frame
        dx = x2 - x1
        dy = y2 - y1
        
        # Wrap yaw delta to [-pi, pi]
        dyaw = yaw2 - yaw1
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        
        deltas.append((float(dx), float(dy), float(dyaw)))
    
    return deltas


def discretize_delta(dx: float, dy: float, dyaw: float, 
                     dist_bins: List[float], yaw_bins: List[float]) -> Tuple[int, int]:
    """
    Discretize continuous deltas into bins for classification training.
    
    Args:
        dx, dy: Translation deltas (meters)
        dyaw: Rotation delta (radians)
        dist_bins: Distance bin edges (e.g., [0, 0.25, 0.5, 1.0, 2.0])
        yaw_bins: Yaw bin edges in degrees (e.g., [-180, -30, -10, 10, 30, 180])
    
    Returns:
        (dist_bin_idx, yaw_bin_idx)
    """
    # Compute distance
    dist = math.sqrt(dx**2 + dy**2)
    
    # Find distance bin
    dist_bin = 0
    for i, threshold in enumerate(dist_bins[1:], start=1):
        if dist >= threshold:
            dist_bin = i
    
    # Convert yaw to degrees
    yaw_deg = math.degrees(dyaw)
    
    # Find yaw bin
    yaw_bin = 0
    for i, threshold in enumerate(yaw_bins[1:], start=1):
        if yaw_deg >= threshold:
            yaw_bin = i
    
    return dist_bin, yaw_bin


def export_poses_for_sample(
    sample: Dict,
    episode: Dict,
    num_frames: int,
    dist_bins: List[float],
    yaw_bins: List[float],
) -> Dict:
    """
    Extract ground-truth poses for a single NaVILA sample.
    
    CORRECTED LOGIC:
    - sample["frames"] contains actual frame list (e.g., 49 frames)
    - video_id suffix represents timestep, NOT frame index
    - Interpolate dense poses for ALL frames, then sample num_frames uniformly
    
    Args:
        sample: NaVILA annotation dict with "video_id", "frames", "q", "a"
        episode: VLN-CE episode dict with "reference_path"
        num_frames: Number of frames to sample (e.g., 8 for NaVILA)
        dist_bins: Distance bin edges for discretization
        yaw_bins: Yaw bin edges (degrees) for discretization
    
    Returns:
        Dict with 'video_id', 'poses', 'deltas', 'dist_bins', 'yaw_bins'
    """
    video_id = sample["video_id"]
    episode_id, timestep_idx = parse_video_id(video_id)
    
    reference_path = episode["reference_path"]
    total_frames = len(sample["frames"])  # Actual number of frames for this sample
    
    # Step 1: Interpolate dense poses for ALL frames in this sample
    dense_poses = interpolate_dense_poses(reference_path, total_frames)
    
    # Step 2: Sample num_frames poses uniformly (matches NaVILA's frame sampling)
    if len(dense_poses) <= num_frames:
        sampled_poses = dense_poses
    else:
        sample_indices = np.linspace(0, len(dense_poses) - 1, num_frames, dtype=int)
        sampled_poses = [dense_poses[i] for i in sample_indices]
    
    # Step 3: Compute deltas between sampled poses
    deltas = compute_deltas(sampled_poses)
    
    # Step 4: Discretize deltas
    dist_bin_labels = []
    yaw_bin_labels = []
    for dx, dy, dyaw in deltas:
        dist_bin, yaw_bin = discretize_delta(dx, dy, dyaw, dist_bins, yaw_bins)
        dist_bin_labels.append(dist_bin)
        yaw_bin_labels.append(yaw_bin)
    
    return {
        "video_id": video_id,
        "episode_id": episode_id,
        "timestep": timestep_idx,
        "scene_id": episode["scene_id"],
        "poses": sampled_poses,
        "deltas": deltas,
        "dist_bins": dist_bin_labels,
        "yaw_bins": yaw_bin_labels,
        "num_total_frames": total_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="Export ground-truth poses for NaVILA training")
    parser.add_argument("--dataset", type=str, required=True, choices=["r2r", "rxr"],
                        help="Dataset name (r2r or rxr)")
    parser.add_argument("--split", type=str, required=True,
                        choices=["train", "val_seen", "val_unseen"],
                        help="Split to export")
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of frames sampled per trajectory (default: 8)")
    parser.add_argument("--navila-root", type=str, default="/home/rithvik/NaVILA-Dataset",
                        help="Path to NaVILA-Dataset root")
    parser.add_argument("--data-root", type=str,
                        default="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets",
                        help="Path to VLN-CE data root")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: {navila-root}/{DATASET}/gru_pose_{split}.jsonl)")
    
    # Binning parameters
    parser.add_argument("--dist-bins", type=float, nargs="+",
                        default=[0, 0.25, 0.5, 1.0, 2.0],
                        help="Distance bin edges in meters")
    parser.add_argument("--yaw-bins", type=float, nargs="+",
                        default=[-180, -30, -10, 10, 30, 180],
                        help="Yaw bin edges in degrees")
    
    args = parser.parse_args()
    
    # Setup paths
    navila_root = Path(args.navila_root)
    data_root = Path(args.data_root)
    
    if args.output is None:
        output_file = navila_root / args.dataset.upper() / f"gru_pose_{args.split}.jsonl"
    else:
        output_file = Path(args.output)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NaVILA Pose Export")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Split: {args.split}")
    print(f"Frames per sample: {args.frames}")
    print(f"NaVILA root: {navila_root}")
    print(f"VLN-CE data root: {data_root}")
    print(f"Output: {output_file}")
    print(f"Distance bins: {args.dist_bins}")
    print(f"Yaw bins (degrees): {args.yaw_bins}")
    print("="*80)
    print()
    
    # Load data
    print("Loading annotations and episodes...")
    annotations = load_navila_annotations(navila_root, args.dataset)
    episodes = load_vlnce_episodes(data_root, args.dataset, args.split)
    print()
    
    # Export poses
    print("Exporting poses...")
    successful = 0
    failed = 0
    
    with open(output_file, "w") as f:
        for sample in tqdm(annotations, desc="Processing samples"):
            try:
                video_id = sample["video_id"]
                episode_id, _ = parse_video_id(video_id)
                
                # Check if episode exists
                if episode_id not in episodes:
                    # This sample might be from a different split
                    continue
                
                episode = episodes[episode_id]
                
                # Export poses
                pose_data = export_poses_for_sample(
                    sample, episode, args.frames,
                    args.dist_bins, args.yaw_bins
                )
                
                # Write to JSONL
                f.write(json.dumps(pose_data) + "\n")
                successful += 1
                
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only print first 5 errors
                    print(f"Error processing {video_id}: {e}")
    
    print()
    print("="*80)
    print("Export complete!")
    print(f"Successfully exported: {successful} samples")
    print(f"Failed: {failed} samples")
    print(f"Output: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
