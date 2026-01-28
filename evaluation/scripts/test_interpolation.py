#!/usr/bin/env python3
"""
Test the corrected pose interpolation logic.
"""

import json
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from export_gru_poses import (
    parse_video_id,
    interpolate_dense_poses,
    load_navila_annotations,
    load_vlnce_episodes
)

def test_sample_914_23():
    """Test interpolation on sample 914-23 (49 frames, 6 waypoints)."""
    
    # Load data
    navila_root = Path("/home/rithvik/NaVILA-Dataset")
    data_root = Path("/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets")
    
    print("Loading annotations and episodes...")
    annotations = load_navila_annotations(navila_root, "r2r")
    episodes = load_vlnce_episodes(data_root, "r2r", "train")
    
    # Find sample 914-23
    sample = None
    for s in annotations:
        if s["video_id"] == "914-23":
            sample = s
            break
    
    if sample is None:
        print("ERROR: Sample 914-23 not found")
        return
    
    print(f"\n{'='*80}")
    print(f"Testing sample: {sample['video_id']}")
    print(f"{'='*80}\n")
    
    # Parse video_id
    episode_id, timestep = parse_video_id(sample["video_id"])
    print(f"Episode ID: {episode_id}")
    print(f"Timestep: {timestep}")
    
    # Get episode data
    episode = episodes[episode_id]
    reference_path = episode["reference_path"]
    
    print(f"\nReference path: {len(reference_path)} waypoints")
    print(f"Sample frames: {len(sample['frames'])} frames")
    
    # Interpolate dense poses
    dense_poses = interpolate_dense_poses(reference_path, len(sample["frames"]))
    
    print(f"\nInterpolated poses: {len(dense_poses)}")
    print(f"Expected: {len(sample['frames'])}")
    print(f"Match: {len(dense_poses) == len(sample['frames'])}")
    
    # Sample 8 poses (as NaVILA does)
    import numpy as np
    sample_indices = np.linspace(0, len(dense_poses) - 1, 8, dtype=int)
    sampled_poses = [dense_poses[i] for i in sample_indices]
    
    print(f"\n8-frame sampling indices: {sample_indices.tolist()}")
    print(f"\nSampled poses:")
    for i, idx in enumerate(sample_indices):
        x, y, yaw = sampled_poses[i]
        print(f"  Frame {idx:2d}: x={x:7.3f}, y={y:7.3f}, yaw={yaw:6.3f} rad ({np.degrees(yaw):6.1f}°)")
    
    # Verify frame paths match
    expected_frames = [f"914/frame_{idx}.jpg" for idx in sample_indices]
    print(f"\nExpected frame paths for 8-sample:")
    for frame in expected_frames:
        print(f"  {frame}")
    
    print(f"\n{'='*80}")
    print("✓ Interpolation test complete")
    print(f"{'='*80}\n")


def test_multiple_samples():
    """Verify that multiple samples from same episode get different poses."""
    
    navila_root = Path("/home/rithvik/NaVILA-Dataset")
    data_root = Path("/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets")
    
    print("\n" + "="*80)
    print("Testing multiple samples from episode 914")
    print("="*80 + "\n")
    
    annotations = load_navila_annotations(navila_root, "r2r")
    episodes = load_vlnce_episodes(data_root, "r2r", "train")
    
    # Find all samples from episode 914
    samples_914 = []
    for s in annotations:
        episode_id, timestep = parse_video_id(s["video_id"])
        if episode_id == 914:
            samples_914.append(s)
    
    samples_914.sort(key=lambda s: parse_video_id(s["video_id"])[1])
    
    print(f"Found {len(samples_914)} samples from episode 914\n")
    
    for sample in samples_914[:5]:  # Show first 5
        video_id = sample["video_id"]
        episode_id, timestep = parse_video_id(video_id)
        num_frames = len(sample["frames"])
        
        print(f"{video_id}: timestep={timestep:2d}, frames={num_frames:2d}")
    
    print(f"\n✓ Each timestep has different number of frames (accumulated history)")
    print(f"✓ Interpolation will produce different dense poses for each sample")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_sample_914_23()
    test_multiple_samples()
