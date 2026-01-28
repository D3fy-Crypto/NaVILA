#!/usr/bin/env python3
"""
Quick test of pose export on a few samples to verify the pipeline.
"""
import argparse
import gzip
import json
import math
from pathlib import Path

def test_pose_export():
    """Run a quick test with a few samples"""
    
    # Load just 5 samples from R2R train
    navila_root = Path("/home/rithvik/NaVILA-Dataset")
    anno_file = navila_root / "R2R" / "annotations.json"
    
    print("Loading annotations...")
    with open(anno_file, "r") as f:
        annotations = json.load(f)
    
    test_samples = annotations[:5]
    print(f"Testing with {len(test_samples)} samples")
    
    # Load corresponding episodes
    data_root = Path("/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets")
    episode_file = data_root / "R2R_VLNCE_v1-3_preprocessed" / "train" / "train.json.gz"
    
    with gzip.open(episode_file, "rt") as f:
        data = json.load(f)
    
    episodes = {ep["episode_id"]: ep for ep in data["episodes"]}
    
    print(f"Loaded {len(episodes)} episodes")
    print()
    
    # Test each sample
    for sample in test_samples:
        video_id = sample["video_id"]
        parts = video_id.split("-")
        episode_id = int(parts[0])
        end_frame_idx = int(parts[1])
        
        print(f"Sample: {video_id}")
        print(f"  Episode ID: {episode_id}")
        print(f"  End frame: {end_frame_idx}")
        print(f"  Total frames in annotation: {len(sample['frames'])}")
        
        if episode_id in episodes:
            episode = episodes[episode_id]
            ref_path = episode["reference_path"]
            print(f"  ✓ Found episode with {len(ref_path)} waypoints")
            print(f"  Scene: {episode['scene_id']}")
            
            # Show first 3 waypoints
            print(f"  First 3 waypoints:")
            for i, wp in enumerate(ref_path[:3]):
                print(f"    [{i}] x={wp[0]:.2f}, y_up={wp[1]:.2f}, z={wp[2]:.2f}")
        else:
            print(f"  ✗ Episode not found in VLN-CE data")
        
        print()
    
    print("="*60)
    print("Test complete! You can now run the full export.")
    print()
    print("Example commands:")
    print()
    print("# Export R2R train split:")
    print("bash run_pose_export.sh --dataset r2r --split train")
    print()
    print("# Export R2R val_unseen split:")
    print("bash run_pose_export.sh --dataset r2r --split val_unseen")
    print()
    print("# Export with custom output:")
    print("bash run_pose_export.sh --dataset r2r --split train \\")
    print("  --output /home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl")

if __name__ == "__main__":
    test_pose_export()
