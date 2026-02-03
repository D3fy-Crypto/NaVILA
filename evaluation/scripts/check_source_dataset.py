#!/usr/bin/env python3
"""
Check source R2R_VLNCE dataset for episode IDs and continuity.

Reads directly from the gzipped JSON dataset file - no Habitat needed.

Usage:
    python check_source_dataset.py --split train
    python check_source_dataset.py --path /path/to/train_gt.json.gz
"""

import json
import gzip
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_dataset(dataset_path: str):
    """Analyze the source dataset for episode continuity"""
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"❌ File not found: {dataset_path}")
        return
    
    print("=" * 80)
    print(f"ANALYZING SOURCE DATASET: {dataset_path.name}")
    print("=" * 80)
    print()
    
    # Read the gzipped JSON
    print("Loading dataset...")
    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if it's episodes list or dict keyed by episode_id
    if isinstance(data, dict) and 'episodes' in data:
        episodes = data.get('episodes', [])
        episode_ids = [int(ep.get('episode_id')) for ep in episodes if ep.get('episode_id') is not None]
    elif isinstance(data, dict):
        # Keys are episode IDs
        episode_ids = [int(k) for k in data.keys()]
        episodes = list(data.values())
    else:
        print("❌ Unknown dataset format!")
        return
    
    if not episode_ids:
        print("❌ No episodes found in dataset!")
        return
    
    print(f"Loaded {len(episode_ids)} episodes")
    print()
    
    # Collect stats from episodes
    scene_counts = defaultdict(int)
    instruction_lengths = []
    ref_path_lengths = []
    
    for ep in episodes:
        # Scene
        scene = ep.get('scene_id', 'unknown').split('/')[-1].replace('.glb', '')
        scene_counts[scene] += 1
        
        # Instruction
        instr = ep.get('instruction', {})
        if isinstance(instr, dict):
            instr_text = instr.get('instruction_text', '')
        else:
            instr_text = str(instr)
        instruction_lengths.append(len(instr_text))
        
        # Reference path
        ref_path = ep.get('reference_path', [])
        ref_path_lengths.append(len(ref_path))
    
    # Analyze IDs
    episode_ids_sorted = sorted(episode_ids)
    episode_ids_set = set(episode_ids)
    
    min_id = min(episode_ids)
    max_id = max(episode_ids)
    total_episodes = len(episodes)
    unique_count = len(episode_ids_set)
    
    # Check for duplicates
    duplicates = total_episodes - unique_count
    
    # Expected range
    expected_range = set(range(min_id, max_id + 1))
    missing_ids = sorted(expected_range - episode_ids_set)
    
    # Find gaps
    gaps = []
    if missing_ids:
        gap_start = missing_ids[0]
        gap_end = missing_ids[0]
        
        for i in range(1, len(missing_ids)):
            if missing_ids[i] == gap_end + 1:
                gap_end = missing_ids[i]
            else:
                gaps.append((gap_start, gap_end))
                gap_start = missing_ids[i]
                gap_end = missing_ids[i]
        gaps.append((gap_start, gap_end))
    
    # Print results
    print("EPISODE COVERAGE IN SOURCE DATASET")
    print("-" * 40)
    print(f"  Total episodes:         {total_episodes}")
    print(f"  Unique episode IDs:     {unique_count}")
    print(f"  First episode ID:       {min_id}")
    print(f"  Last episode ID:        {max_id}")
    print(f"  Expected range:         {min_id} to {max_id} = {max_id - min_id + 1} IDs")
    print()
    
    if duplicates > 0:
        print(f"  ⚠️  Duplicates found:    {duplicates}")
        from collections import Counter
        id_counts = Counter(episode_ids)
        dup_ids = [(ep_id, count) for ep_id, count in id_counts.items() if count > 1]
        print(f"     Duplicated IDs: {dup_ids[:10]}{'...' if len(dup_ids) > 10 else ''}")
        print()
    
    if missing_ids:
        print(f"  ⚠️  GAPS IN ID SEQUENCE: {len(missing_ids)} IDs missing")
        print()
        print("  Gaps in sequence:")
        for gap_start, gap_end in gaps:
            if gap_start == gap_end:
                print(f"     Missing ID: {gap_start}")
            else:
                print(f"     Missing range: {gap_start} to {gap_end} ({gap_end - gap_start + 1} IDs)")
        
        if len(missing_ids) <= 50:
            print()
            print(f"  All missing IDs: {missing_ids}")
    else:
        print(f"  ✅ CONTINUOUS: All IDs from {min_id} to {max_id} are present!")
    
    print()
    
    # Statistics
    print("DATASET STATISTICS")
    print("-" * 40)
    print(f"  Unique scenes:          {len(scene_counts)}")
    print(f"  Avg instruction len:    {sum(instruction_lengths)/len(instruction_lengths):.1f} chars")
    print(f"  Avg reference path len: {sum(ref_path_lengths)/len(ref_path_lengths):.1f} waypoints")
    print()
    
    print("  Top 10 scenes by episode count:")
    top_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for scene, count in top_scenes:
        print(f"     {scene}: {count}")
    
    print()
    print("=" * 80)
    
    return {
        'total': total_episodes,
        'unique': unique_count,
        'min_id': min_id,
        'max_id': max_id,
        'missing': missing_ids,
        'gaps': gaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Check source R2R_VLNCE dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val_seen", "val_unseen"],
                        help="Dataset split to check")
    parser.add_argument("--path", type=str, default=None,
                        help="Direct path to dataset file (overrides --split)")
    
    args = parser.parse_args()
    
    if args.path:
        dataset_path = args.path
    else:
        # Default path structure
        base_path = Path("/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed")
        dataset_path = base_path / args.split / f"{args.split}_gt.json.gz"
    
    analyze_dataset(dataset_path)


if __name__ == "__main__":
    main()
