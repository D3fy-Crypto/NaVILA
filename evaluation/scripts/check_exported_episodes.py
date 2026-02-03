#!/usr/bin/env python3
"""
Check exported oracle deltas JSONL file for continuity and completeness.

Analyzes:
1. Total number of episodes exported
2. First and last episode IDs
3. Whether episode IDs are continuous
4. Which episode IDs are missing (if any)
5. Statistics about the export

Usage:
    python check_exported_episodes.py oracle_exports/oracle_deltas_train.jsonl
    python check_exported_episodes.py --expected 10819  # Check against expected count
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_jsonl(filepath: str, expected_total: int = None):
    """Analyze the JSONL file for episode continuity"""
    
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print("=" * 80)
    print(f"ANALYZING: {filepath}")
    print("=" * 80)
    print()
    
    # Read all episode IDs
    episode_ids = []
    scene_counts = defaultdict(int)
    success_count = 0
    failure_count = 0
    total_steps = 0
    oracle_modes = defaultdict(int)
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                ep_id = record.get('episode_id')
                if ep_id is not None:
                    episode_ids.append(int(ep_id))
                
                # Collect stats
                scene = record.get('scene_id', 'unknown').split('/')[-1].replace('.glb', '')
                scene_counts[scene] += 1
                
                if record.get('success', False):
                    success_count += 1
                else:
                    failure_count += 1
                
                total_steps += record.get('num_steps', 0)
                oracle_modes[record.get('oracle_mode', 'unknown')] += 1
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Line {line_num}: JSON parse error - {e}")
    
    if not episode_ids:
        print("❌ No episodes found in file!")
        return
    
    # Sort and analyze
    episode_ids_sorted = sorted(episode_ids)
    episode_ids_set = set(episode_ids)
    
    min_id = min(episode_ids)
    max_id = max(episode_ids)
    total_exported = len(episode_ids)
    unique_count = len(episode_ids_set)
    
    # Check for duplicates
    duplicates = total_exported - unique_count
    
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
    print("EPISODE COVERAGE")
    print("-" * 40)
    print(f"  Total records in file:  {total_exported}")
    print(f"  Unique episode IDs:     {unique_count}")
    print(f"  First episode ID:       {min_id}")
    print(f"  Last episode ID:        {max_id}")
    print(f"  Expected range:         {min_id} to {max_id} = {max_id - min_id + 1} episodes")
    print()
    
    if duplicates > 0:
        print(f"  ⚠️  Duplicates found:    {duplicates}")
        # Find which IDs are duplicated
        from collections import Counter
        id_counts = Counter(episode_ids)
        dup_ids = [(ep_id, count) for ep_id, count in id_counts.items() if count > 1]
        print(f"     Duplicated IDs: {dup_ids[:10]}{'...' if len(dup_ids) > 10 else ''}")
        print()
    
    if missing_ids:
        print(f"  ❌ MISSING EPISODES:     {len(missing_ids)}")
        print()
        print("  Gaps in sequence:")
        for gap_start, gap_end in gaps:
            if gap_start == gap_end:
                print(f"     Missing ID: {gap_start}")
            else:
                print(f"     Missing range: {gap_start} to {gap_end} ({gap_end - gap_start + 1} episodes)")
        
        if len(missing_ids) <= 50:
            print()
            print(f"  All missing IDs: {missing_ids}")
    else:
        print(f"  ✅ CONTINUOUS: All episodes from {min_id} to {max_id} are present!")
    
    print()
    
    # Check against expected total
    if expected_total is not None:
        print("EXPECTED vs ACTUAL")
        print("-" * 40)
        print(f"  Expected total:   {expected_total}")
        print(f"  Actual unique:    {unique_count}")
        
        if unique_count == expected_total:
            print(f"  ✅ COMPLETE: All {expected_total} episodes exported!")
        elif unique_count < expected_total:
            print(f"  ❌ INCOMPLETE: Missing {expected_total - unique_count} episodes")
            # If we know expected is 1 to expected_total
            if min_id == 1:
                all_expected = set(range(1, expected_total + 1))
                truly_missing = sorted(all_expected - episode_ids_set)
                print(f"     Episodes not yet exported: {len(truly_missing)}")
                if len(truly_missing) <= 20:
                    print(f"     Missing: {truly_missing}")
                else:
                    print(f"     First 10 missing: {truly_missing[:10]}")
                    print(f"     Last 10 missing: {truly_missing[-10:]}")
        else:
            print(f"  ⚠️  MORE than expected? Got {unique_count - expected_total} extra")
        print()
    
    # Statistics
    print("EXPORT STATISTICS")
    print("-" * 40)
    print(f"  Success rate:     {success_count}/{total_exported} ({100*success_count/total_exported:.1f}%)")
    print(f"  Failure rate:     {failure_count}/{total_exported} ({100*failure_count/total_exported:.1f}%)")
    print(f"  Total steps:      {total_steps}")
    print(f"  Avg steps/ep:     {total_steps/total_exported:.1f}")
    print()
    
    print("  Oracle modes:")
    for mode, count in oracle_modes.items():
        print(f"     {mode}: {count}")
    print()
    
    print(f"  Unique scenes:    {len(scene_counts)}")
    print("  Top 5 scenes:")
    top_scenes = sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for scene, count in top_scenes:
        print(f"     {scene}: {count}")
    
    print()
    print("=" * 80)
    
    # Return summary for programmatic use
    return {
        'total': total_exported,
        'unique': unique_count,
        'min_id': min_id,
        'max_id': max_id,
        'missing': missing_ids,
        'gaps': gaps,
        'duplicates': duplicates,
        'success_rate': success_count / total_exported if total_exported > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Check exported oracle deltas for completeness")
    parser.add_argument("filepath", type=str, nargs='?', 
                        default="oracle_exports/oracle_deltas_train.jsonl",
                        help="Path to JSONL file")
    parser.add_argument("--expected", type=int, default=10819,
                        help="Expected total number of episodes (default: 10819 for train)")
    
    args = parser.parse_args()
    
    analyze_jsonl(args.filepath, args.expected)


if __name__ == "__main__":
    main()
