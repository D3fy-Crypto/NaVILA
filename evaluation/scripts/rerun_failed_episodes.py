#!/usr/bin/env python3
"""
Re-run failed episodes from an existing oracle deltas export.

This script:
1. Reads the existing JSONL and identifies episodes with success=False
2. Re-runs only those failed episodes with the fixed WaypointFollower
3. Replaces the failed entries in the JSONL with the new successful ones

Usage:
    python rerun_failed_episodes.py oracle_exports/oracle_deltas_train.jsonl
    python rerun_failed_episodes.py oracle_exports/oracle_deltas_train.jsonl --visualize
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat import Env
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import quaternion_to_yaw

# Import the WaypointFollower from the main script
from extract_pose_deltas_final import WaypointFollower, make_cache_key


def load_jsonl(filepath: str) -> List[Dict]:
    """Load all records from JSONL file"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(filepath: str, records: List[Dict]):
    """Save records to JSONL file"""
    with open(filepath, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')


def setup_environment(split: str = "train", max_steps: int = 500, seed: int = 42) -> Env:
    """Initialize Habitat environment"""
    print("Setting up Habitat environment...")
    
    eval_dir = Path(__file__).parent.parent
    config_path = eval_dir / "vlnce_baselines/config/r2r_baselines/navila.yaml"
    
    original_cwd = os.getcwd()
    os.chdir(eval_dir)
    
    try:
        config = get_config(str(config_path), [
            "TASK_CONFIG.DATASET.SPLIT", split,
            "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", str(max_steps),
            "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE", "False",
            "TASK_CONFIG.SEED", str(seed),
        ])
        
        config.defrost()
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
        config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
        
        if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")
        
        config.freeze()
        
        env = Env(config=config.TASK_CONFIG)
        
        print(f"✓ Environment ready with {len(env.episodes)} episodes")
        return env
    
    finally:
        os.chdir(original_cwd)


def rollout_episode(env: Env, episode: Any, oracle_mode: str = "reference_path", max_steps: int = 500) -> Optional[Dict]:
    """
    Execute oracle rollout for a single episode.
    """
    try:
        # CRITICAL: Use the correct method to load specific episode
        env._episode_iterator._iterator = iter([episode])
        observations = env.reset()
        
        # Verify correct episode loaded
        if int(env.current_episode.episode_id) != int(episode.episode_id):
            print(f"  ERROR: Episode mismatch! Requested {episode.episode_id}, got {env.current_episode.episode_id}")
            return None
            
    except Exception as e:
        if "does not correspond to any existing file" in str(e) or "not found" in str(e).lower():
            return None
        raise
    
    # Initialize waypoint follower for reference_path mode
    waypoint_follower = None
    actual_mode = oracle_mode
    
    if oracle_mode == "reference_path":
        reference_path = getattr(episode, 'reference_path', None)
        if reference_path and len(reference_path) > 0:
            waypoint_follower = WaypointFollower(env, threshold=0.5)
            waypoint_follower.reset(reference_path)
        else:
            actual_mode = "goal"  # Fallback
    
    # Collect trajectory
    poses = []
    actions = []
    
    # Initial pose
    state = env.sim.get_agent_state()
    pos = state.position
    x, y = float(pos[0]), float(pos[2])
    yaw = float(quaternion_to_yaw(state.rotation))
    poses.append((x, y, yaw))
    
    step = 0
    
    while not env.episode_over and step < max_steps:
        # Get oracle action
        if actual_mode == "goal":
            oracle_action = int(observations["shortest_path_sensor"][0])
        else:
            current_pos = np.array([pos[0], pos[1], pos[2]])
            oracle_action = waypoint_follower.get_next_action(observations, current_pos)
        
        actions.append(oracle_action)
        
        # Execute action
        observations = env.step(oracle_action)
        
        # Record new pose
        state = env.sim.get_agent_state()
        pos = state.position
        x, y = float(pos[0]), float(pos[2])
        yaw = float(quaternion_to_yaw(state.rotation))
        poses.append((x, y, yaw))
        
        step += 1
    
    # Compute deltas
    deltas = []
    for i in range(len(poses) - 1):
        x1, y1, yaw1 = poses[i]
        x2, y2, yaw2 = poses[i + 1]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        dyaw = float(np.arctan2(np.sin(yaw2 - yaw1), np.cos(yaw2 - yaw1)))
        deltas.append((dx, dy, dyaw))
    
    # Get final metrics
    metrics = env.get_metrics()
    
    return {
        'poses': poses,
        'deltas': deltas,
        'actions': actions,
        'num_steps': step,
        'success': bool(metrics.get("success", 0) > 0),
        'spl': float(metrics.get("spl", 0)),
        'path_length': float(metrics.get("path_length", 0)),
        'distance_to_goal': float(metrics.get("distance_to_goal", 0)),
        'oracle_mode': actual_mode,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-run failed episodes from oracle deltas export")
    parser.add_argument("jsonl_file", type=str, help="Path to JSONL file")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization for re-run episodes")
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_file)
    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        return
    
    print("=" * 80)
    print("RE-RUN FAILED EPISODES")
    print("=" * 80)
    print(f"Input file: {jsonl_path}")
    print()
    
    # Load existing records
    print("Loading existing records...")
    records = load_jsonl(jsonl_path)
    print(f"  Total records: {len(records)}")
    
    # Find failed episodes
    failed_records = [(i, r) for i, r in enumerate(records) if not r.get('success', True)]
    failed_episode_ids = [r['episode_id'] for _, r in failed_records]
    
    print(f"  Failed episodes: {len(failed_records)}")
    
    if not failed_records:
        print("✅ No failed episodes to re-run!")
        return
    
    print(f"\nFailed episode IDs: {failed_episode_ids[:20]}{'...' if len(failed_episode_ids) > 20 else ''}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would re-run these episodes and update the file.")
        return
    
    # Get the split from the first record
    split = records[0].get('split', 'train')
    oracle_mode = records[0].get('oracle_mode', 'reference_path')
    
    print(f"\nSplit: {split}")
    print(f"Oracle mode: {oracle_mode}")
    print()
    
    # Setup environment
    env = setup_environment(split=split, max_steps=args.max_steps)
    
    # Build episode lookup
    episode_lookup = {int(ep.episode_id): ep for ep in env.episodes}
    
    # Re-run failed episodes
    print(f"\nRe-running {len(failed_records)} failed episodes...")
    
    fixed_count = 0
    still_failed = []
    
    pbar = tqdm(failed_records, desc="Re-running", unit="ep")
    
    for idx, old_record in pbar:
        episode_id = old_record['episode_id']
        
        # Find episode in environment
        episode = episode_lookup.get(episode_id)
        if episode is None:
            print(f"\n  ⚠️  Episode {episode_id} not found in environment!")
            still_failed.append(episode_id)
            continue
        
        # Re-run episode
        rollout = rollout_episode(env, episode, oracle_mode=oracle_mode, max_steps=args.max_steps)
        
        if rollout is None:
            print(f"\n  ⚠️  Episode {episode_id} rollout failed!")
            still_failed.append(episode_id)
            continue
        
        # Update the record
        new_record = old_record.copy()
        new_record['poses'] = rollout['poses']
        new_record['deltas'] = rollout['deltas']
        new_record['actions'] = rollout['actions']
        new_record['num_steps'] = rollout['num_steps']
        new_record['success'] = rollout['success']
        new_record['spl'] = rollout['spl']
        new_record['path_length'] = rollout['path_length']
        new_record['distance_to_goal'] = rollout['distance_to_goal']
        new_record['rerun'] = True  # Mark as re-run
        
        records[idx] = new_record
        
        if rollout['success']:
            fixed_count += 1
            pbar.set_postfix({'fixed': fixed_count, 'still_failed': len(still_failed)})
        else:
            still_failed.append(episode_id)
            pbar.set_postfix({'fixed': fixed_count, 'still_failed': len(still_failed)})
    
    pbar.close()
    env.close()
    
    # Save updated records
    print(f"\nSaving updated records to: {jsonl_path}")
    
    # Backup original file
    backup_path = jsonl_path.with_suffix('.jsonl.backup')
    print(f"  Backing up original to: {backup_path}")
    import shutil
    shutil.copy(jsonl_path, backup_path)
    
    # Save new file
    save_jsonl(jsonl_path, records)
    
    # Summary
    print()
    print("=" * 80)
    print("RE-RUN COMPLETE")
    print("=" * 80)
    print(f"Total failed episodes: {len(failed_records)}")
    print(f"Fixed (now success): {fixed_count}")
    print(f"Still failing: {len(still_failed)}")
    
    if still_failed:
        print(f"\nStill failing episode IDs: {still_failed[:30]}{'...' if len(still_failed) > 30 else ''}")
    
    # Verify final stats
    print("\nVerifying updated file...")
    updated_records = load_jsonl(jsonl_path)
    success_count = sum(1 for r in updated_records if r.get('success', False))
    fail_count = len(updated_records) - success_count
    
    print(f"  Total records: {len(updated_records)}")
    print(f"  Success: {success_count} ({100*success_count/len(updated_records):.1f}%)")
    print(f"  Failed: {fail_count} ({100*fail_count/len(updated_records):.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
