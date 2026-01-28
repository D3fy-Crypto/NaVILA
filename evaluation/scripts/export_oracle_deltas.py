#!/usr/bin/env python3
"""
Export oracle navigation deltas (SE(2) poses) from VLN-CE Habitat episodes.

Outputs per-episode pose trajectories and SE(2) deltas in JSONL format,
ready for GRU training on motion prediction.

Format per episode:
  {
    "episode_id": int,
    "scene_id": str,
    "split": str,
    "poses": [(x, y, yaw), ...],      # length S+1 (start + all steps)
    "deltas": [(dx, dy, dyaw), ...],  # length S (computed from consecutive poses)
    "actions": [0, 1, 2, ...],        # length S (oracle actions)
    "num_steps": int,
    "start_position": [x, y, z],
    "goal_position": [x, y, z],
    "success": bool,
    "spl": float,
    "path_length": float,
  }
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

# Add evaluation directory to path
eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, eval_dir)

from habitat import Env
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import quaternion_to_yaw


def wrap_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def export_oracle_deltas(
    config_path="vlnce_baselines/config/r2r_baselines/navila.yaml",
    split="train",
    max_steps=500,
    output_dir="oracle_exports",
    max_episodes=None,
    resume=True,
):
    """
    Export oracle navigation deltas per episode to JSONL.
    
    Args:
        config_path: Path to config file
        split: Dataset split (train/val_seen/val_unseen)
        max_steps: Maximum steps per episode
        output_dir: Directory to save JSONL exports
        max_episodes: Maximum number of episodes to export (None=all)
        resume: Skip already-exported episodes
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"oracle_deltas_{split}.jsonl")
    
    # Load already-exported episode IDs if resuming
    exported_ids = set()
    if resume and os.path.exists(output_file):
        print(f"[*] Resume mode: loading existing exports from {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    exported_ids.add(rec["episode_id"])
                except:
                    pass
        print(f"[*] Already exported {len(exported_ids)} episodes")
    
    print("=" * 80)
    print(f"Oracle Delta Exporter - Split: {split}")
    print("=" * 80)
    
    # Load config
    config = get_config(config_path, [
        "TASK_CONFIG.DATASET.SPLIT", split,
        "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", str(max_steps),
        "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE", "False",
        "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS", "-1",
    ])
    
    # Set chunking parameters
    config.defrost()
    config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
    config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
    
    # Enable sensors
    if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
        config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")
    
    config.freeze()
    
    # Create environment
    print("\n[1] Creating environment...")
    env = Env(config=config.TASK_CONFIG)
    total_episodes = len(env.episodes)
    print(f"[*] Total episodes in dataset: {total_episodes}")
    
    # Statistics
    stats = defaultdict(lambda: {"count": 0, "translations": [], "rotations": []})
    exported_count = 0
    skipped_count = 0
    
    print(f"\n[2] Starting export (max episodes: {max_episodes or 'all'})...")
    print("-" * 80)
    
    # Reset and run episodes
    observations = env.reset()
    
    while len(env.episodes) > 0 and (max_episodes is None or exported_count < max_episodes):
        episode = env.current_episode
        episode_id = int(episode.episode_id)
        
        # Skip if already exported
        if resume and episode_id in exported_ids:
            skipped_count += 1
            observations = env.reset()
            continue
        
        print(f"Episode {episode_id}: ", end="", flush=True)
        
        # Store poses and actions
        poses = []
        actions = []
        
        # Get initial pose (right after reset)
        st = env.sim.get_agent_state()
        pos = st.position
        x, y = float(pos[0]), float(pos[2])
        yaw = float(quaternion_to_yaw(st.rotation))
        poses.append((x, y, yaw))
        
        start_pos = episode.start_position.copy()
        goal_pos = episode.goals[0].position.copy()
        
        # Run episode
        step = 0
        while not env.episode_over and step < max_steps:
            # Get oracle action
            oracle_action = int(observations["shortest_path_sensor"][0])
            actions.append(oracle_action)
            
            # Execute action
            observations = env.step(oracle_action)
            
            # Get new pose
            st = env.sim.get_agent_state()
            pos = st.position
            x, y = float(pos[0]), float(pos[2])
            yaw = float(quaternion_to_yaw(st.rotation))
            poses.append((x, y, yaw))
            
            step += 1
        
        # Compute deltas
        deltas = []
        for i in range(len(poses) - 1):
            x1, y1, yaw1 = poses[i]
            x2, y2, yaw2 = poses[i + 1]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            dyaw = wrap_pi(yaw2 - yaw1)
            deltas.append((dx, dy, dyaw))
            
            # Collect statistics by action
            action = actions[i]
            action_name = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"][action]
            stats[action_name]["count"] += 1
            stats[action_name]["translations"].append(np.sqrt(dx**2 + dy**2))
            stats[action_name]["rotations"].append(abs(dyaw))
        
        # Get final metrics
        final_metrics = env.get_metrics()
        
        # Create export record
        record = {
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "split": split,
            "poses": poses,
            "deltas": deltas,
            "actions": actions,
            "num_steps": step,
            "start_position": list(start_pos),
            "goal_position": list(goal_pos),
            "success": bool(final_metrics.get("success", 0) > 0),
            "spl": float(final_metrics.get("spl", 0)),
            "path_length": float(final_metrics.get("path_length", 0)),
            "distance_to_goal": float(final_metrics.get("distance_to_goal", 0)),
        }
        
        # Write to JSONL
        with open(output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        success_str = "✓" if record["success"] else "✗"
        print(f"{success_str} steps={step:3d} spl={record['spl']:.3f} path={record['path_length']:.2f}m")
        
        exported_count += 1
        observations = env.reset()
    
    print("-" * 80)
    
    # Print statistics
    print(f"\n[3] Export Complete!")
    print(f"  Total exported: {exported_count}")
    print(f"  Total skipped (already exported): {skipped_count}")
    print(f"  Output file: {output_file}")
    
    print(f"\n[4] Action→Delta Statistics:")
    print("-" * 80)
    for action_name in ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]:
        if action_name in stats and stats[action_name]["count"] > 0:
            s = stats[action_name]
            trans = np.array(s["translations"])
            rots = np.array(s["rotations"])
            
            print(f"\n{action_name} ({s['count']} steps):")
            print(f"  Translation: mean={trans.mean():.4f}m, "
                  f"std={trans.std():.4f}m, "
                  f"min={trans.min():.4f}m, max={trans.max():.4f}m")
            print(f"  Rotation:    mean={np.degrees(rots.mean()):.2f}°, "
                  f"std={np.degrees(rots.std()):.2f}°, "
                  f"min={np.degrees(rots.min()):.2f}°, max={np.degrees(rots.max()):.2f}°")
    
    print("\n" + "=" * 80)
    print("Export Complete!")
    print("=" * 80)
    
    env.close()
    
    return output_file


def load_and_validate_exports(output_file):
    """
    Load and validate exported deltas.
    Checks for anomalies in pose/delta alignment.
    """
    print(f"\n[*] Validating exports from: {output_file}")
    
    issues = []
    total_episodes = 0
    
    with open(output_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                issues.append(f"Line {line_no}: Invalid JSON - {e}")
                continue
            
            total_episodes += 1
            episode_id = rec.get("episode_id")
            poses = rec.get("poses", [])
            deltas = rec.get("deltas", [])
            actions = rec.get("actions", [])
            
            # Check lengths
            if len(poses) != len(deltas) + 1:
                issues.append(
                    f"Episode {episode_id}: len(poses)={len(poses)} != "
                    f"len(deltas)+1={len(deltas)+1}"
                )
            
            if len(actions) != len(deltas):
                issues.append(
                    f"Episode {episode_id}: len(actions)={len(actions)} != "
                    f"len(deltas)={len(deltas)}"
                )
            
            # Check delta computation
            for i, (delta, pose1, pose2) in enumerate(zip(deltas, poses[:-1], poses[1:])):
                x1, y1, yaw1 = pose1
                x2, y2, yaw2 = pose2
                expected_dx = x2 - x1
                expected_dy = y2 - y1
                expected_dyaw = wrap_pi(yaw2 - yaw1)
                
                dx, dy, dyaw = delta
                
                tol = 1e-5
                if not (np.isclose(dx, expected_dx, atol=tol) and 
                        np.isclose(dy, expected_dy, atol=tol) and
                        np.isclose(dyaw, expected_dyaw, atol=tol)):
                    issues.append(
                        f"Episode {episode_id}, step {i}: delta mismatch. "
                        f"Expected ({expected_dx:.4f}, {expected_dy:.4f}, {expected_dyaw:.4f}), "
                        f"got ({dx:.4f}, {dy:.4f}, {dyaw:.4f})"
                    )
    
    print(f"[*] Validated {total_episodes} episodes")
    if issues:
        print(f"[!] Found {len(issues)} issues:")
        for issue in issues[:10]:  # Print first 10
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print("[✓] All exports valid!")
    
    return total_episodes, issues


def main():
    parser = argparse.ArgumentParser(
        description="Export oracle navigation deltas for GRU training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vlnce_baselines/config/r2r_baselines/navila.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val_seen", "val_unseen", "test"],
        help="Dataset split to export"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="oracle_exports",
        help="Directory to save JSONL exports"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to export (default: all)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume; overwrite existing exports"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exports and exit (don't export)"
    )
    
    args = parser.parse_args()
    
    # Change to evaluation directory
    os.chdir(eval_dir)
    
    if args.validate:
        # Just validate
        output_file = os.path.join(
            args.output_dir,
            f"oracle_deltas_{args.split}.jsonl"
        )
        if not os.path.exists(output_file):
            print(f"[!] File not found: {output_file}")
            return
        load_and_validate_exports(output_file)
    else:
        # Run export
        output_file = export_oracle_deltas(
            config_path=args.config,
            split=args.split,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            max_episodes=args.max_episodes,
            resume=not args.no_resume,
        )
        
        # Validate after export
        print("\n[*] Running validation...")
        load_and_validate_exports(output_file)


if __name__ == "__main__":
    main()
