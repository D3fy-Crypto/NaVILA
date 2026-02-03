#!/usr/bin/env python3
"""
DEBUG VERSION: Export oracle navigation trajectories with extensive logging.

This version has print statements EVERYWHERE to debug episode loading issues.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from PIL import Image
import imageio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat import Env
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import quaternion_to_yaw
from habitat.utils.visualizations.utils import append_text_to_image
from habitat.utils.visualizations import maps as habitat_maps
from habitat_extensions import maps

try:
    import cv2
except ImportError:
    cv2 = None


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def setup_environment(split: str, max_steps: int, visualize: bool) -> Env:
    """Initialize Habitat environment"""
    print("\n" + "="*80)
    print("SETTING UP ENVIRONMENT")
    print("="*80)
    
    eval_dir = Path(__file__).parent.parent
    config_path = eval_dir / "vlnce_baselines/config/r2r_baselines/navila.yaml"
    
    original_cwd = os.getcwd()
    os.chdir(eval_dir)
    
    print(f"  eval_dir: {eval_dir}")
    print(f"  config_path: {config_path}")
    print(f"  split: {split}")
    
    try:
        config = get_config(str(config_path), [
            "TASK_CONFIG.DATASET.SPLIT", split,
            "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", str(max_steps),
            "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE", "False",
            "TASK_CONFIG.SEED", "42",
        ])
        
        config.defrost()
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
        config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
        
        if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")
        
        if visualize:
            if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        
        config.freeze()
        
        env = Env(config=config.TASK_CONFIG)
        
        print(f"  ✓ Environment created")
        print(f"  Total episodes in dataset: {len(env.episodes)}")
        
        # Print first 5 episodes to verify order
        print(f"\n  First 5 episodes in env.episodes:")
        for i in range(min(5, len(env.episodes))):
            ep = env.episodes[i]
            print(f"    [{i}] episode_id={ep.episode_id}, scene={ep.scene_id.split('/')[-1]}")
        
        return env
    
    finally:
        os.chdir(original_cwd)


def rollout_single_episode(env: Env, episode: Any, max_steps: int, visualize: bool) -> Optional[Dict]:
    """
    Execute oracle rollout for ONE episode with extensive debugging.
    """
    episode_id = int(episode.episode_id)
    scene_id = str(episode.scene_id)
    
    print("\n" + "-"*80)
    print(f"ROLLOUT START: Episode {episode_id}")
    print("-"*80)
    print(f"  Target episode_id: {episode_id}")
    print(f"  Target scene: {scene_id}")
    print(f"  Target start_position: {list(episode.start_position)}")
    
    # =========================================================================
    # STEP 1: Set current_episode and reset
    # =========================================================================
    print(f"\n  [STEP 1] Setting env.current_episode = episode")
    print(f"    Before: env.current_episode = {env.current_episode.episode_id if env.current_episode else 'None'}")
    
    env.current_episode = episode
    
    print(f"    After assignment: env.current_episode = {env.current_episode.episode_id}")
    
    print(f"\n  [STEP 2] Calling env.reset()")
    observations = env.reset()
    
    # =========================================================================
    # STEP 2: Verify what was actually loaded
    # =========================================================================
    print(f"\n  [STEP 3] Verifying loaded episode")
    actual_episode = env.current_episode
    actual_episode_id = int(actual_episode.episode_id)
    actual_scene = str(actual_episode.scene_id)
    
    print(f"    Actual episode_id after reset: {actual_episode_id}")
    print(f"    Actual scene after reset: {actual_scene}")
    
    # Get actual agent position
    agent_state = env.sim.get_agent_state()
    actual_position = list(agent_state.position)
    print(f"    Actual agent position: {actual_position}")
    
    # Compare
    expected_position = list(episode.start_position)
    position_match = np.allclose(actual_position, expected_position, atol=0.1)
    
    print(f"\n  [VERIFICATION]")
    print(f"    Episode ID match: {actual_episode_id == episode_id} (expected {episode_id}, got {actual_episode_id})")
    print(f"    Scene match: {actual_scene == scene_id}")
    print(f"    Position match: {position_match}")
    print(f"      Expected: {expected_position}")
    print(f"      Actual:   {actual_position}")
    
    if actual_episode_id != episode_id:
        print(f"\n  ❌ CRITICAL ERROR: Episode ID mismatch!")
        print(f"     The environment loaded episode {actual_episode_id} instead of {episode_id}")
        return None
    
    if not position_match:
        print(f"\n  ⚠ WARNING: Position mismatch (might be normal for some episodes)")
    
    # =========================================================================
    # STEP 3: Execute rollout
    # =========================================================================
    print(f"\n  [STEP 4] Starting rollout execution")
    
    poses = []
    actions = []
    frames = [] if visualize else None
    
    # Record initial pose
    pos = agent_state.position
    x, y = float(pos[0]), float(pos[2])
    yaw = float(quaternion_to_yaw(agent_state.rotation))
    poses.append((x, y, yaw))
    
    print(f"    Initial pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
    
    step = 0
    while not env.episode_over and step < max_steps:
        # Get oracle action (shortest path to goal)
        oracle_action = int(observations["shortest_path_sensor"][0])
        actions.append(oracle_action)
        
        # Create visualization frame if needed
        if visualize and frames is not None:
            metrics = env.get_metrics()
            if "top_down_map_vlnce" in metrics:
                frame = create_frame(observations, actual_episode, metrics, step, oracle_action)
                if frame is not None:
                    frames.append(frame)
        
        # Log first 3 steps
        if step < 3:
            action_names = ["STOP", "FORWARD", "LEFT", "RIGHT"]
            metrics = env.get_metrics()
            dist = metrics.get('distance_to_goal', -1)
            print(f"    Step {step}: action={action_names[oracle_action]}, dist_to_goal={dist:.2f}m, pos=({x:.2f}, {y:.2f})")
        
        # Execute action
        observations = env.step(oracle_action)
        
        # Record new pose
        state = env.sim.get_agent_state()
        pos = state.position
        x, y = float(pos[0]), float(pos[2])
        yaw = float(quaternion_to_yaw(state.rotation))
        poses.append((x, y, yaw))
        
        step += 1
        
        # Stop if STOP action
        if oracle_action == 0:
            break
    
    # =========================================================================
    # STEP 4: Compute results
    # =========================================================================
    print(f"\n  [STEP 5] Computing results")
    print(f"    Total steps: {step}")
    print(f"    Final pose: x={x:.2f}, y={y:.2f}")
    
    # Compute deltas
    deltas = []
    for i in range(len(poses) - 1):
        x1, y1, yaw1 = poses[i]
        x2, y2, yaw2 = poses[i + 1]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        dyaw = float(wrap_angle(yaw2 - yaw1))
        deltas.append((dx, dy, dyaw))
    
    # Get final metrics
    metrics = env.get_metrics()
    success = bool(metrics.get("success", 0) > 0)
    spl = float(metrics.get("spl", 0))
    distance_to_goal = float(metrics.get("distance_to_goal", 0))
    
    print(f"    Success: {success}")
    print(f"    SPL: {spl:.4f}")
    print(f"    Final distance to goal: {distance_to_goal:.2f}m")
    
    result = {
        'episode_id': episode_id,
        'scene_id': scene_id,
        'poses': poses,
        'deltas': deltas,
        'actions': actions,
        'num_steps': step,
        'success': success,
        'spl': spl,
        'distance_to_goal': distance_to_goal,
        'frames': frames,
    }
    
    print(f"\n  ✓ ROLLOUT COMPLETE for episode {episode_id}")
    print("-"*80)
    
    return result


def create_frame(observations: Dict, episode: Any, metrics: Dict, step: int, action: int) -> Optional[np.ndarray]:
    """Create a visualization frame"""
    try:
        rgb = observations.get("rgb")
        if rgb is None:
            return None
        
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = (rgb * 255).astype(np.uint8)
        
        # Resize RGB
        target_width = 640
        aspect_ratio = rgb.shape[0] / rgb.shape[1]
        target_height = int(target_width * aspect_ratio)
        rgb_resized = Image.fromarray(rgb).resize((target_width, target_height), Image.LANCZOS)
        rgb_array = np.array(rgb_resized)
        
        # Get top-down map
        map_data = metrics["top_down_map_vlnce"]
        td_map = map_data["map"].copy()
        agent_map_coord = map_data["agent_map_coord"]
        agent_angle = map_data["agent_angle"]
        
        td_map_colored = maps.colorize_topdown_map(
            td_map,
            map_data.get("fog_of_war_mask", np.ones_like(td_map)),
            fog_of_war_desat_amount=0.75,
        )
        
        td_map_colored = habitat_maps.draw_agent(
            image=td_map_colored,
            agent_center_coord=agent_map_coord,
            agent_rotation=agent_angle,
            agent_radius_px=max(5, min(td_map_colored.shape[0:2]) // 24),
        )
        
        # Resize map
        map_height = target_height
        old_h, old_w = td_map_colored.shape[:2]
        map_width = int(float(map_height) / old_h * old_w)
        if cv2:
            td_map_colored = cv2.resize(td_map_colored, (map_width, map_height), interpolation=cv2.INTER_CUBIC)
        
        # Combine
        max_width = max(rgb_array.shape[1], td_map_colored.shape[1])
        
        def pad_img(img, target_w):
            h, w = img.shape[:2]
            if w >= target_w:
                return img
            pad_left = (target_w - w) // 2
            pad_right = target_w - w - pad_left
            return np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
        
        rgb_padded = pad_img(rgb_array, max_width)
        map_padded = pad_img(td_map_colored, max_width)
        combined = np.vstack([rgb_padded, map_padded])
        
        # Add text
        action_names = ["STOP", "FORWARD", "LEFT", "RIGHT"]
        dist = metrics.get('distance_to_goal', -1)
        text = f"Episode: {episode.episode_id} | Step: {step} | Action: {action_names[action]} | Dist: {dist:.2f}m"
        combined_with_text = append_text_to_image(combined, text)
        
        return combined_with_text
    
    except Exception as e:
        print(f"      Warning: Frame creation failed: {e}")
        return None


def save_video(frames: List[np.ndarray], video_path: str, episode_id: int) -> bool:
    """Save frames as video"""
    print(f"\n  [SAVING VIDEO]")
    print(f"    Episode: {episode_id}")
    print(f"    Path: {video_path}")
    print(f"    Frames: {len(frames)}")
    
    if not frames:
        print(f"    ⚠ No frames to save!")
        return False
    
    try:
        frame_arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame_arrays.append(np.array(frame))
            elif isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                frame_arrays.append(frame)
        
        if frame_arrays:
            imageio.mimsave(video_path, frame_arrays, fps=5)
            print(f"    ✓ Video saved successfully")
            return True
        else:
            print(f"    ⚠ No valid frames!")
            return False
    
    except Exception as e:
        print(f"    ❌ Failed to save video: {e}")
        return False


def test_episodes(episode_ids: List[int], visualize: bool = True):
    """Test specific episodes with full debugging"""
    print("\n" + "="*80)
    print("DEBUG TEST: Testing specific episodes")
    print("="*80)
    print(f"Episodes to test: {episode_ids}")
    print(f"Visualize: {visualize}")
    
    # Setup
    env = setup_environment("train", max_steps=500, visualize=visualize)
    
    # Create output directory
    output_dir = Path("debug_visualizations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Build episode lookup
    print("\n  Building episode lookup...")
    episode_lookup = {}
    for ep in env.episodes:
        episode_lookup[int(ep.episode_id)] = ep
    print(f"  Found {len(episode_lookup)} episodes")
    
    # Test each episode
    for target_id in episode_ids:
        print("\n" + "="*80)
        print(f"TESTING EPISODE {target_id}")
        print("="*80)
        
        if target_id not in episode_lookup:
            print(f"  ❌ Episode {target_id} not found in dataset!")
            continue
        
        episode = episode_lookup[target_id]
        
        # Run rollout
        result = rollout_single_episode(env, episode, max_steps=500, visualize=visualize)
        
        if result is None:
            print(f"  ❌ Rollout failed for episode {target_id}")
            continue
        
        # Save video
        if visualize and result['frames']:
            video_path = str(output_dir / f"debug_ep_{target_id:05d}_success_{result['success']}.mp4")
            save_video(result['frames'], video_path, target_id)
        
        # Print summary
        print(f"\n  SUMMARY for episode {target_id}:")
        print(f"    Steps: {result['num_steps']}")
        print(f"    Success: {result['success']}")
        print(f"    Distance to goal: {result['distance_to_goal']:.2f}m")
        print(f"    First pose: {result['poses'][0]}")
        print(f"    Last pose: {result['poses'][-1]}")
    
    env.close()
    
    # List output files
    print("\n" + "="*80)
    print("OUTPUT FILES:")
    print("="*80)
    for f in sorted(output_dir.glob("debug_ep_*.mp4")):
        size = f.stat().st_size
        print(f"  {f.name}: {size/1024:.1f} KB")
    
    print("\n✓ DEBUG TEST COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Debug oracle export")
    parser.add_argument("--episodes", type=str, default="1,2,100",
                       help="Comma-separated episode IDs to test")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualization")
    args = parser.parse_args()
    
    # Parse episode IDs
    episode_ids = [int(x.strip()) for x in args.episodes.split(",")]
    
    print("\n" + "#"*80)
    print("# DEBUG ORACLE EXPORT")
    print("#"*80)
    print(f"# Episodes: {episode_ids}")
    print(f"# Visualize: {not args.no_visualize}")
    print("#"*80 + "\n")
    
    test_episodes(episode_ids, visualize=not args.no_visualize)


if __name__ == "__main__":
    main()
