#!/usr/bin/env python3
"""
Visualize agent following oracle shortest path with 2D top-down map view.
This script shows both first-person view and bird's-eye view of the navigation.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import imageio
import random

# Add evaluation directory to path
eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, eval_dir)

from habitat import Env
from habitat.utils.visualizations.utils import append_text_to_image
from habitat.utils.visualizations import maps as habitat_maps
from vlnce_baselines.config.default import get_config
from habitat_extensions import maps


def visualize_oracle_with_map(
    config_path="vlnce_baselines/config/r2r_baselines/navila.yaml",
    split="train",
    max_steps=500,
    output_dir="oracle_visualization",
    save_video=True,
    episode_index=0,
    episode_id=None,
):
    """
    Run oracle navigation and visualize with both first-person and top-down map views.
    
    Args:
        config_path: Path to config file
        split: Dataset split (train/val_seen/val_unseen)
        max_steps: Maximum steps per episode
        output_dir: Directory to save outputs
        save_video: Whether to save video
        episode_index: Index of episode to run (for display purposes)
        episode_id: Specific episode ID to visualize (if None, uses sequential)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"Oracle Path Visualization with 2D Map - Run {episode_index}")
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
    
    # Enable top-down map
    if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
    
    config.freeze()
    
    # Create environment
    print(f"\n[1] Creating environment...")
    env = Env(config=config.TASK_CONFIG)
    
    # If specific episode_id is provided, find and load it
    if episode_id is not None:
        print(f"[2] Searching for episode ID: {episode_id}...")
        found = False
        max_search = len(env.episodes) if hasattr(env, 'episodes') else 10000
        actual_search = min(max_search, 10000)  # Cap at 10k to avoid infinite search
        
        for i in range(actual_search):
            observations = env.reset()
            current_ep_id = env.current_episode.episode_id
            if current_ep_id == episode_id:
                print(f"    Found episode {episode_id} at index {i}")
                found = True
                break
            if (i + 1) % 100 == 0:
                print(f"    Searched {i+1}/{actual_search} episodes...")
        
        if not found:
            print(f"\n    ERROR: Episode {episode_id} not found in dataset!")
            print(f"    Searched {actual_search} episodes in split '{split}'")
            print(f"    This episode may not exist in this split.")
            print(f"    Skipping this episode...\n")
            env.close()
            return None  # Return None to indicate failure
    else:
        # Just reset once for sequential mode
        print(f"[2] Loading episode {episode_index}...")
        for i in range(episode_index + 1):
            observations = env.reset()
    
    print(f"[3] Starting visualization...")
    
    # Get episode info
    episode = env.current_episode
    print(f"\n[Episode Info]")
    print(f"  Episode ID: {episode.episode_id}")
    print(f"  Scene: {episode.scene_id.split('/')[-1]}")
    print(f"  Start Position: {episode.start_position}")
    print(f"  Goal Position: {episode.goals[0].position}")
    print(f"  Instruction: {episode.instruction.instruction_text[:100]}")
    
    # Action names
    action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    
    # Storage
    frames = []
    path_positions = []  # Store agent positions for visualization
    
    # Run episode
    print(f"\n[4] Running oracle navigation...")
    print("-" * 80)
    
    step = 0
    while not env.episode_over and step < max_steps:
        # Get oracle action
        oracle_action = int(observations["shortest_path_sensor"][0])
        
        # Get metrics
        current_metrics = env.get_metrics()
        distance_to_goal = current_metrics.get('distance_to_goal', -1)
        
        # Store agent position for path visualization
        agent_pos = env.sim.get_agent_state().position
        path_positions.append(agent_pos.copy())
        
        # Log
        print(f"Step {step:3d}: Action={action_names[oracle_action]:12s} | "
              f"Distance={distance_to_goal:.2f}m | Pos={agent_pos[:2]}")
        
        # Create visualization frame with map
        frame = create_combined_visualization(
            observations,
            episode,
            current_metrics,
            step,
            oracle_action,
            action_names[oracle_action],
            distance_to_goal,
            path_positions,
            env.sim,
        )
        frames.append(frame)
        
        # Execute action
        observations = env.step(oracle_action)
        step += 1
    
    print("-" * 80)
    final_metrics = env.get_metrics()
    
    print(f"\n[5] Episode Complete!")
    print(f"\n[Final Metrics]")
    print(f"  Total Steps: {step}")
    print(f"  Success: {final_metrics.get('success', 0) > 0}")
    print(f"  SPL: {final_metrics.get('spl', 0):.3f}")
    print(f"  Distance to Goal: {final_metrics.get('distance_to_goal', 0):.2f}m")
    print(f"  Path Length: {final_metrics.get('path_length', 0):.2f}m")
    
    # Save video with unique filename including episode_index
    if save_video and frames:
        video_path = os.path.join(output_dir, f"oracle_map_ep{episode_index:03d}_id{episode.episode_id}.mp4")
        print(f"\n[6] Saving video to: {video_path}")
        imageio.mimsave(video_path, frames, fps=10)
        print(f"  Saved {len(frames)} frames at 10 FPS")
    
    # Save frames with unique filenames
    if frames:
        first_frame_path = os.path.join(output_dir, f"map_ep{episode_index:03d}_id{episode.episode_id}_start.png")
        last_frame_path = os.path.join(output_dir, f"map_ep{episode_index:03d}_id{episode.episode_id}_end.png")
        Image.fromarray(frames[0]).save(first_frame_path)
        Image.fromarray(frames[-1]).save(last_frame_path)
        print(f"  Saved start frame: {first_frame_path}")
        print(f"  Saved end frame: {last_frame_path}")
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, f"ep{episode_index:03d}_id{episode.episode_id}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Episode Index: {episode_index}\n")
        f.write(f"Episode ID: {episode.episode_id}\n")
        f.write(f"Scene: {episode.scene_id.split('/')[-1]}\n")
        f.write(f"Instruction: {episode.instruction.instruction_text}\n")
        f.write(f"\nFinal Metrics:\n")
        f.write(f"  Total Steps: {step}\n")
        f.write(f"  Success: {final_metrics.get('success', 0) > 0}\n")
        f.write(f"  SPL: {final_metrics.get('spl', 0):.3f}\n")
        f.write(f"  Distance to Goal: {final_metrics.get('distance_to_goal', 0):.2f}m\n")
        f.write(f"  Path Length: {final_metrics.get('path_length', 0):.2f}m\n")
    print(f"  Saved metrics: {metrics_path}")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    
    env.close()
    return True  # Return True to indicate success


def create_combined_visualization(
    observations,
    episode,
    metrics,
    step,
    action,
    action_name,
    distance,
    path_positions,
    sim,
):
    """
    Create visualization with first-person view (top) and 2D map view (bottom).
    """
    
    # Get RGB observation
    rgb = observations["rgb"]
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        rgb = (rgb * 255).astype(np.uint8)
    
    # Resize RGB
    target_width = 640
    aspect_ratio = rgb.shape[0] / rgb.shape[1]
    target_height = int(target_width * aspect_ratio)
    rgb_resized = Image.fromarray(rgb).resize((target_width, target_height), Image.LANCZOS)
    rgb_array = np.array(rgb_resized)
    
    # Get top-down map
    if "top_down_map_vlnce" in metrics:
        map_data = metrics["top_down_map_vlnce"]
        td_map = map_data["map"].copy()
        agent_map_coord = map_data["agent_map_coord"]
        agent_angle = map_data["agent_angle"]
        
        # Colorize map
        td_map_colored = maps.colorize_topdown_map(
            td_map,
            map_data.get("fog_of_war_mask", np.ones_like(td_map)),
            fog_of_war_desat_amount=0.75,
        )
        
        # Draw agent on map
        td_map_colored = habitat_maps.draw_agent(
            image=td_map_colored,
            agent_center_coord=agent_map_coord,
            agent_rotation=agent_angle,
            agent_radius_px=max(5, min(td_map_colored.shape[0:2]) // 24),
        )
        
        # Draw path taken so far
        td_map_colored = draw_path_on_map(
            td_map_colored,
            path_positions,
            agent_map_coord,
            episode.start_position,
            episode.goals[0].position,
            sim,
        )
        
        # Resize map to match RGB width
        map_height = target_height
        old_h, old_w = td_map_colored.shape[:2]
        map_width = int(float(map_height) / old_h * old_w)
        td_map_colored = cv2.resize(
            td_map_colored,
            (map_width, map_height),
            interpolation=cv2.INTER_CUBIC,
        )
        
        # Combine: RGB on top, map on bottom
        # Make them same width by padding
        max_width = max(rgb_array.shape[1], td_map_colored.shape[1])
        
        rgb_padded = pad_image(rgb_array, max_width, 3)
        map_padded = pad_image(td_map_colored, max_width, 3)
        
        combined = np.vstack([rgb_padded, map_padded])
    else:
        combined = rgb_array
    
    # Add text overlay
    instruction = episode.instruction.instruction_text
    if len(instruction) > 80:
        instruction = instruction[:77] + "..."
    
    text_lines = "\n".join([
        f"Episode: {episode.episode_id} | Step: {step}",
        f"Action: {action_name} | Distance: {distance:.2f}m",
        f"Instruction: {instruction}",
    ])
    
    combined_with_text = append_text_to_image(
        combined,
        text_lines,
    )
    
    return combined_with_text


def draw_path_on_map(map_image, path_positions, agent_map_coord, start_pos, goal_pos, sim):
    """
    Draw the agent's path on the top-down map.
    """
    from habitat.utils.visualizations import maps as habitat_maps
    
    map_copy = map_image.copy()
    
    if len(path_positions) < 2:
        return map_copy
    
    # Convert 3D positions to 2D map coordinates
    path_coords = []
    for pos in path_positions:
        # Project 3D position to 2D map
        map_coord = habitat_maps.to_grid(
            pos[2],  # z
            pos[0],  # x
            map_copy.shape[0:2],
            sim,
        )
        path_coords.append(map_coord)
    
    # Draw path line
    if len(path_coords) > 1:
        for i in range(len(path_coords) - 1):
            pt1 = tuple(path_coords[i][::-1])  # Convert to (x, y)
            pt2 = tuple(path_coords[i + 1][::-1])
            try:
                cv2.line(
                    map_copy,
                    pt1,
                    pt2,
                    color=(100, 200, 100),  # Green
                    thickness=2,
                )
            except:
                pass
    
    # Draw start point
    try:
        start_map_coord = habitat_maps.to_grid(
            start_pos[2],
            start_pos[0],
            map_copy.shape[0:2],
            sim,
        )
        cv2.circle(
            map_copy,
            tuple(start_map_coord[::-1]),
            radius=5,
            color=(0, 255, 0),  # Green
            thickness=-1,
        )
    except:
        pass
    
    # Draw goal point
    try:
        goal_map_coord = habitat_maps.to_grid(
            goal_pos[2],
            goal_pos[0],
            map_copy.shape[0:2],
            sim,
        )
        cv2.circle(
            map_copy,
            tuple(goal_map_coord[::-1]),
            radius=5,
            color=(0, 0, 255),  # Red
            thickness=-1,
        )
    except:
        pass
    
    return map_copy


def pad_image(img, target_width, channels):
    """Pad image to target width."""
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    
    pad_width = target_width - w
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    padded = np.pad(
        img,
        ((0, 0), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0,
    )
    return padded


# Import cv2 for drawing functions
try:
    import cv2
except ImportError:
    print("Warning: cv2 not available, some visualization features will be limited")
    cv2 = None


def main():
    parser = argparse.ArgumentParser(description="Visualize oracle path with 2D map view")
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
        help="Dataset split to use"
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
        default="oracle_visualization",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip saving video (only save images)"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Select random episodes instead of sequential"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for episode selection (if --random is used)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Change to evaluation directory
    os.chdir(eval_dir)
    
    # Get list of available episode IDs if using random mode
    episode_ids_to_process = None
    if args.random:
        print(f"\n{'='*80}")
        print(f"Collecting episode IDs from dataset...")
        print(f"{'='*80}\n")
        
        # Load config to get dataset
        config = get_config(args.config, [
            "TASK_CONFIG.DATASET.SPLIT", args.split,
        ])
        config.defrost()
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
        config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
        config.freeze()
        
        # Create temporary environment to get episode list
        temp_env = Env(config=config.TASK_CONFIG)
        all_episode_ids = []
        
        # Collect episode IDs (limit to reasonable number)
        max_collect = min(1000, len(temp_env.episodes))
        print(f"Collecting up to {max_collect} episode IDs...")
        for i in range(max_collect):
            temp_env.reset()
            all_episode_ids.append(temp_env.current_episode.episode_id)
            if (i + 1) % 100 == 0:
                print(f"  Collected {i+1} episode IDs...")
        
        temp_env.close()
        
        print(f"Total episodes collected: {len(all_episode_ids)}")
        
        # Randomly sample episodes
        if args.num_episodes <= len(all_episode_ids):
            episode_ids_to_process = random.sample(all_episode_ids, args.num_episodes)
        else:
            print(f"Warning: Requested {args.num_episodes} episodes but only {len(all_episode_ids)} available")
            episode_ids_to_process = all_episode_ids
        
        print(f"Selected {len(episode_ids_to_process)} random episodes: {episode_ids_to_process[:5]}{'...' if len(episode_ids_to_process) > 5 else ''}")
    
    # Run visualization for multiple episodes
    successful_runs = 0
    skipped_runs = 0
    
    for i in range(args.num_episodes):
        print(f"\n{'='*80}")
        print(f"Running Episode {i+1}/{args.num_episodes}")
        print(f"{'='*80}\n")
        
        # Determine episode ID to use
        episode_id = episode_ids_to_process[i] if episode_ids_to_process else None
        
        result = visualize_oracle_with_map(
            config_path=args.config,
            split=args.split,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            save_video=not args.no_video,
            episode_index=i,
            episode_id=episode_id,
        )
        
        if result is None:
            skipped_runs += 1
            print(f"Skipped episode {i+1}/{args.num_episodes} (not found in dataset)")
        else:
            successful_runs += 1
        
        if args.num_episodes > 1 and i < args.num_episodes - 1:
            print(f"\nCompleted episode {i+1}/{args.num_episodes}, continuing to next episode...\n")
    
    print(f"\n{'='*80}")
    print(f"Summary: {successful_runs} successful, {skipped_runs} skipped")
    print(f"{'='*80}")
    print("All episodes completed!")


if __name__ == "__main__":
    main()
