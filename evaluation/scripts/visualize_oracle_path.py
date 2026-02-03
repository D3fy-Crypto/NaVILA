#!/usr/bin/env python3
"""
Visualize agent following oracle shortest path in NaVILA environment.
This script loads a single episode, follows the oracle path, and saves a video.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import imageio

# Add evaluation directory to path
eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, eval_dir)

from habitat import Env
from habitat.utils.visualizations.utils import append_text_to_image
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import observations_to_image


def visualize_oracle_navigation(
    config_path="vlnce_baselines/config/r2r_baselines/navila.yaml",
    split="train",
    episode_id=None,
    max_steps=500,
    output_dir="oracle_visualization",
    save_video=True,
    show_top_down_map=True,
):
    """
    Run oracle navigation and visualize the agent's path.
    
    Args:
        config_path: Path to config file
        split: Dataset split (train/val_seen/val_unseen)
        episode_id: Specific episode ID to run (None for first episode)
        max_steps: Maximum steps per episode
        output_dir: Directory to save outputs
        save_video: Whether to save video
        show_top_down_map: Whether to show top-down map
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Oracle Path Visualization")
    print("=" * 80)
    
    # Load config
    config = get_config(config_path, [
        "TASK_CONFIG.DATASET.SPLIT", split,
        "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", str(max_steps),
        "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE", "False",
        "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS", "-1",
    ])
    
    # Set chunking parameters (needed for dataset initialization)
    config.defrost()
    config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
    config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
    
    # Enable shortest path sensor
    if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
        config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")
    
    # Enable top-down map if requested
    if show_top_down_map and "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
    
    config.freeze()
    
    # Create environment
    print("\n[1] Creating environment...")
    env = Env(config=config.TASK_CONFIG)
    
    # Reset environment
    print("[2] Resetting environment...")
    observations = env.reset()
    
    # Get episode info
    episode = env.current_episode
    print(f"\n[Episode Info]")
    print(f"  Episode ID: {episode.episode_id}")
    print(f"  Scene: {episode.scene_id.split('/')[-1]}")
    print(f"  Start Position: {episode.start_position}")
    print(f"  Goal Position: {episode.goals[0].position}")
    print(f"  Instruction: {episode.instruction.instruction_text}")
    
    # Action names for logging
    action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
    
    # Storage for visualization
    frames = []
    metrics_history = []
    
    # Run episode
    print(f"\n[3] Running oracle navigation...")
    print("-" * 80)
    
    step = 0
    while not env.episode_over and step < max_steps:
        # Get oracle action from sensor
        oracle_action = int(observations["shortest_path_sensor"][0])
        
        # Get current metrics
        current_metrics = env.get_metrics()
        distance_to_goal = current_metrics.get('distance_to_goal', -1)
        
        # Log step
        print(f"Step {step:3d}: Action={action_names[oracle_action]:12s} | "
              f"Distance to Goal={distance_to_goal:.2f}m")
        
        # Create visualization frame
        frame = create_visualization_frame(
            observations, 
            episode,
            step,
            oracle_action,
            action_names[oracle_action],
            distance_to_goal,
            show_top_down_map and "top_down_map_vlnce" in current_metrics
        )
        frames.append(frame)
        
        # Store metrics
        metrics_history.append({
            'step': step,
            'action': oracle_action,
            'distance': distance_to_goal
        })
        
        # Execute action
        observations = env.step(oracle_action)
        step += 1
    
    # Get final metrics
    print("-" * 80)
    final_metrics = env.get_metrics()
    
    print(f"\n[4] Episode Complete!")
    print(f"\n[Final Metrics]")
    print(f"  Total Steps: {step}")
    print(f"  Success: {final_metrics.get('success', 0) > 0}")
    print(f"  SPL: {final_metrics.get('spl', 0):.3f}")
    print(f"  Distance to Goal: {final_metrics.get('distance_to_goal', 0):.2f}m")
    print(f"  Path Length: {final_metrics.get('path_length', 0):.2f}m")
    print(f"  Oracle Success: {final_metrics.get('oracle_success', 0):.3f}")
    print(f"  NDTW: {final_metrics.get('ndtw', 0):.3f}")
    
    # Save video
    if save_video and frames:
        video_path = os.path.join(output_dir, f"oracle_episode_{episode.episode_id}.mp4")
        print(f"\n[5] Saving video to: {video_path}")
        imageio.mimsave(video_path, frames, fps=10)
        print(f"  Saved {len(frames)} frames at 10 FPS")
    
    # Save first and last frames as images
    if frames:
        first_frame_path = os.path.join(output_dir, f"episode_{episode.episode_id}_start.png")
        last_frame_path = os.path.join(output_dir, f"episode_{episode.episode_id}_end.png")
        Image.fromarray(frames[0]).save(first_frame_path)
        Image.fromarray(frames[-1]).save(last_frame_path)
        print(f"  Saved start frame: {first_frame_path}")
        print(f"  Saved end frame: {last_frame_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"episode_{episode.episode_id}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Episode ID: {episode.episode_id}\n")
        f.write(f"Scene: {episode.scene_id}\n")
        f.write(f"Instruction: {episode.instruction.instruction_text}\n\n")
        f.write("Final Metrics:\n")
        for key, value in sorted(final_metrics.items()):
            f.write(f"  {key}: {value}\n")
    
    print(f"  Saved metrics: {metrics_path}")
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    
    env.close()
    
    return final_metrics


def create_visualization_frame(observations, episode, step, action, action_name, distance, has_map=False):
    """
    Create a visualization frame combining RGB observation and text overlay.
    """
    # Get RGB observation
    rgb = observations["rgb"]
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        rgb = (rgb * 255).astype(np.uint8)
    
    # Convert to PIL Image
    frame = Image.fromarray(rgb)
    
    # Resize if needed (make it larger for better visibility)
    target_width = 640
    aspect_ratio = frame.height / frame.width
    target_height = int(target_width * aspect_ratio)
    frame = frame.resize((target_width, target_height), Image.LANCZOS)
    
    # Prepare text overlay
    instruction = episode.instruction.instruction_text
    # Truncate long instructions
    if len(instruction) > 100:
        instruction = instruction[:97] + "..."
    
    text_lines = [
        f"Episode: {episode.episode_id} | Step: {step}",
        f"Action: {action_name}",
        f"Distance to Goal: {distance:.2f}m",
        f"Instruction: {instruction}",
    ]
    
    # Add text to image (join lines with newline)
    frame_with_text = append_text_to_image(
        np.array(frame),
        "\n".join(text_lines),
    )
    
    return frame_with_text


def main():
    parser = argparse.ArgumentParser(
        description="Visualize agent following oracle shortest path"
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
        help="Dataset split to use"
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Specific episode ID to visualize (optional)"
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
        "--no-map",
        action="store_true",
        help="Disable top-down map"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to run"
    )
    
    args = parser.parse_args()
    
    # Change to evaluation directory
    eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(eval_dir)
    
    # Run visualization
    for i in range(args.num_episodes):
        print(f"\n{'='*80}")
        print(f"Running Episode {i+1}/{args.num_episodes}")
        print(f"{'='*80}\n")
        
        metrics = visualize_oracle_navigation(
            config_path=args.config,
            split=args.split,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            save_video=not args.no_video,
            show_top_down_map=not args.no_map,
        )
        
        if args.num_episodes > 1:
            print(f"\nCompleted episode {i+1}/{args.num_episodes}\n")


if __name__ == "__main__":
    main()
