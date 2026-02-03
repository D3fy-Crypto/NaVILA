"""
Debug script to verify episode loading behavior in Habitat.

This script tests:
1. Whether env.current_episode assignment actually changes the loaded episode
2. Whether env.reset() respects the episode assignment or uses its own iterator
3. Visual display of the environment to confirm the scene matches

Usage:
    python debug_episode_loading.py --display  # Show environment visually
    python debug_episode_loading.py            # Print-only debug
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat import Env
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import quaternion_to_yaw


def display_frame(title: str, observations: Dict, episode: Any, step: int = 0, wait_ms: int = 1):
    """Display current observation with episode info overlay"""
    rgb = observations.get("rgb")
    if rgb is None:
        print(f"  [WARNING] No RGB observation available")
        return
    
    # Convert to uint8 if needed
    if rgb.dtype == np.float32 or rgb.dtype == np.float64:
        rgb = (rgb * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Add text overlay with episode info
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 0)  # Green
    thickness = 2
    
    info_lines = [
        f"Episode ID: {episode.episode_id}",
        f"Scene: {episode.scene_id.split('/')[-1]}",
        f"Step: {step}",
        f"Start pos: ({episode.start_position[0]:.2f}, {episode.start_position[1]:.2f}, {episode.start_position[2]:.2f})",
    ]
    
    y_offset = 30
    for line in info_lines:
        cv2.putText(bgr, line, (10, y_offset), font, font_scale, font_color, thickness)
        y_offset += 25
    
    # Show window
    cv2.imshow(title, bgr)
    cv2.waitKey(wait_ms)


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


def test_episode_loading_basic(env: Env, display: bool = False):
    """
    Test 1: Basic check - does env.current_episode assignment work?
    """
    print("\n" + "=" * 80)
    print("TEST 1: Basic episode loading verification")
    print("=" * 80)
    
    # Get first 3 different episodes
    episodes = env.episodes[:3]
    
    for i, episode in enumerate(episodes):
        print(f"\n--- Testing Episode {i+1} ---")
        print(f"  Requested episode_id: {episode.episode_id}")
        print(f"  Requested scene: {episode.scene_id}")
        print(f"  Requested start_pos: {episode.start_position}")
        
        # Method 1: Direct assignment (what the original code does)
        env.current_episode = episode
        
        print(f"  After assignment, env.current_episode.episode_id: {env.current_episode.episode_id}")
        
        # Now reset and see what actually loads
        observations = env.reset()
        
        loaded_episode = env.current_episode
        print(f"  After reset, env.current_episode.episode_id: {loaded_episode.episode_id}")
        print(f"  After reset, scene: {loaded_episode.scene_id}")
        
        # Get actual agent state
        state = env.sim.get_agent_state()
        actual_pos = state.position
        print(f"  Actual agent position: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f})")
        
        # Check if positions match
        expected_pos = episode.start_position
        pos_diff = np.linalg.norm(np.array(actual_pos) - np.array(expected_pos))
        if pos_diff > 0.1:
            print(f"  ⚠️  POSITION MISMATCH! Difference: {pos_diff:.3f}m")
        else:
            print(f"  ✓ Position matches (diff: {pos_diff:.3f}m)")
        
        # Check if episode IDs match
        if int(loaded_episode.episode_id) != int(episode.episode_id):
            print(f"  ❌ EPISODE ID MISMATCH!")
            print(f"     Requested: {episode.episode_id}, Got: {loaded_episode.episode_id}")
        else:
            print(f"  ✓ Episode ID matches")
        
        if display:
            display_frame(f"Episode {episode.episode_id}", observations, loaded_episode, step=0, wait_ms=2000)


def test_episode_loading_order(env: Env, display: bool = False):
    """
    Test 2: Check if episodes are loaded in sequence regardless of assignment
    """
    print("\n" + "=" * 80)
    print("TEST 2: Episode loading order (does reset() follow iterator?)")
    print("=" * 80)
    
    loaded_ids = []
    
    for i in range(5):
        print(f"\n--- Reset #{i+1} (no explicit episode assignment) ---")
        
        observations = env.reset()
        loaded_episode = env.current_episode
        loaded_ids.append(int(loaded_episode.episode_id))
        
        print(f"  Loaded episode_id: {loaded_episode.episode_id}")
        print(f"  Scene: {loaded_episode.scene_id.split('/')[-1]}")
        
        if display:
            display_frame(f"Reset #{i+1} - Episode {loaded_episode.episode_id}", 
                         observations, loaded_episode, step=0, wait_ms=1500)
    
    print(f"\n  Episode IDs loaded in order: {loaded_ids}")
    print(f"  Expected (first 5 episodes): {[int(ep.episode_id) for ep in env.episodes[:5]]}")


def test_episode_loading_specific(env: Env, display: bool = False):
    """
    Test 3: Load specific episodes out of order to see if assignment works
    """
    print("\n" + "=" * 80)
    print("TEST 3: Specific episode loading (out of order)")
    print("=" * 80)
    
    # Try to load episodes in reverse order: 5th, 3rd, 1st
    test_indices = [4, 2, 0, 9, 7]
    
    for idx in test_indices:
        if idx >= len(env.episodes):
            continue
            
        target_episode = env.episodes[idx]
        
        print(f"\n--- Attempting to load episode at index {idx} ---")
        print(f"  Target episode_id: {target_episode.episode_id}")
        print(f"  Target scene: {target_episode.scene_id.split('/')[-1]}")
        
        # Assign before reset
        env.current_episode = target_episode
        
        # Reset
        observations = env.reset()
        loaded_episode = env.current_episode
        
        print(f"  Actually loaded episode_id: {loaded_episode.episode_id}")
        print(f"  Actually loaded scene: {loaded_episode.scene_id.split('/')[-1]}")
        
        match = int(loaded_episode.episode_id) == int(target_episode.episode_id)
        if match:
            print(f"  ✓ SUCCESS: Correct episode loaded!")
        else:
            print(f"  ❌ FAILURE: Wrong episode loaded!")
            print(f"     This confirms env.reset() ignores env.current_episode assignment")
        
        if display:
            title = f"Target: {target_episode.episode_id} | Got: {loaded_episode.episode_id}"
            display_frame(title, observations, loaded_episode, step=0, wait_ms=2000)


def test_episode_iterator_methods(env: Env, display: bool = False):
    """
    Test 4: Check if there's an episode_iterator and its methods
    """
    print("\n" + "=" * 80)
    print("TEST 4: Episode iterator inspection")
    print("=" * 80)
    
    print(f"\n  env type: {type(env)}")
    print(f"  env.episodes length: {len(env.episodes)}")
    
    # Check for episode_iterator
    if hasattr(env, '_episode_iterator'):
        print(f"  env._episode_iterator: {env._episode_iterator}")
        print(f"  Iterator type: {type(env._episode_iterator)}")
        
        if hasattr(env._episode_iterator, 'episodes'):
            print(f"  Iterator episodes length: {len(env._episode_iterator.episodes)}")
        
        # Check iterator methods
        print(f"\n  Episode iterator methods:")
        for attr in dir(env._episode_iterator):
            if not attr.startswith('_'):
                print(f"    - {attr}")
    else:
        print(f"  No _episode_iterator attribute found")
    
    # Check for _current_episode
    if hasattr(env, '_current_episode'):
        print(f"  env._current_episode: {env._current_episode}")
    
    # Check current_episode property (only if not None)
    try:
        ep = env.current_episode
        print(f"  env.current_episode type: {type(ep)}")
        print(f"  env.current_episode.episode_id: {ep.episode_id if ep else 'None'}")
    except AssertionError:
        print(f"  env.current_episode: Not set (None) - need to call reset() first")
    
    # List relevant methods
    print(f"\n  Environment methods related to episodes:")
    for attr in dir(env):
        if 'episode' in attr.lower():
            print(f"    - {attr}")


def test_correct_loading_method(env: Env, display: bool = False):
    """
    Test 5: Try different methods to correctly load specific episodes
    """
    print("\n" + "=" * 80)
    print("TEST 5: Finding correct method to load specific episodes")
    print("=" * 80)
    
    target_idx = 5
    target_episode = env.episodes[target_idx]
    
    print(f"\n  Target: Episode {target_episode.episode_id} at index {target_idx}")
    print(f"  Scene: {target_episode.scene_id}")
    
    # Method A: Try setting _episode_iterator if it exists
    print("\n--- Method A: Manipulate episode iterator ---")
    if hasattr(env, '_episode_iterator') and env._episode_iterator is not None:
        print(f"  Has _episode_iterator")
        
        # Check if we can set current episode via iterator
        if hasattr(env._episode_iterator, '_current_episode'):
            env._episode_iterator._current_episode = target_episode
            print(f"  Set _episode_iterator._current_episode")
        
        env.current_episode = target_episode
        observations = env.reset()
        loaded = env.current_episode
        print(f"  Result: Loaded episode {loaded.episode_id}")
        
        if int(loaded.episode_id) == int(target_episode.episode_id):
            print(f"  ✓ Method A works!")
        else:
            print(f"  ❌ Method A failed")
    else:
        print(f"  No _episode_iterator or it's None")
    
    # Method B: Recreate episodes list
    print("\n--- Method B: Replace episodes list with single episode ---")
    original_episodes = env.episodes.copy() if hasattr(env.episodes, 'copy') else list(env.episodes)
    
    # This is destructive but let's see if it works
    print(f"  Temporarily setting env.episodes to single-element list")
    
    # Some Habitat versions allow this
    try:
        env._env._dataset.episodes = [target_episode]
        observations = env.reset()
        loaded = env.current_episode
        print(f"  Result: Loaded episode {loaded.episode_id}")
        
        if int(loaded.episode_id) == int(target_episode.episode_id):
            print(f"  ✓ Method B works!")
        else:
            print(f"  ❌ Method B failed")
        
        # Restore
        env._env._dataset.episodes = original_episodes
    except Exception as e:
        print(f"  Method B failed with error: {e}")
    
    # Method C: Use episode_from_id if available
    print("\n--- Method C: Check for episode selection methods ---")
    for method_name in ['set_episode', 'load_episode', 'goto_episode', 'episode_from_id']:
        if hasattr(env, method_name):
            print(f"  Found method: {method_name}")


def run_agent_visualization(env: Env, episode_indices: list = [0, 5, 10], steps: int = 10):
    """
    Run agent for a few steps on different episodes and display in real-time
    """
    print("\n" + "=" * 80)
    print("LIVE VISUALIZATION: Agent in action")
    print("=" * 80)
    
    for idx in episode_indices:
        if idx >= len(env.episodes):
            continue
        
        target_episode = env.episodes[idx]
        print(f"\n--- Episode {target_episode.episode_id} (index {idx}) ---")
        print(f"  Scene: {target_episode.scene_id}")
        print(f"  Instruction: {target_episode.instruction.instruction_text[:80]}...")
        
        # Load episode
        env.current_episode = target_episode
        observations = env.reset()
        
        loaded_episode = env.current_episode
        print(f"  Actually loaded: Episode {loaded_episode.episode_id}")
        
        if int(loaded_episode.episode_id) != int(target_episode.episode_id):
            print(f"  ⚠️  MISMATCH - requested {target_episode.episode_id}, got {loaded_episode.episode_id}")
        
        # Run for a few steps
        for step in range(steps):
            # Get oracle action
            oracle_action = int(observations["shortest_path_sensor"][0])
            
            action_names = ["STOP", "FORWARD", "LEFT", "RIGHT"]
            action_name = action_names[oracle_action] if oracle_action < 4 else "UNKNOWN"
            
            # Display current frame
            rgb = observations.get("rgb")
            if rgb is not None:
                if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    rgb = (rgb * 255).astype(np.uint8)
                
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Add overlay
                state = env.sim.get_agent_state()
                pos = state.position
                
                info_text = [
                    f"Episode: {loaded_episode.episode_id} | Step: {step}",
                    f"Action: {action_name}",
                    f"Scene: {loaded_episode.scene_id.split('/')[-1]}",
                    f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                ]
                
                y = 30
                for text in info_text:
                    cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    y += 25
                
                cv2.imshow("Agent View", bgr)
                key = cv2.waitKey(300)  # 300ms per frame
                
                if key == ord('q'):
                    print("  User requested quit")
                    cv2.destroyAllWindows()
                    return
            
            # Stop if episode is over or STOP action
            if oracle_action == 0 or env.episode_over:
                print(f"  Episode ended at step {step}")
                break
            
            # Take action
            observations = env.step(oracle_action)
        
        print(f"  Completed {step+1} steps")
        time.sleep(0.5)
    
    cv2.destroyAllWindows()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug episode loading in Habitat")
    parser.add_argument("--display", action="store_true", help="Display environment visually")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--live", action="store_true", help="Run live agent visualization")
    args = parser.parse_args()
    
    # Check for display
    if args.display or args.live:
        if cv2 is None:
            print("ERROR: OpenCV (cv2) is required for display mode")
            print("Install with: pip install opencv-python")
            return
        
        # Check if display is available
        display_env = os.environ.get('DISPLAY')
        if not display_env:
            print("WARNING: No DISPLAY environment variable set")
            print("Setting DISPLAY=:1 for VNC...")
            os.environ['DISPLAY'] = ':1'
    
    # Setup environment
    env = setup_environment(split=args.split)
    
    try:
        if args.live:
            # Live visualization
            run_agent_visualization(env, episode_indices=[0, 5, 10], steps=20)
        else:
            # Run all tests
            test_episode_iterator_methods(env, display=args.display)
            test_episode_loading_basic(env, display=args.display)
            test_episode_loading_order(env, display=args.display)
            test_episode_loading_specific(env, display=args.display)
            test_correct_loading_method(env, display=args.display)
        
        print("\n" + "=" * 80)
        print("DEBUG COMPLETE")
        print("=" * 80)
        
    finally:
        env.close()
        if args.display or args.live:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
