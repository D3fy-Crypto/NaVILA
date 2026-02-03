"""
Debug script to FIX episode loading behavior in Habitat.

This script:
1. Confirms the bug: env.reset() ignores env.current_episode assignment
2. Provides the CORRECT way to load a specific episode
3. Shows live visualization to confirm the fix works

Usage:
    python debug_episode_loading_fix.py --display  # Show environment visually
    python debug_episode_loading_fix.py --live     # Live agent visualization
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from habitat import Env
from habitat.core.dataset import EpisodeIterator
from vlnce_baselines.config.default import get_config
from habitat_extensions.utils import quaternion_to_yaw


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


def load_specific_episode(env: Env, episode) -> Dict:
    """
    THE CORRECT WAY to load a specific episode in Habitat.
    
    The bug: env.current_episode = episode; env.reset() DOES NOT WORK!
    
    The fix: Replace the episode iterator with one containing only the target episode,
    then call reset().
    
    Args:
        env: Habitat environment
        episode: Target episode object
    
    Returns:
        observations dict from reset()
    """
    # Create a single-episode iterator for this specific episode
    # The iterator will just return this one episode
    single_episode_iter = iter([episode])
    
    # Replace the internal _iterator (not the full EpisodeIterator object)
    # This is the cleanest way to do it
    env._episode_iterator._iterator = single_episode_iter
    
    # Now reset() will call next() on our iterator, getting exactly our episode
    observations = env.reset()
    
    return observations


def verify_episode_loaded(env: Env, target_episode, label: str = "") -> bool:
    """Verify the correct episode was loaded"""
    loaded = env.current_episode
    
    match_id = int(loaded.episode_id) == int(target_episode.episode_id)
    match_scene = loaded.scene_id == target_episode.scene_id
    
    # Check position match
    state = env.sim.get_agent_state()
    actual_pos = np.array(state.position)
    expected_pos = np.array(target_episode.start_position)
    pos_diff = np.linalg.norm(actual_pos - expected_pos)
    match_pos = pos_diff < 0.1
    
    print(f"{label}")
    print(f"  Target ID: {target_episode.episode_id}, Loaded ID: {loaded.episode_id} - {'✓' if match_id else '❌'}")
    print(f"  Target scene: {target_episode.scene_id.split('/')[-1]}")
    print(f"  Loaded scene: {loaded.scene_id.split('/')[-1]} - {'✓' if match_scene else '❌'}")
    print(f"  Position diff: {pos_diff:.4f}m - {'✓' if match_pos else '❌'}")
    
    all_match = match_id and match_scene and match_pos
    if all_match:
        print(f"  ✅ SUCCESS: Correct episode loaded!")
    else:
        print(f"  ❌ FAILURE: Wrong episode loaded!")
    
    return all_match


def test_fix_basic(env: Env):
    """Test the fix on basic episode loading"""
    print("\n" + "=" * 80)
    print("TEST: Verify the FIX works for loading specific episodes")
    print("=" * 80)
    
    # Test loading episodes in non-sequential order
    test_indices = [0, 10, 5, 100, 50, 1, 9]
    
    successes = 0
    for idx in test_indices:
        if idx >= len(env.episodes):
            continue
        
        target_episode = env.episodes[idx]
        print(f"\n--- Loading episode index {idx} (ID: {target_episode.episode_id}) ---")
        
        observations = load_specific_episode(env, target_episode)
        
        if verify_episode_loaded(env, target_episode, label="  Result:"):
            successes += 1
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {successes}/{len(test_indices)} episodes loaded correctly")
    print(f"{'=' * 80}")
    
    return successes == len(test_indices)


def test_scene_switching(env: Env, display: bool = False):
    """Test loading episodes from DIFFERENT scenes to verify scene changes"""
    print("\n" + "=" * 80)
    print("TEST: Scene switching (load episodes from different scenes)")
    print("=" * 80)
    
    # Find episodes from different scenes
    scene_episodes = {}
    for ep in env.episodes:
        scene = ep.scene_id.split('/')[-1].replace('.glb', '')
        if scene not in scene_episodes:
            scene_episodes[scene] = ep
        if len(scene_episodes) >= 5:
            break
    
    print(f"Found {len(scene_episodes)} unique scenes to test")
    
    for scene_name, episode in scene_episodes.items():
        print(f"\n--- Scene: {scene_name} (Episode {episode.episode_id}) ---")
        
        observations = load_specific_episode(env, episode)
        
        verify_episode_loaded(env, episode, label="  Result:")
        
        if display:
            rgb = observations.get("rgb")
            if rgb is not None:
                if rgb.dtype == np.float32:
                    rgb = (rgb * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Add overlay
                cv2.putText(bgr, f"Scene: {scene_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(bgr, f"Episode: {episode.episode_id}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Scene Switching Test", bgr)
                cv2.waitKey(2000)


def live_agent_test(env: Env, episode_indices: List[int] = [0, 10, 50], steps: int = 15):
    """Live visualization of agent in action on different episodes"""
    print("\n" + "=" * 80)
    print("LIVE TEST: Agent navigating in different episodes")
    print("Press 'q' to quit, 'n' for next episode")
    print("=" * 80)
    
    for idx in episode_indices:
        if idx >= len(env.episodes):
            continue
        
        target_episode = env.episodes[idx]
        scene_name = target_episode.scene_id.split('/')[-1].replace('.glb', '')
        
        print(f"\n--- Episode {target_episode.episode_id} (Scene: {scene_name}) ---")
        print(f"Instruction: {target_episode.instruction.instruction_text[:100]}...")
        
        # CORRECT WAY: Use our fix function
        observations = load_specific_episode(env, target_episode)
        
        loaded = env.current_episode
        print(f"Loaded episode: {loaded.episode_id} (Scene: {loaded.scene_id.split('/')[-1]})")
        
        if int(loaded.episode_id) != int(target_episode.episode_id):
            print(f"⚠️ MISMATCH STILL! Something is wrong with the fix.")
            continue
        
        # Run agent for steps
        for step in range(steps):
            oracle_action = int(observations["shortest_path_sensor"][0])
            action_names = ["STOP", "FORWARD", "LEFT", "RIGHT"]
            action_name = action_names[oracle_action] if oracle_action < 4 else "?"
            
            # Get agent state
            state = env.sim.get_agent_state()
            pos = state.position
            
            # Display
            rgb = observations.get("rgb")
            if rgb is not None:
                if rgb.dtype == np.float32:
                    rgb = (rgb * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Add info overlay
                info = [
                    f"Episode: {loaded.episode_id} | Step: {step}",
                    f"Scene: {scene_name}",
                    f"Action: {action_name}",
                    f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
                ]
                y = 30
                for text in info:
                    cv2.putText(bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, (0, 255, 0), 2)
                    y += 25
                
                cv2.imshow("Live Agent Test", bgr)
                key = cv2.waitKey(400)
                
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    break
            
            if oracle_action == 0 or env.episode_over:
                print(f"  Episode ended at step {step}")
                break
            
            observations = env.step(oracle_action)
        
        print(f"  Completed {step+1} steps")
        time.sleep(0.5)
    
    cv2.destroyAllWindows()
    print("\n✅ Live test complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug and FIX episode loading")
    parser.add_argument("--display", action="store_true", help="Show visual display")
    parser.add_argument("--live", action="store_true", help="Live agent test")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    
    if args.display or args.live:
        os.environ.setdefault('DISPLAY', ':1')
    
    env = setup_environment(split=args.split)
    
    try:
        # Always run the basic fix test
        success = test_fix_basic(env)
        
        if args.display:
            test_scene_switching(env, display=True)
        
        if args.live:
            live_agent_test(env, episode_indices=[0, 10, 50, 100], steps=20)
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Basic fix test: {'✅ PASSED' if success else '❌ FAILED'}")
        print()
        print("THE CORRECT WAY TO LOAD A SPECIFIC EPISODE:")
        print("  env._episode_iterator._iterator = iter([episode])")
        print("  observations = env.reset()")
        print()
        print("DO NOT USE:")
        print("  env.current_episode = episode  # This gets overwritten by reset()!")
        print("=" * 80)
        
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
