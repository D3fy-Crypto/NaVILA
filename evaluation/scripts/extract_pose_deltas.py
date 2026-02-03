"""
Export oracle navigation trajectories with pose deltas from R2R_VLNCE episodes.

Features:
- Full dataset export (all 10,819+ train episodes)
- Duplicate-path caching for speed optimization
- Oracle behavior verification (goal vs waypoint following)
- Resume support for interrupted runs
- Optional visualization with top-down map view
- Progress tracking and statistics

Usage:
    python extract_pose_deltas_final.py --split train --output oracle_exports/train_oracle_deltas.jsonl
    python extract_pose_deltas_final.py --split train --cache-duplicates --verify-oracle
    python extract_pose_deltas_final.py --split train --visualize --max-episodes 5
    python extract_pose_deltas_final.py --split train --resume --verify-oracle
"""

import sys
import os
import json
import gzip
import argparse
import hashlib
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
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


class WaypointFollower:
    """
    Navigate sequentially through waypoints using shortest path actions.
    
    Why not just use observations["shortest_path_sensor"][0] for all modes?
    - SHORTEST_PATH_SENSOR always targets the final goal set in the episode
    - For reference_path mode, we need to target intermediate waypoints sequentially
    - This follower computes shortest path actions to each waypoint as a subgoal
    
    Implementation: Uses Habitat's ShortestPathFollower to compute greedy actions
    toward each waypoint position, advancing to the next when within threshold.
    """
    
    def __init__(self, env: Env, threshold: float = 0.5):
        """
        Args:
            env: Habitat environment
            threshold: Distance threshold to consider waypoint reached (meters)
        """
        self.env = env
        self.sim = env.sim
        self.threshold = threshold
        self.current_waypoint_idx = 0
        self.reference_path = []
        
        # Use Habitat's ShortestPathFollower for computing actions to arbitrary goals
        # CRITICAL: goal_radius must be SMALLER than threshold!
        # - threshold (0.5m): When to advance to next waypoint
        # - goal_radius (0.1m): When ShortestPathFollower returns STOP
        # If goal_radius >= threshold, the follower returns STOP before we advance!
        try:
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
            self.follower = ShortestPathFollower(
                self.sim,
                goal_radius=0.1,  # Much smaller than threshold to keep navigating
                return_one_hot=False,
            )
            self.has_follower = True
        except ImportError:
            self.follower = None
            self.has_follower = False
    
    def reset(self, reference_path: Optional[List[Tuple[float, float, float]]] = None):
        """Reset follower state for new episode"""
        self.current_waypoint_idx = 0
        self.reference_path = reference_path or []
    
    def get_next_action(
        self, 
        observations: Dict,
        current_position: np.ndarray,
    ) -> int:
        """
        Get next action toward current waypoint.
        
        Args:
            observations: Current observations
            current_position: Current (x, y, z) position
        
        Returns:
            Action index (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
        """
        # Check if we've completed all waypoints
        if self.current_waypoint_idx >= len(self.reference_path):
            return 0  # STOP
        
        # Get current waypoint
        current_wp = self.reference_path[self.current_waypoint_idx]
        waypoint_pos = np.array(current_wp[:3], dtype=np.float32)
        
        # Distance to current waypoint
        distance = np.linalg.norm(current_position - waypoint_pos)
        
        # Advance to next waypoint if within threshold
        if distance < self.threshold:
            self.current_waypoint_idx += 1
            
            # All waypoints reached - STOP
            if self.current_waypoint_idx >= len(self.reference_path):
                return 0  # STOP
            
            # Update waypoint for next iteration
            current_wp = self.reference_path[self.current_waypoint_idx]
            waypoint_pos = np.array(current_wp[:3], dtype=np.float32)
        
        # Compute action toward current waypoint using ShortestPathFollower
        if self.has_follower:
            try:
                # ShortestPathFollower.get_next_action() takes a goal position
                # and computes the greedy action using geodesic distance
                action = self.follower.get_next_action(waypoint_pos.tolist())
                
                # Habitat action codes: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
                # If we get None, try manual fallback
                if action is None:
                    print(f"    Warning: ShortestPathFollower returned None, using manual fallback")
                    return self._compute_greedy_action(current_position, waypoint_pos)
                
                # If follower returns STOP but we're not at the last waypoint,
                # there might be a navigation issue
                if action == 0 and self.current_waypoint_idx < len(self.reference_path) - 1:
                    dist = np.linalg.norm(current_position - waypoint_pos)
                    if dist > 0.2:  # Still far from waypoint - use manual fallback
                        print(f"    Warning: ShortestPathFollower returned STOP at dist={dist:.2f}m, using manual fallback")
                        return self._compute_greedy_action(current_position, waypoint_pos)
                    else:
                        # Waypoint is very close (< 0.2m) - skip to next waypoint
                        # This handles reference paths with redundant/overlapping waypoints
                        while self.current_waypoint_idx < len(self.reference_path) - 1:
                            self.current_waypoint_idx += 1
                            current_wp = self.reference_path[self.current_waypoint_idx]
                            waypoint_pos = np.array(current_wp[:3], dtype=np.float32)
                            dist = np.linalg.norm(current_position - waypoint_pos)
                            if dist > 0.2:  # Found a waypoint that's far enough
                                break
                        
                        # Try again with new waypoint
                        if self.current_waypoint_idx >= len(self.reference_path):
                            return 0  # STOP - all waypoints exhausted
                        
                        action = self.follower.get_next_action(waypoint_pos.tolist())
                        if action is None or (action == 0 and dist > 0.2):
                            return self._compute_greedy_action(current_position, waypoint_pos)
                        return int(action) if action else self._compute_greedy_action(current_position, waypoint_pos)
                
                return int(action)
            except Exception as e:
                # If follower fails, compute manually
                print(f"    Warning: Follower failed with {e}, computing manually")
                pass
        
        # Manual fallback: compute greedy action based on angle to waypoint
        return self._compute_greedy_action(current_position, waypoint_pos)
    
    def _compute_greedy_action(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> int:
        """
        Compute greedy action toward goal using simple angle-based heuristic.
        
        Args:
            current_pos: Current (x, y, z) position
            goal_pos: Goal (x, y, z) position
        
        Returns:
            Action index (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
        """
        # Get agent state
        agent_state = self.sim.get_agent_state()
        
        # Current heading (yaw)
        from habitat_extensions.utils import quaternion_to_yaw
        current_yaw = quaternion_to_yaw(agent_state.rotation)
        
        # Vector to goal (in xz plane)
        to_goal = goal_pos - current_pos
        goal_direction = np.arctan2(to_goal[2], to_goal[0])  # Note: z is forward in Habitat
        
        # Angle difference
        angle_diff = wrap_angle(goal_direction - current_yaw)
        
        # Thresholds
        forward_threshold = np.pi / 6  # 30 degrees
        
        # Decide action
        if abs(angle_diff) < forward_threshold:
            return 1  # MOVE_FORWARD
        elif angle_diff > 0:
            return 3  # TURN_RIGHT
        else:
            return 2  # TURN_LEFT
    
    def get_waypoint_progress(self) -> Tuple[int, int]:
        """Return (current_waypoint_idx, total_waypoints)"""
        return (self.current_waypoint_idx, len(self.reference_path))


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))


def round_vec3(pos: List[float], decimals: int = 3) -> Tuple[float, ...]:
    """Round 3D position coordinates (xyz) for cache key generation"""
    return tuple(round(x, decimals) for x in pos[:3])


def round_quat4(quat: List[float], decimals: int = 3) -> Tuple[float, ...]:
    """Round quaternion (4 floats: wxyz) for cache key generation"""
    return tuple(round(x, decimals) for x in quat[:4])


def round_position(pos: List[float], decimals: int = 3) -> Tuple[float, ...]:
    """Round position coordinates for cache key generation (deprecated: use round_vec3)"""
    return round_vec3(pos, decimals)


def make_cache_key(episode: Any, oracle_mode: str = "goal", decimals: int = 3) -> str:
    """
    Create cache key from episode metadata for duplicate detection.
    
    Cache key includes:
    - scene_id
    - start_position (rounded xyz)
    - start_rotation (rounded quaternion: wxyz)
    - goal_position (rounded xyz)
    - reference_path waypoints (rounded xyz)
    - oracle_mode (CRITICAL: goal vs reference_path are different rollouts)
    
    Args:
        episode: Habitat episode object
        oracle_mode: 'goal' or 'reference_path' - MUST be included to avoid cache collision
        decimals: Rounding precision for positions
    
    Returns:
        SHA256 hash string (16 chars)
    """
    # Extract scene ID
    scene_id = str(episode.scene_id).split('/')[-1].replace('.glb', '')
    
    # Round positions for stability (xyz only)
    start_pos = round_vec3(episode.start_position, decimals)
    
    # Round quaternion rotation (wxyz format, 4 values)
    # Note: start_rotation is a quaternion, not a position vector
    start_rot = round_quat4(episode.start_rotation, decimals)
    
    # Goal position (first goal, xyz only)
    if hasattr(episode, 'goals') and len(episode.goals) > 0:
        goal_pos = round_vec3(episode.goals[0].position, decimals)
    else:
        goal_pos = (0, 0, 0)
    
    # Reference path (critical for duplicate detection)
    # Each waypoint is xyz coordinates
    if hasattr(episode, 'reference_path') and episode.reference_path:
        ref_path = tuple(
            round_vec3(wp, decimals) 
            for wp in episode.reference_path
        )
    else:
        ref_path = ()
    
    # Build deterministic cache key
    # CRITICAL: Include oracle_mode so that goal-following and reference_path-following
    # trajectories don't collide in the cache (they will have different rollouts)
    key_data = {
        'scene': scene_id,
        'start_pos': start_pos,
        'start_rot': start_rot,
        'goal': goal_pos,
        'ref_path': ref_path,
        'oracle_mode': oracle_mode,
    }
    
    # Hash for compact key
    key_str = json.dumps(key_data, sort_keys=True)
    cache_key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    return cache_key


class OracleExporter:
    """
    Export oracle navigation trajectories with duplicate-path caching and optional visualization.
    """
    
    def __init__(
        self,
        split: str = "train",
        output_file: str = "oracle_deltas_train.jsonl",
        max_episodes: Optional[int] = None,
        max_steps: int = 500,
        cache_duplicates: bool = True,
        oracle_mode: str = "goal",
        verify_oracle: bool = False,
        visualize: bool = False,
        visualization_dir: str = "oracle_visualizations",
        resume: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            split: Dataset split (train/val_seen/val_unseen)
            output_file: Output JSONL file path
            max_episodes: Maximum episodes to export (None = all)
            max_steps: Maximum steps per episode
            cache_duplicates: Enable duplicate-path caching
            oracle_mode: 'goal' (shortest path to final goal) or 'reference_path' (waypoint following)
            verify_oracle: Log oracle behavior verification
            visualize: Save visualization videos
            visualization_dir: Directory for visualization outputs
            resume: Resume from existing output file
            seed: Random seed
        """
        self.split = split
        self.output_file = Path(output_file).resolve()
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.cache_duplicates = cache_duplicates
        self.oracle_mode = oracle_mode
        self.verify_oracle = verify_oracle
        self.visualize = visualize
        self.visualization_dir = Path(visualization_dir).resolve()
        self.resume = resume
        self.seed = seed
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'successes': 0,
            'failures': 0,
            'total_steps': 0,
            'scene_counts': defaultdict(int),
            'ref_path_follows': 0,  # Episodes that follow reference path
            'goal_follows': 0,  # Episodes that go to final goal
        }
        
        # Duplicate cache: cache_key -> rollout data
        self.cache: Dict[str, Dict] = {}
        
        # Track processed episodes (for resume)
        self.processed_episode_ids = set()
        
        # Create visualization directory
        if self.visualize:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("ORACLE POSE-DELTA EXPORTER")
        print("=" * 80)
        print(f"Split: {split}")
        print(f"Output: {self.output_file}")
        print(f"Max episodes: {max_episodes or 'ALL'}")
        print(f"Max steps: {max_steps}")
        print(f"Cache duplicates: {cache_duplicates}")
        print(f"Oracle mode: {oracle_mode}")
        print(f"Verify oracle: {verify_oracle}")
        print(f"Visualize: {visualize}")
        if visualize:
            print(f"Visualization dir: {self.visualization_dir}")
        print(f"Resume: {resume}")
        print(f"Seed: {seed}")
        print("=" * 80)
        print()
    
    def load_processed_episodes(self):
        """Load already-processed episode IDs from existing output file"""
        if not self.resume or not self.output_file.exists():
            return
        
        print(f"Loading processed episodes from: {self.output_file}")
        
        with open(self.output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    self.processed_episode_ids.add(record['episode_id'])
                    
                    # Rebuild cache from previous runs (if cache_key is in record)
                    if 'cache_key' in record and not record.get('metrics_reused', False):
                        cache_key = record['cache_key']
                        if cache_key not in self.cache:
                            # Reconstruct cache entry
                            self.cache[cache_key] = {
                                'poses': record['poses'],
                                'deltas': record['deltas'],
                                'actions': record['actions'],
                                'num_steps': record['num_steps'],
                                'success': record['success'],
                                'spl': record['spl'],
                                'path_length': record['path_length'],
                                'distance_to_goal': record['distance_to_goal'],
                                'oracle_mode': record.get('oracle_mode', 'goal'),
                                'frames': None,
                                'first_episode_id': record['episode_id'],
                            }
                except Exception as e:
                    continue
        
        print(f"  Found {len(self.processed_episode_ids)} already-processed episodes")
        print(f"  Rebuilt cache with {len(self.cache)} unique paths")
        print()
    
    def setup_environment(self) -> Env:
        """Initialize Habitat environment with oracle sensor"""
        print("Setting up Habitat environment...")
        
        # Get the evaluation directory path
        eval_dir = Path(__file__).parent.parent  # scripts -> evaluation
        config_path = eval_dir / "vlnce_baselines/config/r2r_baselines/navila.yaml"
        
        # Change to evaluation directory for config loading
        original_cwd = os.getcwd()
        os.chdir(eval_dir)
        
        try:
            config = get_config(str(config_path), [
                "TASK_CONFIG.DATASET.SPLIT", self.split,
                "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", str(self.max_steps),
                "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE", "False",
                "TASK_CONFIG.SEED", str(self.seed),
            ])
            
            config.defrost()
            config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
            config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
            
            # Add shortest path sensor for oracle actions
            if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
                config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")
            
            # Add top-down map if visualizing
            if self.visualize:
                if "TOP_DOWN_MAP_VLNCE" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
                    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            
            config.freeze()
            
            print(f"  Config loaded: {config_path}")
            print(f"  Split: {self.split}")
            print(f"  Sensors: {config.TASK_CONFIG.TASK.SENSORS}")
            if self.visualize:
                print(f"  Measurements: {config.TASK_CONFIG.TASK.MEASUREMENTS}")
            print()
            
            # Create environment
            env = Env(config=config.TASK_CONFIG)
            
            print(f"✓ Environment ready")
            print(f"  Total episodes: {len(env.episodes)}")
            print()
            
            return env
        
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def verify_oracle_behavior(
        self,
        env: Env,
        observations: Dict,
        episode: Any,
        step: int,
        oracle_action: int,
        waypoint_follower: Optional[WaypointFollower] = None,
    ) -> Dict[str, float]:
        """
        Verify what the oracle is computing for the current step.
        
        Logs different metrics based on oracle mode:
        - goal mode: dist_to_goal before and after
        - reference_path mode: also dist_to_current_waypoint and waypoint index
        
        Returns dict with:
        - dist_to_goal: Distance to final goal
        - dist_to_current_wp: Distance to current reference waypoint (if ref_path mode)
        - waypoint_idx: Current waypoint index (if ref_path mode)
        - oracle_action: Action to be taken
        - step: Step number
        """
        # Get current state
        state = env.sim.get_agent_state()
        pos = state.position
        current_pos = np.array([pos[0], pos[1], pos[2]])
        
        # Distance to final goal
        goal_pos = np.array(episode.goals[0].position)
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)
        
        # Distance to current waypoint (if reference_path mode)
        dist_to_current_wp = None
        waypoint_idx = None
        
        if waypoint_follower is not None and self.oracle_mode == "reference_path":
            if waypoint_follower.reference_path and waypoint_follower.current_waypoint_idx < len(waypoint_follower.reference_path):
                current_wp = waypoint_follower.reference_path[waypoint_follower.current_waypoint_idx]
                wp_pos = np.array(current_wp[:3])
                dist_to_current_wp = np.linalg.norm(current_pos - wp_pos)
                waypoint_idx = waypoint_follower.current_waypoint_idx
        
        result = {
            'dist_to_goal': dist_to_goal,
            'dist_to_current_wp': dist_to_current_wp,
            'waypoint_idx': waypoint_idx,
            'oracle_action': oracle_action,
            'step': step,
        }
        
        if self.verify_oracle and step < 5:
            action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
            log_msg = (f"    [Step {step}] pos=({pos[0]:.2f},{pos[2]:.2f}), "
                      f"action={action_names[oracle_action]}, "
                      f"dist_to_goal={dist_to_goal:.2f}m")
            
            if waypoint_follower is not None and self.oracle_mode == "reference_path":
                log_msg += f", wp_idx={waypoint_idx}/{len(waypoint_follower.reference_path)}"
                if dist_to_current_wp is not None:
                    log_msg += f", dist_to_wp={dist_to_current_wp:.2f}m"
            
            print(log_msg)
        
        return result
    
    def create_visualization_frame(
        self,
        observations: Dict,
        episode: Any,
        metrics: Dict,
        step: int,
        oracle_action: int,
        distance_to_goal: float,
        verify_data: Dict,
        agent_height: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Create a visualization frame with RGB and top-down map"""
        
        if not self.visualize or "top_down_map_vlnce" not in metrics:
            return None
        
        try:
            # Get RGB observation
            rgb = observations.get("rgb")
            if rgb is None:
                return None
            
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                rgb = (rgb * 255).astype(np.uint8)
            
            # Resize RGB
            target_width = 640
            aspect_ratio = rgb.shape[0] / rgb.shape[1]
            target_height = int(target_width * aspect_ratio)
            rgb_resized = Image.fromarray(rgb).resize(
                (target_width, target_height),
                Image.LANCZOS
            )
            rgb_array = np.array(rgb_resized)
            
            # Get and colorize top-down map
            map_data = metrics["top_down_map_vlnce"]
            td_map = map_data["map"].copy()
            agent_map_coord = map_data["agent_map_coord"]
            agent_angle = map_data["agent_angle"]
            
            td_map_colored = maps.colorize_topdown_map(
                td_map,
                map_data.get("fog_of_war_mask", np.ones_like(td_map)),
                fog_of_war_desat_amount=0.75,
            )
            
            # Draw agent on map - ensure map is fresh for each step
            # This prevents cached waypoint markers from different floor levels
            td_map_colored = habitat_maps.draw_agent(
                image=td_map_colored,
                agent_center_coord=agent_map_coord,
                agent_rotation=agent_angle,
                agent_radius_px=max(5, min(td_map_colored.shape[0:2]) // 24),
            )
            
            # Resize map to match RGB width
            map_height = target_height
            old_h, old_w = td_map_colored.shape[:2]
            map_width = int(float(map_height) / old_h * old_w)
            td_map_colored = cv2.resize(
                td_map_colored,
                (map_width, map_height),
                interpolation=cv2.INTER_CUBIC,
            ) if cv2 else td_map_colored
            
            # Combine: RGB on top, map on bottom
            max_width = max(rgb_array.shape[1], td_map_colored.shape[1])
            rgb_padded = self._pad_image(rgb_array, max_width, 3)
            map_padded = self._pad_image(td_map_colored, max_width, 3)
            combined = np.vstack([rgb_padded, map_padded])
            
            # Add text overlay
            action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
            instruction = episode.instruction.instruction_text
            if len(instruction) > 80:
                instruction = instruction[:77] + "..."
            
            # Build verification text
            verify_text = f"Dist to Goal: {distance_to_goal:.2f}m"
            if verify_data.get('dist_to_next_wp') is not None:
                verify_text += f" | Dist to WP: {verify_data['dist_to_next_wp']:.2f}m"
            
            text_lines = "\n".join([
                f"Episode: {episode.episode_id} | Step: {step} | Height: {agent_height:.2f}m",
                f"Action: {action_names[oracle_action]} | {verify_text}",
                f"Instruction: {instruction}",
            ])
            
            combined_with_text = append_text_to_image(combined, text_lines)
            
            return combined_with_text
        
        except Exception as e:
            print(f"    Warning: Visualization frame creation failed: {e}")
            return None
    
    def _pad_image(self, img: np.ndarray, target_width: int, channels: int) -> np.ndarray:
        """Pad image to target width"""
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
    
    def rollout_episode(
        self,
        env: Env,
        episode: Any,
    ) -> Optional[Dict]:
        """
        Execute oracle rollout for a single episode.
        
        Supports two oracle modes:
        - 'goal': Use shortest path to final goal (observations["shortest_path_sensor"][0])
        - 'reference_path': Follow episode.reference_path sequentially (WaypointFollower)
        
        Args:
            env: Habitat environment
            episode: Episode object
        
        Returns:
            Rollout data dict or None if failed
        """
        try:
            # CRITICAL FIX: env.current_episode = episode does NOT work!
            # The reset() method always calls next(self._episode_iterator), ignoring the assignment.
            # We must replace the iterator to force loading our specific episode.
            env._episode_iterator._iterator = iter([episode])
            observations = env.reset()
            
            # Verify we actually loaded the right episode
            actual_episode_id = int(env.current_episode.episode_id)
            expected_episode_id = int(episode.episode_id)
            if actual_episode_id != expected_episode_id:
                print(f"ERROR: Episode ID mismatch!")
                print(f"  Requested episode_id: {expected_episode_id}")
                print(f"  Loaded episode_id: {actual_episode_id}")
                print(f"  Scene requested: {episode.scene_id}")
                print(f"  Scene loaded: {env.current_episode.scene_id}")
                return None

        except Exception as e:
            # Skip episodes with missing scenes or other reset errors
            if "does not correspond to any existing file" in str(e) or "not found" in str(e).lower():
                return None
            raise
        
        # Initialize waypoint follower if using reference_path mode
        waypoint_follower = None
        if self.oracle_mode == "reference_path":
            reference_path = getattr(episode, 'reference_path', None)
            if reference_path and len(reference_path) > 0:
                waypoint_follower = WaypointFollower(env, threshold=0.5)
                waypoint_follower.reset(reference_path)
            else:
                # Fallback to goal mode if no reference path
                if self.verify_oracle:
                    print(f"    Warning: Episode {episode.episode_id} has no reference_path, "
                          f"falling back to goal mode")
                self.oracle_mode = "goal"
        
        # Collect trajectory
        poses = []
        actions = []
        verification_log = []
        
        # Track height for multi-floor debugging
        height_changes = []
        
        # Initial pose
        state = env.sim.get_agent_state()
        pos = state.position
        x, y = float(pos[0]), float(pos[2])
        yaw = float(quaternion_to_yaw(state.rotation))
        poses.append((x, y, yaw))
        
        initial_height = float(pos[1])
        current_height = initial_height
        height_changes.append({
            'step': 0,
            'height': initial_height,
            'pos': (x, y),
            'map_shape': None
        })
        
        # Visualization frames
        frames = [] if self.visualize else None
        
        step = 0
        
        while not env.episode_over and step < self.max_steps:
            # Get metrics first (for visualization)
            metrics = env.get_metrics()
            distance_to_goal = metrics.get('distance_to_goal', -1)
            
            # Get oracle action based on mode
            if self.oracle_mode == "goal":
                # Use shortest path to final goal
                oracle_action = int(observations["shortest_path_sensor"][0])
            else:  # reference_path mode
                # Use waypoint follower to navigate through reference path
                assert waypoint_follower is not None, "Waypoint follower not initialized"
                current_pos = np.array([pos[0], pos[1], pos[2]])
                oracle_action = waypoint_follower.get_next_action(
                    observations,
                    current_pos,
                )
            
            actions.append(oracle_action)
            
            # Verify oracle behavior
            verify_data = self.verify_oracle_behavior(
                env, 
                observations, 
                episode, 
                step,
                oracle_action,
                waypoint_follower,
            )
            verification_log.append(verify_data)
            
            # Create visualization frame
            if self.visualize and frames is not None:
                # Get current agent height for visualization
                agent_state = env.sim.get_agent_state()
                current_agent_height = float(agent_state.position[1])
                
                frame = self.create_visualization_frame(
                    observations,
                    episode,
                    metrics,
                    step,
                    oracle_action,
                    distance_to_goal,
                    verify_data,
                    agent_height=current_agent_height,
                )
                if frame is not None:
                    frames.append(frame)
            
            # Execute action
            observations = env.step(oracle_action)
            
            # Record new pose
            state = env.sim.get_agent_state()
            pos = state.position
            x, y = float(pos[0]), float(pos[2])
            yaw = float(quaternion_to_yaw(state.rotation))
            poses.append((x, y, yaw))
            
            # Track height for debugging multi-floor navigation
            new_height = float(pos[1])
            if new_height != current_height:
                height_changes.append({
                    'step': step + 1,
                    'height': new_height,
                    'pos': (x, y),
                    'height_delta': new_height - current_height
                })
                current_height = new_height
            
            step += 1
        
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
        
        # Build rollout data
        rollout = {
            'poses': poses,
            'deltas': deltas,
            'actions': actions,
            'num_steps': step,
            'success': bool(metrics.get("success", 0) > 0),
            'spl': float(metrics.get("spl", 0)),
            'path_length': float(metrics.get("path_length", 0)),
            'distance_to_goal': float(metrics.get("distance_to_goal", 0)),
            'height_changes': height_changes,
            'verification_log': verification_log if self.verify_oracle else [],
            'frames': frames,
            'oracle_mode': self.oracle_mode,
        }
        
        return rollout
    
    def export_episode(
        self,
        env: Env,
        episode: Any,
        f_out,
    ) -> bool:
        """
        Export a single episode (with caching support).
        
        Args:
            env: Habitat environment
            episode: Episode object
            f_out: Output file handle
        
        Returns:
            True if exported successfully
        """
        episode_id = int(episode.episode_id)
        scene_id = str(episode.scene_id)
        
        # Skip if already processed (resume mode)
        if episode_id in self.processed_episode_ids:
            return True
        
        # Check cache
        cache_key = None
        rollout = None
        metrics_reused = False
        reused_from = None
        
        if self.cache_duplicates:
            cache_key = make_cache_key(episode, self.oracle_mode)
            
            if cache_key in self.cache:
                # Cache hit!
                rollout = self.cache[cache_key].copy()
                # Don't copy frames for duplicate episodes
                rollout['frames'] = None
                metrics_reused = True
                reused_from = self.cache[cache_key].get('first_episode_id')
                self.stats['cache_hits'] += 1
            else:
                # Cache miss - need to roll out
                self.stats['cache_misses'] += 1
        
        # Roll out if not cached
        if rollout is None:
            rollout = self.rollout_episode(env, episode)
            
            if rollout is None:
                # Episode failed (missing scene, etc) - skip it
                return True  # Return True to continue to next episode
            
            # Store in cache (without frames)
            if self.cache_duplicates and cache_key is not None:
                cache_entry = rollout.copy()
                cache_entry['frames'] = None  # Don't cache frames
                cache_entry['first_episode_id'] = episode_id
                self.cache[cache_key] = cache_entry
        
        # Save visualization video if available
        video_path = None
        if self.visualize and rollout['frames']:
            video_path = str(
                self.visualization_dir / f"ep_{episode_id:05d}_success_{rollout['success']}.mp4"
            )
            try:
                # Convert frames to numpy arrays if needed
                frame_arrays = []
                for frame in rollout['frames']:
                    if isinstance(frame, Image.Image):
                        frame_arrays.append(np.array(frame))
                    elif isinstance(frame, np.ndarray):
                        # Ensure uint8
                        if frame.dtype != np.uint8:
                            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                        frame_arrays.append(frame)
                    else:
                        continue
                
                if frame_arrays:
                    imageio.mimsave(video_path, frame_arrays, fps=5)
                else:
                    video_path = None
            except Exception as e:
                print(f"    Warning: Failed to save video: {e}")
                video_path = None
        
        # Build output record
        record = {
            'episode_id': episode_id,
            'scene_id': scene_id,
            'split': self.split,
            'oracle_mode': rollout['oracle_mode'],
            'poses': rollout['poses'],
            'deltas': rollout['deltas'],
            'actions': rollout['actions'],
            'num_steps': rollout['num_steps'],
            'start_position': list(episode.start_position),
            'start_rotation': list(episode.start_rotation),
            'goal_position': list(episode.goals[0].position) if len(episode.goals) > 0 else [],
            'success': rollout['success'],
            'spl': rollout['spl'],
            'path_length': rollout['path_length'],
            'distance_to_goal': rollout['distance_to_goal'],
        }
        
        # Add cache metadata
        if self.cache_duplicates:
            record['cache_key'] = cache_key
            record['metrics_reused'] = metrics_reused
            if reused_from is not None:
                record['reused_from'] = reused_from
        
        # Add visualization path
        if video_path:
            record['visualization_video'] = video_path
        
        # Write to file
        f_out.write(json.dumps(record) + '\n')
        f_out.flush()
        
        # Track this episode as processed (CRITICAL: prevent reprocessing on resume)
        self.processed_episode_ids.add(episode_id)
        
        # Update statistics
        self.stats['total_processed'] += 1
        self.stats['total_steps'] += rollout['num_steps']
        if rollout['success']:
            self.stats['successes'] += 1
        else:
            self.stats['failures'] += 1
        
        scene_name = scene_id.split('/')[-1].replace('.glb', '')
        self.stats['scene_counts'][scene_name] += 1
        
        return True
    
    def sanity_test(self):
        """
        Quick sanity test: run 3 episodes in each oracle mode and verify behavior.
        
        Prints:
        - Total steps taken
        - Final distance to goal
        - Whether STOP was issued (indicates proper termination)
        - Waypoint progression (for reference_path mode)
        """
        print()
        print("=" * 80)
        print("SANITY TEST: Quick validation of oracle modes")
        print("=" * 80)
        print()
        
        env = self.setup_environment()
        
        for test_mode in ["goal", "reference_path"]:
            print(f"Testing oracle_mode = '{test_mode}':")
            print("-" * 80)
            
            self.oracle_mode = test_mode
            episodes_tested = 0
            
            for episode_idx, episode in enumerate(env.episodes):
                if episodes_tested >= 3:
                    break
                
                rollout = self.rollout_episode(env, episode)
                
                if rollout is None:
                    continue
                
                episodes_tested += 1
                
                # Extract test metrics
                num_steps = rollout['num_steps']
                dist_to_goal = rollout['distance_to_goal']
                success = rollout['success']
                actions = rollout['actions']
                
                # Check if STOP was issued
                stop_issued = 0 in actions
                
                print(f"  Ep {episode.episode_id}: steps={num_steps:3d}, "
                      f"dist={dist_to_goal:6.2f}m, success={success}, stop={stop_issued}")
                
                # For reference_path mode, check waypoint progression
                if test_mode == "reference_path" and self.verify_oracle:
                    wp_indices = [
                        v.get('waypoint_idx') 
                        for v in rollout['verification_log'] 
                        if v.get('waypoint_idx') is not None
                    ]
                    if wp_indices:
                        final_wp = max(wp_indices)
                        total_wps = len(getattr(episode, 'reference_path', []))
                        print(f"      Waypoints: reached wp {final_wp}/{total_wps}")
            
            print()
        
        env.close()
        print("=" * 80)
        print("SANITY TEST COMPLETE")
        print("=" * 80)
        print()
    
    def test_single_episode(self, episode_id: int):
        """
        Test a single episode by ID for debugging.
        
        Shows:
        - Episode metadata (scene, goals, reference path)
        - Oracle rollout with detailed step-by-step info
        - Final metrics (success, distance, SPL)
        - Verification logs (for both oracle modes)
        - Visualization videos (if --visualize flag is set)
        """
        print()
        print("=" * 80)
        print(f"TESTING SINGLE EPISODE: {episode_id}")
        print("=" * 80)
        print()
        
        env = self.setup_environment()
        
        # Find the episode by ID
        target_episode = None
        for episode in env.episodes:
            if int(episode.episode_id) == episode_id:
                target_episode = episode
                break
        
        if target_episode is None:
            print(f"❌ Episode {episode_id} not found in dataset")
            env.close()
            return
        
        episode = target_episode
        
        # Print episode info
        print(f"Episode ID: {episode.episode_id}")
        print(f"Scene: {episode.scene_id}")
        print(f"Instruction: {episode.instruction.instruction_text}")
        print(f"Start position: {episode.start_position}")
        print(f"Start rotation: {episode.start_rotation}")
        print(f"Goal position: {episode.goals[0].position if episode.goals else 'N/A'}")
        print(f"Reference path length: {len(getattr(episode, 'reference_path', []))} waypoints")
        print()
        
        # Test both modes
        for test_mode in ["goal", "reference_path"]:
            print("-" * 80)
            print(f"Oracle Mode: {test_mode}")
            print("-" * 80)
            
            self.oracle_mode = test_mode
            self.verify_oracle = True  # Enable detailed logging
            
            rollout = self.rollout_episode(env, episode)
            
            if rollout is None:
                print(f"  ❌ Episode rollout failed (missing scene or error)")
                continue
            
            # Print results
            print(f"  Success: {rollout['success']}")
            print(f"  Steps taken: {rollout['num_steps']}")
            print(f"  Distance to goal: {rollout['distance_to_goal']:.2f}m")
            print(f"  Path length: {rollout['path_length']:.2f}m")
            print(f"  SPL: {rollout['spl']:.4f}")
            print()
            
            # Print first 10 steps of verification log
            print(f"  Verification log (first 10 steps):")
            for i, step_log in enumerate(rollout['verification_log'][:10]):
                action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
                action_name = action_names[step_log['oracle_action']] if step_log['oracle_action'] < 4 else "UNKNOWN"
                
                log_str = f"    Step {i:2d}: action={action_name:12s}, dist_to_goal={step_log['dist_to_goal']:6.2f}m"
                
                if step_log.get('dist_to_current_wp') is not None:
                    log_str += f", dist_to_wp={step_log['dist_to_current_wp']:6.2f}m"
                    if step_log.get('waypoint_idx') is not None:
                        log_str += f", wp_idx={step_log['waypoint_idx']}"
                
                print(log_str)
            
            if len(rollout['verification_log']) > 10:
                print(f"    ... ({len(rollout['verification_log']) - 10} more steps)")
            
            print()
            
            # Print height changes (for multi-floor debugging)
            if rollout['height_changes']:
                height_changes = rollout['height_changes']
                if len(height_changes) > 1:  # Only show if height actually changed
                    print(f"  Height changes during navigation:")
                    for change in height_changes:
                        delta_str = ""
                        if 'height_delta' in change:
                            delta_str = f" (delta: {change['height_delta']:+.2f}m)"
                        print(f"    Step {change['step']:2d}: height={change['height']:6.2f}m, pos=({change['pos'][0]:7.2f}, {change['pos'][1]:7.2f}){delta_str}")
                    print()
            
            # Save visualization video if frames are available
            if self.visualize and rollout['frames']:
                video_path = str(
                    self.visualization_dir / f"test_ep_{episode_id:05d}_{test_mode}_success_{rollout['success']}.mp4"
                )
                try:
                    # Convert frames to numpy arrays if needed
                    frame_arrays = []
                    for frame in rollout['frames']:
                        if isinstance(frame, Image.Image):
                            frame_arrays.append(np.array(frame))
                        elif isinstance(frame, np.ndarray):
                            # Ensure uint8
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                            frame_arrays.append(frame)
                        else:
                            continue
                    
                    if frame_arrays:
                        imageio.mimsave(video_path, frame_arrays, fps=5)
                        print(f"  ✓ Video saved: {video_path}")
                    else:
                        print(f"  ⚠ No video frames captured")
                except Exception as e:
                    print(f"  ❌ Failed to save video: {e}")
        
        env.close()
        print("=" * 80)
        print("EPISODE TEST COMPLETE")
        print("=" * 80)
        print()
    
    def run(self):
        """Main export loop"""
        start_time = time.time()
        
        # Load resume state
        self.load_processed_episodes()
        
        # Setup environment
        env = self.setup_environment()
        
        # Prepare output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if self.resume else 'w'
        
        print(f"Starting export...")
        print(f"  Output mode: {mode}")
        if self.resume:
            print(f"  Resuming from {len(self.processed_episode_ids)} already-processed episodes")
        print()
        
        with open(self.output_file, mode) as f_out:
            # Calculate remaining episodes
            total_episodes = len(env.episodes)
            remaining_episodes = total_episodes - len(self.processed_episode_ids)
            
            if self.max_episodes:
                remaining_episodes = min(remaining_episodes, self.max_episodes - len(self.processed_episode_ids))
            
            print(f"Total episodes available: {total_episodes}")
            print(f"Already processed: {len(self.processed_episode_ids)}")
            print(f"Remaining to process: {remaining_episodes}")
            print()
            
            pbar = tqdm(total=remaining_episodes, desc="Exporting")
            
            episodes_processed = 0
            episodes_skipped = 0
            
            # Iterate through all episodes by index to avoid infinite loops
            for episode_idx in range(len(env.episodes)):
                # Check max episodes limit (considering already-processed)
                if self.max_episodes and len(self.processed_episode_ids) + episodes_processed >= self.max_episodes:
                    break
                
                # Get episode at this index
                episode = env.episodes[episode_idx]
                episode_id = int(episode.episode_id)
                
                # Skip if already processed
                if episode_id in self.processed_episode_ids:
                    episodes_skipped += 1
                    continue
                
                # Export episode
                success = self.export_episode(env, episode, f_out)
                
                if success:
                    episodes_processed += 1
                    
                    # Update progress
                    cache_rate = 0
                    if self.cache_duplicates and self.stats['cache_misses'] > 0:
                        total_checks = self.stats['cache_hits'] + self.stats['cache_misses']
                        cache_rate = 100 * self.stats['cache_hits'] / total_checks
                    
                    pbar.set_postfix({
                        'cache': f"{cache_rate:.1f}%",
                        'success': f"{100*self.stats['successes']/(episodes_processed or 1):.1f}%",
                        'skipped': episodes_skipped,
                    })
                    pbar.update(1)
        
        pbar.close()
        env.close()
        
        # Final statistics
        elapsed = time.time() - start_time
        
        print()
        print("=" * 80)
        print("EXPORT COMPLETE")
        print("=" * 80)
        print(f"Output file: {self.output_file}")
        print(f"File size: {self.output_file.stat().st_size / (1024**2):.1f} MB")
        print()
        print("Statistics:")
        print(f"  Episodes processed: {self.stats['total_processed']}")
        print(f"  Successes: {self.stats['successes']} ({100*self.stats['successes']/(self.stats['total_processed'] or 1):.1f}%)")
        print(f"  Failures: {self.stats['failures']}")
        print(f"  Total steps: {self.stats['total_steps']}")
        print(f"  Avg steps/episode: {self.stats['total_steps']/(self.stats['total_processed'] or 1):.1f}")
        print()
        
        if self.cache_duplicates:
            total_cache_checks = self.stats['cache_hits'] + self.stats['cache_misses']
            cache_rate = 100 * self.stats['cache_hits'] / (total_cache_checks or 1)
            print("Cache statistics:")
            print(f"  Cache hits: {self.stats['cache_hits']}")
            print(f"  Cache misses: {self.stats['cache_misses']}")
            print(f"  Hit rate: {cache_rate:.1f}%")
            print(f"  Unique paths: {len(self.cache)}")
            print()
        
        if self.verify_oracle:
            total_analyzed = self.stats['ref_path_follows'] + self.stats['goal_follows']
            if total_analyzed > 0:
                print("Oracle behavior analysis:")
                print(f"  Episodes following reference path: {self.stats['ref_path_follows']} ({100*self.stats['ref_path_follows']/total_analyzed:.1f}%)")
                print(f"  Episodes going to final goal: {self.stats['goal_follows']} ({100*self.stats['goal_follows']/total_analyzed:.1f}%)")
                print()
        
        print("Top 10 scenes by episode count:")
        top_scenes = sorted(self.stats['scene_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (scene, count) in enumerate(top_scenes, 1):
            print(f"  {i:2d}. {scene:30s}: {count:4d} episodes")
        print()
        
        print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"Speed: {self.stats['total_processed']/elapsed:.2f} episodes/sec")
        
        if self.visualize:
            print()
            print(f"Visualizations saved to: {self.visualization_dir}")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Export oracle pose deltas from R2R_VLNCE")
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val_seen", "val_unseen"],
        help="Dataset split to export"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="oracle_exports/oracle_deltas_train.jsonl",
        help="Output JSONL file path"
    )
    
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to export (default: all)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    
    parser.add_argument(
        "--cache-duplicates",
        action="store_true",
        help="Enable duplicate-path caching (default: True)"
    )
    
    parser.add_argument(
        "--no-cache",
        dest="cache_duplicates",
        action="store_false",
        help="Disable duplicate-path caching"
    )
    
    # Set default to True
    parser.set_defaults(cache_duplicates=True)
    
    parser.add_argument(
        "--oracle-mode",
        type=str,
        default="reference_path",
        choices=["goal", "reference_path"],
        help="Oracle mode: 'goal' (shortest to final) or 'reference_path' (waypoint following)"
    )
    
    parser.add_argument(
        "--verify-oracle",
        action="store_true",
        help="Log oracle behavior verification for debugging"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization videos with top-down map view"
    )
    
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default="oracle_visualizations",
        help="Directory for visualization outputs"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file (skip processed episodes)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--sanity-test",
        action="store_true",
        help="Run quick sanity test (3 episodes per oracle mode) and exit"
    )
    
    parser.add_argument(
        "--test-episode",
        type=int,
        default=None,
        help="Test a single episode by ID (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = OracleExporter(
        split=args.split,
        output_file=args.output,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        cache_duplicates=args.cache_duplicates,
        oracle_mode=args.oracle_mode,
        verify_oracle=args.verify_oracle,
        visualize=args.visualize,
        visualization_dir=args.visualization_dir,
        resume=args.resume,
        seed=args.seed,
    )
    
    # Run sanity test if requested
    if args.sanity_test:
        exporter.sanity_test()
        return
    
    # Run single episode test if requested
    if args.test_episode is not None:
        exporter.test_single_episode(args.test_episode)
        return
    
    # Run export
    exporter.run()


if __name__ == "__main__":
    main()