# NaVILA Training Pipeline & Oracle Shortest Path Follower Guide

## Overview

This guide explains how NaVILA loads environments from training datasets, how agents interact with the environment, and how to use the shortest path follower as an oracle agent.

---

## 1. Environment Loading from Training Dataset

### Key Files:
- **`evaluation/vlnce_baselines/common/environments.py`** - Environment class definitions
- **`evaluation/vlnce_baselines/common/env_utils.py`** - Environment construction utilities
- **`evaluation/habitat_extensions/config/vlnce_task.yaml`** - Task configuration
- **`evaluation/vlnce_baselines/config/default.py`** - Training configuration defaults

### Environment Creation Flow:

```python
# 1. Load config
config = get_config("vlnce_baselines/config/r2r_baselines/navila.yaml")

# 2. Set dataset split (train/val_seen/val_unseen)
config.TASK_CONFIG.DATASET.SPLIT = "train"
config.TASK_CONFIG.DATASET.DATA_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"

# 3. Create environment
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class

env_class = get_env_class(config.ENV_NAME)  # e.g., "VLNCEDaggerEnv"
envs = construct_envs(config, env_class)

# 4. Reset to get first observation
observations = envs.reset()
```

### Environment Types:

**`VLNCEDaggerEnv`** (training with DAGGER):
- Used during imitation learning training
- Supports expert oracle actions via `SHORTEST_PATH_SENSOR`
- Location: `evaluation/vlnce_baselines/common/environments.py:44`

**`VLNCEInferenceEnv`** (evaluation):
- Used during model evaluation
- Returns position, heading, and stop status in `get_info()`
- Location: `evaluation/vlnce_baselines/common/environments.py:64`

**`VLNCECollectorEnv`** (data collection):
- Used for collecting demonstrations
- Tracks oracle success metrics
- Location: `evaluation/vlnce_baselines/common/environments.py:15`

---

## 2. Agent-Environment Interaction

### Observation → Action → Environment Update Cycle

```python
# STEP 1: Get observations from environment
observations = envs.reset()

# Observation contains:
# - 'rgb': RGB image [H, W, 3]
# - 'depth': Depth image [H, W, 1]
# - 'instruction': Navigation instruction text
# - 'shortest_path_sensor': Oracle action (if sensor enabled)
# - 'gps': Agent position
# - 'compass': Agent heading

# STEP 2: Agent processes observation → decides action
action = agent.act(observations)  # Returns action ID (0-3)

# Action space (discrete):
# 0 = STOP
# 1 = MOVE_FORWARD (0.25m)
# 2 = TURN_LEFT (15°)
# 3 = TURN_RIGHT (15°)

# STEP 3: Execute action in environment
outputs = envs.step([action])
observations, rewards, dones, infos = outputs

# STEP 4: Check if episode done
if dones[0]:
    metrics = envs.get_metrics()
    # metrics contains: success, spl, distance_to_goal, etc.
```

### Input Processing (NaVILA Model):

**Location**: `evaluation/vlnce_baselines/navila_trainer.py:150-250`

```python
# 1. Extract RGB frame
curr_rgb = Image.fromarray(batch[0]["rgb"].cpu().numpy())

# 2. Collect historical frames (8 frames)
past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=8)

# 3. Build multimodal prompt
instruction = current_episodes[0].instruction.instruction_text
question = (
    f"Imagine you are a robot programmed for navigation tasks. "
    f"You have been given a video of historical observations {interleaved_images}, "
    f'and current observation <image>. Your assigned task is: "{instruction}" '
    f"Analyze this series of images to decide your next action..."
)

# 4. Model inference
images_tensor = process_images(past_and_current_rgbs, image_processor, model.config)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
output_ids = model.generate(input_ids, images=images_tensor, ...)

# 5. Parse output text → action
# Model outputs: "The action is move forward 25 cm"
# Parse to get action ID (0-3)
```

---

## 3. Shortest Path Follower as Oracle Agent

### How It Works:

The shortest path follower provides **oracle (expert) actions** for imitation learning. It computes the optimal next action to reach the goal using the simulator's geodesic distance function.

### Key Components:

**Sensor**: `ShortestPathSensor` 
- **Location**: `evaluation/habitat_extensions/sensors.py:118`
- Provides oracle action at each timestep
- Returns action ID in `observations['shortest_path_sensor']`

**Follower**: `ShortestPathFollowerCompat`
- **Location**: `evaluation/habitat_extensions/shortest_path_follower.py:23`
- Computes next action to minimize geodesic distance
- Two modes: `geodesic_path` or `greedy`

### Configuration:

**In `habitat_extensions/config/default.py`**:
```python
_C.TASK.SHORTEST_PATH_SENSOR = CN()
_C.TASK.SHORTEST_PATH_SENSOR.TYPE = "ShortestPathSensor"
_C.TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5  # Success threshold
_C.TASK.SHORTEST_PATH_SENSOR.USE_ORIGINAL_FOLLOWER = False
```

**In `vlnce_task.yaml`**:
```yaml
TASK:
  SENSORS: [INSTRUCTION_SENSOR, SHORTEST_PATH_SENSOR, VLN_ORACLE_PROGRESS_SENSOR]
```

---

## 4. Using Shortest Path Follower in Training (DAGGER)

### DAGGER Training Loop:

**Location**: `evaluation/vlnce_baselines/dagger_trainer.py:210-450`

```python
# 1. Enable shortest path sensor
expert_uuid = config.IL.DAGGER.expert_policy_sensor_uuid  # "shortest_path_sensor"
EPS = config.IL.DAGGER.expert_policy_sensor  # "SHORTEST_PATH_SENSOR"
config.TASK_CONFIG.TASK.SENSORS.append(EPS)

# 2. Create environments
envs = construct_envs(config, get_env_class(config.ENV_NAME))

# 3. Collection loop
observations = envs.reset()
beta = p ** data_it  # Expert mixing probability

for step in range(max_steps):
    # 4. Get model action
    actions, rnn_states = policy.act(batch, rnn_states, prev_actions, not_done_masks)
    
    # 5. Mix model action with oracle action (ε-greedy)
    oracle_actions = batch[expert_uuid].long()  # From shortest_path_sensor
    actions = torch.where(
        torch.rand_like(actions, dtype=torch.float) < beta,
        oracle_actions,  # Use oracle
        actions,         # Use model
    )
    
    # 6. Store trajectory (obs, prev_action, oracle_action)
    episodes[i].append((
        observations[i],
        prev_actions[i].item(),
        oracle_actions[i].item(),  # Oracle label for training
    ))
    
    # 7. Execute action
    outputs = envs.step([a[0].item() for a in actions])
    observations, _, dones, _ = zip(*outputs)

# 8. Save trajectories to LMDB
# Dataset contains: (observations, prev_actions, oracle_actions)
```

### Oracle Action Flow:

```
Episode Goal Position
         ↓
ShortestPathSensor.get_observation(episode)
         ↓
ShortestPathFollowerCompat.get_next_action(goal_position)
         ↓
    [Computes geodesic distance]
    [Estimates max gradient direction]
    [Selects STOP/FORWARD/LEFT/RIGHT]
         ↓
observation['shortest_path_sensor'] = [action_id]
```

---

## 5. Standalone Oracle Agent Example

### Option 1: Using ShortestPathSensor in Environment

```python
from habitat import Env
from vlnce_baselines.config.default import get_config

# Configure with shortest path sensor
config = get_config("vlnce_baselines/config/r2r_baselines/navila.yaml", [
    "TASK_CONFIG.DATASET.SPLIT", "train",
    "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", "500",
])

# Ensure sensor is enabled
if "SHORTEST_PATH_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
    config.TASK_CONFIG.TASK.SENSORS.append("SHORTEST_PATH_SENSOR")

# Create environment
env = Env(config=config.TASK_CONFIG)

# Run episode with oracle
observations = env.reset()
print(f"Episode: {env.current_episode.episode_id}")
print(f"Goal: {env.current_episode.goals[0].position}")

step = 0
while not env.episode_over:
    # Get oracle action from sensor
    oracle_action = int(observations["shortest_path_sensor"][0])
    
    print(f"Step {step}: Action={oracle_action}")
    
    # Execute oracle action
    observations = env.step(oracle_action)
    step += 1

# Check metrics
metrics = env.get_metrics()
print(f"Success: {metrics['success']}")
print(f"SPL: {metrics['spl']:.3f}")
print(f"Path Length: {metrics['path_length']:.2f}m")
```

### Option 2: Direct ShortestPathFollower Usage

```python
from habitat import Env
from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat

# Create environment
env = Env(config=config.TASK_CONFIG)
observations = env.reset()

# Create follower
goal_radius = 0.5
follower = ShortestPathFollowerCompat(
    env.sim, 
    goal_radius=goal_radius, 
    return_one_hot=False
)

# Run episode
goal_position = env.current_episode.goals[0].position
while not env.episode_over:
    # Get next action from follower
    action = follower.get_next_action(goal_position)
    
    if action is None:
        action = 0  # STOP - reached goal
    
    observations = env.step(action)
```

---

## 6. Dataset Structure for Training

### LMDB Dataset (Created by DAGGER):

**Location**: `config.IL.DAGGER.lmdb_features_dir`

```python
# Each entry contains:
{
    'observations': {
        'rgb': [T, H, W, 3],
        'depth': [T, H, W, 1],
        'instruction': str,
        # ... other sensors
    },
    'prev_actions': [T],      # Agent's previous actions
    'oracle_actions': [T],    # Oracle labels (from shortest_path_sensor)
}
```

### Training Dataset Loader:

**Location**: `evaluation/vlnce_baselines/dagger_trainer.py:115-200`

```python
class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        # Load trajectories from LMDB
        obs, prev_actions, oracle_actions = msgpack_numpy.unpackb(data)
        
        # Apply inflection weights
        oracle_actions = torch.from_numpy(oracle_actions)
        not_same = (oracle_actions[1:] != oracle_actions[:-1]).long()
        weights = self.inflec_weights[not_same]
        
        return (obs, prev_actions, oracle_actions, weights)
```

---

## 7. Key Configuration Parameters

### Dataset:
```yaml
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: train  # train, val_seen, val_unseen, test
  DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz
```

### DAGGER Training:
```python
IL.DAGGER.iterations = 10           # Dataset aggregation rounds
IL.DAGGER.update_size = 5000        # Episodes per iteration
IL.DAGGER.p = 0.75                  # Expert probability
IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
```

### Shortest Path:
```python
TASK.SHORTEST_PATH_SENSOR.GOAL_RADIUS = 0.5
TASK.SHORTEST_PATH_SENSOR.USE_ORIGINAL_FOLLOWER = False
```

---

## 8. Summary: Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATASET LOADING                                          │
│    - Load R2R episodes from .json.gz                        │
│    - Each episode: start_pos, goal_pos, instruction        │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ENVIRONMENT CREATION                                     │
│    - VLNCEDaggerEnv (train) or VLNCEInferenceEnv (eval)   │
│    - Initialize Habitat-Sim with scenes (MP3D)             │
│    - Enable sensors: RGB, Depth, Instruction,              │
│                      ShortestPathSensor (oracle)           │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. EPISODE RESET                                            │
│    - Reset simulator to episode.start_position             │
│    - Load episode.instruction                               │
│    - Set episode.goal_position                              │
│    - Return initial observations                            │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. AGENT DECISION (Multiple Options)                        │
│                                                             │
│    A) Oracle (Shortest Path Follower):                      │
│       observations['shortest_path_sensor'][0]               │
│                                                             │
│    B) Learned Policy:                                       │
│       action = policy.act(observations)                     │
│                                                             │
│    C) DAGGER (Mix both):                                    │
│       action = oracle (prob β) or policy (prob 1-β)        │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ACTION EXECUTION                                         │
│    - env.step(action)                                       │
│    - Simulator moves agent: MOVE_FORWARD/TURN_LEFT/etc.    │
│    - Return new observations                                │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. TRAJECTORY STORAGE (Training)                            │
│    - Store (observation, prev_action, oracle_action)        │
│    - Save to LMDB for supervised learning                   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. EPISODE END                                              │
│    - Check: env.episode_over or success                     │
│    - Compute metrics: SPL, Success, NDTW, Path Length      │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Quick Commands

### Run Oracle on Training Split (Single Episode):
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation

python -c "
from habitat import Env
from vlnce_baselines.config.default import get_config

config = get_config('vlnce_baselines/config/r2r_baselines/navila.yaml', [
    'TASK_CONFIG.DATASET.SPLIT', 'train',
    'EVAL.EPISODE_COUNT', '1',
])
config.TASK_CONFIG.TASK.SENSORS.append('SHORTEST_PATH_SENSOR')

env = Env(config=config.TASK_CONFIG)
obs = env.reset()

while not env.episode_over:
    action = int(obs['shortest_path_sensor'][0])
    obs = env.step(action)

print('Metrics:', env.get_metrics())
"
```

### Train with DAGGER (Uses Oracle as Expert):
```bash
cd /home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation

python run.py \
    --exp-config vlnce_baselines/config/r2r_baselines/dagger.yaml \
    --run-type train
```

---

## Key Takeaways:

1. **Environment**: Habitat-Sim loads MP3D scenes + VLN-CE episodes
2. **Observations**: RGB, Depth, Instruction, GPS, Compass, (optional) ShortestPathSensor
3. **Actions**: Discrete (STOP=0, FORWARD=1, LEFT=2, RIGHT=3)
4. **Oracle**: ShortestPathSensor provides optimal action using geodesic distance
5. **Training**: DAGGER mixes oracle actions (β probability) with model actions
6. **Dataset**: Trajectories stored as (obs, prev_action, oracle_action) tuples

