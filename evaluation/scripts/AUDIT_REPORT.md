# Audit Report: Existing Frame Rendering & Pose Export Infrastructure

**Date**: January 27, 2026  
**Purpose**: Determine if NaVILA already has scripts to render R2R episodes into frames and/or export agent poses

---

## Executive Summary

**Finding**: NaVILA **does NOT** have existing scripts to render VLN-CE R2R episodes into the training dataset format (`NaVILA-Dataset/R2R/<episode_id>/frame_*.jpg`).

**What exists**:
- ✅ Evaluation-time video generation for qualitative analysis
- ✅ Agent state queries during inference
- ✅ Frame extraction from YouTube videos (Human dataset)
- ❌ **No training data frame rendering pipeline**
- ❌ **No pose logging during frame rendering**

**Recommendation**: Create a new script `evaluation/scripts/render_r2r_episodes.py` that renders VLN-CE episodes into the NaVILA training format **while simultaneously exporting poses**.

---

## Part A: Candidate Files & What They Do

### 1. **Video Generation During Evaluation**
**Files**: 
- `evaluation/vlnce_baselines/cma_trainer.py` (lines 400-445)
- `evaluation/vlnce_baselines/navila_trainer.py` (lines 1-400)
- `evaluation/habitat_extensions/utils.py::generate_video()` (lines 658-693)

**What they do**:
```python
# During eval, accumulate RGB frames
rgb_frames[i].append(frame)

# At episode end, generate video
generate_video(
    video_option=config.VIDEO_OPTION,
    video_dir=config.VIDEO_DIR,
    images=rgb_frames[i],
    episode_id=ep_id,
    ...
)
```

**Purpose**: Generate **evaluation videos** for qualitative analysis (saved to `eval_out/`)

**Key differences from training data**:
- ❌ Saves videos, not individual frame JPGs
- ❌ Uses agent's inferred trajectory (not ground-truth reference path)
- ❌ Output directory is `eval_out/`, not `NaVILA-Dataset/`
- ❌ No pose logging

---

### 2. **Agent State Queries**
**Files**:
- `evaluation/habitat_extensions/measures.py` (lines 49, 53, 160, 169, etc.)
- `evaluation/habitat_extensions/sensors.py` (lines 42, 69)
- `evaluation/vlnce_baselines/common/environments.py` (lines 78-82)
- `evaluation/habitat_extensions/shortest_path_follower.py` (lines 81-82, 129)

**What they do**:
```python
# Query agent state during episode
agent_state = self._sim.get_agent_state()
position = agent_state.position  # [x, y_up, z]
rotation = agent_state.rotation  # quaternion

# Teleport agent to specific pose
self._sim.set_agent_state(position, rotation, reset_sensors=False)
```

**Purpose**: 
- Compute navigation metrics (distance traveled, NDTW, etc.)
- Shortest path following
- Goal distance checks

**Key observations**:
- ✅ Infrastructure exists for reading/setting agent poses
- ❌ Not used for frame rendering or pose export
- ✅ Can be reused for our pose export script

---

### 3. **Frame Extraction from Videos**
**File**: `scripts/extract_rawframes.py`

**What it does**:
```python
# Extract frames from YouTube videos (Human dataset)
subprocess.call([
    "ffmpeg", "-i", videopath,
    "-vf", "fps=1",
    dest + "/%04d.jpg",
])
```

**Purpose**: Extract frames from downloaded YouTube videos for Human touring dataset

**Key differences**:
- ✅ Writes individual frames as JPGs (good template!)
- ❌ Works on pre-recorded videos, not Habitat simulator
- ❌ No pose data
- ❌ Not applicable to R2R/RxR (which need simulator rendering)

---

### 4. **VLN-CE Dataset Loading**
**Files**:
- `evaluation/habitat_extensions/task.py::VLNCEDatasetV1` (lines 56-115)
- `evaluation/habitat_extensions/config/vlnce_task.yaml`

**What they do**:
```python
# Load R2R episodes from json.gz
dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
# e.g., "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"

with gzip.open(dataset_filename, "rt") as f:
    deserialized = json.loads(f.read())
    
for episode in deserialized["episodes"]:
    episode = VLNExtendedEpisode(**episode)
    # episode.reference_path: List[List[float]] - ground truth waypoints
    # episode.scene_id: str - MP3D scene
```

**Purpose**: Load VLN-CE episodes for evaluation

**Key observations**:
- ✅ Already reads R2R_VLNCE json.gz files
- ✅ Has `reference_path` (ground-truth waypoints)
- ✅ Can be reused for our rendering script

---

## Part B: Do They Log Poses?

**Short answer: NO**

**Detailed findings**:

### Evaluation Scripts
- **Video generation**: Only saves RGB frames to video
- **No pose logging**: Agent states are queried for metrics but never written to disk
- **Output format**: Videos (`.mp4`), not individual frames + poses

### Shortest Path Follower
- **Purpose**: Compute shortest path actions for metrics
- **Pose usage**: Temporarily teleports agent to test actions, then resets
- **No persistence**: Poses are never saved

### Frame Extraction Script
- **Purpose**: Extract frames from pre-recorded YouTube videos
- **No simulator**: Works on videos, not Habitat episodes
- **No pose data**: Videos don't contain pose information

### Data Loading
- **Reference paths**: Episodes contain `reference_path` (waypoint positions)
- **Not rendered**: These are loaded for evaluation, not for rendering training data
- **No pose export**: Dataset loader doesn't write poses to files

---

## Part C: Proposed Solution

### Missing Component: Training Data Renderer

**Location**: `evaluation/scripts/render_r2r_episodes.py`

**Purpose**: 
1. Load VLN-CE R2R episodes from `train.json.gz`
2. For each episode, render frames by stepping through `reference_path`
3. Save frames as `NaVILA-Dataset/R2R/<episode_id>/frame_<idx>.jpg`
4. **Simultaneously** export poses to `NaVILA-Dataset/R2R/gru_pose_train.jsonl`

**Why create a new script?**
- Existing eval scripts are optimized for inference (agent-driven trajectory)
- Training data needs **ground-truth reference path** rendering
- Need specific output format: individual JPGs + pose sidecar
- Decouple from evaluation infrastructure

---

### Recommended Architecture

```python
#!/usr/bin/env python3
"""
Render VLN-CE R2R episodes into NaVILA training dataset format.

This script:
1. Loads R2R episodes from VLN-CE json.gz
2. Creates Habitat sim environment
3. For each episode:
   - Teleports agent to each waypoint in reference_path
   - Captures RGB observation
   - Records agent_state (position + rotation)
   - Saves frame as JPG
   - Logs pose to JSONL
4. Outputs:
   - Frames: NaVILA-Dataset/R2R/<episode_id>/frame_<idx>.jpg
   - Poses: NaVILA-Dataset/R2R/gru_pose_train.jsonl
"""
```

---

### What to Reuse from Existing Code

#### 1. **Episode Loading** (from `task.py`)
```python
# Reuse VLNCEDatasetV1 class
from habitat_extensions.task import VLNCEDatasetV1

dataset = VLNCEDatasetV1(config)
for episode in dataset.episodes:
    scene_id = episode.scene_id
    reference_path = episode.reference_path  # List of waypoint positions
    episode_id = episode.episode_id
```

#### 2. **Habitat Environment Setup** (from trainers)
```python
# Reuse environment construction utilities
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false

# Or direct sim creation
import habitat
env = habitat.Env(config=config)
```

#### 3. **Agent Teleportation** (from `shortest_path_follower.py`)
```python
# Reuse teleportation pattern
sim.set_agent_state(position, rotation, reset_sensors=False)
agent_state = sim.get_agent_state()
```

#### 4. **RGB Observation Capture** (from trainers)
```python
# After teleporting to waypoint
observations = env.sim.get_sensor_observations()
rgb_frame = observations["rgb"]

# Convert to PIL Image and save
from PIL import Image
img = Image.fromarray(rgb_frame)
img.save(f"NaVILA-Dataset/R2R/{episode_id}/frame_{idx}.jpg")
```

#### 5. **Quaternion to Yaw** (we already added this!)
```python
# Reuse our new utility
from habitat_extensions.utils import quaternion_to_yaw

yaw = quaternion_to_yaw(agent_state.rotation)
```

#### 6. **Config Files** (reuse existing)
```python
# Base config
from habitat import get_config
config = get_config("evaluation/habitat_extensions/config/vlnce_task.yaml")
```

---

### Key Differences from Existing Code

| Aspect | Existing Eval Scripts | Proposed Renderer |
|--------|----------------------|-------------------|
| **Purpose** | Evaluate trained models | Generate training data |
| **Trajectory** | Agent-inferred actions | Ground-truth reference path |
| **Output** | Videos (mp4) | Individual frames (jpg) + poses (jsonl) |
| **Output Dir** | `eval_out/` | `NaVILA-Dataset/R2R/` |
| **Pose Logging** | No | **Yes** |
| **Navigation** | `env.step(action)` | `sim.set_agent_state(waypoint)` |

---

### Proposed Script Structure

```
evaluation/scripts/render_r2r_episodes.py
├── Load R2R episodes (VLNCEDatasetV1)
├── Create Habitat sim (reuse env_utils)
├── For each episode:
│   ├── Load scene (episode.scene_id)
│   ├── For each waypoint in reference_path:
│   │   ├── Teleport agent (sim.set_agent_state)
│   │   ├── Capture RGB (sim.get_sensor_observations)
│   │   ├── Query pose (sim.get_agent_state)
│   │   ├── Save frame JPG
│   │   └── Log pose to JSONL
│   └── Write episode metadata to annotations.json
└── Close sim
```

---

### Integration with `export_gru_poses.py`

**Option 1: Combine into single script** (Recommended)
- Render frames AND export poses in one pass
- Guarantees perfect alignment
- More efficient (single scene load per episode)

**Option 2: Keep separate** (Current approach)
- `render_r2r_episodes.py` - Render frames only
- `export_gru_poses.py` - Export poses from reference_path
- Pros: Separation of concerns
- Cons: Assumes frames already exist, potential alignment issues

**Recommendation**: Create `render_r2r_episodes.py` that does **both** rendering and pose export simultaneously. This is the cleanest approach and matches your "Path A" strategy from the original plan.

---

## Summary

### What NaVILA Currently Has
✅ Evaluation video generation (for qualitative analysis)  
✅ Agent state queries (for metrics)  
✅ VLN-CE episode loading  
✅ Habitat sim infrastructure  
✅ Frame extraction from videos (Human dataset)  

### What's Missing
❌ Training data frame renderer (ground-truth reference path)  
❌ Pose export during rendering  
❌ Output to `NaVILA-Dataset/` directory structure  

### Recommendation
**Create**: `evaluation/scripts/render_r2r_episodes.py`

**Reuse**:
- `habitat_extensions.task.VLNCEDatasetV1` - Episode loading
- `vlnce_baselines.common.env_utils` - Habitat env setup
- `sim.set_agent_state()` - Agent teleportation
- `sim.get_agent_state()` - Pose queries
- `habitat_extensions.utils.quaternion_to_yaw()` - Yaw extraction
- Config files: `evaluation/habitat_extensions/config/vlnce_task.yaml`

**Benefits of combined script**:
- Perfect frame-pose alignment (render + export in one pass)
- More efficient (single scene load)
- Matches "Path A" strategy from original plan
- Cleaner than post-hoc pose extraction

---

## Next Step

Should I create `render_r2r_episodes.py` that:
1. Renders frames to `NaVILA-Dataset/R2R/<episode_id>/frame_*.jpg`
2. Exports poses to `NaVILA-Dataset/R2R/gru_pose_train.jsonl`
3. Combines the best of both approaches?

This would replace the need for `export_gru_poses.py` (which assumes frames exist) and give you the **ground-truth aligned** data you need.
