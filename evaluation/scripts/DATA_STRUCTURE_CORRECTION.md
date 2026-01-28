# 🚨 CRITICAL: Data Structure Correction

## What I Got Wrong

### ❌ Original Assumption
- `video_id="914-23"` means "episode 914, ending at frame 23"
- One sample per episode
- Suffix is arbitrary

### ✅ Actual Structure
- `video_id="914-23"` means "episode 914, AT TIMESTEP 23"
- **Multiple samples per episode** (one for each timestep)
- Suffix = **current timestep** in sequential navigation
- Frames = **full history** from start up to current timestep

## Evidence

```
Episode 914 samples:
Step  0: 1 frame  (frame_0)        → action: turn left 45°
Step  1: 4 frames (frame_0→3)      → action: turn left 45°
Step  2: 7 frames (frame_0→6)      → action: turn left 45°
Step  3: 10 frames (frame_0→9)     → action: turn left 45°
Step  4: 13 frames (frame_0→12)    → action: move forward 75cm
...
Step 23: 49 frames (frame_0→48)    → action: move forward 75cm
```

**Pattern**: ~3 frames per timestep (likely 1 frame per low-level action)

## Implications for Pose Export

### Problem 1: Frame Density vs Reference Path
- **Frames**: Dense (49 frames for 6-waypoint trajectory)
- **Reference path**: Sparse (6 waypoints)
- **Gap**: 49 frames → 6 waypoints = ~8 frames per waypoint

**You cannot directly map frame_i → waypoint_j**

### Problem 2: Multiple Samples Need Different Poses
- Sample `914-1`: needs poses for frames 0→3
- Sample `914-23`: needs poses for frames 0→48
- **Same episode, different frame ranges!**

Your current exporter assumes one pose sequence per episode, but you need **per-sample** pose sequences.

### Problem 3: NaVILA Samples 8 Frames from History
The training code samples 8 frames uniformly from the history:
- Sample `914-23` has 49 frames in history
- NaVILA samples 8 frames uniformly: indices `[0, 7, 14, 21, 28, 35, 42, 48]`
- You need poses for THOSE specific 8 frames, not just any 8 waypoints

## What You Need to Fix

### Option A: Interpolate Dense Poses (Recommended)
1. Load episode with all frames
2. Interpolate reference_path to create dense pose trajectory
3. For each frame_i, compute interpolated pose
4. Export poses indexed by frame number
5. Sample script then picks poses for the 8 frames NaVILA uses

**Advantage**: Accurate pose for every frame  
**Disadvantage**: More computation

### Option B: Sample-Specific Export
1. For each NaVILA sample (video_id)
2. Determine which 8 frames NaVILA will actually use
3. Interpolate poses only for those 8 frames
4. Export directly

**Advantage**: Minimal computation  
**Disadvantage**: Tight coupling to NaVILA's sampling logic

### Option C: Approximate with Waypoint Sampling (Current)
1. Ignore frame density
2. Sample 8 waypoints from reference_path
3. Hope NaVILA's frame sampling aligns roughly with waypoints

**Advantage**: Simple  
**Disadvantage**: ⚠️ **INACCURATE** - frame 7 might not align with waypoint 1

## Recommended Fix

### Update `export_gru_poses.py`:

```python
def interpolate_poses_for_frames(reference_path, num_frames):
    """
    Interpolate dense poses for all frames in trajectory.
    
    Args:
        reference_path: List of waypoint positions [N_waypoints, 3]
        num_frames: Total number of frames rendered (e.g., 49)
    
    Returns:
        poses: [(x, y, yaw)] for each frame
    """
    # Create dense trajectory by linear interpolation
    waypoint_indices = np.linspace(0, len(reference_path)-1, len(reference_path))
    frame_indices = np.linspace(0, len(reference_path)-1, num_frames)
    
    # Interpolate x, y positions
    x_interp = np.interp(frame_indices, waypoint_indices, 
                         [wp[0] for wp in reference_path])
    y_interp = np.interp(frame_indices, waypoint_indices, 
                         [wp[2] for wp in reference_path])  # Habitat z → y
    
    # Interpolate yaw (handle wraparound)
    yaws = []
    for i in range(len(reference_path)-1):
        dx = reference_path[i+1][0] - reference_path[i][0]
        dz = reference_path[i+1][2] - reference_path[i][2]
        yaws.append(np.arctan2(dz, dx))
    yaws.append(yaws[-1])  # Repeat last yaw
    
    yaw_interp = np.interp(frame_indices, waypoint_indices, yaws)
    
    poses = [(x, y, yaw) for x, y, yaw in zip(x_interp, y_interp, yaw_interp)]
    return poses
```

Then for each sample:
```python
def export_poses_for_sample(sample, episode, num_frames_to_sample=8):
    video_id = sample["video_id"]
    frames = sample["frames"]  # List of frame paths
    
    # Get total frames in history
    last_frame_idx = int(frames[-1].split('frame_')[1].split('.')[0])
    num_frames_in_history = last_frame_idx + 1
    
    # Interpolate dense poses for ALL frames
    all_poses = interpolate_poses_for_frames(
        episode["reference_path"], 
        num_frames_in_history
    )
    
    # Sample 8 frames uniformly (matching NaVILA's logic)
    frame_indices = np.linspace(0, last_frame_idx, num_frames_to_sample, dtype=int)
    sampled_poses = [all_poses[i] for i in frame_indices]
    
    # Compute deltas
    deltas = compute_deltas(sampled_poses)
    
    return {
        "video_id": video_id,
        "poses": sampled_poses,  # 8 poses
        "deltas": deltas,        # 7 deltas
        ...
    }
```

## Action Items

1. ✅ Understand data structure (DONE)
2. 🔧 Fix `export_gru_poses.py` to interpolate dense poses
3. 🔧 Update `sample_frames_from_path()` to handle dense interpolation
4. ✅ Verify alignment with actual frame indices
5. 🧪 Test on sample `914-23` to ensure poses match frames 0,7,14,21,28,35,42,48

## Testing Command

```python
# Verify interpolation works
sample = navila_data[0]  # 914-23
episode = episodes[914]

# Should have 49 frames (0→48)
assert len(sample['frames']) == 49

# Interpolate 49 poses
poses = interpolate_poses_for_frames(episode['reference_path'], 49)
assert len(poses) == 49

# Sample 8 poses matching NaVILA's sampling
indices = np.linspace(0, 48, 8, dtype=int)  # [0, 6, 12, 18, 24, 30, 36, 48]
sampled_poses = [poses[i] for i in indices]
assert len(sampled_poses) == 8
```

---

**Bottom line**: Your current exporter samples from sparse waypoints (6 points), but needs to interpolate to dense frames (49 points), then sample the 8 frames NaVILA actually uses. This ensures pose-frame alignment.
