# GRU Pose Export Pipeline for NaVILA

This directory contains scripts to export ground-truth poses from VLN-CE simulator data for training NaVILA's GRU motion encoding branch.

## Overview

The pose export pipeline creates JSONL sidecar files that contain ground-truth (x, y, yaw) poses and motion deltas for each NaVILA training sample. These poses are extracted from the VLN-CE reference trajectories using the Habitat simulator.

## Prerequisites

✓ **Environment**: Use the `navila-eval` conda environment (see main README)
✓ **VLN-CE Data**: `evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed/` and `RxR_VLNCE_v0/`
✓ **MP3D Scenes**: `evaluation/data/scene_datasets/mp3d/`
✓ **NaVILA Dataset**: `/home/rithvik/NaVILA-Dataset/R2R/` and `RxR/` with `annotations.json`

Run the diagnostic to verify your setup:
```bash
cd evaluation/scripts
python check_setup.py
```

## Quick Start

### 1. Test the pipeline (recommended first step)
```bash
cd evaluation/scripts
python test_pose_export.py
```

This will verify that:
- NaVILA annotations are properly formatted
- VLN-CE episodes can be matched to NaVILA samples
- Reference paths contain valid waypoints

### 2. Export poses for R2R

```bash
# Train split (~353K samples)
bash run_pose_export.sh --dataset r2r --split train

# Val seen split
bash run_pose_export.sh --dataset r2r --split val_seen

# Val unseen split
bash run_pose_export.sh --dataset r2r --split val_unseen
```

### 3. Export poses for RxR

```bash
# Train split
bash run_pose_export.sh --dataset rxr --split train

# Val unseen split
bash run_pose_export.sh --dataset rxr --split val_unseen
```

## Output Format

Each JSONL file contains one record per NaVILA training sample:

```json
{
  "video_id": "914-23",
  "episode_id": 914,
  "scene_id": "mp3d/8WUmhLawc2A/8WUmhLawc2A.glb",
  "poses": [
    [x0, y0, yaw0],
    [x1, y1, yaw1],
    ...
    [x7, y7, yaw7]
  ],
  "deltas": [
    [dx01, dy01, dyaw01],
    [dx12, dy12, dyaw12],
    ...
    [dx67, dy67, dyaw67]
  ],
  "dist_bins": [0, 1, 1, 2, 1, 0, 1],
  "yaw_bins": [2, 2, 3, 2, 2, 2, 1]
}
```

**Fields:**
- `poses`: List of 8 absolute poses `(x, y, yaw)` in meters and radians
  - `x, y`: 2D position in scene coordinates (Habitat x, z)
  - `yaw`: heading angle in radians (wrapped to [-π, π])
  
- `deltas`: List of 7 relative motion deltas between consecutive poses
  - `dx, dy`: Translation in meters
  - `dyaw`: Rotation delta in radians (wrapped to [-π, π])
  
- `dist_bins`: Discretized distance bins for classification (0-4)
- `yaw_bins`: Discretized yaw bins for classification (0-5)

## Default Binning

**Distance bins** (meters): `[0, 0.25, 0.5, 1.0, 2.0]`
- Bin 0: [0, 0.25)
- Bin 1: [0.25, 0.5)
- Bin 2: [0.5, 1.0)
- Bin 3: [1.0, 2.0)
- Bin 4: ≥ 2.0

**Yaw bins** (degrees): `[-180, -30, -10, 10, 30, 180]`
- Bin 0: [-180, -30) — sharp left
- Bin 1: [-30, -10) — left
- Bin 2: [-10, 10) — straight
- Bin 3: [10, 30) — right
- Bin 4: [30, 180] — sharp right

## Advanced Usage

### Custom output path
```bash
bash run_pose_export.sh \
  --dataset r2r \
  --split train \
  --output /custom/path/gru_pose_train.jsonl
```

### Custom binning
```bash
bash run_pose_export.sh \
  --dataset r2r \
  --split train \
  --dist-bins 0 0.5 1.0 2.0 \
  --yaw-bins -180 -45 -15 15 45 180
```

### Different frame count (if you change NaVILA's frame sampling)
```bash
bash run_pose_export.sh \
  --dataset r2r \
  --split train \
  --frames 16
```

## Sanity Checks

After export, verify the output:

```bash
# Check file was created
ls -lh /home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl

# Count records
wc -l /home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl

# Inspect first record
head -1 /home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl | python -m json.tool
```

Expected output:
- R2R train: ~353K records
- R2R val_seen: varies by split
- R2R val_unseen: varies by split

### Verify alignment manually

```python
import json

# Load a sample
with open('/home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl') as f:
    sample = json.loads(f.readline())

print(f"Video ID: {sample['video_id']}")
print(f"Poses: {len(sample['poses'])} (expect 8)")
print(f"Deltas: {len(sample['deltas'])} (expect 7)")

# Check delta computation
poses = sample['poses']
deltas = sample['deltas']
for i in range(len(deltas)):
    dx_computed = poses[i+1][0] - poses[i][0]
    dy_computed = poses[i+1][1] - poses[i][1]
    print(f"Delta {i}: stored=({deltas[i][0]:.3f}, {deltas[i][1]:.3f}), "
          f"computed=({dx_computed:.3f}, {dy_computed:.3f})")
```

## Files Created

After running all exports:

```
/home/rithvik/NaVILA-Dataset/
├── R2R/
│   ├── annotations.json
│   ├── gru_pose_train.jsonl         ← NEW
│   ├── gru_pose_val_seen.jsonl      ← NEW
│   └── gru_pose_val_unseen.jsonl    ← NEW
└── RxR/
    ├── annotations.json
    ├── gru_pose_train.jsonl         ← NEW
    └── gru_pose_val_unseen.jsonl    ← NEW
```

## Next Steps

After exporting poses, update the NaVILA training code:

1. **Register pose paths** in `llava/data/datasets_mixture.py`
2. **Load poses in dataset** class (see integration guide below)
3. **Train GRU** on motion deltas
4. **Fuse GRU embeddings** into NaVILA's multimodal architecture

## Troubleshooting

### "Episode not found" errors
Some NaVILA samples may reference episodes from different splits. This is normal—the script will skip them.

### "Failed to load scene" errors
Ensure MP3D scene data is properly downloaded and extracted in `evaluation/data/scene_datasets/mp3d/`.

### Memory issues with large exports
The script processes samples sequentially and writes incrementally, so memory usage should be stable even for large splits.

### Coordinate system confusion
- Habitat uses `[x, y_up, z]` where `y_up` is vertical
- We convert to 2D navigation: `(x, z) → (x, y)` and extract yaw from quaternion

## Technical Notes

### Pose extraction method
The script extracts poses from VLN-CE `reference_path` which contains ground-truth waypoint positions. Since NaVILA samples 8 frames uniformly from a trajectory, we:
1. Parse `video_id` to get `episode_id` and `end_frame_idx`
2. Sample 8 indices uniformly from `[0, end_frame_idx]`
3. Map those indices to `reference_path` waypoints
4. Compute heading from waypoint-to-waypoint direction vectors

### Alternative: Using Habitat simulator live
For higher accuracy, you could teleport the Habitat agent to each waypoint and query `sim.get_agent_state()`. The current implementation is faster and sufficient for most use cases.

## Scripts Reference

- `export_gru_poses.py` — Main export script
- `run_pose_export.sh` — Wrapper to activate navila-eval environment
- `test_pose_export.py` — Quick sanity check with 5 samples
- `check_setup.py` — Diagnostic to verify prerequisites

## Questions?

If you encounter issues or need clarification on the coordinate system, binning strategy, or frame sampling, please refer to the main NaVILA README or open an issue.
