# GRU Pose Export Summary

## What We Built

A complete pipeline to export ground-truth poses from VLN-CE/Habitat simulator data for training NaVILA's GRU motion encoding branch.

## Files Created

### Core Scripts
1. **`export_gru_poses.py`** — Main export script that:
   - Loads NaVILA annotations (R2R/RxR)
   - Maps samples to VLN-CE episodes
   - Extracts poses from reference paths
   - Computes motion deltas (dx, dy, dyaw)
   - Discretizes into bins for classification
   - Exports to JSONL sidecars

2. **`run_pose_export.sh`** — Wrapper to activate navila-eval environment and run exports

3. **`export_all_poses.sh`** — Batch script to export all splits (train/val_seen/val_unseen) for both R2R and RxR

### Testing & Utilities
4. **`test_pose_export.py`** — Quick sanity check with 5 samples to verify data alignment

5. **`check_setup.py`** — Diagnostic script to verify all prerequisites are installed

### Documentation
6. **`POSE_EXPORT_README.md`** — Complete guide for running pose export
   - Prerequisites checklist
   - Quick start commands
   - Output format specification
   - Binning strategy
   - Sanity checks
   - Troubleshooting

7. **`INTEGRATION_GUIDE.md`** — Step-by-step guide for integrating poses into NaVILA training
   - Update `datasets_mixture.py`
   - Modify `LazyVLNCEDataset`
   - Update data collator
   - Use poses in model forward pass
   - Testing instructions

8. **`utils.py`** — Added `quaternion_to_yaw()` helper function

## Current Status

✅ **Complete and Ready to Run**:
- All scripts created and tested
- Environment diagnostic confirms all prerequisites installed
- Test script verified data alignment
- Comprehensive documentation provided

⏳ **Next Steps for You**:
1. Run pose export for all splits
2. Integrate poses into training code
3. Train GRU module
4. Evaluate results

## Quick Start (Step-by-Step)

### Phase 1: Export Poses (~30-60 min depending on dataset size)

```bash
cd evaluation/scripts

# Test first (5 samples)
python test_pose_export.py

# Export all splits (this will take a while)
bash export_all_poses.sh
```

**Expected output files**:
```
/home/rithvik/NaVILA-Dataset/
├── R2R/
│   ├── gru_pose_train.jsonl         (~353K records, ~50MB)
│   ├── gru_pose_val_seen.jsonl
│   └── gru_pose_val_unseen.jsonl
└── RxR/
    ├── gru_pose_train.jsonl
    └── gru_pose_val_unseen.jsonl
```

### Phase 2: Integrate into Training

Follow `INTEGRATION_GUIDE.md`:

1. **Update `llava/data/datasets_mixture.py`**:
   ```python
   r2r = Dataset(
       ...,
       gru_pose_path="/home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl",
   )
   ```

2. **Update `llava/data/dataset.py`** (`LazyVLNCEDataset`):
   - Add `gru_pose_path` parameter to `__init__`
   - Load pose sidecar as dict[video_id -> pose_record]
   - Attach poses/deltas to samples in `__getitem__`

3. **Update model** to use pose data in forward pass

### Phase 3: Train & Evaluate

Use existing NaVILA training scripts with your updated dataset:

```bash
# Training (example)
bash scripts/train/sft_8frames.sh

# Evaluation (example)
cd evaluation
bash scripts/eval/r2r.sh CKPT_PATH 1 0 "0"
```

## Data Format Details

### Pose Record Schema

```json
{
  "video_id": "914-23",              // Matches NaVILA annotation
  "episode_id": 914,                 // VLN-CE episode ID
  "scene_id": "mp3d/...",           // Matterport3D scene
  
  "poses": [                         // 8 frames × (x, y, yaw)
    [x0, y0, yaw0],                 // Frame 0 pose
    [x1, y1, yaw1],                 // Frame 1 pose
    ...,
    [x7, y7, yaw7]                  // Frame 7 pose
  ],
  
  "deltas": [                        // 7 transitions × (dx, dy, dyaw)
    [dx01, dy01, dyaw01],           // Frame 0→1 motion
    [dx12, dy12, dyaw12],           // Frame 1→2 motion
    ...,
    [dx67, dy67, dyaw67]            // Frame 6→7 motion
  ],
  
  "dist_bins": [0, 1, 1, 2, 1, 0, 1],  // Discretized distances
  "yaw_bins": [2, 2, 3, 2, 2, 2, 1]    // Discretized yaw changes
}
```

### Coordinate System

- **Habitat** uses: `[x, y_up, z]` where `y_up` is vertical
- **We export**: `(x, y, yaw)` where:
  - `x` = Habitat x (meters)
  - `y` = Habitat z (meters, forward direction)
  - `yaw` = rotation around y_up axis (radians, wrapped to [-π, π])

### Binning Strategy

**Distance bins** (meters):
```
[0, 0.25, 0.5, 1.0, 2.0]
→ Bin 0: [0, 0.25)      short
→ Bin 1: [0.25, 0.5)    medium
→ Bin 2: [0.5, 1.0)     medium-long
→ Bin 3: [1.0, 2.0)     long
→ Bin 4: ≥ 2.0          very long
```

**Yaw bins** (degrees):
```
[-180, -30, -10, 10, 30, 180]
→ Bin 0: [-180, -30)    sharp left
→ Bin 1: [-30, -10)     left
→ Bin 2: [-10, 10)      straight
→ Bin 3: [10, 30)       right
→ Bin 4: [30, 180]      sharp right
```

## Architecture Choices

### Why JSONL Sidecars?

1. **Separation of concerns**: Training data (frames + instructions) separate from pose data
2. **Flexible**: Easy to regenerate/update poses without touching original data
3. **Optional**: Training can work without poses (backward compatible)
4. **Fast**: Loads entire JSONL into memory dict for O(1) lookup by video_id

### Why 8 Frames?

Matches NaVILA's `sft_8frames.sh` training script. Each sample uses exactly 8 uniformly-sampled frames from the trajectory.

### Why Deltas Instead of Absolute Poses?

1. **Relative motion** is more informative for action prediction
2. **Translation invariant**: Works regardless of starting position
3. **Better for GRU**: Sequential relative changes capture motion patterns
4. **Easier to learn**: Smaller value ranges, more stable gradients

## Verification Checklist

Before training, verify:

- [ ] All export scripts completed without errors
- [ ] JSONL files exist and have correct number of records
- [ ] Sample inspection shows 8 poses and 7 deltas per record
- [ ] Delta computation is correct (manually check a few samples)
- [ ] Coordinate system makes sense (forward motion → positive dy or distance increase)
- [ ] Yaw deltas are wrapped to [-π, π]
- [ ] Bins are distributed reasonably (not all in one bin)

## Performance Expectations

### Export Speed
- ~1-2K samples/sec on CPU
- R2R train (~353K samples): ~5-10 minutes
- Memory usage: <2GB RAM

### Storage
- ~150 bytes/record on disk
- R2R train: ~50MB
- All splits combined: ~100-200MB

### Training Impact
- Pose data adds minimal overhead (<1% training time)
- Memory impact: ~100 bytes/sample (negligible)

## Troubleshooting

### Common Issues

1. **"Module not found: habitat"**
   - You're not in navila-eval environment
   - Solution: Use `run_pose_export.sh` wrapper

2. **"Episode not found" warnings**
   - Normal for samples from different splits
   - Export script skips them automatically

3. **Mismatched frame counts**
   - Check `num_video_frames` in data_args matches export
   - Default: 8 frames

4. **Large memory usage**
   - JSONL loads entire file into memory
   - For huge datasets, consider streaming mode

### Getting Help

If stuck:
1. Check `POSE_EXPORT_README.md` for detailed troubleshooting
2. Run `python check_setup.py` to verify environment
3. Run `python test_pose_export.py` to check data alignment
4. Inspect a few records manually with `head -1 *.jsonl | python -m json.tool`

## What's Next?

### Immediate Next Steps (You)
1. **Run exports**: `bash export_all_poses.sh`
2. **Verify outputs**: Check file sizes and sample records
3. **Integrate**: Follow `INTEGRATION_GUIDE.md` step-by-step
4. **Test loading**: Verify poses appear in dataset samples

### Research Next Steps
1. **GRU architecture**: Design your motion encoder
2. **Fusion strategy**: Decide how to combine motion + vision
3. **Loss function**: Choose next-step vs k-step vs bins
4. **Training**: Run NaVILA SFT with GRU branch
5. **Evaluation**: Compare with baseline on VLN-CE metrics

## Questions?

Refer to:
- `POSE_EXPORT_README.md` — Export process details
- `INTEGRATION_GUIDE.md` — Training code changes
- Main NaVILA `README.md` — Overall project setup

Or check the original plan document for context on why we're doing this.

---

**Created**: January 27, 2026
**Status**: Ready to use
**Dependencies**: navila-eval environment, VLN-CE data, NaVILA training dataset
