# Quick Reference: GRU Pose Export Commands

## Check Your Setup
```bash
cd evaluation/scripts
python check_setup.py
```

Expected: All ✓ marks, especially for habitat, VLN-CE data, and NaVILA dataset.

---

## Test Export (5 samples, ~1 second)
```bash
cd evaluation/scripts
python test_pose_export.py
```

Expected output:
```
Sample: 914-23
  ✓ Found episode with 6 waypoints
  Scene: mp3d/8WUmhLawc2A/8WUmhLawc2A.glb
```

---

## Export Single Split
```bash
cd evaluation/scripts

# R2R train
bash run_pose_export.sh --dataset r2r --split train

# R2R val_unseen
bash run_pose_export.sh --dataset r2r --split val_unseen

# RxR train
bash run_pose_export.sh --dataset rxr --split train
```

---

## Export All Splits (Recommended)
```bash
cd evaluation/scripts
bash export_all_poses.sh
```

This exports:
- R2R: train, val_seen, val_unseen
- RxR: train, val_unseen

**Time**: ~10-30 minutes total
**Output**: `~/NaVILA-Dataset/{R2R,RxR}/gru_pose_*.jsonl`

---

## Verify Exports
```bash
# Check files exist
ls -lh ~/NaVILA-Dataset/R2R/gru_pose_*.jsonl
ls -lh ~/NaVILA-Dataset/RxR/gru_pose_*.jsonl

# Count records
wc -l ~/NaVILA-Dataset/R2R/gru_pose_train.jsonl

# Inspect first record
head -1 ~/NaVILA-Dataset/R2R/gru_pose_train.jsonl | python -m json.tool
```

Expected:
- R2R train: ~353K records, ~50MB
- Each record has: 8 poses, 7 deltas

---

## Inspect a Sample Manually

```bash
cd evaluation/scripts
python << 'EOF'
import json

# Load first record
with open('/home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl') as f:
    sample = json.loads(f.readline())

print(f"Video ID: {sample['video_id']}")
print(f"Episode: {sample['episode_id']}")
print(f"Scene: {sample['scene_id']}")
print(f"\nPoses (8 frames):")
for i, (x, y, yaw) in enumerate(sample['poses']):
    print(f"  [{i}] x={x:7.2f}, y={y:7.2f}, yaw={yaw:6.3f} rad")

print(f"\nDeltas (7 transitions):")
for i, (dx, dy, dyaw) in enumerate(sample['deltas']):
    dist = (dx**2 + dy**2)**0.5
    print(f"  [{i}→{i+1}] dx={dx:6.2f}, dy={dy:6.2f}, dyaw={dyaw:6.3f}, dist={dist:5.2f}m")

print(f"\nBins:")
print(f"  Distance: {sample['dist_bins']}")
print(f"  Yaw:      {sample['yaw_bins']}")
EOF
```

---

## Integration (After Export)

### 1. Update `datasets_mixture.py`
```python
# llava/data/datasets_mixture.py
r2r = Dataset(
    ...,
    gru_pose_path="/home/rithvik/NaVILA-Dataset/R2R/gru_pose_train.jsonl",
)
```

### 2. Update `LazyVLNCEDataset`
```python
# llava/data/dataset.py
def __init__(self, ..., gru_pose_path=None):
    ...
    # Load pose sidecar
    self.pose_data = {}
    if gru_pose_path and os.path.exists(gru_pose_path):
        with open(gru_pose_path) as f:
            for line in f:
                rec = json.loads(line)
                self.pose_data[rec['video_id']] = rec

def __getitem__(self, i):
    ...
    video_id = sources[0]["video_id"]
    if video_id in self.pose_data:
        data_dict["deltas"] = np.array(self.pose_data[video_id]["deltas"])
    ...
```

### 3. Use in Model
```python
# In forward pass
deltas = batch.get("deltas")  # [B, 7, 3]
if deltas is not None:
    motion_embeds = self.gru_encoder(deltas)
    # Fuse with vision/text...
```

---

## Custom Export Options

### Different frame count
```bash
bash run_pose_export.sh --dataset r2r --split train --frames 16
```

### Custom output path
```bash
bash run_pose_export.sh \
  --dataset r2r \
  --split train \
  --output /custom/path/poses.jsonl
```

### Custom binning
```bash
bash run_pose_export.sh \
  --dataset r2r \
  --split train \
  --dist-bins 0 0.5 1.0 2.0 \
  --yaw-bins -180 -45 -15 15 45 180
```

---

## File Locations

### Scripts
```
evaluation/scripts/
├── export_gru_poses.py           # Main export script
├── run_pose_export.sh            # Wrapper (activates navila-eval)
├── export_all_poses.sh           # Batch export all splits
├── test_pose_export.py           # Quick test (5 samples)
└── check_setup.py                # Diagnostic

evaluation/habitat_extensions/
└── utils.py                      # Added quaternion_to_yaw()
```

### Documentation
```
evaluation/scripts/
├── POSE_EXPORT_README.md         # Detailed export guide
├── INTEGRATION_GUIDE.md          # Training integration steps
├── SUMMARY.md                    # Complete overview
└── QUICKREF.md                   # This file
```

### Output Data
```
/home/rithvik/NaVILA-Dataset/
├── R2R/
│   ├── gru_pose_train.jsonl
│   ├── gru_pose_val_seen.jsonl
│   └── gru_pose_val_unseen.jsonl
└── RxR/
    ├── gru_pose_train.jsonl
    └── gru_pose_val_unseen.jsonl
```

---

## Troubleshooting One-Liners

### "Module not found: habitat"
```bash
# Wrong environment - use wrapper
bash run_pose_export.sh ...
```

### Check environment
```bash
conda activate navila-eval
python -c "import habitat; print(habitat.__version__)"
```

### Verify episode mapping
```bash
python test_pose_export.py | grep "✓ Found episode"
```

### Check pose data shape
```bash
head -1 ~/NaVILA-Dataset/R2R/gru_pose_train.jsonl | \
  python -c "import json,sys; d=json.load(sys.stdin); \
  print(f'poses: {len(d[\"poses\"])}x{len(d[\"poses\"][0])}'); \
  print(f'deltas: {len(d[\"deltas\"])}x{len(d[\"deltas\"][0])}')"
```

---

## Full Workflow Summary

```bash
# 1. Check setup
cd evaluation/scripts
python check_setup.py

# 2. Test (optional but recommended)
python test_pose_export.py

# 3. Export all data
bash export_all_poses.sh

# 4. Verify
ls -lh ~/NaVILA-Dataset/{R2R,RxR}/gru_pose_*.jsonl
wc -l ~/NaVILA-Dataset/R2R/gru_pose_train.jsonl

# 5. Inspect sample
head -1 ~/NaVILA-Dataset/R2R/gru_pose_train.jsonl | python -m json.tool

# 6. Integrate (see INTEGRATION_GUIDE.md)
#    - Update datasets_mixture.py
#    - Update dataset.py
#    - Update model forward pass

# 7. Train
cd ../..
bash scripts/train/sft_8frames.sh

# 8. Evaluate
cd evaluation
bash scripts/eval/r2r.sh CKPT_PATH 1 0 "0"
```

---

**For detailed explanations, see:**
- Export details → `POSE_EXPORT_README.md`
- Integration steps → `INTEGRATION_GUIDE.md`
- Full overview → `SUMMARY.md`
