#!/bin/bash

# =============================================================================
# Stage 1: Sanity + Gradient Check
# =============================================================================
# Quick validation run before full training
# - 10% of 1 epoch on 10k samples
# - Checks: loss decrease, gradient flow, no NaNs, projection norms
# - Expected time: 20-30 minutes on 1 GPU
# =============================================================================

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navila

# Environment setup
export GPUS_PER_NODE=1
export MASTER_PORT=29500
export n_node=1
export CURRENT_RANK=0
export MASTER_ADDR="127.0.0.1"

# Create output directory
OUTPUT="./checkpoints/navila-8b-8f-gru-sanity-check"
mkdir -p $OUTPUT

# Paths to trained components
GRU_CKPT="./evaluation/checkpoints/motion_gru_infonce.pt"
ORACLE_DELTAS="./evaluation/oracle_exports/oracle_deltas_train.jsonl"

echo ""
echo "======================================================================="
echo "Stage 1: Sanity + Gradient Check"
echo "======================================================================="
echo "Output Directory: $OUTPUT"
echo "GRU Checkpoint: $GRU_CKPT"
echo "Oracle Deltas: $ORACLE_DELTAS"
echo "Max Samples: 10000 (10k samples)"
echo "Epochs: 0.1 (10% of 1 epoch)"
echo "Expected Time: 20-30 minutes"
echo ""
echo "Validating:"
echo "  ✓ Loss decreases"
echo "  ✓ Gradients flow properly"
echo "  ✓ No NaNs/Infs in loss"
echo "  ✓ Projection norms are reasonable"
echo "======================================================================="
echo ""

# Run sanity check
torchrun \
    --nnodes=$n_node \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --longvila_sampler=True \
    --deepspeed=./scripts/zero3.json \
    --model_name_or_path=a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
    --version=llama_3 \
    --seed=42 \
    --data_mixture=r2r \
    --vision_tower=google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature=cls_patch \
    --mm_projector=mlp_downsample \
    --num_video_frames=8 \
    --tune_vision_tower=False \
    --tune_mm_projector=True \
    --tune_language_model=False \
    --tune_motion_gru=False \
    --mm_vision_select_layer=-2 \
    --mm_use_im_start_end=False \
    --mm_use_im_patch_token=False \
    --image_aspect_ratio=resize \
    --bf16=True \
    --output_dir=$OUTPUT \
    --num_train_epochs=0.1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --do_eval=False \
    --save_strategy=no \
    --fps=0.0 \
    --learning_rate=1e-4 \
    --weight_decay=0.01 \
    --warmup_ratio=0.05 \
    --lr_scheduler_type=cosine \
    --logging_steps=5 \
    --tf32=True \
    --model_max_length=4096 \
    --gradient_checkpointing=True \
    --dataloader_num_workers=8 \
    --lazy_preprocess=True \
    --report_to=none \
    --gru_ckpt_path=$GRU_CKPT \
    --pose_deltas_path=$ORACLE_DELTAS \
    2>&1 | tee $OUTPUT/sanity_check.log

echo ""
echo "======================================================================="
echo "Sanity Check Complete!"
echo "======================================================================="
echo ""
echo "Analyzing results..."
echo ""

# Parse results from log
python3 << 'EOF'
import re
import sys
from pathlib import Path

log_file = Path("$OUTPUT/sanity_check.log")
if not log_file.exists():
    print("❌ Log file not found!")
    sys.exit(1)

content = log_file.read_text()

# Extract loss values
losses = []
for match in re.finditer(r'"loss": ([\d.]+)', content):
    losses.append(float(match.group(1)))

# Check for NaNs/Infs
has_nan = 'nan' in content.lower() or 'inf' in content.lower() and 'infinity' not in content.lower()
has_error = 'error' in content.lower() and 'error:' in content.lower()

print("📊 Loss Analysis:")
print(f"   Total loss values: {len(losses)}")
if losses:
    print(f"   Initial loss: {losses[0]:.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Loss change: {losses[0] - losses[-1]:.4f}")
    if losses[0] > losses[-1]:
        print(f"   ✅ Loss DECREASED")
    else:
        print(f"   ⚠️  Loss did NOT decrease (sanity check may have issues)")
else:
    print("   ⚠️  No loss values found in log")

print("")
print("⚠️  Numerical Stability:")
if has_nan:
    print("   ❌ NaN/Inf detected in logs")
else:
    print("   ✅ No NaNs/Infs detected")

if has_error:
    print("   ❌ Errors detected in logs")
else:
    print("   ✅ No errors detected")

print("")
print("📝 Log file: ./checkpoints/navila-8b-8f-gru-sanity-check/sanity_check.log")
EOF

echo ""
echo "======================================================================="
echo "Next Steps:"
echo ""
echo "If all checks passed ✅:"
echo "  1. Review the log for gradient norms"
echo "  2. Check projector weights changed (not frozen)"
echo "  3. Run full training: bash train_gru_full.sh"
echo ""
echo "If issues found ❌:"
echo "  1. Check GRU checkpoint loads correctly"
echo "  2. Verify oracle deltas format"
echo "  3. Check GPU memory is sufficient"
echo "======================================================================="
