#!/bin/bash

# GRU Integration Training Launcher
# This script verifies the implementation and starts training

echo "======================================"
echo "GRU Integration Training Launcher"
echo "======================================"
echo ""

# Activate conda environment
echo "Activating navila conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navila
echo "✅ Environment activated"
echo ""

# Step 1: Verify implementation
echo "Step 1: Running implementation verification..."
echo "--------------------------------------"
python verify_gru_integration.py
VERIFY_EXIT_CODE=$?

if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Verification failed! Please fix issues before training."
    echo "Check the output above for details."
    exit 1
fi

echo ""
echo "✅ Verification passed!"
echo ""

# Step 2: Set training environment variables
echo "Step 2: Setting environment variables..."
echo "--------------------------------------"
export GPUS_PER_NODE=4
export n_node=1
export MASTER_PORT=29500
export MASTER_ADDR=localhost
export CURRENT_RANK=0

echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "n_node=$n_node"
echo "MASTER_PORT=$MASTER_PORT"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CURRENT_RANK=$CURRENT_RANK"
echo ""

# Step 3: Start training
echo "Step 3: Starting training with GRU integration..."
echo "--------------------------------------"
echo "Training script: scripts/train/sft_8frames_gru.sh"
echo "Expected training time: 8-12 hours"
echo "Loss trajectory: 2.5 → 1.8"
echo ""
echo "Monitoring features enabled:"
echo "  ✅ Checkpoint loading logs"
echo "  ✅ Pose deltas loading logs"
echo "  ✅ Forward pass motion encoding logs"
echo "  ✅ Gradient flow monitoring (every 10 steps)"
echo "  ✅ WANDB integration (if configured)"
echo ""

read -p "Press Enter to start training or Ctrl+C to cancel..."

bash scripts/train/sft_8frames_gru.sh

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
