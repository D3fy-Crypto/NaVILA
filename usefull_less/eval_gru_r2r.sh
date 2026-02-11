#!/bin/bash
# Evaluation script for trained GRU+NaVILA model on R2R

# Exit on error
set -e

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAVILA_ROOT="${SCRIPT_DIR}"
EVAL_DIR="${NAVILA_ROOT}/evaluation"

# Trained checkpoint paths
FINAL_MODEL="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/final_model_gru"
SANITY_CKPT="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/checkpoints/navila-8b-8f-gru-sanity-check"

# Configuration
CKPT_PATH="${1:-${SANITY_CKPT}}"  # Default to sanity check checkpoint
NUM_CHUNKS="${2:-1}"
CHUNK_START_IDX="${3:-0}"
GPU_IDS="${4:-0}"

echo "=================================================================="
echo "Evaluation: GRU+NaVILA on R2R"
echo "=================================================================="
echo "Checkpoint Path: $CKPT_PATH"
echo "GPU IDs: $GPU_IDS"
echo "Chunks: $NUM_CHUNKS (start: $CHUNK_START_IDX)"
echo "=================================================================="

# Check checkpoint exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "❌ Checkpoint not found: $CKPT_PATH"
    echo ""
    echo "Available checkpoints:"
    echo "  1. Final Model: $FINAL_MODEL"
    echo "  2. Sanity Check: $SANITY_CKPT"
    exit 1
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navila

# Change to evaluation directory
cd "$EVAL_DIR"

# Run evaluation
echo ""
echo "Starting R2R evaluation..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_IDS python run.py \
    --model_path "$CKPT_PATH" \
    --num_chunks "$NUM_CHUNKS" \
    --chunk_start_idx "$CHUNK_START_IDX" \
    --dataset_name r2r \
    --task VLNCE \
    --split_env eval_seen \
    --output_dir "${CKPT_PATH}/r2r_eval_results"

echo ""
echo "=================================================================="
echo "Evaluation complete!"
echo "Results saved to: ${CKPT_PATH}/r2r_eval_results"
echo "=================================================================="
