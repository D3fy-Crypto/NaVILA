#!/bin/bash
# ==============================================================================
# Finetune NaVILA with GRU - PROJECTOR ONLY Training
# ==============================================================================
# This script trains ONLY the multimodal projector (and grid_to_vision projector)
# while keeping frozen:
#   - Vision Tower (SigLIP)
#   - LLM (LLaMA-3)
#   - GRU encoder (motion_gru_infonce)
# ==============================================================================

set -euo pipefail

# ============ PATHS - CUSTOMIZE THESE ============
# Base NaVILA model (your pretrained navila checkpoint)
MODEL_PATH="/home/rithvik/NaVILA_Results/navila-llama3-8b-8f"

# GRU checkpoint path
GRU_CKPT="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"

# Output directory for fine-tuned projector
OUTPUT="./checkpoints/navila-8b-8f-projector-only"
mkdir -p "$OUTPUT"

# ============ DISTRIBUTED TRAINING CONFIG ============
n_node=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
CURRENT_RANK=${SLURM_NODEID:-0}
MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============ WANDB (optional) ============
export WANDB_PROJECT="NaVILA-Projector-Finetune"
# export WANDB_API_KEY="your_key_here"

# ============ CUDA CONFIG ============
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH=$CUDA_HOME/bin:$PATH

echo "=============================================="
echo "NaVILA Projector-Only Finetuning"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "GRU Checkpoint: $GRU_CKPT"
echo "Output: $OUTPUT"
echo "Nodes: $n_node, GPUs/Node: $GPUS_PER_NODE"
echo "=============================================="

# Verify paths exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$GRU_CKPT" ]; then
    echo "ERROR: GRU checkpoint does not exist: $GRU_CKPT"
    exit 1
fi

torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_PATH \
    --version llama_3 \
    --seed 42 \
    --data_mixture r2r \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 200 \
    --fps 0.0 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

echo "=============================================="
echo "Training Complete!"
echo "Checkpoints saved to: $OUTPUT"
echo "=============================================="
