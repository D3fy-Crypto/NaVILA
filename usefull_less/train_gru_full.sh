#!/bin/bash

# =============================================================================
# Full Training Script: NaVILA with MotionGRU Integration
# =============================================================================
# This script trains the combined model:
#   - Vision tower (frozen)
#   - MM Projector (trainable)  
#   - Language model (frozen)
#   - Motion GRU (frozen)
#   - Grid-to-Vision Projector (trainable)
# With R2R, RxR, EnvDrop, Human, and other vision-language datasets
# =============================================================================

set -e

# Environment setup
export GPUS_PER_NODE=1
export MASTER_PORT=29500
export n_node=1
export CURRENT_RANK=0
export MASTER_ADDR="127.0.0.1"

# Create output directory
OUTPUT="./checkpoints/navila-8b-8f-gru-full-training"
mkdir -p $OUTPUT

# Paths to trained components
GRU_CKPT="./evaluation/checkpoints/motion_gru_infonce.pt"
ORACLE_DELTAS="./evaluation/oracle_exports/oracle_deltas_train.jsonl"

echo "======================================================================="
echo "Starting Full Training: NaVILA with MotionGRU"
echo "======================================================================="
echo "Output Directory: $OUTPUT"
echo "GRU Checkpoint: $GRU_CKPT"
echo "Oracle Deltas: $ORACLE_DELTAS"
echo "Data Mixture: r2r+rxr+envdrop+human+scanqa+video_chatgpt+sharegpt_video+sharegpt4v_sft"
echo "======================================================================="

# Run training
python -m torch.distributed.launch \
    --nnodes=$n_node \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
    --version llama_3 \
    --seed 42 \
    --data_mixture r2r+rxr+envdrop+human+scanqa+video_chatgpt+sharegpt_video+sharegpt4v_sft \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --tune_motion_gru False \
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
    --save_steps 500 \
    --fps 0.0 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --gru_ckpt_path $GRU_CKPT \
    --pose_deltas_path $ORACLE_DELTAS \
    2>&1 | tee -a $OUTPUT/training.log

echo ""
echo "======================================================================="
echo "Training completed successfully!"
echo "======================================================================="
echo "Output saved to: $OUTPUT"
echo "Checkpoints: $OUTPUT/checkpoint-*"
echo "Logs: $OUTPUT/training.log"
echo "======================================================================="
