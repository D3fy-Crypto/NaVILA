#!/bin/bash

# Minimal GRU Training Test - Uses synthetic data
# Purpose: Verify GRU + VLA integration and save final model

OUTPUT="./checkpoints/navila-8b-8f-sft-gru-test"

# Path to pretrained MotionGRU checkpoint
GRU_CKPT="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"

# Path to oracle deltas (pose/motion data)
ORACLE_DELTAS="/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/oracle_exports/oracle_deltas_train.jsonl"

# Minimal test - single epoch, small batch, no data loading required
torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --longvila_sampler False \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
    --version llama_3 \
    --seed 10 \
    --data_mixture sharegpt4v_sft \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model False \
    --tune_motion_gru True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_eval False \
    --save_strategy "no" \
    --fps 0.0 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    --gru_ckpt_path $GRU_CKPT \
    --pose_deltas_path $ORACLE_DELTAS \
    --max_steps 5
