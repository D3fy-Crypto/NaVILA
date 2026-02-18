#!/bin/bash
# Train with LLM LoRA (1-GPU friendly) starting from navila-8b-8f-sft_gru_18/checkpoint-100.
# Example:
#   nohup bash scripts/train/sft_8frames_launchjson_lora_gru18.sh > train_lora_gru18.log 2>&1 &

set -euo pipefail

BASE_MODEL=${BASE_MODEL:-"/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/checkpoint-100"}
OUTPUT=${OUTPUT:-"./checkpoints/navila-8b-8f-sft_gru_18_lora_llm"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}

if [ ! -d "$BASE_MODEL" ]; then
    echo "Base model checkpoint not found: $BASE_MODEL"
    exit 1
fi

if [ "${n_node:-1}" -eq 1 ] && [ "${GPUS_PER_NODE:-1}" -eq 1 ]; then
    LAUNCHER=(python llava/train/train_mem.py)
else
    LAUNCHER=(
        torchrun
        --nnodes=${n_node:-1}
        --nproc_per_node=${GPUS_PER_NODE:-1}
        --master_port=${MASTER_PORT:-29501}
        --master_addr ${MASTER_ADDR:-127.0.0.1}
        --node_rank=${CURRENT_RANK:-0}
        llava/train/train_mem.py
    )
fi

LLAVA_DEBUG_MOTION=1 \
"${LAUNCHER[@]}" \
    --longvila_sampler True \
    --model_name_or_path "$BASE_MODEL" \
    --version llama_3 \
    --seed 42 \
    --data_mixture r2r \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model False \
    --tune_motion_gru False \
    --tune_motion_projector True \
    --lora_enable True \
    --lora_llm True \
    --lora_vt False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT" \
    --num_train_epochs 0.05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --do_eval False \
    --save_strategy steps \
    --save_steps 100 \
    --fps 0.0 \
    --save_total_limit 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --ddp_find_unused_parameters False \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to wandb
