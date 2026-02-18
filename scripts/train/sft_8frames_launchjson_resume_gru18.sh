#!/bin/bash
# Resume training from navila-8b-8f-sft_gru_18/checkpoint-100 with LLM unfrozen.
# Example:
#   nohup bash scripts/train/sft_8frames_launchjson_resume_gru18.sh > train_resume_gru18.log 2>&1 &

set -euo pipefail

# You can pass either:
# 1) a concrete checkpoint dir, e.g. .../checkpoint-100
# 2) a run dir containing checkpoint-* dirs, e.g. .../navila-8b-8f-sft_gru_18
RESUME_CKPT=${RESUME_CKPT:-"/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18"}
OUTPUT=${OUTPUT:-"./checkpoints/navila-8b-8f-sft_gru_18_resume_llm"}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"./scripts/zero3_offload.json"}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

if [ ! -d "$RESUME_CKPT" ]; then
    echo "Resume source not found: $RESUME_CKPT"
    exit 1
fi

# Match llava/train/train.py logic:
# - train_mem.py -> train.py
# - get_checkpoint_path(output_dir) scans output_dir/checkpoint-*
# So we must ensure OUTPUT contains a real checkpoint-* entry.
if [[ "$(basename "$RESUME_CKPT")" =~ ^checkpoint-[0-9]+$ ]]; then
    RESUME_STEP_DIR="$RESUME_CKPT"
else
    RESUME_STEP_DIR="$(find "$RESUME_CKPT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
fi

if [ -z "${RESUME_STEP_DIR:-}" ] || [ ! -d "$RESUME_STEP_DIR" ]; then
    echo "No checkpoint-* directory found under: $RESUME_CKPT"
    exit 1
fi

CKPT_LINK_NAME="$(basename "$RESUME_STEP_DIR")"
mkdir -p "$OUTPUT"

# llava/train/utils.py:get_checkpoint_path returns 'finished' if OUTPUT/config.json exists.
# Fail early to avoid silent 'Skipp training'.
if [ -f "$OUTPUT/config.json" ]; then
    echo "Output already looks like a finished run (found $OUTPUT/config.json)."
    echo "Use a fresh OUTPUT path to resume training."
    exit 1
fi

ln -sfn "$RESUME_STEP_DIR" "$OUTPUT/$CKPT_LINK_NAME"
echo "Resume checkpoint linked: $OUTPUT/$CKPT_LINK_NAME -> $RESUME_STEP_DIR"

LLAVA_DEBUG_MOTION=1 \
torchrun --nnodes=${n_node:-1} --nproc_per_node=${GPUS_PER_NODE:-1} --master_port=${MASTER_PORT:-29501} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} --node_rank=${CURRENT_RANK:-0} \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
    --version llama_3 \
    --seed 42 \
    --data_mixture r2r \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model True \
    --tune_motion_gru False \
    --tune_motion_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT" \
    --num_train_epochs 0.2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --do_eval False \
    --save_strategy steps \
    --save_steps 100 \
    --fps 0.0 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to wandb
