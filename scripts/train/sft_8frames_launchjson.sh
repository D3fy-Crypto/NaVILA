#!/bin/bash
# nohup bash /home/rithvik/IROS_proj/NaVILA_iros/scripts/train/sft_8frames_launchjson.sh > train.log 2>&1 &
export MASTER_PORT=29501
export GPUS_PER_NODE=1
export n_node=1
export MASTER_ADDR=127.0.0.1
export CURRENT_RANK=0
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
TUNE_LANGUAGE_MODEL=${TUNE_LANGUAGE_MODEL:-False}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
#nohup bash [sft_8frames_launchjson.sh](http://_vscodecontentref_/0) > [train.log](http://_vscodecontentref_/1) 2>&1 &
OUTPUT="./checkpoints/navila-8b-8f-sft_motion_alligned"


LLAVA_DEBUG_MOTION=0 \
torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain \
    --version llama_3 \
    --seed 42 \
    --data_mixture r2r \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --motion_alignment_debug False \
    --motion_alignment_debug_max_prints 0 \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model $TUNE_LANGUAGE_MODEL \
    --tune_motion_gru False \
    --tune_motion_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 0.02 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --do_train True \
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
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to wandb
