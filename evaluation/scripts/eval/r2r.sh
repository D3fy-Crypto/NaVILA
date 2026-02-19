#!/bin/bash
set -e

# Usage message
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 MODEL_PATH TOTAL_CHUNKS IDX_START GPU_LIST [EXTRA_OPTS...]"
    echo "Example: $0 /path/to/ckpt 4 0 '0,1' NAVILA.ENABLE_MOTION True"
    echo "Adapter eval: BASE_MODEL_PATH=/path/to/base/checkpoint-100 $0 /path/to/adapter/checkpoint-100 1 0 '0'"
    exit 1
fi
# bash scripts/eval/r2r.sh /home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/ 1 0 "0" EVAL.EPISODE_COUNT 2


MODEL_PATH=$1
TOTAL_CHUNKS=$2
IDX_START=$3
GPU_LIST=$4  # GPU list as a string (e.g., "0,2,4,6")
EXTRA_OPTS=("${@:5}")
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

BASE_MODEL_OPTS=()
if [ -n "$BASE_MODEL_PATH" ]; then
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        echo "Error: BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH"
        exit 1
    fi
    BASE_MODEL_OPTS=(EVAL_MODEL_BASE_PATH "$BASE_MODEL_PATH")
    echo "Using base checkpoint for adapter eval: $BASE_MODEL_PATH"
fi


IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

if [ "$CHUNKS" -eq 0 ]; then
    echo "Error: GPU_LIST is empty or malformed."
    exit 1
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    echo "Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
        --run-type eval \
        --num-chunks $TOTAL_CHUNKS \
        --chunk-idx $CHUNK_IDX \
        EVAL_CKPT_PATH_DIR $MODEL_PATH \
        "${BASE_MODEL_OPTS[@]}" \
        "${EXTRA_OPTS[@]}" &
done

wait


#From repo root, run:

# FOR ADAPTER EVAL:
#cd /home/rithvik/IROS_proj/NaVILA_iros/evaluation

#BASE_MODEL_PATH=/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/checkpoint-100 \
#bash scripts/eval/r2r.sh \
#/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18_lora_llm_v3/checkpoint-200 \
#1 0 "0"

# FOR FULL MODEL EVAL:
#cd /home/rithvik/IROS_proj/NaVILA_iros/evaluation

#bash scripts/eval/r2r.sh \
#/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/checkpoint-100 \
#1 0 "0"

# FOR FULL MODEL EVAL WITH MOTION:
#cd /home/rithvik/IROS_proj/NaVILA_iros/evaluation

#BASE_MODEL_PATH=/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/checkpoint-100 \
#bash scripts/eval/r2r.sh \
#/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18_lora_llm_v3/checkpoint-200 \
#1 0 "0" \
#NAVILA.ENABLE_MOTION True

#FOR FULL MODEL EVAL WITHOUT MOTION:
#cd /home/rithvik/IROS_proj/NaVILA_iros/evaluation

#BASE_MODEL_PATH=/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/checkpoint-100 \
#bash scripts/eval/r2r.sh \
#/home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18_lora_llm_v3/checkpoint-200 \
#1 0 "0" \
#NAVILA.ENABLE_MOTION False

