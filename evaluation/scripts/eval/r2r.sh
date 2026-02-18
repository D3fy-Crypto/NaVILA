#!/bin/bash
set -e

# Usage message
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 MODEL_PATH TOTAL_CHUNKS IDX_START GPU_LIST [EXTRA_OPTS...]"
    echo "Example: $0 /path/to/ckpt 4 0 '0,1' NAVILA.ENABLE_MOTION True"
    exit 1
fi
# bash scripts/eval/r2r.sh /home/rithvik/IROS_proj/NaVILA_iros/checkpoints/navila-8b-8f-sft_gru_18/ 1 0 "0" EVAL.EPISODE_COUNT 2


MODEL_PATH=$1
TOTAL_CHUNKS=$2
IDX_START=$3
GPU_LIST=$4  # GPU list as a string (e.g., "0,2,4,6")
EXTRA_OPTS=("${@:5}")


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
        "${EXTRA_OPTS[@]}" &
done

wait
