#!/bin/bash

MODEL_PATH=${1:-/home/rithvik/IROS_proj/cvpr_proj/model}
TOTAL_CHUNKS=$2
IDX_START=$3
GPU_LIST=$4  # GPU list as a string (e.g., "0,1,2,3")

if [ -z "$TOTAL_CHUNKS" ] || [ -z "$IDX_START" ] || [ -z "$GPU_LIST" ]; then
    echo "Usage: $0 [MODEL_PATH] TOTAL_CHUNKS IDX_START GPU_LIST"
    echo "Example: $0 /home/rithvik/IROS_proj/cvpr_proj/model 8 0 0,1,2,3"
    exit 1
fi

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

if [ "$CHUNKS" -le 0 ]; then
    echo "Error: GPU_LIST produced zero workers."
    exit 1
fi

if [ "$TOTAL_CHUNKS" -le 0 ]; then
    echo "Error: TOTAL_CHUNKS must be > 0"
    exit 1
fi

if [ "$IDX_START" -lt 0 ]; then
    echo "Error: IDX_START must be >= 0"
    exit 1
fi

LAST_CHUNK_IDX=$((IDX_START + CHUNKS - 1))
if [ "$LAST_CHUNK_IDX" -ge "$TOTAL_CHUNKS" ]; then
    echo "Error: chunk range [$IDX_START, $LAST_CHUNK_IDX] exceeds TOTAL_CHUNKS=$TOTAL_CHUNKS"
    exit 1
fi

for IDX in $(seq 0 $((CHUNKS - 1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    echo "Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/qwen.yaml \
        --run-type eval \
        --num-chunks $TOTAL_CHUNKS \
        --chunk-idx $CHUNK_IDX \
        EVAL_CKPT_PATH_DIR $MODEL_PATH &
done

wait
