#!/bin/bash
# VLN-CE R2R evaluation of the Stage-3 Qwen3-VL + spatial-memory-GRU model.
#
#   bash scripts/eval/r2r_qwen_gru.sh CKPT_PATH ARM TOTAL_CHUNKS CHUNK_START_IDX "GPU_IDS"
#
#   smoke (20 episodes, one chunk):
#     EPISODE_COUNT=20 bash scripts/eval/r2r_qwen_gru.sh $CKPT gru 1 0 "1"
#   full split, two chunks per GPU:
#     bash scripts/eval/r2r_qwen_gru.sh $CKPT gru 4 0 "0,1,0,1"
#
# ARM: gru | gru-memzero | gru-noline | gru-gate (with GATE_M=<m>) | nogru. Results go to
#   eval_out/<ckpt-name><RUN_TAG or -ARM>/VLN-CE-v1/val_unseen/val_unseen_<chunks>-<idx>.json
# A chunk whose JSON already exists is skipped, so delete stale files before re-running.
set -eo pipefail

MODEL_PATH=$1
ARM=${2:-gru}
TOTAL_CHUNKS=${3:-1}
IDX_START=${4:-0}
GPU_LIST=${5:-0}

ENV_DIR=${ENV_DIR:-/home/djonna1/.conda/envs/vln-eval-gru}
SP=$ENV_DIR/lib/python3.10/site-packages
PY=$ENV_DIR/bin/python
export QWEN_FT_DIR=${QWEN_FT_DIR:-/home/djonna1/brain_inspired_project/Qwen3-VL/.claude/worktrees/integrating_gru/qwen-vl-finetune}

# habitat_sim finds its bundled .so files and libminizip here; without this it will not import
export LD_LIBRARY_PATH=$SP/habitat_sim/_ext:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$QWEN_FT_DIR:${PYTHONPATH:-}
export NAVILA_BASE=${NAVILA_BASE:-/home/djonna1/brain_inspired_project/IROS_proj/NaVILA-Dataset}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
# habitat-sim picks its EGL device by matching EGL_CUDA_DEVICE_NV. With the mesa ICD in
# the list that query fails on every device and the context cannot be created
# ("unable to find EGL device for CUDA device N"). Restrict EGL to the NVIDIA ICD.
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}
export ATTN_IMPL=${ATTN_IMPL:-sdpa} PATCH_EMBED_LINEAR=${PATCH_EMBED_LINEAR:-1}
export TOKENIZERS_PARALLELISM=false
unset DISPLAY

case "$ARM" in gru|gru-memzero|gru-noline|gru-gate|nogru) ;; *) echo "ARM must be gru, gru-memzero, gru-noline, gru-gate or nogru"; exit 2;; esac
if [[ "$ARM" == "gru-gate" ]]; then
    [[ -n "${GATE_M:-}" ]] || { echo "ARM=gru-gate needs GATE_M=<metres>"; exit 2; }
fi
[[ -f "$MODEL_PATH/config.json" ]] || { echo "no checkpoint at $MODEL_PATH"; exit 1; }

EXTRA=(ARM "$ARM")
[[ "$ARM" == "gru-gate" ]] && EXTRA+=(GATE_M "$(printf "%.2f" "$GATE_M")")   # yacs wants a float
[[ -n "${EPISODE_COUNT:-}" ]] && EXTRA+=(EVAL.EPISODE_COUNT "$EPISODE_COUNT")
[[ -n "${RUN_TAG:-}" ]] && EXTRA+=(RUN_TAG "$RUN_TAG")

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}
LOG_DIR=${LOG_DIR:-$PWD/eval_logs}; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    LOG=$LOG_DIR/r2r_${ARM}${GATE_M:+${GATE_M}m}_${TOTAL_CHUNKS}-${CHUNK_IDX}_$STAMP.log
    echo "arm=$ARM chunk $CHUNK_IDX/$TOTAL_CHUNKS on GPU ${GPULIST[$IDX]} -> $LOG"
    # Assign the GPU through the config, not CUDA_VISIBLE_DEVICES: habitat-sim matches EGL
    # devices by physical CUDA index, and hiding devices makes it fail with
    # "unable to find EGL device for CUDA device 0".
    GPU=${GPULIST[$IDX]}
    $PY run.py \
        --exp-config "${EXP_CONFIG:-vlnce_baselines/config/r2r_baselines/qwen3vl_gru.yaml}" \
        --run-type eval \
        --num-chunks "$TOTAL_CHUNKS" \
        --chunk-idx "$CHUNK_IDX" \
        EVAL_CKPT_PATH_DIR "$MODEL_PATH" \
        SIMULATOR_GPU_IDS "[$GPU]" TORCH_GPU_ID "$GPU" "${EXTRA[@]}" > "$LOG" 2>&1 &
done

wait
echo "done: arm=$ARM chunks=$TOTAL_CHUNKS"
