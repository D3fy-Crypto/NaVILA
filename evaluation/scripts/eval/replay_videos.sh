#!/bin/bash
# Re-render a finished VLN-CE R2R eval into videos by replaying its logged actions. No model runs.
#
#   bash scripts/eval/replay_videos.sh CKPT_DIR LOG "JSON_GLOB" [GPUS]
#
#   e.g.
#     D=/home/shared/mann/VLN_Qwen_2B_instruct/ckpt/qwen2b_ep1_1of3_epochs
#     bash scripts/eval/replay_videos.sh $D $D/eval_val_unseen/eval_vega_1chunk.log \
#          "$D/eval_val_unseen/val_unseen_*.json" "0,1,0,1"
#   smoke: EPISODE_COUNT=20 bash scripts/eval/replay_videos.sh ... "1"
#
# CKPT_DIR only names the run (no weights are read). Results:
#   eval_out/<basename CKPT_DIR>-replay/VLN-CE-v1/val_unseen/{val_unseen_<N>-<i>.json, replay_check_<N>-<i>.json, videos/}
set -eo pipefail
cd "$(dirname "$0")/../.."

CKPT=$1; LOG=$2; JSON=$3; GPU_LIST=${4:-"0,1,0,1"}
[[ -f "$LOG" ]] || { echo "no log at $LOG"; exit 1; }
compgen -G "$JSON" > /dev/null || { echo "no JSON matches $JSON"; exit 1; }

ENV_DIR=${ENV_DIR:-/home/djonna1/.conda/envs/vln-eval-gru}
SP=$ENV_DIR/lib/python3.10/site-packages
PY=$ENV_DIR/bin/python
export LD_LIBRARY_PATH=$SP/habitat_sim/_ext:$ENV_DIR/lib:${LD_LIBRARY_PATH:-}
# vlnce_baselines/__init__ also imports the GRU trainer, which needs qwenvl importable
export QWEN_FT_DIR=${QWEN_FT_DIR:-/home/djonna1/brain_inspired_project/Qwen3-VL/.claude/worktrees/integrating_gru/qwen-vl-finetune}
export PYTHONPATH=$QWEN_FT_DIR:${PYTHONPATH:-}
export NAVILA_BASE=${NAVILA_BASE:-/home/djonna1/brain_inspired_project/IROS_proj/NaVILA-Dataset}
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet GLOG_minloglevel=2
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}
unset DISPLAY

EXTRA=(REPLAY_LOG "$LOG" REPLAY_JSON "$JSON")
[[ -n "${EPISODE_COUNT:-}" ]] && EXTRA+=(EVAL.EPISODE_COUNT "$EPISODE_COUNT")
[[ -n "${RUN_TAG:-}" ]] && EXTRA+=(RUN_TAG "$RUN_TAG")

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}
LOG_DIR=${LOG_DIR:-$PWD/eval_logs}; mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
for IDX in $(seq 0 $((CHUNKS-1))); do
    OUTLOG=$LOG_DIR/replay_$(basename "$CKPT")_${CHUNKS}-${IDX}_$STAMP.log
    echo "chunk $IDX/$CHUNKS on GPU ${GPULIST[$IDX]} -> $OUTLOG"
    GPU=${GPULIST[$IDX]}   # through the config, never CUDA_VISIBLE_DEVICES (habitat EGL matching)
    $PY run.py --exp-config vlnce_baselines/config/r2r_baselines/replay.yaml --run-type eval \
        --num-chunks "$CHUNKS" --chunk-idx "$IDX" \
        EVAL_CKPT_PATH_DIR "$CKPT" SIMULATOR_GPU_IDS "[$GPU]" TORCH_GPU_ID "$GPU" "${EXTRA[@]}" > "$OUTLOG" 2>&1 &
done
wait
echo "done: $CHUNKS chunks"
