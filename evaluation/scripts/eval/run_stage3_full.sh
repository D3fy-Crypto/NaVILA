#!/bin/bash
# Full R2R val_unseen for both Stage-3 arms, 4 chunks each (two per GPU), one arm after the other.
set -uo pipefail
cd "$(dirname "$0")/../.."
CKPT=${CKPT:-/home/djonna1/brain_inspired_project/IROS_proj/stage3/runs/stage3-full-20260913-2333}
GPUS=${GPUS:-"0,1,0,1"}
CHUNKS=${CHUNKS:-4}
PY=/home/djonna1/.conda/envs/vln-eval-gru/bin/python
for ARM in gru gru-memzero; do
  OUT=eval_out/$(basename $CKPT)-$ARM/VLN-CE-v1/val_unseen
  echo "=== $(date +%H:%M:%S) arm=$ARM -> $OUT"
  bash scripts/eval/r2r_qwen_gru.sh "$CKPT" "$ARM" "$CHUNKS" 0 "$GPUS"
  echo "=== $(date +%H:%M:%S) arm=$ARM aggregate"
  $PY scripts/eval_jsons.py "./$OUT" "$CHUNKS"
done
echo "=== ALL ARMS DONE $(date +%H:%M:%S)"
