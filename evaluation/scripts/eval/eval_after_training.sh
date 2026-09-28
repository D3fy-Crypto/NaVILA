#!/bin/bash
# Wait for a training run to finish, then run the full R2R val_unseen eval on its final model.
#
#   bash scripts/eval/eval_after_training.sh OUTPUT_DIR [ARM] [GPUS]
#
#   e.g. (plain SFT, no GRU):
#     nohup bash scripts/eval/eval_after_training.sh \
#       /home/djonna1/brain_inspired_project/IROS_proj/output_plain-sft-2b-1ep-matched-20260924-0020 nogru \
#       > eval_logs/after_training_$(date +%Y%m%d_%H%M%S).out 2>&1 &
#
# ARM: nogru for plain Qwen3-VL checkpoints; gru / gru-memzero for GRU checkpoints.
# "Finished" means: no train_qwen.py / train_gru_qwen_chang.py process is writing to OUTPUT_DIR any
# more AND the final model + processor files exist in OUTPUT_DIR itself (train_qwen.py writes the
# processor last). If the process dies without those files the script stops and eval is skipped,
# so a crashed run is never evaluated from a half-written directory.
#
# Results: eval_out/<basename OUTPUT_DIR>-<ARM>/VLN-CE-v1/val_unseen/val_unseen_4-<i>.json
# (.json.partial files while running; watch them with scripts/eval_progress.py).
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=${1:?usage: eval_after_training.sh OUTPUT_DIR [ARM] [GPUS]}
OUT=${OUT%/}
ARM=${2:-nogru}
GPUS=${3:-"0,1,0,1"}
CHUNKS=$(awk -F, '{print NF}' <<< "$GPUS")
POLL=${POLL:-300}                        # seconds between checks while training runs
PY=${PY:-/home/djonna1/.conda/envs/vln-eval-gru/bin/python}
RES=eval_out/$(basename "$OUT")-$ARM/VLN-CE-v1/val_unseen

say() { echo "[$(date '+%a %H:%M:%S')] $*"; }

training_alive() { pgrep -f "train(_gru)?_qwen(_chang)?\.py.*--output_dir $OUT( |$)" > /dev/null; }

final_files_present() {
    # transformers >= 5 saves the processor as processor_config.json; older versions as preprocessor_config.json
    [[ -f "$OUT/config.json" && -f "$OUT/tokenizer_config.json" ]] || return 1
    [[ -f "$OUT/processor_config.json" || -f "$OUT/preprocessor_config.json" ]] || return 1
    [[ -f "$OUT/model.safetensors" || -f "$OUT/model.safetensors.index.json" ]]
}

say "watching $OUT (arm=$ARM, gpus=$GPUS, $CHUNKS chunks, poll ${POLL}s)"
[[ -d "$OUT" ]] || { say "FATAL: $OUT does not exist"; exit 1; }

while training_alive; do
    last=$(ls -1 "$OUT" 2>/dev/null | grep -E '^checkpoint-[0-9]+$' | sort -t- -k2 -n | tail -1)
    say "training still running (latest ${last:-no checkpoint yet})"
    sleep "$POLL"
done
say "training process has exited"

# the final save can lag the process exit slightly on a busy disk
for _ in $(seq 1 20); do final_files_present && break; sleep 30; done
if ! final_files_present; then
    say "FATAL: training exited but $OUT has no final model/processor files -- it probably crashed."
    say "check the training log; not starting eval."
    exit 1
fi
sleep 60   # let the last writes settle before four processes read the weights
say "final model found: $(du -sh "$OUT/model.safetensors"* 2>/dev/null | head -1)"

# never mix a new run with stale results in the same results dir
if compgen -G "$RES/val_unseen_${CHUNKS}-*.json" > /dev/null; then
    say "FATAL: $RES already holds results; move them away first (chunks with a JSON are skipped)."
    exit 1
fi

say "starting eval -> $RES"
bash scripts/eval/r2r_qwen_gru.sh "$OUT" "$ARM" "$CHUNKS" 0 "$GPUS"
say "eval processes finished"

n=$(compgen -G "$RES/val_unseen_${CHUNKS}-*.json" | wc -l)
if [[ "$n" -ne "$CHUNKS" ]]; then
    say "WARNING: only $n of $CHUNKS chunk JSONs written; see eval_logs/r2r_${ARM}_${CHUNKS}-*.log"
    $PY scripts/eval_progress.py "$RES" || true
    exit 1
fi
say "aggregate:"
$PY scripts/eval_jsons.py "./$RES" "$CHUNKS"
say "DONE"
