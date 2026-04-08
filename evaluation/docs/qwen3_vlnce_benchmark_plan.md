# Qwen3-VL VLNCE Benchmark Plan (Repo-Aligned)

## 1. Active Defaults
- Model path: /home/rithvik/IROS_proj/cvpr_proj/model
- Model family: unsloth/Qwen3-VL-8B-Instruct (full precision)
- Split: val_unseen only
- Seed: 3407
- Generation defaults: top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.0, presence_penalty=1.5

## 2. Relevant Code Flow
1. Launcher starts chunked eval workers and forwards checkpoint override through trailing opts: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L16), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L21).
2. Entrypoint parses opts and builds unified config: [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L34), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L50).
3. Trainer is selected by TRAINER_NAME via registry: [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L75).
4. navila path currently maps config to registered navila trainer: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L3), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L55).
5. Package init imports trainer modules so registration side effects exist: [NaVILA/evaluation/vlnce_baselines/__init__.py](NaVILA/evaluation/vlnce_baselines/__init__.py#L1).
6. Model load and inference-to-action mapping happen inside trainer eval loop: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L85), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L222), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L355).

Audit/code mismatch check:
- No blocking mismatch found for config->trainer selection flow.

## 3. Benchmark Goal
- Integrate Qwen3-VL into the existing VLNCE eval pipeline with minimal structural change.
- Keep NaVILA path untouched.
- Produce comparable metrics and artifact layout for NaVILA vs Qwen.

## 4. Integration Design
Low-risk path:
- Mirror NaVILA structure exactly:
  - new config file: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml)
  - new trainer: [NaVILA/evaluation/vlnce_baselines/qwen_trainer.py](NaVILA/evaluation/vlnce_baselines/qwen_trainer.py)
  - register trainer via package import: [NaVILA/evaluation/vlnce_baselines/__init__.py](NaVILA/evaluation/vlnce_baselines/__init__.py#L1)
- Keep run.py unchanged, relying on existing registry lookup.

Higher-performance path (HIGH-RISK):
- Shared multimodal inference abstraction to remove duplicated eval loop code.
- Batched multi-env support (`NUM_ENVIRONMENTS > 1`) with model batching.
- Constrained decoding for action schema at generation-time.

## 5. Minimal Patch Plan
- [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml): added qwen benchmark config and defaults.
- [NaVILA/evaluation/vlnce_baselines/qwen_trainer.py](NaVILA/evaluation/vlnce_baselines/qwen_trainer.py): added qwen trainer with mirrored eval pipeline.
- [NaVILA/evaluation/vlnce_baselines/__init__.py](NaVILA/evaluation/vlnce_baselines/__init__.py): added qwen trainer import for registration.
- [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py): added QWEN config schema defaults.
- [NaVILA/evaluation/scripts/eval/r2r_qwen.sh](NaVILA/evaluation/scripts/eval/r2r_qwen.sh): added qwen launcher with chunk guardrails.

## 6. Applied Safe Changes
- Added `TRAINER_NAME: qwen` config path and val_unseen default: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml#L3), [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/qwen.yaml#L24).
- Added Qwen generation config in schema to avoid unknown-key merge failures: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L28).
- Added qwen trainer registration import: [NaVILA/evaluation/vlnce_baselines/__init__.py](NaVILA/evaluation/vlnce_baselines/__init__.py#L1).
- Added qwen trainer with safe default STOP fallback for unmatched text: [NaVILA/evaluation/vlnce_baselines/qwen_trainer.py](NaVILA/evaluation/vlnce_baselines/qwen_trainer.py#L104), [NaVILA/evaluation/vlnce_baselines/qwen_trainer.py](NaVILA/evaluation/vlnce_baselines/qwen_trainer.py#L130).
- Added queue clear on episode reset to avoid cross-episode leakage: [NaVILA/evaluation/vlnce_baselines/qwen_trainer.py](NaVILA/evaluation/vlnce_baselines/qwen_trainer.py#L234).
- Added qwen launcher chunk validation checks: [NaVILA/evaluation/scripts/eval/r2r_qwen.sh](NaVILA/evaluation/scripts/eval/r2r_qwen.sh#L22), [NaVILA/evaluation/scripts/eval/r2r_qwen.sh](NaVILA/evaluation/scripts/eval/r2r_qwen.sh#L33).

## 7. Benchmark Matrix
- Baseline A: NaVILA existing path (navila.yaml + navila_trainer)
- Candidate B: Qwen low-risk path (qwen.yaml + qwen_trainer)
- Split: val_unseen
- Seeds: 3407 for qwen path; keep NaVILA as-is and record seed used
- Scale points:
  - 1 GPU, single chunk smoke run
  - N GPUs chunked run (same TOTAL_CHUNKS for both models)

## 8. Execution Commands
Single GPU smoke test:
```bash
cd NaVILA/evaluation
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/qwen.yaml \
  --run-type eval \
  --num-chunks 1 \
  --chunk-idx 0 \
  --log-file logs/qwen_eval_2026_04_06.log \
  EVAL_CKPT_PATH_DIR /home/rithvik/IROS_proj/cvpr_proj/model
```

Multi GPU chunked benchmark (Qwen):
```bash
cd NaVILA/evaluation/scripts/eval
./r2r_qwen.sh /home/rithvik/IROS_proj/cvpr_proj/model 8 0 0,1,2,3
./r2r_qwen.sh /home/rithvik/IROS_proj/cvpr_proj/model 8 4 4,5,6,7
```

Result aggregation (JSON merge sketch):
```bash
python - <<'PY'
import glob, json
from pathlib import Path

root = Path('NaVILA/evaluation/eval_out_qwen/model/VLN-CE-v1/val_unseen')
out = root / 'val_unseen_merged.json'
merged = {}
for fp in glob.glob(str(root / 'val_unseen_*.json')):
    with open(fp) as f:
        merged.update(json.load(f))
with open(out, 'w') as f:
    json.dump(merged, f, indent=2)
print(f'wrote {out} with {len(merged)} episodes')
PY
```

## 9. Metrics + Artifacts
- Metrics source remains Habitat infos per episode (same as NaVILA path).
- Primary files:
  - per-chunk JSON: `<split>_<num_chunks>-<chunk_idx>.json`
  - optional videos under `videos/`
- Qwen output root default: `eval_out_qwen/<model_name>/<dataset_type>/<split>/`

## 10. Validation + Failure Modes
Validation checks:
- Verify trainer registration by confirming qwen resolves via run.py registry path.
- Confirm qwen path writes results JSON for each chunk.
- Confirm merged episode count matches expected chunk coverage.

Known failure modes:
- Free-form generation may still produce noisy text; mitigation: STOP default and strict regex.
- Chunk overlap/miss across launches; mitigation: guardrails in r2r_qwen.sh and explicit chunk plan.
- Dependency/runtime mismatch for Qwen model API in installed transformers version.

## 11. Comparison Template
Use this table for run logs:

| Model | Split | Seed | GPUs/Chunks | Episodes | SR | SPL | nDTW | Notes |
|---|---|---:|---|---:|---:|---:|---:|---|
| NaVILA | val_unseen | (recorded) | ... | ... | ... | ... | ... | ... |
| Qwen3-VL | val_unseen | 3407 | ... | ... | ... | ... | ... | ... |

## 12. Next-step Prompts
1. Tighten Qwen action decoding with schema-constrained output and fallback unit tests.
2. Add robust chunk merger utility script under evaluation/scripts with duplicate-episode detection.
3. Add optional NaVILA parity mode in qwen_trainer for deterministic generation settings.
