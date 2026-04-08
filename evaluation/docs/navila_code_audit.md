# NaVILA R2R Code Audit

## 1. Scope and Files Reviewed
- [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh)
- [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py)
- [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml)
- [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml)
- [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py)
- [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py)

## 2. End-to-End Execution Flow
1. The launcher reads positional arguments for model path, total chunks, start index, and GPU list: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L3), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L4), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L5), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L8).
2. It computes local worker count from GPUs and maps each local worker to a global chunk index: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L10), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L12), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L13).
3. One background eval process is launched per GPU with chunk args and checkpoint override, then the script waits for all: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L16), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L20), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L24).
4. The Python entrypoint parses command line including trailing config options, then executes run_exp: [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L15), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L34), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L42).
5. run_exp resolves merged config, instantiates trainer via registry, and dispatches eval mode: [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L50), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L75), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L78), [NaVILA/evaluation/run.py](NaVILA/evaluation/run.py#L83).
6. navila.yaml binds this run to NaVILA trainer and the task yaml, while the shell passes CLI override for checkpoint path: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L2), [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L3), [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L7).
7. NaVILATrainer.eval sets device and evaluates only when checkpoint path is a directory: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L384), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L395).

## 3. Config Resolution Chain and Override Behavior
Merge order in get_config:
1. Habitat-baselines default config: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L294).
2. VLN-CE local defaults in _C: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L296).
3. Experiment yaml files via merge_from_file: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L307).
4. Task config reload using BASE_TASK_CONFIG_PATH: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L309).
5. Trailing CLI opts last (highest priority): [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L314).

Effective NaVILA overrides:
- Default trainer is dagger but navila.yaml switches to navila: [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L17), [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L3).
- Evaluation split becomes val_unseen from navila.yaml: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L14).
- Shell command replaces EVAL_CKPT_PATH_DIR at runtime using trailing opts: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L21), [NaVILA/evaluation/vlnce_baselines/config/default.py](NaVILA/evaluation/vlnce_baselines/config/default.py#L314).

## 4. Episode Inference Loop Behavior
- Trainer registration and checkpoint evaluation entry: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L55), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L70).
- Model load occurs from checkpoint directory name, then model is moved to CUDA: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L85).
- Runtime config mutation sets split, language, chunking, result dir, and video dir before freezing: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L91), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L98), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L100), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L104).
- Environment creation and single-env constraint: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L121), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L145).
- Temporal frame handling and prompt generation path: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L38), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L166), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L176).
- Decoding happens via model.generate with deterministic settings: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L202).

## 5. Action Parsing and Queued Action Execution
- Regex map defines four classes and returns None on no match: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L222), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L230).
- Action is parsed from text into actions list, then branch logic selects forward, left, right, or fallback stop branch: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L237), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L282).
- Forward distance parse and normalization: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L244), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L249).
- Turn parse and normalization: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L257), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L271).
- Queue first policy and replay behavior: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L147), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L153), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L253), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L266), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L280).

## 6. Chunking and Multi-GPU Behavior
- GPU list controls local process fan-out: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L8), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L10), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L12).
- Global chunk index is local index plus start offset: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L13).
- Each process receives common total chunks and unique chunk index in run.py args: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L19), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L20).
- Trainer writes chunk selectors into dataset config and emits per-chunk result file names: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L98), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L99), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L115), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L355).

## 7. Inputs and Outputs
Inputs:
- Checkpoint directory from shell arg into EVAL_CKPT_PATH_DIR override: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L21), [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L7).
- Task action and motion constants: [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L8), [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L9), [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L26).
- Dataset and GT paths for R2R/nDTW: [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L32), [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L37), [NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml](NaVILA/evaluation/habitat_extensions/config/vlnce_task.yaml#L38).

Outputs:
- Results root from config, then runtime nested directory by model and split: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L8), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L100).
- Metrics json saved per chunk: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L354), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L355).
- Optional videos saved under videos directory: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L104).

## 8. Risks, Edge Cases, and Assumptions
1. High: Invalid action propagation on unmatched model text.
The mapper can return None, but the fallback branch steps the environment with actions directly, which may contain None instead of STOP: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L230), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L237), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L283).

2. High: Queued actions can leak across episode boundaries.
queue_actions is created once per eval and consumed before new inference, but it is not cleared when an episode ends and reset_at is called: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L147), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L153), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L301).

3. Medium: Single-env hard assumption conflicts with potential config scaling.
Code asserts one env, while this can silently block future multi-env settings: [NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml](NaVILA/evaluation/vlnce_baselines/config/r2r_baselines/navila.yaml#L6), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L145).

4. Medium: Chunk assignment has no guardrails in launcher.
No checks ensure total chunks, start index, and GPU count produce a valid non-overlapping range: [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L3), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L4), [NaVILA/evaluation/scripts/eval/r2r.sh](NaVILA/evaluation/scripts/eval/r2r.sh#L13).

5. Medium: Broad exception handlers hide parse failures.
except without error typing appears in action parse paths and defaults to fixed motion, which may mask systematic failures: [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L236), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L246), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L259), [NaVILA/evaluation/vlnce_baselines/navila_trainer.py](NaVILA/evaluation/vlnce_baselines/navila_trainer.py#L273).

## 9. Suggested Follow-up Prompts
1. Harden NaVILA action parsing so unknown language maps deterministically to STOP and never steps with invalid actions.
2. Refactor queue handling to clear or scope queue_actions per episode, then add a regression test for episode boundary behavior.
3. Add launcher validation in r2r.sh for chunk range sanity and duplicate chunk detection.
4. Introduce a dry-run mode that prints resolved config and output file targets per chunk without model inference.
