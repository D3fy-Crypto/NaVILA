## Brief purpose
NaVILA is a research codebase that extends VILA/LLAVA-style multimodal models for vision-language-action navigation. The repository contains two main workflows you will see in the code: model training / pretraining (in `llava/`) and environment-based evaluation (in `evaluation/`, built on VLN-CE + Habitat). This file gives targeted, actionable hints for an AI coding agent to be immediately productive.

## High-level architecture (what to read first)
- `README.md` — project overview, datasets, and high-level commands. Start here to understand goals and dataset expectations.
- `pyproject.toml` — package name `vila`, pinned dependency versions (Torch 2.3, Transformers 4.37.2, specific webdataset, etc.). Use this to infer exact runtime versions.
- `llava/` — core multimodal model and training code. Key files:
  - `llava/entry.py` — top-level programmatic model loader (calls `llava.model.builder.load_pretrained_model`). Use this to learn how checkpoints are laid out (many checkpoints have a `model/` subfolder).
  - `llava/cli/run.py` — SLURM wrapper used by team runs; expects env vars `VILA_SLURM_ACCOUNT` and `VILA_SLURM_PARTITION` and shows the job-run conventions (output in `runs/<mode>/<job-name>`).
  - `llava/train/` — training entrypoints and Deepspeed/transformers compatibility patches (see `transformers_replace/` and `deepspeed_replace/`).
- `evaluation/` — VLN-CE based evaluation and habitat integration. Look at `evaluation/scripts/` and `evaluation/habitat_extensions/` for Habitat-specific adapters.

## Developer workflows and gotchas (exact, discoverable steps)
- Environment: the canonical setup script is `./environment_setup.sh navila` which (a) creates a conda env, (b) installs the package in editable mode, installs extras `.[train]` and `.[eval]`, installs FlashAttention2 wheel, and patches Transformers/Deepspeed by copying files from `llava/train/transformers_replace` and `llava/train/deepspeed_replace` into site-packages. Prefer following this script; it is the single source of truth for env setup.
- Training: look for training scripts in `llava/train/train.py` (and `scripts/train/sft_8frames.sh` for example launch scripts). The repo provides a Slurm `llava/cli/run.py` wrapper for cluster runs.
- Evaluation: evaluation depends on VLN-CE and Habitat-Sim v0.1.7. The README lists necessary manual steps (build Habitat-Sim from source, run `evaluation/scripts/habitat_sim_autofix.py` to patch NumPy compatibility). Evaluation entry: `evaluation/scripts/eval/r2r.sh` (see README for how to pass checkpoint path and GPU list). Visual outputs go to `eval_out/<CKPT_NAME>/...`.

## Patterns & conventions to follow in code changes
- Checkpoint layout: loaders (e.g., `llava/entry.py`) expect a `model` subfolder inside checkpoints. When writing code that saves/loads models, follow the same layout.
- Monkey-patching approach: instead of forking library code, the project copies compatibility patches into the site-packages at install time (`transformers_replace/`, `deepspeed_replace/`). For local experiments prefer replicating the same patching approach (copy files) rather than changing upstream libs directly.
- SLURM & run naming: run names often contain `%t` placeholder (replaced by timestamp in `llava/cli/run.py`). Output directories use `runs/<mode>/<job-name>` and expect `RUN_NAME` and `OUTPUT_DIR` env vars for downstream scripts.
- Pinned versions: `pyproject.toml` pins exact versions (Torch, Transformers, webdataset older/modified versions). Match these versions for reproducing reported results.

## Integration points & external dependencies
- HuggingFace model/weights: README references checkpoints hosted on HF (e.g., `a8cheng/navila-siglip-llama3-8b-v1.5-pretrain`). Code often uses HF-style repo layouts — when testing loaders, look for `model/` under the repo path.
- FlashAttention2: installed via a specific wheel in `environment_setup.sh`; missing this wheel will break training performance code paths.
- VLN-CE / Habitat: evaluation integrates VLN-CE; building Habitat from source is required and the repo provides `evaluation/scripts/habitat_sim_autofix.py` to patch compatibility.

## Useful concrete examples to copy/paste
- Install & setup (canonical): run `./environment_setup.sh navila` then `conda activate navila` (this installs edits, wheel, and copies patches).
- Run a SLURM-backed training job: the maintainers use `llava/cli/run.py` to compose `srun` commands. It requires `VILA_SLURM_ACCOUNT` and `VILA_SLURM_PARTITION` env vars and places logs in `runs/<mode>/<job-name>/slurm/`.
- Load a model programmatically: use `llava.entry.load(model_path)` — it auto-detects the `model/` subfolder and calls `llava.model.builder.load_pretrained_model`.

## What NOT to change without checking with maintainers
- Do not alter the pinned versions in `pyproject.toml` when reproducing experiments; changes require retesting both training and evaluation pipelines.
- Avoid in-place edits of system site-packages in CI; the repo’s current pattern is to copy small patches into site-packages at install time—mirror that workflow locally instead of editing packages directly.

## Where to look next when you need more context
- `llava/train/` — for trainer behavior and parameter parsing.
- `llava/model/` — for model architecture construction and where new modalities would be wired in.
- `evaluation/habitat_extensions/` — for Habitat-specific observation/action adapters used by VLN-CE.
- `scripts/train/sft_8frames.sh` — concrete example of a training launch.

If any of these points are unclear or you'd like me to expand any section (for example, add exact command snippets for evaluation or show how a checkpoint is structured in the filesystem), tell me which part to expand and I'll iterate.
