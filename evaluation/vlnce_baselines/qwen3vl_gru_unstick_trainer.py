"""Qwen3-VL VLN-CE evaluation with an "unstick" intervention (timeout investigation, 2026-11-10).

Identical to qwen3vl_gru_trainer.Qwen3VLGRUTrainer except for what happens when a forward step
is blocked. Greedy decoding is deterministic, so a model that walks into an obstacle keeps
choosing forward with an unchanged view until the 500-step limit. Here:

  detect   a MOVE_FORWARD primitive whose simulator path_length grows by < BLOCK_EPS_M (5 cm)
  rewind   the blocked step is not added to the model's history (frames + executed actions), so
           the next input is exactly the input at the moment before the blocked step; any queued
           primitives of the same answer are dropped
  choose   at that rewound input, all 10 canonical training answers are scored by teacher-forced
           log-probability (forward 25/50/75 cm, turn left/right 15/30/45 degree, stop), and the
           best answer that is NOT a forward move is executed instead of the greedy one

Everything else (prompt, frame sampling, memory inputs, parsing, the 500-step limit, measures,
output JSON, videos) is the stock harness. The simulator step that was blocked still counts toward
steps_taken. The prompt for scoring is captured from the policy's own act() (the model call is
intercepted before the first forward), so the scored prompt is bit-identical to the greedy one for
every policy version (integrating_gru and continous_mem_tokens).

Extra output per chunk: unstick-<chunks>-<idx>.json with, per episode, the executed primitive
actions (with blocked flags) and every intervention (step, scores of all 10 answers, choice).
Env UNSTICK_CHECK=N: at the first N ordinary decisions also score the candidates and log whether
the top-scored candidate equals the greedy answer (a parity check of the scorer).
"""

import json
import os
import time

import numpy as np
import torch
import tqdm
from habitat import logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.qwen3vl_gru_trainer import (MOVE_FORWARD, GRUVLNPolicy, Qwen3VLGRUTrainer,
                                                 parse_action)

BLOCK_EPS_M = 0.05
CANDIDATES = [
    "The next action is move forward 25 cm.",
    "The next action is move forward 50 cm.",
    "The next action is move forward 75 cm.",
    "The next action is turn left 15 degree.",
    "The next action is turn left 30 degree.",
    "The next action is turn left 45 degree.",
    "The next action is turn right 15 degree.",
    "The next action is turn right 30 degree.",
    "The next action is turn right 45 degree.",
    "I think I should stop because I have finished the instruction.",
]
NONFORWARD = [c for c in CANDIDATES if "move forward" not in c]
ACT_CHAR = {0: "S", 1: "F", 2: "L", 3: "R"}


class _Captured(Exception):
    pass


class _CaptureProxy:
    """Stands in for policy.model: records the first model(**kwargs) call, then aborts act()."""

    def __init__(self, model):
        self._model = model
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _Captured

    def __getattr__(self, name):           # compute_memory etc. go to the real model
        return getattr(self._model, name)


def capture_prompt(policy, instruction, frames, codes):
    real, proxy = policy.model, _CaptureProxy(policy.model)
    policy.model = proxy
    try:
        policy.act(instruction, frames, codes)
    except _Captured:
        pass
    finally:
        policy.model = real
    if proxy.kwargs is None:
        raise RuntimeError("policy.act() returned without calling the model")
    return proxy.kwargs


@torch.no_grad()
def score_candidates(policy, kw, cands=CANDIDATES):
    """Sum log p(answer tokens + eos | prompt) for each candidate; positions continue exactly as
    the policy's greedy loop extends them (nxt + 1 + k)."""
    ids0, att0, pos0 = kw["input_ids"], kw["attention_mask"], kw["position_ids"]
    extras = {k: v for k, v in kw.items() if k not in ("input_ids", "attention_mask", "position_ids")}
    nxt = pos0.amax(dim=2)                                   # (3, 1)
    eos = policy.tok.eos_token_id
    start = ids0.size(1)
    out = {}
    for text in cands:
        toks = policy.tok(text, add_special_tokens=False)["input_ids"] + [eos]
        t = torch.tensor([toks], dtype=ids0.dtype, device=ids0.device)
        k = torch.arange(len(toks), device=pos0.device, dtype=pos0.dtype)
        ids = torch.cat([ids0, t], 1)
        att = torch.cat([att0, torch.ones_like(t)], 1)
        pos = torch.cat([pos0, (nxt[:, :, None] + 1 + k[None, None, :])], 2)
        logits = policy.model(input_ids=ids, attention_mask=att, position_ids=pos, **extras).logits
        lp = torch.log_softmax(logits[0, start - 1:start - 1 + len(toks)].float(), -1)
        out[text] = float(lp.gather(1, t[0, :, None]).sum())
    return out


@baseline_registry.register_trainer(name="qwen3vl_gru_unstick")
class Qwen3VLGRUUnstickTrainer(Qwen3VLGRUTrainer):

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter) -> None:
        arm = self.config.ARM
        logger.info(f"checkpoint_path: {checkpoint_path}  arm: {arm}  [unstick: best non-forward when blocked]")
        policy = GRUVLNPolicy(checkpoint_path, arm=arm, device=f"cuda:{self.config.TORCH_GPU_ID}", log=logger.info)
        n_check = int(os.environ.get("UNSTICK_CHECK", "0"))

        config = self.config.clone()
        split = config.EVAL.SPLIT
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        run_name = model_name + (config.RUN_TAG or f"-{arm}-unstick")

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, run_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        config.freeze()

        fname = os.path.join(config.RESULTS_DIR, f"{split}_{self.num_chunks}-{self.chunk_idx}.json")
        uname = os.path.join(config.RESULTS_DIR, f"unstick-{self.num_chunks}-{self.chunk_idx}.json")
        if config.EVAL.SAVE_RESULTS and os.path.exists(fname):
            logger.info("skipping -- evaluation exists.")
            return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes, unstick_log = {}, {}
        past_rgbs, past_actions = [[]], [[]]
        travelled_m = [0.0]
        rgb_frames = [[]]
        ep_trace = {"acts": [], "interventions": []}
        pending_alt = False
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)
        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        start_time = time.time()
        assert envs.num_envs == 1

        queue_actions = []
        n_decisions, n_interv, n_checked, n_check_agree, decision_time = 0, 0, 0, 0, 0.0

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()
            from_decision = False
            if len(queue_actions) > 0:
                executed = queue_actions.pop(0)
            else:
                from_decision = True
                curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
                instruction = current_episodes[0].instruction.instruction_text
                frames, codes = past_rgbs[0] + [curr_rgb], past_actions[0]
                t0 = time.time()
                if pending_alt:
                    kw = capture_prompt(policy, instruction, frames, codes)
                    scores = score_candidates(policy, kw)
                    answer = max(NONFORWARD, key=scores.get)
                    top = max(CANDIDATES, key=scores.get)
                    ep_trace["interventions"].append({"step": len(ep_trace["acts"]), "hist": len(codes),
                                                      "top": top, "chosen": answer,
                                                      "scores": {k: round(v, 3) for k, v in scores.items()}})
                    n_interv += 1
                    pending_alt = False
                else:
                    answer = policy.act(instruction, frames, codes)
                    if n_checked < n_check:
                        kw = capture_prompt(policy, instruction, frames, codes)
                        scores = score_candidates(policy, kw)
                        top = max(CANDIDATES, key=scores.get)
                        n_checked += 1
                        n_check_agree += int(top == answer)
                        logger.info(f"[unstick-check {n_checked}] greedy={answer!r} top-scored={top!r} "
                                    f"{'OK' if top == answer else 'DIFF'}")
                decision_time += time.time() - t0
                n_decisions += 1
                action, repeats, _ = parse_action(answer)
                executed = action
                queue_actions.extend([action] * (repeats - 1))

            pre_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
            path_before = travelled_m[0]
            outputs = envs.step([executed])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            i = 0
            path_now = float(infos[i].get("path_length", path_before))
            blocked = executed == MOVE_FORWARD and (path_now - path_before) < BLOCK_EPS_M
            travelled_m[i] = path_now
            ep_trace["acts"].append(ACT_CHAR[executed] + ("b" if blocked else ""))
            if blocked and not dones[i]:
                # rewind: the blocked step never enters the model's history
                queue_actions.clear()
                pending_alt = True
            else:
                past_rgbs[i].append(pre_rgb)
                past_actions[i].append(executed)

            if len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(observations[i], infos[i])
                frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                rgb_frames[i].append(frame)

            if dones[i]:
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                unstick_log[ep_id] = {"acts": "".join(a[0] if len(a) == 1 else a[0].lower() for a in ep_trace["acts"]),
                                      "n_interventions": len(ep_trace["interventions"]),
                                      "interventions": ep_trace["interventions"]}
                if config.EVAL.SAVE_RESULTS:
                    self._write_partial(fname, stats_episodes, num_eps, time.time() - start_time)
                    with open(uname + ".tmp", "w") as f:
                        json.dump(unstick_log, f)
                    os.replace(uname + ".tmp", uname)
                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i], past_actions[i] = [], []
                travelled_m[i] = 0.0
                queue_actions.clear()
                pending_alt = False
                ep_trace = {"acts": [], "interventions": []}
                if config.use_pbar:
                    pbar.update()
                logger.info(f"[{run_name}] episode {ep_id} done: steps {stats_episodes[ep_id].get('steps_taken')} "
                            f"success {stats_episodes[ep_id].get('success')} interventions {unstick_log[ep_id]['n_interventions']} "
                            f"({len(stats_episodes)}/{num_eps}, {round(time.time() - start_time)} s)")
                if len(config.VIDEO_OPTION) > 0:
                    generate_video(video_option=config.VIDEO_OPTION, video_dir=config.VIDEO_DIR,
                                   images=rgb_frames[i], episode_id=ep_id, checkpoint_idx="0",
                                   metrics={"spl": stats_episodes[ep_id]["spl"]}, tb_writer=writer)
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()
            for j in range(envs.num_envs):
                if next_episodes[j].episode_id in stats_episodes:
                    envs_to_pause.append(j)
            (envs, batch, rgb_frames) = self._pause_envs(envs_to_pause, envs, batch, rgb_frames)

        envs.close()
        if config.use_pbar:
            pbar.close()
        logger.info(f"[{run_name}] {len(stats_episodes)} episodes, {n_decisions} decisions, {n_interv} interventions, "
                    f"{decision_time / max(1, n_decisions):.2f}s per decision"
                    + (f", scorer parity {n_check_agree}/{n_checked}" if n_checked else ""))
        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)
            with open(uname, "w") as f:
                json.dump(unstick_log, f)
            if os.path.exists(fname + ".partial"):
                os.remove(fname + ".partial")
