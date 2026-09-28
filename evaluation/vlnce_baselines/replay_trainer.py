"""Re-render a finished VLN-CE eval from its per-step action log. No model is loaded.

VLN-CE is deterministic given the start pose and the primitive action sequence, so stepping the
simulator through the logged actions reproduces the original trajectory exactly, and the videos
come out as if the eval had recorded them. The env, video and results-dir code is the same as
qwen3vl_gru_trainer.py, so the output looks and is laid out like any other eval here.

The log must carry every executed primitive, one line per env step:
    [Qwen][ep=E][step=S] raw='...' parsed=ACT xK     a model decision (its first primitive)
    [Qwen][ep=E][step=S] queued_action=ACT           a queued sub-step of that decision
Primitives are replayed as logged; the answer text is never re-parsed, so a difference between the
original harness's parser and ours cannot change the trajectory.

Every episode is checked against the original run's JSON (steps, path length, final distance,
success, oracle success). Differences go to replay_check_<N>-<i>.json; nothing is hidden.
"""

import glob
import json
import os
import re
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
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

for _name, _alias in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):  # numpy 1.26 removed these; habitat-lab 0.1.7 still uses them
        setattr(np, _name, _alias)

ACTION = {"STOP": 0, "MOVE_FORWARD": 1, "TURN_LEFT": 2, "TURN_RIGHT": 3}
STEP_LINE = re.compile(r"\[Qwen\]\[ep=(\d+)\]\[step=(\d+)\] (?:raw='.*?' parsed=(\w+) x\d+|queued_action=(\w+))")
CHECK = {"steps_taken": 0.0, "path_length": 1e-3, "distance_to_goal": 1e-3, "success": 0.0, "oracle_success": 0.0}


def load_actions(log_path):
    """{episode_id: [primitive action ids in execution order]}; raises if a step is missing."""
    seq = {}
    for line in open(log_path, errors="ignore"):
        m = STEP_LINE.search(line)
        if m:
            seq.setdefault(m.group(1), []).append((int(m.group(2)), ACTION[m.group(3) or m.group(4)]))
    out = {}
    for ep, steps in seq.items():
        nums = [s for s, _ in steps]
        if nums != list(range(1, len(nums) + 1)):
            raise ValueError(f"episode {ep}: logged steps are not 1..{len(nums)} in order")
        out[ep] = [a for _, a in steps]
    return out


@baseline_registry.register_trainer(name="action_replay")
class ActionReplayTrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        super().__init__(config)

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter) -> None:
        actions = load_actions(self.config.REPLAY_LOG)
        reference = {}
        for f in sorted(glob.glob(self.config.REPLAY_JSON)):
            reference.update(json.load(open(f)))
        logger.info(f"[replay] {len(actions)} episodes of actions from {self.config.REPLAY_LOG}, "
                    f"{len(reference)} reference results from {self.config.REPLAY_JSON}")

        config = self.config.clone()
        split = config.EVAL.SPLIT
        run_name = os.path.basename(os.path.normpath(checkpoint_path)) + (config.RUN_TAG or "-replay")
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(config.RESULTS_DIR, run_name, config.TASK_CONFIG.DATASET.TYPE,
                                          config.TASK_CONFIG.DATASET.SPLIT)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        config.freeze()

        fname = os.path.join(config.RESULTS_DIR, f"{split}_{self.num_chunks}-{self.chunk_idx}.json")
        check_name = os.path.join(config.RESULTS_DIR, f"replay_check_{self.num_chunks}-{self.chunk_idx}.json")
        if os.path.exists(fname):
            logger.info("skipping -- replay exists.")
            return
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        assert envs.num_envs == 1

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)
        stats_episodes, problems = {}, {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        cursor = 0                    # index into the current episode's logged actions
        pbar = tqdm.tqdm(total=num_eps)
        start = time.time()

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            ep_id = envs.current_episodes()[0].episode_id
            seq = actions.get(ep_id)
            if seq is None:
                raise KeyError(f"episode {ep_id} is not in the replay log")
            if cursor < len(seq):
                action = seq[cursor]
            else:                     # log exhausted but env not done: record it, stop the episode
                problems.setdefault(ep_id, {})["log_exhausted_at"] = cursor
                action = ACTION["STOP"]
            cursor += 1
            outputs = envs.step([action])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(observations[0], infos[0])
                frame = append_text_to_image(frame, envs.current_episodes()[0].instruction.instruction_text)
                rgb_frames[0].append(frame)

            if dones[0]:
                stats_episodes[ep_id] = infos[0]
                if cursor < len(seq):
                    problems.setdefault(ep_id, {})["env_done_early"] = {"at": cursor, "logged": len(seq)}
                ref = reference.get(ep_id)
                if ref is not None:
                    diff = {k: {"replay": float(infos[0][k]), "original": float(ref[k])}
                            for k, tol in CHECK.items() if abs(float(infos[0][k]) - float(ref[k])) > tol}
                    if diff:
                        problems.setdefault(ep_id, {})["metrics"] = diff
                else:
                    problems.setdefault(ep_id, {})["no_reference"] = True
                if len(config.VIDEO_OPTION) > 0:
                    generate_video(video_option=config.VIDEO_OPTION, video_dir=config.VIDEO_DIR,
                                   images=rgb_frames[0], episode_id=ep_id, checkpoint_idx="0",
                                   metrics={"spl": stats_episodes[ep_id]["spl"]}, tb_writer=writer)
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[0] = []
                observations[0] = envs.reset_at(0)[0]
                cursor = 0
                pbar.update()

            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            if envs.current_episodes()[0].episode_id in stats_episodes:
                envs.pause_at(0)

        envs.close()
        pbar.close()
        n = len(stats_episodes)
        logger.info(f"[replay] {n} episodes in {time.time() - start:.0f}s; {len(problems)} with differences")
        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)
            with open(check_name, "w") as f:
                json.dump({"episodes": n, "with_differences": len(problems), "tolerances": CHECK,
                           "problems": problems}, f, indent=2)

    def eval(self) -> None:
        self.device = torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
