"""VLN-CE evaluation of the Stage-3 Qwen3-VL + spatial-memory-GRU model.

A copy of NaVILATrainer with the VILA model swapped for ours. Everything that decides the
SCORE is left exactly as NaVILA has it -- env construction, the action regexes, the magnitude
parsing, the primitive-action queue (forward 75 cm -> three 25 cm steps, turn 45 deg -> three
15 deg steps), the measures and the output JSON -- so numbers stay comparable with runs of the
stock harness. What changes:

  model     qwenvl.eval.gru_vln_policy.GRUVLNPolicy, which builds the prompt, samples the
            frames and decodes with the SAME code the model was trained under.
  history   NaVILA's evaluator keeps past frames but throws away the actions it takes. The GRU
            needs them, so executed primitive actions are recorded next to the frame append
            (same cadence, queued sub-steps included) and reset per episode.
  queue     cleared on episode end. NaVILA leaks queue_actions across episodes, which would
            execute the tail of one episode's turn at the start of the next.

Arms are selected by config.ARM: gru | gru-memzero | nogru.
"""

import json
import os
import re
import sys
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
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

# numpy 1.26 removed these aliases; habitat-lab v0.1.7 still uses them in code we do not own
# (habitat/tasks/nav/nav.py, sims/pyrobot/pyrobot.py). Restore them before habitat builds a task.
for _name, _alias in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

_QWEN_FT = os.environ.get(
    "QWEN_FT_DIR",
    "/home/djonna1/brain_inspired_project/Qwen3-VL/.claude/worktrees/integrating_gru/qwen-vl-finetune",
)
if _QWEN_FT not in sys.path:
    sys.path.insert(0, _QWEN_FT)

from qwenvl.data.data_processor_gru_chang import action_codes_from_answer  # noqa: E402
from qwenvl.eval.gru_vln_policy import GRUVLNPolicy  # noqa: E402

# VLN-CE action indices, and the action codes the GRU was trained on. They agree by
# construction: STOP=0, MOVE_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3.
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


def parse_action(answer: str):
    """NaVILA's own text -> (action, repeats) mapping, kept verbatim so scores stay comparable.

    First matching pattern wins; magnitudes are snapped to the simulator's primitives
    (25 cm, 15 deg) exactly as evaluation/vlnce_baselines/navila_trainer.py does. Returns
    (action, repeats, parsed); an unparseable answer steps forward once, which is what the
    stock harness ends up doing.
    """
    low = answer.lower()
    if re.search(r"\bstop\b", low):
        return STOP, 1, True
    if re.search(r"\bis move forward\b", low):
        m = re.search(r"move forward (\d+) cm", low)
        cm = int(m.group(1)) if m else 25
        if cm % 25:
            cm = min([25, 50, 75], key=lambda x: abs(x - cm))
        return MOVE_FORWARD, max(1, cm // 25), True
    for verb, act in (("left", TURN_LEFT), ("right", TURN_RIGHT)):
        if re.search(rf"\bis turn {verb}\b", low):
            m = re.search(rf"turn {verb} (\d+) degree", low)
            deg = int(m.group(1)) if m else 15
            if deg % 15:
                deg = min([15, 30, 45], key=lambda x: abs(x - deg))
            return act, max(1, deg // 15), True
    return MOVE_FORWARD, 1, False


@baseline_registry.register_trainer(name="qwen3vl_gru")
class Qwen3VLGRUTrainer(BaseVLNCETrainer):
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
        arm = self.config.ARM
        logger.info(f"checkpoint_path: {checkpoint_path}  arm: {arm}")

        policy = GRUVLNPolicy(checkpoint_path, arm=arm, device=f"cuda:{self.config.TORCH_GPU_ID}",
                              log=logger.info)

        config = self.config.clone()
        split = config.EVAL.SPLIT
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        run_name = model_name + (config.RUN_TAG or f"-{arm}")

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
            config.RESULTS_DIR, run_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(config.RESULTS_DIR, f"{split}_{self.num_chunks}-{self.chunk_idx}.json")
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}
        past_rgbs = [[] for _ in range(envs.num_envs)]
        past_actions = [[] for _ in range(envs.num_envs)]   # executed primitives -> the GRU history
        rgb_frames = [[] for _ in range(envs.num_envs)]

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = "[Ckpt: {ckpt}] [Episodes evaluated: {evaluated}/{total}] [Time elapsed (s): {time}]"
        start_time = time.time()
        assert envs.num_envs == 1

        queue_actions = []
        n_decisions, n_unparsed, decision_time = 0, 0, 0.0

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            if len(queue_actions) > 0:
                executed = queue_actions.pop(0)
                outputs = envs.step([executed])
            else:
                curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
                instruction = current_episodes[0].instruction.instruction_text
                t0 = time.time()
                answer = policy.act(instruction, past_rgbs[0] + [curr_rgb], past_actions[0])
                decision_time += time.time() - t0
                n_decisions += 1

                action, repeats, parsed = parse_action(answer)
                if not parsed:
                    n_unparsed += 1
                # Cross-check against the parser the GRU history was built with in training.
                # They agree on well-formed answers; NaVILA additionally snaps magnitudes.
                train_codes = action_codes_from_answer(answer)
                if train_codes and (train_codes[0] != action or len(train_codes) != repeats):
                    logger.info(f"[parse] harness {[action] * repeats} != training {train_codes} "
                                f"for {answer!r}")
                if n_decisions <= 20 or n_decisions % 100 == 0:
                    logger.info(f"[decision {n_decisions}] hist={len(past_actions[0])} "
                                f"-> {answer!r} -> {[action] * repeats}")

                executed = action
                outputs = envs.step([action])
                queue_actions.extend([action] * (repeats - 1))

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))
                past_actions[i].append(executed)           # same cadence as the frame history

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(frame, current_episodes[i].instruction.instruction_text)
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i] = []
                past_actions[i] = []
                queue_actions.clear()                      # never carry a turn into the next episode

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(log_str.format(ckpt=run_name, evaluated=len(stats_episodes),
                                               total=num_eps, time=round(time.time() - start_time)))

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION, video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i], episode_id=ep_id, checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]}, tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)
            (envs, batch, rgb_frames) = self._pause_envs(envs_to_pause, envs, batch, rgb_frames)

        envs.close()
        if config.use_pbar:
            pbar.close()

        logger.info(f"[{run_name}] {len(stats_episodes)} episodes, {n_decisions} decisions, "
                    f"{n_unparsed} unparsed, {decision_time / max(1, n_decisions):.2f}s per decision")

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(envs_to_pause, envs, batch, rgb_frames=None):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            for k, v in batch.items():
                batch[k] = v[state_index]
            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]
        return envs, batch, rgb_frames

    def eval(self) -> None:
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
        )
        if "tensorboard" in self.config.VIDEO_OPTION:
            assert len(self.config.TENSORBOARD_DIR) > 0, "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert len(self.config.VIDEO_DIR) > 0, "Must specify a directory for storing videos on disk"

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            if os.path.isdir(self.config.EVAL_CKPT_PATH_DIR):
                self._eval_checkpoint(self.config.EVAL_CKPT_PATH_DIR, writer)
            else:
                raise ValueError(f"EVAL_CKPT_PATH_DIR is not a directory: {self.config.EVAL_CKPT_PATH_DIR}")
