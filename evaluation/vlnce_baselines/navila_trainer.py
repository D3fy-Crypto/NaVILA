import copy
import gc
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict

import lmdb
import msgpack_numpy
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
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MOTION_TOKEN, IMAGE_TOKEN_INDEX, MOTION_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    tokenizer_mm_token,
)
from llava.model.builder import load_pretrained_model


def _sample_indices(total_frames, num_frames):
    if total_frames <= 0:
        return [0] * num_frames
    padded = max(0, num_frames - total_frames)
    effective_len = total_frames + padded
    sampled = np.linspace(0, effective_len - 1, num=num_frames - 1, endpoint=False, dtype=int).tolist()
    sampled.append(effective_len - 1)
    return [max(0, idx - padded) for idx in sampled]


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    frames = copy.deepcopy(images)
    if len(frames) == 0:
        frames = [Image.new("RGB", (width, height), color=(0, 0, 0))]

    sampled_indices = _sample_indices(len(frames), num_frames)
    sampled_frames = [frames[i] for i in sampled_indices]
    return sampled_frames, sampled_indices


def _wrap_to_pi(delta):
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def _pose_delta(prev_pose, cur_pose):
    dx = float(cur_pose["x"]) - float(prev_pose["x"])
    dy = float(cur_pose["z"]) - float(prev_pose["z"])
    dyaw = _wrap_to_pi(float(cur_pose["heading"]) - float(prev_pose["heading"]))
    return dx, dy, dyaw


def _normalize_delta(dx, dy, dyaw, trans_norm=0.25):
    return [
        float(dx) / trans_norm,
        float(dy) / trans_norm,
        math.sin(float(dyaw)),
        math.cos(float(dyaw)),
    ]


def _make_motion_windows(pose_deltas_step, num_frames, window_size=10, trans_norm=0.25):
    out = torch.zeros(num_frames, window_size, 4, dtype=torch.float32)
    for t in range(1, num_frames):
        start = max(0, t - window_size)
        chunk = pose_deltas_step[start:t]
        pad = window_size - len(chunk)
        for j, (dx, dy, dyaw) in enumerate(chunk):
            out[t, pad + j] = torch.tensor(_normalize_delta(dx, dy, dyaw, trans_norm), dtype=torch.float32)
    return out


@baseline_registry.register_trainer(name="navila")
class NaVILATrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

        super().__init__(config)

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
        """
        logger.info(f"checkpoint_path: {checkpoint_path}")

        # build model
        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)
        model = model.cuda()

        config = self.config.clone()
        split = config.EVAL.SPLIT

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
            config.RESULTS_DIR, model_name, config.TASK_CONFIG.DATASET.TYPE, config.TASK_CONFIG.DATASET.SPLIT
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()
        navila_cfg = getattr(config, "NAVILA", None)
        enable_motion = bool(getattr(navila_cfg, "ENABLE_MOTION", True))
        motion_window_size = int(getattr(navila_cfg, "MOTION_WINDOW_SIZE", 10))
        motion_trans_norm = float(getattr(navila_cfg, "MOTION_TRANS_NORM", 0.25))
        strict_motion = bool(getattr(navila_cfg, "STRICT_MOTION", True))
        if strict_motion:
            if motion_window_size <= 0:
                raise ValueError(f"NAVILA.MOTION_WINDOW_SIZE must be > 0, got {motion_window_size}")
            if motion_trans_norm <= 0:
                raise ValueError(f"NAVILA.MOTION_TRANS_NORM must be > 0, got {motion_trans_norm}")

        model_motion_encoder = model.get_motion_encoder() if hasattr(model, "get_motion_encoder") else None
        use_motion = enable_motion and model_motion_encoder is not None
        if enable_motion and not use_motion:
            logger.warning("NAVILA.ENABLE_MOTION=True but checkpoint has no motion encoder; using image-only eval.")

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"{split}_{self.num_chunks}-{self.chunk_idx}.json",
            )
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
        past_poses = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]  # this is for visualization, contains text and map

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_path}]" " [Episodes evaluated: {evaluated}/{total}]" " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        assert envs.num_envs == 1

        queue_actions = []
        logged_motion_stats = False

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            current_episodes = envs.current_episodes()
            step_poses = [None for _ in range(envs.num_envs)]
            if use_motion:
                for i in range(envs.num_envs):
                    try:
                        step_poses[i] = envs.call_at(i, "get_agent_pose")
                    except Exception as exc:
                        msg = (
                            f"Failed to query pose for episode_id={current_episodes[i].episode_id}: {exc}. "
                            "Ensure VLNCEDaggerEnv.get_agent_pose is available."
                        )
                        if strict_motion:
                            raise ValueError(msg) from exc
                        logger.warning(msg)

            if len(queue_actions) > 0:
                print(f"using queue...{queue_actions[0]}")
                outputs = envs.step([queue_actions[0]])
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    past_and_current_rgbs_raw = past_rgbs[0] + [curr_rgb]
                    num_video_frames = model.config.num_video_frames

                    past_and_current_rgbs, sampled_indices = sample_and_pad_images(
                        past_and_current_rgbs_raw,
                        num_frames=num_video_frames,
                    )

                    instruction = current_episodes[0].instruction.instruction_text
                    use_motion_this_step = use_motion and (step_poses[0] is not None)
                    motion_tensor = None

                    if use_motion and step_poses[0] is None and strict_motion:
                        raise ValueError(
                            f"Motion enabled but pose is unavailable for episode_id={current_episodes[0].episode_id}"
                        )

                    if use_motion_this_step:
                        past_and_current_poses = past_poses[0] + [step_poses[0]]
                        if strict_motion and len(past_and_current_poses) != len(past_and_current_rgbs_raw):
                            raise ValueError(
                                "Pose/frame history mismatch: "
                                f"poses={len(past_and_current_poses)} frames={len(past_and_current_rgbs_raw)}"
                            )

                        step_pose_deltas = []
                        for idx in range(1, len(past_and_current_poses)):
                            step_pose_deltas.append(_pose_delta(past_and_current_poses[idx - 1], past_and_current_poses[idx]))

                        pose_deltas_step = []
                        for j in range(1, len(sampled_indices)):
                            prev_idx = sampled_indices[j - 1]
                            curr_idx = sampled_indices[j]
                            dx = dy = dyaw = 0.0
                            max_k = min(curr_idx, len(step_pose_deltas))
                            for k in range(prev_idx, max_k):
                                d = step_pose_deltas[k]
                                dx += float(d[0])
                                dy += float(d[1])
                                dyaw += float(d[2])
                            pose_deltas_step.append((dx, dy, dyaw))
                        if len(pose_deltas_step) != max(0, num_video_frames - 1):
                            pose_deltas_step = pose_deltas_step[: num_video_frames - 1] + [(0.0, 0.0, 0.0)] * max(
                                0, (num_video_frames - 1) - len(pose_deltas_step)
                            )

                        motion_tensor = _make_motion_windows(
                            pose_deltas_step=pose_deltas_step,
                            num_frames=num_video_frames,
                            window_size=motion_window_size,
                            trans_norm=motion_trans_norm,
                        )
                        if strict_motion and tuple(motion_tensor.shape) != (num_video_frames, motion_window_size, 4):
                            raise ValueError(
                                f"Invalid motion tensor shape {tuple(motion_tensor.shape)}; "
                                f"expected ({num_video_frames}, {motion_window_size}, 4)"
                            )

                        hist_pairs = (DEFAULT_MOTION_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n") * max(
                            0, num_video_frames - 1
                        )
                        cur_pair = DEFAULT_MOTION_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n"
                        question = (
                            "Imagine you are a robot programmed for navigation tasks. "
                            f"You have been given a video of historical observations {hist_pairs}, "
                            f"and current observation {cur_pair}. "
                            f'Your assigned task is: "{instruction}" '
                            "Analyze this series of observations to decide your next action, which could be "
                            "turning left or right by a specific degree, moving forward a certain distance, "
                            "or stop if the task is completed."
                        )
                    else:
                        interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)
                        question = (
                            "Imagine you are a robot programmed for navigation tasks. You have been given a video "
                            f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                            "Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                            "degree, moving forward a certain distance, or stop if the task is completed."
                        )

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    if use_motion_this_step:
                        input_ids = tokenizer_mm_token(
                            prompt,
                            tokenizer,
                            image_token_index=IMAGE_TOKEN_INDEX,
                            motion_token_index=MOTION_TOKEN_INDEX,
                            return_tensors="pt",
                        ).unsqueeze(0)
                    else:
                        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                    input_ids = input_ids.to(model.device)

                    num_image_tokens = int((input_ids[0] == IMAGE_TOKEN_INDEX).sum().item())
                    num_motion_tokens = int((input_ids[0] == MOTION_TOKEN_INDEX).sum().item())
                    if use_motion_this_step and strict_motion:
                        if num_image_tokens != num_video_frames or num_motion_tokens != num_video_frames:
                            raise ValueError(
                                "Prompt multimodal token mismatch: "
                                f"image_tokens={num_image_tokens} motion_tokens={num_motion_tokens} "
                                f"expected={num_video_frames}"
                            )
                        if motion_tensor is None:
                            raise ValueError("Motion enabled but motion tensor was not created.")
                    if motion_tensor is not None:
                        motion_tensor = motion_tensor.to(model.device, dtype=torch.float32)
                        if strict_motion and motion_tensor.shape[0] != num_motion_tokens:
                            raise ValueError(
                                f"Motions/token mismatch: motions={motion_tensor.shape[0]} motion_tokens={num_motion_tokens}"
                            )
                    if use_motion_this_step and not logged_motion_stats:
                        logger.info(
                            "[NaVILA motion] image_tokens=%s motion_tokens=%s motions_shape=%s",
                            num_image_tokens,
                            num_motion_tokens,
                            tuple(motion_tensor.shape),
                        )
                        logged_motion_stats = True

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        if use_motion_this_step:
                            output_ids = model.generate(
                                input_ids,
                                images=images_tensor,
                                motions=motion_tensor,
                                do_sample=False,
                                temperature=0.0,
                                max_new_tokens=32,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        else:
                            output_ids = model.generate(
                                input_ids,
                                images=images_tensor,
                                do_sample=False,
                                temperature=0.0,
                                max_new_tokens=32,
                                use_cache=True,
                                stopping_criteria=[stopping_criteria],
                                pad_token_id=tokenizer.eos_token_id,
                            )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()

                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                    outputs = outputs.strip()
                    print(outputs)

                    # Define the regex patterns for each action
                    patterns = {
                        0: re.compile(r"\bstop\b", re.IGNORECASE),
                        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
                        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
                        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
                    }

                    # Function to map a string to an action integer
                    def map_string_to_action(s):
                        for action, pattern in patterns.items():
                            if pattern.search(s):
                                return action
                        return None  # Return None if no match is found

                    try:
                        actions = [map_string_to_action(outputs)]
                    except:
                        actions = [1]
                    print(actions)

                if actions[0] == 1:
                    try:
                        match = re.search(r"move forward (\d+) cm", outputs)
                        distance = int(match.group(1))
                    except:
                        distance = 25
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    outputs = envs.step([1])

                    for _ in range(int(distance // 25) - 1):
                        queue_actions.append(1)

                elif actions[0] == 2:
                    try:
                        match = re.search(r"turn left (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([2])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(2)
                    print(f"queue length: {len(queue_actions)}")

                elif actions[0] == 3:
                    try:
                        match = re.search(r"turn right (\d+) degree", outputs)
                        degree = int(match.group(1))
                    except:
                        degree = 15
                    if (degree % 15) != 0:
                        degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                    outputs = envs.step([3])

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)

                else:  # 0, stop
                    outputs = envs.step(actions)

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))
                if use_motion and step_poses[i] is not None:
                    past_poses[i].append(step_poses[i])

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
                past_poses[i] = []
                queue_actions = []

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (envs, batch, rgb_frames, past_rgbs, past_poses,) = self._pause_envs(
                envs_to_pause,
                envs,
                batch,
                rgb_frames,
                past_rgbs=past_rgbs,
                past_poses=past_poses,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        batch,
        rgb_frames=None,
        past_rgbs=None,
        past_poses=None,
    ):
        # pausing envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]
            if past_rgbs is not None:
                past_rgbs = [past_rgbs[i] for i in state_index]
            if past_poses is not None:
                past_poses = [past_poses[i] for i in state_index]

        return (
            envs,
            batch,
            rgb_frames,
            past_rgbs,
            past_poses,
        )

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
                self._eval_checkpoint(
                    self.config.EVAL_CKPT_PATH_DIR,
                    writer,
                )
