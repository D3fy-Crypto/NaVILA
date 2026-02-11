import copy
import gc
import json
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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.grid_rnn import MotionEncoderWithProjector


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        padding_frames = num_frames - len(frames)
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    else:
        padding_frames = 0

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


@baseline_registry.register_trainer(name="navila")
class NaVILATrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

        super().__init__(config)

        # Oracle deltas cache (loaded on first eval)
        self._oracle_deltas_by_episode = None

    def _load_oracle_deltas(self, split_name: str):
        """Load oracle pose deltas for the given split into a dict by episode_id."""
        if self._oracle_deltas_by_episode is not None:
            return self._oracle_deltas_by_episode

        oracle_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "oracle_exports",
            f"oracle_deltas_{split_name}.jsonl",
        )
        oracle_path = os.path.abspath(oracle_path)

        deltas_by_episode = {}
        if not os.path.exists(oracle_path):
            logger.warning(f"[Eval] Oracle deltas file not found: {oracle_path}")
            self._oracle_deltas_by_episode = deltas_by_episode
            return deltas_by_episode

        logger.info(f"[Eval] Loading oracle deltas from {oracle_path}")
        with open(oracle_path, "r") as fp:
            for line in fp:
                data = json.loads(line)
                episode_id = data.get("episode_id")
                deltas = data.get("deltas", [])
                if episode_id is None or not deltas:
                    continue
                deltas_by_episode[int(episode_id)] = deltas

        logger.info(f"[Eval] ✅ Loaded oracle deltas for {len(deltas_by_episode)} episodes")
        self._oracle_deltas_by_episode = deltas_by_episode
        return deltas_by_episode

    def _build_pose_deltas_tensor(self, deltas_list, num_frames, device):
        """
        Convert raw deltas to normalized tensor shape [1, num_frames, 4].
        Format: (dx, dy, dyaw) -> (dx/0.25, dy/0.25, sin(dyaw), cos(dyaw))
        """
        if not deltas_list:
            return None

        # Sample num_frames-1 deltas to match video frames
        if len(deltas_list) >= num_frames:
            indices = np.linspace(0, len(deltas_list) - 1, num_frames - 1, endpoint=False, dtype=int)
            sampled_deltas = [deltas_list[idx] for idx in indices]
        else:
            sampled_deltas = deltas_list + [[0, 0, 0]] * (num_frames - 1 - len(deltas_list))

        processed = []
        for delta in sampled_deltas:
            dx, dy, dyaw = delta[0], delta[1], delta[2]
            dx_norm = dx / 0.25
            dy_norm = dy / 0.25
            processed.append([dx_norm, dy_norm, np.sin(dyaw), np.cos(dyaw)])

        # Pad to num_frames
        if len(processed) < num_frames:
            processed.extend([[0, 0, 0, 1]] * (num_frames - len(processed)))

        tensor = torch.tensor(processed[:num_frames], dtype=torch.float32, device=device)
        return tensor.unsqueeze(0)  # [1, num_frames, 4]

    def _action_to_delta(self, action):
        """
        Convert discrete action to pose delta [dx, dy, dyaw].
        Actions: 0=stop, 1=forward 25cm, 2=turn left 15deg, 3=turn right 15deg
        Deltas are in agent's local frame.
        """
        if action == 1:  # forward
            return [0.25, 0.0, 0.0]  # dx=0.25m forward
        elif action == 2:  # turn left
            return [0.0, 0.0, np.radians(15)]  # positive yaw = left
        elif action == 3:  # turn right
            return [0.0, 0.0, -np.radians(15)]  # negative yaw = right
        else:  # stop or unknown
            return [0.0, 0.0, 0.0]

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

        # Optional GRU checkpoint path (for motion encoder init)
        default_gru_ckpt = "/home/rithvik/NaVILA_Env/brain_inspired/NaVILA/evaluation/checkpoints/motion_gru_infonce.pt"
        gru_ckpt_path = os.environ.get("GRU_CKPT_PATH")
        if not gru_ckpt_path and os.path.exists(default_gru_ckpt):
            gru_ckpt_path = default_gru_ckpt

        final_model_pt = os.path.join(checkpoint_path, "final_model.pt")
        final_model_safetensors = os.path.join(checkpoint_path, "final_model.safetensors")

        if os.path.exists(final_model_pt) or os.path.exists(final_model_safetensors):
            # Load base model then apply full trained state_dict (includes motion encoder weights)
            base_model_path = "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain"
            tokenizer, model, image_processor, context_len = load_pretrained_model(base_model_path, "navila")

            # Ensure motion encoder exists before loading state_dict
            if hasattr(model, "motion_encoder") and model.motion_encoder is None and gru_ckpt_path:
                model.motion_encoder = MotionEncoderWithProjector(
                    gru_ckpt_path=gru_ckpt_path,
                    gru_hidden_size=256,
                    gru_num_layers=2,
                    gru_embedding_dim=128,
                    projector_intermediate_dim=256,
                    output_dim=model.config.hidden_size,
                    freeze_gru=True,
                    dropout=0.1,
                )

            # Load trained weights
            if os.path.exists(final_model_safetensors):
                from safetensors.torch import load_file

                state_dict = load_file(final_model_safetensors)
                model.load_state_dict(state_dict, strict=False)
            else:
                saved = torch.load(final_model_pt, map_location="cpu")
                state_dict = saved.get("model_state_dict", saved)
                model.load_state_dict(state_dict, strict=False)
        else:
            # Load standard checkpoint (llm/vision/mm_projector)
            tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint_path, model_name)

            # If GRU path provided, attach motion encoder for inference
            if hasattr(model, "motion_encoder") and model.motion_encoder is None and gru_ckpt_path:
                model.motion_encoder = MotionEncoderWithProjector(
                    gru_ckpt_path=gru_ckpt_path,
                    gru_hidden_size=256,
                    gru_num_layers=2,
                    gru_embedding_dim=128,
                    projector_intermediate_dim=256,
                    output_dim=model.config.hidden_size,
                    freeze_gru=True,
                    dropout=0.1,
                )

        model = model.cuda()

        # Print full model architecture once loaded
        logger.info("=" * 80)
        logger.info("MODEL ARCHITECTURE:")
        logger.info("=" * 80)
        logger.info(f"\n{model}")
        logger.info("=" * 80)

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

        # NOTE: We no longer use oracle deltas - instead we track actual agent movements
        # oracle_deltas = self._load_oracle_deltas(split)

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
        rgb_frames = [[] for _ in range(envs.num_envs)]  # this is for visualization, contains text and map
        
        # Track actual agent movements for GRU input (not oracle)
        # Each entry is [dx, dy, dyaw] in agent's local frame
        agent_deltas_history = [[] for _ in range(envs.num_envs)]

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

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:

            current_episodes = envs.current_episodes()

            if len(queue_actions) > 0:
                queued_action = queue_actions[0]
                print(f"using queue...{queued_action}")
                outputs = envs.step([queued_action])
                # Record the actual movement from queued action
                agent_deltas_history[0].append(self._action_to_delta(queued_action))
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")

                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    num_video_frames = model.config.num_video_frames

                    # Build pose_deltas tensor from agent's ACTUAL movement history (not oracle)
                    pose_deltas_tensor = None
                    try:
                        deltas_list = agent_deltas_history[0]
                        if deltas_list:
                            pose_deltas_tensor = self._build_pose_deltas_tensor(
                                deltas_list,
                                num_video_frames,
                                device=model.device,
                            )
                            logger.info(f"[Eval] Built pose_deltas_tensor from {len(deltas_list)} actual agent moves: {pose_deltas_tensor.shape if pose_deltas_tensor is not None else 'None'}")
                        else:
                            logger.info(f"[Eval] No agent movement history yet (first step)")
                    except Exception as e:
                        logger.warning(f"[Eval] Failed to build pose deltas: {e}")

                    past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=num_video_frames)

                    instruction = current_episodes[0].instruction.instruction_text

                    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)

                    frame_length = len(past_and_current_rgbs)
                    print(f"input frame length {frame_length}")

                    question = (
                        f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                        f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                        f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                        f"degree, moving forward a certain distance, or stop if the task is completed."
                    )

                    conv_mode = "llama_3"
                    conv = conv_templates[conv_mode].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0)
                        .cuda()
                    )

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    # Add newline and period as stop tokens to prevent rambling
                    keywords = [stop_str, "\n", ".\n"]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor.half().cuda(),
                            pose_deltas=pose_deltas_tensor,
                            do_sample=False,  # Greedy decoding
                            max_new_tokens=24,  # Shorter to avoid rambling
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()

                    if outputs.endswith(stop_str):
                        outputs = outputs[: -len(stop_str)]
                    outputs = outputs.strip()
                    
                    # Log full model output for debugging
                    print("=" * 50)
                    print(f"[MODEL OUTPUT]: '{outputs}'")
                    print("=" * 50)

                    # Define the regex patterns for each action
                    # Order matters: check specific actions before stop to avoid false positives
                    def map_string_to_action(s):
                        s_lower = s.lower()
                        # Check for movement actions first (more specific)
                        if re.search(r"move forward|forward \d+", s_lower):
                            return 1
                        if re.search(r"turn left|left \d+", s_lower):
                            return 2
                        if re.search(r"turn right|right \d+", s_lower):
                            return 3
                        # Check for stop/completed
                        if re.search(r"\bstop\b|\bcompleted\b|\bfinished\b|\bdone\b", s_lower):
                            return 0
                        return None

                    try:
                        actions = [map_string_to_action(outputs)]
                        # If no action matched, default to STOP (safer than random walking)
                        if actions[0] is None:
                            print(f"[WARNING] No action pattern matched in '{outputs}', defaulting to STOP")
                            actions = [0]
                    except Exception as e:
                        print(f"[ERROR] Action parsing failed: {e}, defaulting to STOP")
                        actions = [0]
                    print(f"[PARSED ACTION]: {actions} (0=STOP, 1=FWD, 2=LEFT, 3=RIGHT)")

                if actions[0] == 1:
                    try:
                        match = re.search(r"move forward (\d+) cm", outputs)
                        distance = int(match.group(1))
                    except:
                        distance = 25
                    if (distance % 25) != 0:
                        distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                    outputs = envs.step([1])
                    # Record actual movement
                    agent_deltas_history[0].append(self._action_to_delta(1))

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
                    # Record actual movement
                    agent_deltas_history[0].append(self._action_to_delta(2))

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
                    # Record actual movement
                    agent_deltas_history[0].append(self._action_to_delta(3))

                    for _ in range(int(degree // 15) - 1):
                        queue_actions.append(3)

                else:  # 0, stop
                    outputs = envs.step(actions)
                    # Record stop (no movement)
                    agent_deltas_history[0].append(self._action_to_delta(0))

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                past_rgbs[i].append(Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB"))

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
                agent_deltas_history[i] = []  # Reset movement history for new episode

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

            (envs, batch, rgb_frames,) = self._pause_envs(
                envs_to_pause,
                envs,
                batch,
                rgb_frames,
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

        return (
            envs,
            batch,
            rgb_frames,
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
