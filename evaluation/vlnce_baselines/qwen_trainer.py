import copy
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
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    frames = copy.deepcopy(images)

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))

    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]

    return sampled_frames


@baseline_registry.register_trainer(name="qwen")
class QwenTrainer(BaseVLNCETrainer):
    def __init__(self, config=None, num_chunks=1, chunk_idx=0):
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        super().__init__(config)

    def _make_dirs(self) -> None:
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        raise NotImplementedError

    def _load_qwen_model(self, checkpoint_path: str):
        # Import lazily so this module is only required when qwen trainer is used.
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = None

        # Qwen3-VL is a vision-language model and is not compatible with
        # AutoModelForCausalLM. Prefer dedicated VL loaders, then fallback.
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except Exception:
            try:
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
            except Exception:
                from transformers import AutoModelForVision2Seq

                model = AutoModelForVision2Seq.from_pretrained(
                    checkpoint_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )

        model = model.to(self.device)
        model.eval()
        return processor, model

    def _qwen_generate_action_text(self, processor, model, images, instruction: str) -> str:
        prompt_text = (
            "You are a robot for indoor navigation. "
            "You are given historical observations and the current view. "
            f'Task: "{instruction}". '
            "Respond with only one action command in one of these forms: "
            "stop | move forward <cm> cm | turn left <degree> degree | turn right <degree> degree"
        )

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        qcfg = self.config.QWEN
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=qcfg.DO_SAMPLE,
                top_p=float(qcfg.TOP_P),
                top_k=int(qcfg.TOP_K),
                temperature=float(qcfg.TEMPERATURE),
                repetition_penalty=float(qcfg.REPETITION_PENALTY),
                max_new_tokens=int(qcfg.MAX_NEW_TOKENS),
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        decoded = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        return decoded[0].strip() if len(decoded) > 0 else "stop"

    @staticmethod
    def _map_action_and_steps(text: str):
        normalized = text.strip().lower()

        if "turn left" in normalized:
            match = re.search(r"turn left\s+(\d+)\s+degree", normalized)
            degree = int(match.group(1)) if match else 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            return 2, max(1, int(degree // 15))

        if "turn right" in normalized:
            match = re.search(r"turn right\s+(\d+)\s+degree", normalized)
            degree = int(match.group(1)) if match else 15
            if (degree % 15) != 0:
                degree = min([15, 30, 45], key=lambda x: abs(x - degree))
            return 3, max(1, int(degree // 15))

        if "move forward" in normalized:
            match = re.search(r"move forward\s+(\d+)\s+cm", normalized)
            distance = int(match.group(1)) if match else 25
            if (distance % 25) != 0:
                distance = min([25, 50, 75], key=lambda x: abs(x - distance))
            return 1, max(1, int(distance // 25))

        # Default to STOP for invalid/unmatched outputs.
        return 0, 1

    @staticmethod
    def _action_name(action_id: int) -> str:
        action_map = {
            0: "STOP",
            1: "MOVE_FORWARD",
            2: "TURN_LEFT",
            3: "TURN_RIGHT",
        }
        return action_map.get(action_id, f"UNKNOWN({action_id})")

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter) -> None:
        logger.info(f"checkpoint_path: {checkpoint_path}")

        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        processor, model = self._load_qwen_model(checkpoint_path)

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
        rgb_frames = [[] for _ in range(envs.num_envs)]

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
        step_id = 0

        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()
            ep_id = current_episodes[0].episode_id

            if len(queue_actions) > 0:
                queued_action = queue_actions.pop(0)
                step_id += 1
                logger.info(
                    "[Qwen][ep=%s][step=%d] queued_action=%s",
                    ep_id,
                    step_id,
                    self._action_name(queued_action),
                )
                outputs = envs.step([queued_action])
            else:
                with torch.no_grad():
                    curr_rgb = Image.fromarray(np.uint8(batch[0]["rgb"].cpu().numpy())).convert("RGB")
                    past_and_current_rgbs = past_rgbs[0] + [curr_rgb]
                    # Keep frame count aligned with the existing NaVILA path for fair benchmarking.
                    past_and_current_rgbs = sample_and_pad_images(past_and_current_rgbs, num_frames=8)

                    instruction = current_episodes[0].instruction.instruction_text
                    raw_text = self._qwen_generate_action_text(processor, model, past_and_current_rgbs, instruction)
                    action, steps = self._map_action_and_steps(raw_text)

                step_id += 1
                logger.info(
                    "[Qwen][ep=%s][step=%d] raw='%s' parsed=%s x%d",
                    ep_id,
                    step_id,
                    raw_text,
                    self._action_name(action),
                    steps,
                )

                outputs = envs.step([action])
                if action != 0 and steps > 1:
                    queue_actions.extend([action] * (steps - 1))

            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

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
                queue_actions.clear()
                step_id = 0

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

            (envs, batch, rgb_frames) = self._pause_envs(
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

        return (envs, batch, rgb_frames)

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
