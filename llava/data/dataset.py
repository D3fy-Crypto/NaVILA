# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



"""_summary_
    This file implements the dataset class for supervised fine-tuning.
    
    Code trace: 
    Called in llava/train/train.py - make_supervised_data_module -> build_dataset(from .builer.py) -> LazyVLNCEDataset(from .dataset.py) 
    Called in llava/train/train.py - make_supervised_data_module -> DataCollatorForSupervisedDataset
    
    The LazyVLNCEDataset dataset class is responsible for loading the data, processing the images, and tokenizing the conversations.
    DataCollatorForSupervisedDataset is responsible for collating the data into batches and applying necessary padding.
    
    def preprocess_plain and def preprocess getting used called in the LazyVLNCEDataset class, which is the main dataset class used for r2r dataset
    
    Remaining all are useless classes that are not used in the current training and evaluation pipeline, but we keep them for potential future use and reference.
"""

import base64
import copy
import io
import json
import math
import os
import os.path as osp
import pickle
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MOTION_TOKEN_INDEX,
    DEFAULT_MOTION_TOKEN
)
from llava.eval.mmmu_utils.data_utils import CAT_SHORT2LONG, construct_prompt, load_yaml, process_single_sample
from llava.mm_utils import opencv_extract_frames, process_image, tokenizer_image_token, tokenizer_mm_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.train.sequence_parallel import (
    extract_local_from_list,
    extract_local_input_ids,
    extract_local_position_ids,
    get_pg_manager,
)
from llava.utils.logging import logger
from llava.utils.tokenizer import preprocess_conversation

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000

_DATAFLOW_DEBUG_DATASET_PRINTED = False
_DATAFLOW_DEBUG_COLLATOR_PRINTED = False
_MOTION_ALIGNMENT_DEBUG_PRINTED = 0
_POSE_DELTAS_CACHE: Dict[str, Dict[int, List[List[float]]]] = {}


def _env_flag_enabled(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


_DATAFLOW_DEBUG_ENABLED = _env_flag_enabled(
    "LLAVA_DEBUG_DATAFLOW",
    default=_env_flag_enabled("LLAVA_DEBUG_MOTION", default=True),
)


def _summarize_positions(pos):
    if len(pos) <= 20:
        return str(pos)
    return f"{pos[:10]} ... {pos[-10:]} (count={len(pos)})"


def _normalize_delta(dx, dy, dyaw, trans_norm=0.25):
    # [dx/0.25, dy/0.25, sin(dyaw), cos(dyaw)]
    return [
        float(dx) / trans_norm,
        float(dy) / trans_norm,
        math.sin(float(dyaw)),
        math.cos(float(dyaw)),
    ]


def _make_motion_windows(pose_deltas_step, num_frames, window_size=10, trans_norm=0.25):
    """
    pose_deltas_step: list length (num_frames-1), each (dx,dy,dyaw) for transition (t-1)->t
    returns motion tensor [num_frames, window_size, 4] where token t contains history up to frame t.
    """
    W = window_size
    out = torch.zeros(num_frames, W, 4, dtype=torch.float32)
    for t in range(1, num_frames):
        start = max(0, t - W)
        chunk = pose_deltas_step[start:t]
        pad = W - len(chunk)
        for j, (dx, dy, dyaw) in enumerate(chunk):
            out[t, pad + j] = torch.tensor(_normalize_delta(dx, dy, dyaw, trans_norm), dtype=torch.float32)
    return out


def _vlnce_sample_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return [0] * num_frames
    padded = max(0, num_frames - total_frames)
    effective_len = total_frames + padded
    sampled = np.linspace(0, effective_len - 1, num=num_frames - 1, endpoint=False, dtype=int).tolist()
    sampled.append(effective_len - 1)
    return [max(0, idx - padded) for idx in sampled]


def _is_json_motion_sample(sample: Dict[str, Any]) -> bool:
    if not isinstance(sample, dict):
        return False
    if "img_8_indices" not in sample:
        return False
    return all(f"motion_{idx}" in sample for idx in range(1, 9))


def _parse_img_slots(sample: Dict[str, Any], num_slots: int) -> List[Optional[int]]:
    raw_slots = sample.get("img_8_indices", [])
    if not isinstance(raw_slots, list):
        raw_slots = []

    slots: List[Optional[int]] = []
    for idx in range(num_slots):
        value = raw_slots[idx] if idx < len(raw_slots) else "x"
        if isinstance(value, (int, np.integer)):
            slots.append(int(value))
            continue
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"", "x"}:
                slots.append(None)
                continue
            try:
                slots.append(int(stripped))
            except Exception:
                slots.append(None)
            continue
        slots.append(None)
    return slots


def _parse_motion_slot(value: Any) -> Tuple[Optional[List[int]], str]:
    if isinstance(value, list):
        actions: List[int] = []
        for item in value:
            if isinstance(item, (int, np.integer)):
                actions.append(int(item))
            elif isinstance(item, str):
                stripped = item.strip().lower()
                if stripped in {"", "x"}:
                    continue
                try:
                    actions.append(int(stripped))
                except Exception:
                    continue
        return actions, "list"

    if isinstance(value, (int, np.integer)):
        return [int(value)], "int"

    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "x"}:
            return None, "x"
        try:
            return [int(stripped)], "str_int"
        except Exception:
            return None, "str_other"

    if value is None:
        return None, "none"
    return None, type(value).__name__


def _actions_to_delta(
    actions: Sequence[int],
    heading: float,
    forward_step_m: float = 0.25,
    turn_deg: float = 15.0,
) -> Tuple[float, float, float, float]:
    dx = 0.0
    dy = 0.0
    dyaw = 0.0
    cur_heading = float(heading)
    turn_rad = math.radians(float(turn_deg))
    for action in actions:
        if int(action) == 1:
            dx += float(forward_step_m) * math.cos(cur_heading)
            dy += float(forward_step_m) * math.sin(cur_heading)
        elif int(action) == 2:
            cur_heading += turn_rad
            dyaw += turn_rad
        elif int(action) == 3:
            cur_heading -= turn_rad
            dyaw -= turn_rad
        elif int(action) == 0:
            continue
    return dx, dy, dyaw, cur_heading


def _build_pose_deltas_from_json_motion(
    sample: Dict[str, Any],
    num_frames: int,
    forward_step_m: float = 0.25,
    turn_deg: float = 15.0,
) -> Tuple[List[Tuple[float, float, float]], List[str], List[str]]:
    parsed_slots: List[Optional[List[int]]] = []
    motion_slot_types: List[str] = []
    for slot_idx in range(1, num_frames + 1):
        raw_value = sample.get(f"motion_{slot_idx}", "x")
        parsed, slot_type = _parse_motion_slot(raw_value)
        parsed_slots.append(parsed)
        motion_slot_types.append(slot_type)

    pose_deltas_step: List[Tuple[float, float, float]] = []
    segment_descriptions: List[str] = ["token 0: ZERO"]
    heading = 0.0

    for token_idx in range(1, num_frames):
        actions = parsed_slots[token_idx]  # motion_{token_idx + 1}
        slot_type = motion_slot_types[token_idx]
        field_name = f"motion_{token_idx + 1}"
        if actions is None:
            pose_deltas_step.append((0.0, 0.0, 0.0))
            segment_descriptions.append(f"token {token_idx}: {field_name} type={slot_type} -> ZERO")
            continue

        dx, dy, dyaw, heading = _actions_to_delta(
            actions,
            heading=heading,
            forward_step_m=forward_step_m,
            turn_deg=turn_deg,
        )
        pose_deltas_step.append((dx, dy, dyaw))
        segment_descriptions.append(
            f"token {token_idx}: {field_name} type={slot_type}, num_actions={len(actions)}"
        )

    return pose_deltas_step, segment_descriptions, motion_slot_types


def _parse_frame_id(frame_relpath: str) -> int:
    match = re.search(r"frame_(\d+)\.[^.]+$", frame_relpath)
    if match is None:
        raise ValueError(f"Could not parse frame id from path: {frame_relpath}")
    return int(match.group(1))


def _aggregate_pose_deltas_for_sampled_frames(
    all_deltas: Sequence[Sequence[float]],
    sampled_frame_ids: Sequence[int],
) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    pose_deltas_step: List[Tuple[float, float, float]] = []
    segment_descriptions: List[str] = ["token 0: ZERO"]
    total_steps = len(all_deltas)
    for token_idx in range(1, len(sampled_frame_ids)):
        prev_frame = int(sampled_frame_ids[token_idx - 1])
        curr_frame = int(sampled_frame_ids[token_idx])
        if curr_frame <= prev_frame:
            pose_deltas_step.append((0.0, 0.0, 0.0))
            segment_descriptions.append(
                f"token {token_idx}: prev={prev_frame}, curr={curr_frame} -> ZERO (non-increasing or repeated frame)"
            )
            continue

        start = max(0, prev_frame)
        end = max(0, curr_frame)
        clipped_start = min(start, total_steps)
        clipped_end = min(end, total_steps)

        dx = dy = dyaw = 0.0
        for k in range(clipped_start, clipped_end):
            d = all_deltas[k]
            dx += float(d[0])
            dy += float(d[1])
            dyaw += float(d[2])
        pose_deltas_step.append((dx, dy, dyaw))

        if clipped_end > clipped_start:
            frame_range = f"{prev_frame + 1}..{curr_frame}"
            delta_range = f"[{clipped_start}, {clipped_end})"
            segment_descriptions.append(
                f"token {token_idx}: prev={prev_frame}, curr={curr_frame}, frame_ids={frame_range}, delta_idx={delta_range}"
            )
        else:
            segment_descriptions.append(
                f"token {token_idx}: prev={prev_frame}, curr={curr_frame}, frame_ids=NONE, delta_idx=EMPTY -> ZERO"
            )

    return pose_deltas_step, segment_descriptions


def _pose_deltas_cache_key(pose_deltas_dir: str, filenames: Sequence[str]) -> str:
    return f"{pose_deltas_dir}::{'|'.join(filenames)}"


def _load_pose_deltas_dir(
    pose_deltas_dir: Optional[str],
    filenames: Optional[Sequence[str]] = None,
) -> Dict[int, List[List[float]]]:
    if not pose_deltas_dir:
        return {}
    files_to_load = tuple(
        filenames
        if filenames is not None
        else (
            "oracle_deltas_train.jsonl",
            "oracle_deltas_val_seen.jsonl",
            "oracle_deltas_val_unseen.jsonl",
        )
    )
    cache_key = _pose_deltas_cache_key(pose_deltas_dir, files_to_load)
    if cache_key in _POSE_DELTAS_CACHE:
        return _POSE_DELTAS_CACHE[cache_key]
    if not os.path.isdir(pose_deltas_dir):
        print(f"[PoseDeltas] directory not found: {pose_deltas_dir}")
        _POSE_DELTAS_CACHE[cache_key] = {}
        return {}
    cache: Dict[int, List[List[float]]] = {}
    loaded = 0
    for fname in files_to_load:
        fpath = os.path.join(pose_deltas_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                episode_id = obj.get("episode_id", None)
                deltas = obj.get("deltas", None)
                if episode_id is None or deltas is None:
                    continue
                cache[int(episode_id)] = deltas
                loaded += 1
    print(
        f"[PoseDeltas] loaded {loaded} entries from {pose_deltas_dir} "
        f"(files={','.join(files_to_load)})"
    )
    _POSE_DELTAS_CACHE[cache_key] = cache
    return cache

# def preprocess_multimodal is not getting called in the current class used for r2r dataset, which is LazyVLNCEDataset class.
def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            # ensure every DEFAULT_IMAGE_TOKEN is followed by a newline character.
            # If it has one already, we don't add another one.
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_IMAGE_TOKEN}\n\n", f"{DEFAULT_IMAGE_TOKEN}\n")

    return sources


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        # Preserve motion tokens if present; otherwise keep legacy behavior.
        if DEFAULT_MOTION_TOKEN not in source[0]["value"]:
            source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_mm_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_mm_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    return default_collate(
        [
            preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
            for conversation in sources
        ]
    )


class DummyDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        self.num_dummy_samples = 32768
        import random
        import string

        def generate_random_string(length):
            letters = string.ascii_letters
            result_str = "".join(random.choice(letters) for _ in range(length))
            return result_str

        self.list_data_dict = []
        for i in range(self.num_dummy_samples):
            question = generate_random_string(32)
            answer = question + generate_random_string(8)
            data_dict = {
                "id": i,
                "image": "empty",
                "conversations": [
                    {
                        "from": "human",
                        "value": question,
                    },
                    {
                        "from": "gpt",
                        "value": answer,
                    },
                ],
            }
            self.list_data_dict.append(data_dict)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif "images" in sources[0]:
            all_images = []
            for image_file in self.list_data_dict[i]["images"]:
                image = process_image(image_file, self.data_args, self.image_folder)
                all_images.append(image)
            image_tensor = torch.stack(all_images)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image.unsqueeze(0)
        elif "images" in self.list_data_dict[i]:
            data_dict["image"] = image_tensor
        else:
            data_dict["image"] = None
        return data_dict


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
                list_data_dict = [json.loads(q) for q in fp]

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        pose_deltas_dir = getattr(data_args, "pose_deltas_dir", None)
        self.delta_cache = _load_pose_deltas_dir(pose_deltas_dir)
        if pose_deltas_dir and len(self.delta_cache) == 0:
            raise ValueError(f"Pose deltas not found or empty: {pose_deltas_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @staticmethod
    def _load_video(video_path, num_video_frames, loader_fps, data_args, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        # frames_loaded = 0
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        elif "longest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["longest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
        # toTensor = transforms.ToTensor()

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            # pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * empty_num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack([process_image(img, self.data_args, self.image_folder) for img in image_file])
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif "images" in sources[0]:
            all_images = []
            for image_file in self.list_data_dict[i]["images"]:
                if isinstance(image_file, dict):
                    image_file = image_file["path"]
                image = process_image(image_file, self.data_args, self.image_folder)
                all_images.append(image)
            image_tensor = torch.stack(all_images)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            # num_video_frames = self.data_args.num_video_frames
            if "video_path" in sources[0]:
                video_file = sources[0]["video_path"]
            elif "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)
            num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
            loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

            if "fps" in sources[0]:
                fps = sources[0]["fps"]
            else:
                fps = None
            if "frame_count" in sources[0]:
                frame_count = sources[0]["frame_count"]
            else:
                frame_count = None

            images, frames_loaded = self._load_video(
                video_path, num_video_frames, loader_fps, self.data_args, fps=fps, frame_count=frame_count
            )

            image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

            if "captions" in sources[0]:
                question = "Elaborate on the visual and narrative elements of the video in detail."
                assert sources[0]["captions"][-1]["idx"] == "-1"
                answer = sources[0]["captions"][-1]["content"]
            elif "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                if isinstance(sources[0]["conversations"][1]["value"], str):
                    answer = sources[0]["conversations"][1]["value"].rstrip()
                else:
                    answer = str(sources[0]["conversations"][1]["value"]).rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]
                if isinstance(answer, list):
                    # for ScanQA compatiability
                    answer = random.choice(answer)

            if frames_loaded == 0:
                answer = "Empty video."
            num_frames_loaded_successfully = len(images)

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * num_frames_loaded_successfully + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
        elif "images" in self.list_data_dict[i]:
            data_dict["image"] = image_tensor
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if frames_loaded == 0:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        return data_dict


class DummyDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.num_dummy_samples = 1024 * world_size
        import random
        import string

        def generate_random_string(length):
            letters = string.ascii_letters
            result_str = "".join(random.choice(letters) for _ in range(length))
            return result_str

        self.list_data_dict = []
        for i in range(self.num_dummy_samples):
            question = generate_random_string(32)
            answer = question + generate_random_string(8)
            data_dict = {
                "id": i,
                "image": "empty",
                "conversations": [
                    {
                        "from": "human",
                        "value": question,
                    },
                    {
                        "from": "gpt",
                        "value": answer,
                    },
                ],
            }
            self.list_data_dict.append(data_dict)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @staticmethod
    def _process_image(image_file, data_args, image_folder, resize=False):
        processor = data_args.image_processor
        # if isinstance(image_file, str):
        #     if image_folder is not None:
        #         image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        #     else:
        #         image = Image.open(image_file).convert("RGB")
        # else:
        #     # image is stored in bytearray
        #     image = image_file
        image = Image.new("RGB", (256, 256), color="white")
        if resize:
            if hasattr(data_args.image_processor, "crop_size"):
                # CLIP vision tower
                crop_size = data_args.image_processor.crop_size
            else:
                # SIGLIP vision tower
                assert hasattr(data_args.image_processor, "size")
                crop_size = data_args.image_processor.size
            image = image.resize((crop_size["height"], crop_size["width"]))
        if data_args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image = self._process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image.unsqueeze(0)
        else:
            data_dict["image"] = None
        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print(f"* loaded totally {len(full_data_list)} samples")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for info in self.data_list:
            num_images = min(6, len(info["image_info"]))
            sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        # kentang-mit@: remove existing <image> tokens in the sentences
        for ix in range(len(sentences)):
            # if this is an html tag, we still preserve its semantic meaning
            sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)

                sim_ix = sample_image["matched_text_index"]
                # sim_ix = np.argmax(sim_vec)
                # sim_score = sim_vec[sim_ix]

                # filter to images >= 5KB
                # if len(rawbytes) // 1000 <= 5:
                #     continue
                # if sim_score < 0.24:
                #     continue
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

                images.append(image)
                sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        for ix in sentence_ixs:
            sentences[ix] = f"<image>\n{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            images = torch.stack([process_image(image, self.data_args, self.image_folder) for image in images])

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        # im_patch_token = self.tokenizer.convert_tokens_to_ids(
        #     [DEFAULT_IMAGE_PATCH_TOKEN]
        # )[0]
        # print(text, len(images))
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )

        # now check the case where the last token is image patch token
        if input_ids[-1] == IMAGE_TOKEN_INDEX:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        images = images[:n_im_patch]
        assert len(images) == n_im_patch, print(text, input_ids)
        assert len(input_ids.shape) == 1, "Unexpected shape of 'input_ids' from MMC4."
        input_ids = (
            torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids])
            if self.tokenizer.bos_token_id is not None and input_ids[0] != self.tokenizer.bos_token_id
            else input_ids
        )
        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != IMAGE_TOKEN_INDEX:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != IMAGE_TOKEN_INDEX:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        return dict(input_ids=input_ids, labels=targets, image=images)


class LazyCoyoDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print(f"* loaded totally {len(full_data_list)} samples")

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print(f"split into {n_groups} groups")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = (
                "text" if "text" in sample else "caption"
            )  # kentang-mit@: remove existing <image> tokens in the sentences
            # kentang-mit@: remove existing <image> token.
            # if this is an html tag, we still preserve its semantic meaning
            sample[caption_key] = sample[caption_key].replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + "\n" + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack([process_image(image, self.data_args, self.image_folder) for image in image_list])

        # the same size for all images, so we concat
        # cur_token_len = (
        #     image_list[0].shape[-2] // self.multimodal_cfg["patch_size"]
        # ) * (image_list[0].shape[-1] // self.multimodal_cfg["patch_size"])
        # cur_token_len += self.multimodal_cfg["n_extra_patch"]

        # replace_token = DEFAULT_IMAGE_TOKEN
        # if self.multimodal_cfg["use_im_start_end"]:
        #     replace_token = (
        #         DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        #     )
        # text_list = [
        #     text.replace(DEFAULT_IMAGE_TOKEN, replace_token) for text in text_list
        # ]

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            # print([x.shape[0] for x in input_ids], [len(x.split()) for x in text_list], [len(re.findall(r"<image[^>]*>", x)) for x in text_list])

            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyCoyoDataset_LONGSEQ(Dataset):
    """Dataset for supervised fine-tuning, with sequence parallelism support.
    This class is implemented by Dacheng Li."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        raise ValueError(
            f"This class LazyCoyoDataset_LONGSEQ has deprecated. You can use naive dataset directly for supporting seq parallel"
        )

        import pickle

        from llava.train.llama_dpsp_attn_monkey_patch import get_sequence_parallel_rank, get_sequence_parallel_size

        self.sequence_parallel_size = get_sequence_parallel_size()
        self.sequence_parallel_rank = get_sequence_parallel_rank()

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        rank = training_args.process_index  # int(os.environ["RANK"])
        world_size = training_args.world_size // self.sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = (rank // self.sequence_parallel_size) * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print(f"* loaded totally {len(full_data_list)} samples")

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print(f"split into {n_groups} groups")

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = (
                "text" if "text" in sample else "caption"
            )  # kentang-mit@: remove existing <image> tokens in the sentences
            # kentang-mit@: remove existing <image> token.
            # if this is an html tag, we still preserve its semantic meaning
            sample[caption_key] = sample[caption_key].replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack(
            [LazySupervisedDataset._process_image(image, self.data_args, self.image_folder) for image in image_list]
        )

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            # print([x.shape[0] for x in input_ids], [len(x.split()) for x in text_list], [len(re.findall(r"<image[^>]*>", x)) for x in text_list])

            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        # input_ids shape: (batch_size, sequence_length)
        subsequence_length = len(input_ids[0]) // self.sequence_parallel_size
        input_ids = input_ids[
            :, self.sequence_parallel_rank * subsequence_length : (self.sequence_parallel_rank + 1) * subsequence_length
        ].contiguous()
        targets = targets[
            :, self.sequence_parallel_rank * subsequence_length : (self.sequence_parallel_rank + 1) * subsequence_length
        ].contiguous()

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyWDSDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        n_samples = []
        n_shards = len(os.listdir(data_path)) // 3
        for shard in range(n_shards):
            with open(os.path.join(data_path, f"{shard:05d}_stats.json")) as f:
                info = json.load(f)
                n_samples.append(info["successes"])

        # print(f"[DEBUG] {data_path} total samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size
        print("rank", rank, "world_size", world_size, "shared_size", shared_size)
        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = f"/tmp/ccs{tar}"
            tar_path = os.path.join(data_path, tar)

            if PROCESS_GROUP_MANAGER is not None:
                dist.barrier()
                if PROCESS_GROUP_MANAGER.sp_rank == 0:
                    os.makedirs(tmp_path, exist_ok=True)
                    os.system(f"tar -xkf {tar_path} -C {tmp_path}")
                dist.barrier()
            else:
                os.makedirs(tmp_path, exist_ok=True)
                os.system(f"tar -xkf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt)).read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # print("i", i, "idx_offset", self.idx_offset, "len", len(self.data_list))
        info = self.data_list[i - self.idx_offset]
        caption, image_path = info["caption"], info["image"]

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, self.image_folder)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


class LazyVFlanDataset(Dataset):
    """Dataset for supervised fine-tuning from flan mixture.
    This class is implemented by Ji Lin and Haotian Tang."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        import pickle

        self.list_data_dict = []

        logger.warning("Loading data...")
        pkl_list = os.listdir(data_path)

        self.sharded = False
        # The original unsharded implementation loads the entire vflan dataset
        # on each GPU. So 80x8=640G host memory per device.
        # If we use the sharded implementation, only 80G per device.
        for pkl in pkl_list:
            if ".count" in pkl:
                self.sharded = True
                break
        if not self.sharded:
            for pkl in pkl_list:
                if pkl.endswith(".pkl"):
                    with open(os.path.join(data_path, pkl), "rb") as f:
                        data = pickle.load(f)
                        self.list_data_dict.extend(data)
            self.n_samples = len(self.list_data_dict)
            logger.warning(f"Loaded {len(self.list_data_dict)} samples...")
        else:
            # kentang-mit@: memory efficient loading of vflan via sharding.
            n_samples = []
            # actually shards and stats info
            n_shards = len(os.listdir(data_path)) // 2
            count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
            n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]
            self.n_samples = sum(n_samples)
            print("total VFlan samples", sum(n_samples))  # 10,881,869

            PROCESS_GROUP_MANAGER = get_pg_manager()
            if PROCESS_GROUP_MANAGER is not None:
                import torch.distributed as dist

                sequence_parallel_size = training_args.seq_parallel_size
            else:
                sequence_parallel_size = 1
            print("sequence_parallel_size", sequence_parallel_size)
            rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
            world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
            shared_size = n_shards // world_size

            gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
            self.n_samples = min(gpu_samples) * world_size  # total size
            self.idx_offset = rank * min(gpu_samples)
            shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
            print(f" * loading data from shard {shard_start}-{shard_end}")

            shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
            shard_names = shard_names[shard_start:shard_end]

            full_data_list = []
            # now load data
            for shard_name in shard_names:
                # load shard
                with open(os.path.join(data_path, shard_name), "rb") as f:
                    data_list = pickle.load(f)

                full_data_list.extend(data_list)

            print(f"* loaded totally {len(full_data_list)} samples")

            self.list_data_dict = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if not self.sharded:
            data = self.list_data_dict[i]
        else:
            data = self.list_data_dict[i - self.idx_offset]
        question = data["question"].rstrip()
        answer = data["answer:" if "answer:" in data else "answer"].rstrip()
        images = data["image:" if "image:" in data else "image"]

        if isinstance(images, str):
            images = [images]
        assert len(images) <= 8, f"Too many images in one sample {len(images)}"
        if len(images) == 8:  # sample it to be 4
            if hasattr(self.data_args, "downsample_video") and self.data_args.downsample_video:
                images = images[::2]
        n_images = len(images)

        decode_images = []
        for image_str in images:
            if image_str.endswith(".jpg"):
                decode_images.append(image_str)  # a path
            else:  # jpeg bytes
                rawbytes = base64.b64decode(image_str)
                decode_images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

        images = [process_image(img, self.data_args, image_folder=self.image_folder) for img in decode_images]

        # kentang-mit@: num_shots is not part of data_args. not included now.
        # if self.multimodal_cfg["num_shots"] > 0:
        #     raise NotImplementedError  # do not support multi-shot for FLAN

        # let's make sure there is no <image> in the question...
        if "Image Descriptions" in question:  # NOTE: specicial handlement for generation_visual-dialog_train.pkl
            question_split = question.split("\nQuestion: ")[1:]
            qa_pairs = []
            for qa in question_split:
                qa_pairs.append(qa.split("\nAnswer: "))

            qa_pairs[0][0] = "<image>\n" + qa_pairs[0][0]
            assert len(qa_pairs[-1]) == 1
            qa_pairs[-1][0] = qa_pairs[-1][0].replace("\n", "")
            qa_pairs[-1].append(answer)
            conversation = []
            for q, a in qa_pairs:
                conversation.append({"from": "human", "value": q})
                conversation.append({"from": "gpt", "value": a})
        else:
            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = "<image>\n" * n_images + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

        # the same size for all images, so we concat
        if len(images) == 0:
            assert not "<image>" in question

        # sources = replace_image_patch_tokens([conversation], self.multimodal_cfg)
        sources = [conversation]

        # NOTE: here we use the simple version without the system prompt
        # if n_images == 8:
        #     conv_version = "vicuna_v1_1"
        # else:
        #     conv_version = "vicuna_v1_1_nosys"

        # kentang-mit@: the newest conversation template does not have system prompt.
        if hasattr(self.data_args, "vflan_no_system_prompt"):
            no_system_prompt = self.data_args.vflan_no_system_prompt
        else:
            no_system_prompt = False
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=len(images) > 0,
            no_system_prompt=no_system_prompt,
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if len(images) > 0:
            data_dict["image"] = torch.stack(images)
        else:
            # llava 1.5 way of handling text-only data
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # data_dict['image'] = data_dict['image'].unsqueeze(0)
            # vila way of handling text-only data
            data_dict["image"] = None

        return data_dict


class LazyCCSWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        t1 = time.time()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path))

        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # info = self.data_list[i - self.idx_offset]
        # caption, image_path = info["caption"], info["image"]
        info = self.dataset[i]
        if ".jpg" in info:
            caption, image_path = info[".txt"], info[".jpg"]
        elif ".png" in info:
            caption, image_path = info[".txt"], info[".png"]
        elif ".webp" in info:
            caption, image_path = info[".txt"], info[".webp"]
        elif ".bmp" in info:
            caption, image_path = info[".txt"], info[".bmp"]
        elif ".tiff" in info:
            caption, image_path = info[".txt"], info[".tiff"]
        else:
            print(info.keys())
            print(info)
            raise KeyError

        caption = caption.replace("<image>", "<IMAGE>")
        if isinstance(image_path, io.BytesIO):
            image_path = Image.open(image_path).convert("RGB")

        if not isinstance(image_path, PIL.Image.Image):
            print(image_path)
            print(info.keys())
            print(type(image_path))
            raise NotImplementedError

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, image_folder=None)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


from functools import lru_cache


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    with open(fpath) as fp:
        return json.load(fp)


class LazyEvaluateDataset(LazySupervisedDataset):
    def __init__(
        self,
        data_path: str,
        data_args: dict,
        tokenizer: PreTrainedTokenizer,
        config_path: str = "llava/eval/mmmu_utils/configs/llava1.5.yaml",
        split="validation",
        **kwargs,
    ):
        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        all_datasets = concatenate_datasets(sub_dataset_list)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = None
        self.config = self.get_config(config_path)
        self.list_data_dict = self.get_processed_prompt(all_datasets)

    def get_config(self, config_path: str) -> str:
        config = load_yaml(config_path)
        for key, value in config.items():
            if key != "eval_params" and type(value) == list:
                assert len(value) == 1, f"key {key} has more than one value"
                config[key] = value[0]
        return config

    def get_processed_prompt(self, dataset: list) -> list:
        processed_dataset = []
        for d in dataset:
            sample = process_single_sample(d)
            processed_dict = construct_prompt(sample, self.config)

            if "<image>" in processed_dict["gt_content"]:
                processed_dict["gt_content"] = processed_dict["gt_content"].replace("<image>", "image")
            sample["conversations"] = [
                {"from": "human", "value": processed_dict["final_input_prompt"]},
                {"from": "gpt", "value": processed_dict["gt_content"]},
            ]
            processed_dataset.append(sample)
        return processed_dataset


class LazyCoyoWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path), meta_path=data_args.meta_path)

        if data_args.start_idx >= 0 and data_args.end_idx >= 0:
            # Ligeng: support slicing for ablate different subsets.
            total = len(self.dataset)
            start_idx = int(total * data_args.start_idx)
            end_idx = int(total * data_args.end_idx)
            print(f"loading subset from {start_idx} to {end_idx}, total {total}")
            self.dataset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        # For caption choice,
        #   if None: use original caption
        #   if a folder path: use specified caption to override original one (choice1)
        #   if a folder path: use specified caption and concat with original one (choice2)
        self.caption_choice = None
        self.caption_choice_2 = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override coyo caption using ", self.caption_choice)

        if data_args.caption_choice_2 is not None:
            self.caption_choice_2 = data_args.caption_choice_2
            print("[recapv2] Override coyo caption using ", self.caption_choice_2)

        print("total samples", len(self.dataset))
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )

        self.n_samples_per_idx = n_samples_per_idx
        # self.n_samples = len(self.dataset) // n_samples_per_idx
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset) // self.n_samples_per_idx

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.dataset[i - self.idx_offset]

        begin_idx, end_idx = (
            i * self.n_samples_per_idx,
            (i + 1) * self.n_samples_per_idx,
        )
        end_idx = min(end_idx, len(self.dataset))

        text_list = []
        image_list = []

        for idx in range(begin_idx, end_idx):
            info = self.dataset[idx]
            if ".jpg" in info:
                caption, image_path = info[".txt"], info[".jpg"]
            elif ".png" in info:
                caption, image_path = info[".txt"], info[".png"]
            elif ".webp" in info:
                caption, image_path = info[".txt"], info[".webp"]
            elif ".bmp" in info:
                caption, image_path = info[".txt"], info[".bmp"]
            elif ".tiff" in info:
                caption, image_path = info[".txt"], info[".tiff"]
            else:
                print(info.keys())
                print(info)
                raise KeyError

            if self.caption_choice is not None:
                # load new captions
                shard = info["__shard__"]
                url = info[".json"]["url"]
                tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
                # tar_name = osp.dirname(shard)
                shard_json_path = osp.join(self.caption_choice, tar_name + ".json")
                try:
                    shard_json = lru_json_load(shard_json_path)
                    try:
                        caption = shard_json[url]["output"]
                    except KeyError:
                        print(f"{url} not in caption. fallback to original caption temporarially")
                except:
                    print(f"shard_json_path {shard_json_path} not found. fallback to original caption temporarially")
            caption = caption.replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + caption + self.tokenizer.eos_token)

            if isinstance(image_path, io.BytesIO):
                image_path = Image.open(image_path).convert("RGB")

            if not isinstance(image_path, PIL.Image.Image):
                print(image_path)
                print(info.keys())
                print(type(image_path))
                raise NotImplementedError

            image_list.append(image_path)

        image_list = torch.stack([process_image(image, self.data_args, image_folder=None) for image in image_list])

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]
        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            input_ids = [
                (
                    torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids_i])
                    if input_ids_i[0] != self.tokenizer.bos_token_id
                    else input_ids_i
                )
                for input_ids_i in input_ids
            ]

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyVideoWebDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # cache_path: str,
        # n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=f"{osp.abspath(data_path)}/wids-meta.json",
            # cache_dir=cache_path,
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_choice = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override LazyVideo caption using ", self.caption_choice)

        print("total samples", len(self.dataset))
        # InternVid: TODO
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )
        self.rank = rank
        # rank = int(os.environ["RANK"]) if "RANK" in os.environ else 2
        # world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 32

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.missing_uids = set()

    def __len__(self):
        return len(self.dataset)

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ADD_TEXT_PROMPT = False
        num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
        loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

        info = self.dataset[i]

        caption = ""
        # print(info)
        if ".mp4" in info:
            caption, video_path = info[".txt"], info[".mp4"]
        else:
            video_path = None
            caption = "Empty video."

        images, frames_loaded = LazySupervisedDataset._load_video(
            video_path, num_video_frames, loader_fps, self.data_args
        )

        if frames_loaded == 0:
            caption = "Empty video."

        if self.caption_choice is not None:
            shard = info["__shard__"]
            uuid = osp.join(info["__shard__"], info["__key__"])
            url = info["__key__"]
            tar_name = osp.basename(info["__shard__"])

            try:
                shard_json_path = osp.join(self.caption_choice, tar_name.replace(".tar", ".json"))
                shard_json = lru_json_load(shard_json_path)
                caption = shard_json[url]["summary"]["output"]
            except (KeyError, FileNotFoundError, json.decoder.JSONDecodeError):
                if uuid not in self.missing_uids:
                    print("override caption not found for ", uuid)
                    self.missing_uids.add(uuid)

            # print(f"[DEBUG {uuid}]", caption)

        frames_loaded_successfully = len(images)
        if caption is None:
            caption = ""
        prompt = "<image>\n" * frames_loaded_successfully + caption
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


class VILAPanda70m_LongSeq(Dataset):
    """Dataset for training on video dataset, with sequence parallelism support.
    This class is implemented by Qinghao Hu."""

    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__()

        raise ValueError(
            f"This class VILAPanda70m_LongSeq has deprecated. You can use naive dataset directly for supporting seq parallel"
        )

        from llava.data.simple_vila_webdataset import VILAWebDataset

        data_path = osp.expanduser(data_path)
        self.dataset = VILAWebDataset(data_path=data_path)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_video_frames = data_args.num_video_frames if hasattr(data_args, "num_video_frames") else 8

        PROCESS_GROUP_MANAGER = get_pg_manager()

        self.sp_degree = training_args.seq_parallel_size
        assert self.sp_degree > 1, "Please use this class only when sequence parallelism is enabled."
        assert self.sp_degree < self.num_video_frames, "Sequence parallelism degree should be smaller than the frames."
        # assert (
        #     self.num_video_frames % self.sp_degree == 0
        # ), f"num_video_frames ({self.num_video_frames}) % sp_degree ({self.sp_degree}) != 0. Currently, we only support sequence evenly split across images (`IMAGE_TOKEN_INDEX`)."
        self.sp_rank = PROCESS_GROUP_MANAGER.sp_rank

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # TODO: we shall make sure no key is missing in panda70m.
        try:
            video_path = data[".mp4"]
        except KeyError:
            video_path = None
            print("bad data", data)

        if ".json" in data:
            jinfo = data[".json"]
            caption = jinfo["caption"]
        else:
            caption = "This is a sample video from Youtube."

        imgs = opencv_extract_frames(video_path, self.num_video_frames)
        cap = caption
        # print(imgs.shape, cap, secs)
        # num_video_frames = self.num_video_frames
        if len(imgs) < self.num_video_frames:
            # pad the video to be consistent
            # print(imgs)
            imgs = [
                imgs[0],
            ] * self.num_video_frames
        prompt = "<image>\n" * self.num_video_frames + cap

        processor = self.data_args.image_processor
        image_tensor = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in imgs]

        local_image_tensor = extract_local_from_list(image_tensor, self.sp_rank, self.sp_degree)

        image_tensor = torch.stack(local_image_tensor)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )

        # Split for sequence parallelism
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        local_input_indices = extract_local_from_list(image_token_indices, self.sp_rank, self.sp_degree)
        local_input_ids = extract_local_input_ids(
            input_ids, local_input_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
        )

        targets = copy.deepcopy(local_input_ids)
        data_dict = dict(input_ids=local_input_ids, labels=targets, image=image_tensor)

        return data_dict


class LazyEnvDropDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
                list_data_dict = [json.loads(q) for q in fp]
        print(f"Loaded EnvDrop with {len(list_data_dict)} samples")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        pose_deltas_dir = getattr(data_args, "pose_deltas_dir", None)
        self.delta_cache = _load_pose_deltas_dir(pose_deltas_dir)
        if pose_deltas_dir and len(self.delta_cache) == 0:
            raise ValueError(f"Pose deltas not found or empty: {pose_deltas_dir}")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @staticmethod
    def _load_video(video_path, num_video_frames, data_args):
        import cv2
        from llava.mm_utils import get_frame_from_vcap_vlnce

        try:
            vidcap = cv2.VideoCapture(video_path)
            pil_imgs, frame_length = get_frame_from_vcap_vlnce(vidcap, num_video_frames)

        except Exception as e:
            print(f"[Error] bad data path {video_path}: {e}")
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames
            frame_length = 0

        return pil_imgs, frame_length

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack([process_image(img, self.data_args, self.image_folder) for img in image_file])
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif "images" in sources[0]:
            all_images = []
            for image_file in self.list_data_dict[i]["images"]:
                if isinstance(image_file, dict):
                    image_file = image_file["path"]
                image = process_image(image_file, self.data_args, self.image_folder)
                all_images.append(image)
            image_tensor = torch.stack(all_images)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            if "video_path" in sources[0]:
                video_file = sources[0]["video_path"]
            elif "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"

            video_path = os.path.join(self.image_folder, video_file)
            num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
            images, frames_loaded = self._load_video(video_path, num_video_frames, self.data_args)
            image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

            answer = sources[0]["instruction"]
            if frames_loaded == 0:
                answer = "Empty video."
            num_frames_loaded_successfully = len(images)

            image_token = "<image>\n"
            question = f"Assume you are a robot designed for navigation. You are provided with captured images sequences {image_token * num_frames_loaded_successfully}. Based on this image sequence, please describe the navigation trajectory of the robot."

            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
        elif "images" in self.list_data_dict[i]:
            data_dict["image"] = image_tensor
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if frames_loaded == 0:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            data_dict["image"] = None
        return data_dict


class LazyVLNCEDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
                list_data_dict = [json.loads(q) for q in fp]

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder
        self.pose_deltas_dir = getattr(data_args, "pose_deltas_dir", None)
        self.delta_cache: Optional[Dict[int, List[List[float]]]] = None
        self._delta_cache_loaded = False
        self.vlnce_motion_source = str(getattr(data_args, "vlnce_motion_source", "auto")).strip().lower()
        if self.vlnce_motion_source not in {"auto", "pose_deltas", "json_actions"}:
            raise ValueError(
                "Invalid `vlnce_motion_source` value: "
                f"{self.vlnce_motion_source}. Expected one of auto|pose_deltas|json_actions."
            )

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _ensure_delta_cache_loaded(self):
        if self._delta_cache_loaded:
            return
        self._delta_cache_loaded = True
        self.delta_cache = _load_pose_deltas_dir(
            self.pose_deltas_dir,
            filenames=("oracle_deltas_train.jsonl",),
        )
        if self.pose_deltas_dir and len(self.delta_cache) == 0:
            raise ValueError(f"Pose deltas not found or empty: {self.pose_deltas_dir}")

    @staticmethod
    def _load_video(video_paths, num_video_frames, data_args):
        video_loading_succeed = True
        try:
            pil_imgs = [Image.open(path).convert("RGB") for path in video_paths]

        except Exception as e:
            video_loading_succeed = False
            print(f"[Error] bad data paths {video_paths}: {e}")
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        if len(pil_imgs) != num_video_frames:
            if len(pil_imgs) < num_video_frames:
                pil_imgs = pil_imgs + [Image.new("RGB", (448, 448), (0, 0, 0))] * (num_video_frames - len(pil_imgs))
            else:
                pil_imgs = pil_imgs[:num_video_frames]

        return pil_imgs, video_loading_succeed

    @staticmethod
    def _load_video_slots(slot_frame_paths: Sequence[Optional[str]], num_video_frames: int):
        pil_imgs: List[Image.Image] = []
        valid_slot_mask: List[bool] = []
        frame_ids: List[Optional[int]] = []
        video_loading_succeed = True

        for frame_path in slot_frame_paths[:num_video_frames]:
            if frame_path is None:
                pil_imgs.append(Image.new("RGB", (448, 448), (0, 0, 0)))
                valid_slot_mask.append(False)
                frame_ids.append(None)
                continue

            try:
                image = Image.open(frame_path).convert("RGB")
                pil_imgs.append(image)
                valid_slot_mask.append(True)
                frame_ids.append(_parse_frame_id(os.path.basename(frame_path)))
            except Exception as e:
                video_loading_succeed = False
                print(f"[Error] bad data path {frame_path}: {e}")
                pil_imgs.append(Image.new("RGB", (448, 448), (0, 0, 0)))
                valid_slot_mask.append(False)
                frame_ids.append(None)

        if len(pil_imgs) < num_video_frames:
            pad_count = num_video_frames - len(pil_imgs)
            pil_imgs.extend([Image.new("RGB", (448, 448), (0, 0, 0))] * pad_count)
            valid_slot_mask.extend([False] * pad_count)
            frame_ids.extend([None] * pad_count)
        elif len(pil_imgs) > num_video_frames:
            pil_imgs = pil_imgs[:num_video_frames]
            valid_slot_mask = valid_slot_mask[:num_video_frames]
            frame_ids = frame_ids[:num_video_frames]

        return pil_imgs, video_loading_succeed, valid_slot_mask, frame_ids

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if ("frames" in sources[0]) and ("video_id" in sources[0]):
            num_video_frames = self.data_args.num_video_frames
            frames = sources[0]["frames"]
            video_id = sources[0]["video_id"]
            video_folder = self.image_folder
            has_json_motion = _is_json_motion_sample(sources[0])
            if self.vlnce_motion_source == "auto":
                use_json_motion = has_json_motion
            elif self.vlnce_motion_source == "json_actions":
                if not has_json_motion:
                    raise ValueError(
                        f"`vlnce_motion_source=json_actions` but sample missing JSON motion fields: video_id={video_id}"
                    )
                use_json_motion = True
            else:
                use_json_motion = False

            motion_source_used = "json_actions" if use_json_motion else "pose_deltas"
            raw_img_8_indices = sources[0].get("img_8_indices", None) if has_json_motion else None
            motion_slot_types: Optional[List[str]] = None
            all_deltas = None

            if use_json_motion:
                indices_to_sample = _parse_img_slots(sources[0], num_video_frames)
                sampled_frames: List[Optional[str]] = []
                slot_frame_paths: List[Optional[str]] = []
                for slot_idx in indices_to_sample:
                    if slot_idx is None or slot_idx < 0 or slot_idx >= len(frames):
                        sampled_frames.append(None)
                        slot_frame_paths.append(None)
                    else:
                        frame_relpath = frames[slot_idx]
                        sampled_frames.append(frame_relpath)
                        slot_frame_paths.append(os.path.join(video_folder, frame_relpath))
                images, video_loading_succeed, valid_slot_mask, sampled_frame_ids = self._load_video_slots(
                    slot_frame_paths,
                    num_video_frames=num_video_frames,
                )
            else:
                indices_to_sample = _vlnce_sample_indices(len(frames), num_video_frames)
                sampled_frames = [frames[idx] for idx in indices_to_sample]
                sampled_frame_ids = [_parse_frame_id(frame_path) for frame_path in sampled_frames]
                video_paths = [os.path.join(video_folder, frame) for frame in sampled_frames]
                images, video_loading_succeed = self._load_video(video_paths, num_video_frames, self.data_args)
                valid_slot_mask = [True] * len(images)

            num_frames_loaded_successfully = len(images)
            image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

            # TODO: Remove extra spaces before punctuation
            instruction = sources[0]["q"].replace("\r\n", " ").replace("\n", " ")
            instruction = re.sub(r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), instruction.capitalize())
            instruction = re.sub(r"\s+\.", ".", instruction)
            answer = sources[0]["a"]
            # Build per-transition deltas aligned to sampled frames
            T = num_frames_loaded_successfully
            segment_descriptions = ["token 0: ZERO"]
            if use_json_motion:
                pose_deltas_step, segment_descriptions, motion_slot_types = _build_pose_deltas_from_json_motion(
                    sample=sources[0],
                    num_frames=T,
                    forward_step_m=float(getattr(self.data_args, "motion_action_forward_m", 0.25)),
                    turn_deg=float(getattr(self.data_args, "motion_action_turn_deg", 15.0)),
                )
            else:
                episode_id = None
                try:
                    self._ensure_delta_cache_loaded()
                    episode_id = int(video_id.split("-")[0])  # "914-23" -> 914
                    all_deltas = (self.delta_cache or {}).get(episode_id)
                except Exception:
                    all_deltas = None

                if all_deltas is None:
                    raise ValueError(
                        f"Pose deltas missing for episode_id={episode_id}. "
                        f"Check pose_deltas_dir={getattr(self.data_args, 'pose_deltas_dir', None)}"
                    )

                if video_loading_succeed:
                    pose_deltas_step, segment_descriptions = _aggregate_pose_deltas_for_sampled_frames(
                        all_deltas=all_deltas,
                        sampled_frame_ids=[frame_id if frame_id is not None else 0 for frame_id in sampled_frame_ids],
                    )
                else:
                    pose_deltas_step = [(0.0, 0.0, 0.0)] * max(0, T - 1)
                    segment_descriptions = ["token 0: ZERO"] + [
                        f"token {t}: ZERO (video load failed)" for t in range(1, T)
                    ]

            expected_steps = max(0, T - 1)
            if len(pose_deltas_step) > expected_steps:
                pose_deltas_step = pose_deltas_step[:expected_steps]
                segment_descriptions = segment_descriptions[: expected_steps + 1]
            elif len(pose_deltas_step) < expected_steps:
                pad_count = expected_steps - len(pose_deltas_step)
                pose_deltas_step = pose_deltas_step + [(0.0, 0.0, 0.0)] * pad_count
                segment_descriptions.extend(
                    [f"token {len(segment_descriptions) + j}: PAD_ZERO" for j in range(pad_count)]
                )

            motion_tensor = _make_motion_windows(
                pose_deltas_step=pose_deltas_step,
                num_frames=T,
                window_size=getattr(self.data_args, "motion_window_size", 10),
                trans_norm=getattr(self.data_args, "motion_trans_norm", 0.25),
            )

            global _MOTION_ALIGNMENT_DEBUG_PRINTED
            if getattr(self.data_args, "motion_alignment_debug", False):
                max_prints = max(0, int(getattr(self.data_args, "motion_alignment_debug_max_prints", 1)))
                if _MOTION_ALIGNMENT_DEBUG_PRINTED < max_prints:
                    _MOTION_ALIGNMENT_DEBUG_PRINTED += 1
                    print("[MOTION_ALIGN] video_id:", video_id, flush=True)
                    print("[MOTION_ALIGN] num_video_frames:", num_video_frames, flush=True)
                    print("[MOTION_ALIGN] total_frames_in_sample:", len(frames), flush=True)
                    print("[MOTION_ALIGN] motion_source:", motion_source_used, flush=True)
                    print("[MOTION_ALIGN] sampled_indices:", indices_to_sample, flush=True)
                    print("[MOTION_ALIGN] sampled_frame_ids:", sampled_frame_ids, flush=True)
                    print(
                        "[MOTION_ALIGN] last_sampled_frame_id:",
                        (sampled_frame_ids[-1] if len(sampled_frame_ids) > 0 else None),
                        flush=True,
                    )
                    if all_deltas is not None:
                        print("[MOTION_ALIGN] delta_len:", len(all_deltas), flush=True)
                    if raw_img_8_indices is not None:
                        print("[MOTION_ALIGN] raw_img_8_indices:", raw_img_8_indices, flush=True)
                    print("[MOTION_ALIGN] valid_slot_mask:", valid_slot_mask, flush=True)
                    if use_json_motion:
                        raw_motion_slots = [sources[0].get(f"motion_{idx}", "x") for idx in range(1, num_video_frames + 1)]
                        print("[MOTION_ALIGN] raw_motion_slots:", raw_motion_slots, flush=True)
                    if motion_slot_types is not None:
                        print("[MOTION_ALIGN] motion_slot_types:", motion_slot_types, flush=True)
                    print("[MOTION_ALIGN] segment_ranges:", segment_descriptions, flush=True)
                    print("[MOTION_ALIGN] pose_deltas_step_len:", len(pose_deltas_step), flush=True)
                    print("[MOTION_ALIGN] pose_deltas_step_values:", pose_deltas_step, flush=True)
                    print("[MOTION_ALIGN] motion_tensor_shape:", tuple(motion_tensor.shape), flush=True)
                    preview_tokens = min(2, motion_tensor.shape[0])
                    if preview_tokens > 0:
                        print(
                            "[MOTION_ALIGN] motion_tensor_preview_first_tokens:",
                            motion_tensor[:preview_tokens].detach().cpu().tolist(),
                            flush=True,
                        )

            hist_pairs = (DEFAULT_MOTION_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n") * max(0, T - 1)
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

            if not video_loading_succeed:
                answer = "Empty video."

            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            raise ValueError(f"Unknown data type: {sources[0]}")

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            data_dict["motion"] = motion_tensor
            data_dict["pose_deltas"] = torch.tensor(pose_deltas_step, dtype=torch.float32)
            if not video_loading_succeed:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            data_dict["image"] = None
            data_dict["motion"] = None

        # Hard stop if motion tokens are missing.
        input_ids = data_dict["input_ids"]
        if torch.is_tensor(input_ids):
            mot_pos = (input_ids == MOTION_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
        else:
            mot_pos = [idx for idx, tok in enumerate(input_ids) if tok == MOTION_TOKEN_INDEX]
        if len(mot_pos) == 0:
            raise ValueError("No <motion> tokens found in input_ids; aborting training.")

        global _DATAFLOW_DEBUG_DATASET_PRINTED
        if _DATAFLOW_DEBUG_ENABLED and not _DATAFLOW_DEBUG_DATASET_PRINTED:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None or worker_info.id == 0:
                _DATAFLOW_DEBUG_DATASET_PRINTED = True
                input_ids = data_dict["input_ids"]
                labels = data_dict["labels"]
                images = data_dict.get("image", None)
                motions = data_dict.get("motion", None)

                if torch.is_tensor(input_ids):
                    input_len = input_ids.numel()
                    img_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
                    mot_pos = (input_ids == MOTION_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
                else:
                    input_len = len(input_ids)
                    img_pos = [idx for idx, tok in enumerate(input_ids) if tok == IMAGE_TOKEN_INDEX]
                    mot_pos = [idx for idx, tok in enumerate(input_ids) if tok == MOTION_TOKEN_INDEX]

                if torch.is_tensor(labels):
                    label_len = labels.numel()
                    loss_pos = (labels != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
                else:
                    label_len = len(labels)
                    loss_pos = [idx for idx, tok in enumerate(labels) if tok != IGNORE_INDEX]

                img_shape = tuple(images.shape) if torch.is_tensor(images) else None
                mot_shape = tuple(motions.shape) if torch.is_tensor(motions) else None

                print("[DATAFLOW][LazyVLNCEDataset] sample keys:", list(data_dict.keys()), flush=True)
                print(
                    "[DATAFLOW][LazyVLNCEDataset] input_ids shape/len:",
                    (tuple(input_ids.shape) if torch.is_tensor(input_ids) else (input_len,)),
                    "num_tokens:",
                    input_len,
                    flush=True,
                )
                print(
                    "[DATAFLOW][LazyVLNCEDataset] labels shape/len:",
                    (tuple(labels.shape) if torch.is_tensor(labels) else (label_len,)),
                    "num_tokens:",
                    label_len,
                    flush=True,
                )
                print("[DATAFLOW][LazyVLNCEDataset] image tensor shape:", img_shape, flush=True)
                print("[DATAFLOW][LazyVLNCEDataset] motion tensor shape:", mot_shape, flush=True)
                if torch.is_tensor(motions) and motions.ndim == 3 and motions.shape[0] > 0:
                    print(
                        "[DATAFLOW][LazyVLNCEDataset] first_motion_token_window (W x 4):",
                        motions[0].detach().cpu().tolist(),
                        flush=True,
                    )
                print(
                    "[DATAFLOW][LazyVLNCEDataset] <image> token positions:",
                    _summarize_positions(img_pos),
                    flush=True,
                )
                print(
                    "[DATAFLOW][LazyVLNCEDataset] <motion> token positions:",
                    _summarize_positions(mot_pos),
                    flush=True,
                )
                print(
                    "[DATAFLOW][LazyVLNCEDataset] loss(label != IGNORE_INDEX) positions:",
                    _summarize_positions(loss_pos),
                    flush=True,
                )
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        input_ids, labels, images, motions = [], [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance.get("image") is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])
            if instance.get("motion") is not None:
                cur_motion = instance["motion"]
                assert cur_motion.ndim == 3, f"Expected motion tensor [T,W,4], got {cur_motion.shape}"
                if not isinstance(instance["input_ids"], list):
                    motions.append(cur_motion)
                else:
                    motions.extend(cur_motion.chunk(cur_motion.size(0), dim=0))
            else:
                motions.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _motions, _input_ids in zip(images, motions, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"
            assert len(_motions) == (_input_ids == MOTION_TOKEN_INDEX).sum().item(), (
                "Mismatch motion tensors vs <motion> placeholders.\n"
                f"len(motions)={len(_motions)} but #<motion>={(_input_ids == MOTION_TOKEN_INDEX).sum().item()}\n"
                f"decoded={self.tokenizer.decode([x if x >= 0 else 200 for x in _input_ids])}"
            )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        new_images = []
        new_motions = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            num_motions = (input_ids[ix] == MOTION_TOKEN_INDEX).sum().item()
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            cur_motions = motions[ix][:num_motions] if len(motions[ix]) > 0 else []
            if len(cur_images) > 0:
                new_images.append(cur_images)
            if len(cur_motions) > 0:
                new_motions.append(cur_motions)
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])
        if len(new_motions) > 0:
            batch["motions"] = torch.cat(new_motions, dim=0)
        else:
            W = getattr(self.data_args, "motion_window_size", 10)
            batch["motions"] = torch.zeros(1, W, 4, dtype=torch.float32)

        global _DATAFLOW_DEBUG_COLLATOR_PRINTED
        if _DATAFLOW_DEBUG_ENABLED and not _DATAFLOW_DEBUG_COLLATOR_PRINTED and batch["input_ids"].shape[0] == 1:
            _DATAFLOW_DEBUG_COLLATOR_PRINTED = True
            ids0 = batch["input_ids"][0]
            labels0 = batch["labels"][0]
            img_pos = (ids0 == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
            mot_pos = (ids0 == MOTION_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
            loss_pos = (labels0 != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
            print("[DATAFLOW][DataCollator] batch keys:", list(batch.keys()), flush=True)
            print(
                "[DATAFLOW][DataCollator] input_ids shape:",
                tuple(batch["input_ids"].shape),
                "labels shape:",
                tuple(batch["labels"].shape),
                "attention_mask shape:",
                tuple(batch["attention_mask"].shape),
                flush=True,
            )
            print("[DATAFLOW][DataCollator] images shape:", tuple(batch["images"].shape), flush=True)
            print("[DATAFLOW][DataCollator] motions shape:", tuple(batch["motions"].shape), flush=True)
            if torch.is_tensor(batch["motions"]) and batch["motions"].ndim == 3 and batch["motions"].shape[0] > 0:
                print(
                    "[DATAFLOW][DataCollator] first_motion_window (W x 4):",
                    batch["motions"][0].detach().cpu().tolist(),
                    flush=True,
                )
            print(
                "[DATAFLOW][DataCollator] <image> token positions (sample 0):",
                _summarize_positions(img_pos),
                flush=True,
            )
            print(
                "[DATAFLOW][DataCollator] <motion> token positions (sample 0):",
                _summarize_positions(mot_pos),
                flush=True,
            )
            print(
                "[DATAFLOW][DataCollator] loss(label != IGNORE_INDEX) positions (sample 0):",
                _summarize_positions(loss_pos),
                flush=True,
            )

        return batch


@dataclass
class DataCollatorForSupervisedDatasetSeqParallel:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments
    training_args: TrainingArguments
    sp_degree: int
    sp_rank: int
    ring_degree: int
    ring_type: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images = [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if cur_image.shape[0] == 0:
                    warnings.warn("loaded one sample without images.")
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                warnings.warn("loaded one sample without images.")
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        max_num_images = max([len(_images) for _images in images])
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        # TODO: Remove the hard coding of NUM_TOKENS_PER_IMAGE
        NUM_TOKENS_PER_IMAGE = 196
        if hasattr(self.data_args.image_processor, "crop_size"):
            crop_size = self.data_args.image_processor.crop_size
        else:
            crop_size = self.data_args.image_processor.size

        # Init the padding sample
        seq_id = 0
        while seq_id < len(input_ids):
            # Skip the samples without images
            dummy_image = torch.ones((1, 3, crop_size["height"], crop_size["width"]), device=input_ids[seq_id].device)
            # dummy input_ids include one bos, one image token, and one eos
            dummy_input_ids = torch.zeros_like(input_ids[seq_id][:3])
            dummy_input_ids[0] = self.tokenizer.bos_token_id
            dummy_input_ids[1] = IMAGE_TOKEN_INDEX
            dummy_input_ids[2] = self.tokenizer.eos_token_id
            dummy_labels = copy.deepcopy(dummy_input_ids)
            dummy_labels[:2] = IGNORE_INDEX
            dummy_seqlen = NUM_TOKENS_PER_IMAGE + 2  # TODO: Check the hard coding of 2
            dummy_position_ids = torch.arange(start=0, end=dummy_seqlen, dtype=torch.int32)
            break

        # Sort with the real length of the sequence
        combined = sorted(
            zip(input_ids, labels, images),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_IMAGE - 1) + x[0].size(-1),
            reverse=True,  # Start Packing from the sequence with most images.
        )
        sorted_ids, sorted_labels, sorted_images = zip(*combined)
        sorted_ids, sorted_labels, sorted_images = list(sorted_ids), list(sorted_labels), list(sorted_images)
        max_seq_length = self.tokenizer.model_max_length  # len(sorted_ids[0])
        max_sample_len = 0

        batches = []
        label_batches = []
        position_ids = []
        batch_images = []
        seqlens_in_batch = []

        i = 0
        while i < len(sorted_ids):
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_images = []
            current_num_images = 0
            current_len = 0
            current_num_samples = 0

            # Pack a few samples into one sample
            while i < len(sorted_ids):
                num_images = (sorted_ids[i] == IMAGE_TOKEN_INDEX).sum().item()
                num_image_tokens_added = num_images * (NUM_TOKENS_PER_IMAGE - 1)
                num_incoming_tokens = sorted_ids[i].size(-1) + num_image_tokens_added

                # Handle RingAttn_Varlen which requires `seqlens_in_batch` should be divisible by `ring_degree`
                if self.ring_degree > 1:
                    RING_PAD_TOKEN_INDEX = 2
                    if self.ring_type == "ring_varlen":
                        if num_incoming_tokens % self.sp_degree != 0:
                            pad_len = self.sp_degree - num_incoming_tokens % self.sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    elif self.ring_type == "zigzag_ring_varlen":
                        self.zigzag_sp_degree = self.sp_degree * 2
                        if num_incoming_tokens % self.zigzag_sp_degree != 0:
                            pad_len = self.zigzag_sp_degree - num_incoming_tokens % self.zigzag_sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    else:
                        raise ValueError(f"Invalid ring_type: {self.ring_type}")

                if num_incoming_tokens > max_seq_length:
                    print(
                        f"Warning: Skipping one packed sample with {num_incoming_tokens} tokens,\
                        please consider increase max seq len {max_seq_length}."
                    )
                    i += 1
                    continue

                if (
                    (current_num_images == 0)
                    or (current_num_images < self.sp_degree)
                    or (
                        (current_num_images + num_images <= max_num_images)
                        and (current_len + num_incoming_tokens <= max_sample_len)
                    )
                ) and (current_len + num_incoming_tokens <= max_seq_length):
                    current_num_images += num_images
                    current_len += num_incoming_tokens
                    current_num_samples += 1
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    sorted_labels[i][0] = IGNORE_INDEX
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_images.extend(sorted_images[i])
                    i += 1
                    assert current_num_images == len(current_batch_images)
                else:
                    break

            # Padding the sample with the dummy image sample, if there are no enough images
            MAX_RETRY = self.sp_degree
            num_retry = 0
            while current_num_images < self.sp_degree and current_len < max_seq_length and num_retry <= MAX_RETRY:
                current_num_images += dummy_image.size(0)
                current_len += dummy_seqlen
                current_num_samples += 1
                current_position_ids = torch.cat((current_position_ids, dummy_position_ids), dim=0)
                current_batch = torch.cat((current_batch, dummy_input_ids), dim=0)
                current_label_batch = torch.cat((current_label_batch, dummy_labels), dim=0)
                seqlens_in_batch.append(dummy_seqlen)
                current_batch_images.extend(dummy_image)
                # We pad from left side to ensure correct grad flow
                # current_batch = torch.cat((dummy_input_ids, current_batch), dim=0)
                # current_label_batch = torch.cat((dummy_labels, current_label_batch), dim=0)
                # seqlens_in_batch.insert(0, dummy_seqlen)
                # current_batch_images = torch.cat((dummy_image, current_batch_images), dim=0)
                num_retry += 1

            # Drop the samples that do not have enough images
            if current_num_images < self.sp_degree:
                print(f"Warning: Skipping one packed sample with {current_num_images} images")
                seqlens_in_batch = seqlens_in_batch[:-current_num_samples]
                continue

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_images.append(current_batch_images)

            try:
                assert current_num_images == len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", current_num_images)
                print("current_batch", current_batch)
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(current_batch == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                print(f"Error len(current_batch_images) on {self.sp_rank}:", len(current_batch_images))
                raise AssertionError

        # Split for sequence parallelism
        for i in range(len(batches)):
            image_token_indices = torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist()
            image_ids = torch.arange(0, len(image_token_indices), dtype=torch.int32)
            batches[i] = extract_local_input_ids(
                batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            label_batches[i] = extract_local_input_ids(
                label_batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            batch_images[i] = torch.concat(
                extract_local_from_list(batch_images[i], self.sp_rank, self.sp_degree), dim=0
            )
            H, W = batch_images[i].size(-2), batch_images[i].size(-1)
            batch_images[i] = batch_images[i].reshape(-1, 3, W, H)
            num_images = len(batch_images[i])

            try:
                assert num_images == len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", num_images)
                print("batches[i]", batches[i])
                print(
                    f"Error len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(batches[i] == IMAGE_TOKEN_INDEX)[0].tolist()),
                )
                print(f"Error batch_images[i] on {self.sp_rank}:", batch_images[i].shape)
                raise AssertionError
            position_ids[i] = extract_local_position_ids(
                position_ids[i], image_token_indices, image_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_IMAGE - 1
            )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = [torch.tensor(x) for x in seqlens_in_batch]
        seqlens_in_batch = torch.stack(seqlens_in_batch, axis=0)
        seqlens_in_batch = seqlens_in_batch.flatten()
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)

        if batch_images:
            flat_batch_images = torch.concat(batch_images, dim=0)
        else:
            flat_batch_images = None
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            images=flat_batch_images,
            position_ids=position_ids,
        )

        return batch


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and
    modified by Jason Lu, Haotian Tang and Ligeng Zhu."""
    datasets_mixture.register_datasets_mixtures()

    from .builder import build_dataset

    train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
    # eval_dataset = build_dataset(data_args.eval_data_mixture, data_args, training_args, tokenizer)
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]
    # training_args.eval_sample_lens = [len(d) for d in eval_dataset.datasets]

    PROCESS_GROUP_MANAGER = get_pg_manager() #not used
    if PROCESS_GROUP_MANAGER is None: #not going to do sequence parallelism so no else
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    else:
        sp_degree = training_args.seq_parallel_size
        sp_rank = PROCESS_GROUP_MANAGER.sp_rank
        ring_degree = PROCESS_GROUP_MANAGER.ring_degree
        ring_type = PROCESS_GROUP_MANAGER.ring_type
        data_collator = DataCollatorForSupervisedDatasetSeqParallel(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            sp_degree=sp_degree,
            sp_rank=sp_rank,
            ring_degree=ring_degree,
            ring_type=ring_type,
        )

    return dict(
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
