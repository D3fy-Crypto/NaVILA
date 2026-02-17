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
from typing import Any, Dict, List, Optional, Sequence

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
from llava.mm_utils import opencv_extract_frames, process_image, tokenizer_image_token,tokenizer_mm_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.utils.logging import logger
from llava.utils.tokenizer import preprocess_conversation

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000

_DATAFLOW_DEBUG_DATASET_PRINTED = False
_DATAFLOW_DEBUG_COLLATOR_PRINTED = False
_POSE_DELTAS_CACHE: Dict[str, Dict[int, List[List[float]]]] = {}


def _summarize_positions(pos):
    if len(pos) <= 20:
        return str(pos)
    return f"{pos[:10]} ... {pos[-10:]} (count={len(pos)})"


import math
import numpy as np
from PIL import Image


def _normalize_delta(dx, dy, dyaw, trans_norm=0.25):
    # [dx/0.25, dy/0.25, sin(dyaw), cos(dyaw)]
    return [
        float(dx) / trans_norm,
        float(dy) / trans_norm,
        math.sin(float(dyaw)),
        math.cos(float(dyaw)),
    ]


def _load_pose_deltas_dir(pose_deltas_dir: Optional[str]) -> Dict[int, List[List[float]]]:
    if not pose_deltas_dir:
        return {}
    if pose_deltas_dir in _POSE_DELTAS_CACHE:
        return _POSE_DELTAS_CACHE[pose_deltas_dir]
    if not os.path.isdir(pose_deltas_dir):
        print(f"[PoseDeltas] directory not found: {pose_deltas_dir}")
        _POSE_DELTAS_CACHE[pose_deltas_dir] = {}
        return {}
    cache: Dict[int, List[List[float]]] = {}
    filenames = [
        "oracle_deltas_train.jsonl",
        # "oracle_deltas_val_seen.jsonl",
        # "oracle_deltas_val_unseen.jsonl",
    ]
    loaded = 0
    for fname in filenames:
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
    print(f"[PoseDeltas] loaded {loaded} episodes from {pose_deltas_dir}")
    _POSE_DELTAS_CACHE[pose_deltas_dir] = cache
    return cache

def _make_motion_windows(pose_deltas_step, num_frames, window_size=10, trans_norm=0.25):
    """
    pose_deltas_step: list length (num_frames-1), each (dx,dy,dyaw) for transition (t-1)->t
    returns motion tensor [num_frames, window_size, 4] where token t contains history up to frame t.
    """
    W = window_size
    out = torch.zeros(num_frames, W, 4, dtype=torch.float32)

    # frame 0 = no history (zeros)
    for t in range(1, num_frames):
        start = max(0, t - W)
        chunk = pose_deltas_step[start:t]  # transitions ending at t, length <= W
        pad = W - len(chunk)
        for j, (dx, dy, dyaw) in enumerate(chunk):
            out[t, pad + j] = torch.tensor(_normalize_delta(dx, dy, dyaw, trans_norm), dtype=torch.float32)

    return out

def _load_images_in_order(video_paths):
    # deterministic load (no resampling)
    pil_imgs = []
    for p in video_paths:
        pil_imgs.append(Image.open(p).convert("RGB"))
    return pil_imgs



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
        print("conversation_lib.default_conversation.sep_style:", conversation_lib.default_conversation.sep_style)
        print("conversation_lib.SeparatorStyle.PLAIN:", conversation_lib.SeparatorStyle.PLAIN)
        return preprocess_plain(sources, tokenizer)
    return default_collate(
        [
            preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
            for conversation in sources
        ]
    )

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
    def _load_video(video_paths, num_video_frames, data_args):
        from llava.mm_utils import vlnce_frame_sampling

        video_loading_succeed = True
        try:
            pil_imgs = vlnce_frame_sampling(video_paths, num_video_frames)

        except Exception as e:
            video_loading_succeed = False
            print(f"[Error] bad data paths {video_paths}: {e}")
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        sample = sources[0]
        if not (("frames" in sample) and ("video_id" in sample)):
            raise ValueError(f"Unknown data type: {sample}")

        num_video_frames = self.data_args.num_video_frames
        frames = sample["frames"]
        video_folder = self.image_folder
        video_id = sample["video_id"]

        # ----------------------------
        # 1) Decide frame indices ONCE (so images + deltas align)
        # ----------------------------
        total_frames = len(frames)
        if total_frames <= 0:
            raise ValueError(f"Empty frames list for sample: {sample}")

        indices_to_sample = np.linspace(
            0,
            total_frames - 1,
            num=num_video_frames,
            dtype=int
        ).tolist()

        sampled_frames = [frames[idx] for idx in indices_to_sample]
        video_paths = [os.path.join(video_folder, f) for f in sampled_frames]

        # ----------------------------
        # 2) Load images deterministically (no hidden re-sampling)
        # ----------------------------
        video_loading_succeed = True
        try:
            pil_imgs = [Image.open(p).convert("RGB") for p in video_paths]
        except Exception as e:
            video_loading_succeed = False
            print(f"[Error] bad data paths {video_paths}: {e}")
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        # Always keep tensor length == num_video_frames
        if len(pil_imgs) != num_video_frames:
            # pad/truncate defensively
            if len(pil_imgs) < num_video_frames:
                pil_imgs = pil_imgs + [Image.new("RGB", (448, 448), (0, 0, 0))] * (num_video_frames - len(pil_imgs))
            else:
                pil_imgs = pil_imgs[:num_video_frames]

        T = len(pil_imgs)  # should be num_video_frames
        image_tensor = torch.stack([process_image(img, self.data_args, None) for img in pil_imgs])

        # ----------------------------
        # 3) Build per-transition deltas aligned to sampled frames
        #    oracle all_deltas[k] = motion from step k -> k+1 in original episode timeline
        # ----------------------------
        episode_id = None
        all_deltas = None
        try:
            episode_id = int(video_id.split("-")[0])  # "914-23" -> 914
            all_deltas = self.delta_cache.get(episode_id)  # list of [dx,dy,dyaw]
        except Exception:
            all_deltas = None

        pose_deltas_step = []  # length (T-1): delta from sampled frame (t-1) -> sampled frame t
        if all_deltas is None:
            raise ValueError(
                f"Pose deltas missing for episode_id={episode_id}. "
                f"Check pose_deltas_dir={getattr(self.data_args, 'pose_deltas_dir', None)}"
            )
        if video_loading_succeed:
            for j in range(1, len(indices_to_sample)):
                prev_idx = indices_to_sample[j - 1]
                curr_idx = indices_to_sample[j]

                dx = dy = dyaw = 0.0
                # accumulate oracle deltas for steps prev_idx .. curr_idx-1
                # (cap to available oracle deltas length)
                max_k = min(curr_idx, len(all_deltas))
                for k in range(prev_idx, max_k):
                    d = all_deltas[k]
                    dx += float(d[0])
                    dy += float(d[1])
                    dyaw += float(d[2])

                pose_deltas_step.append((dx, dy, dyaw))

            # pad/truncate defensively
            if len(pose_deltas_step) != T - 1:
                pose_deltas_step = pose_deltas_step[: T - 1] + [(0.0, 0.0, 0.0)] * max(
                    0, (T - 1) - len(pose_deltas_step)
                )
        else:
            pose_deltas_step = [(0.0, 0.0, 0.0)] * max(0, T - 1)

        # ----------------------------
        # 4) Create motion windows: one motion token PER FRAME (C2 pairs with each <image>)
        #    motion_tensor: [T, motion_window_size, 4]
        # ----------------------------
        motion_tensor = _make_motion_windows(
            pose_deltas_step=pose_deltas_step,
            num_frames=T,
            window_size=getattr(self.data_args, "motion_window_size", 10),
            trans_norm=getattr(self.data_args, "motion_trans_norm", 0.25),
        )

        # ----------------------------
        # 5) Build C2 prompt: (<motion>\n<image>\n) repeated per frame
        # ----------------------------
        instruction = sample["q"].replace("\r\n", " ").replace("\n", " ")
        instruction = re.sub(r"(?<=\.\s)([a-z])", lambda x: x.group().upper(), instruction.capitalize())
        instruction = re.sub(r"\s+\.", ".", instruction)
        answer = sample["a"]

        hist_pairs = (DEFAULT_MOTION_TOKEN + "\n<image>\n") * max(0, T - 1)
        cur_pair = DEFAULT_MOTION_TOKEN + "\n<image>\n"

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

        # ----------------------------
        # 6) Tokenize (NOTE: tokenize_conversation must use tokenizer_mm_token)
        # ----------------------------
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # ----------------------------
        # 7) Attach modalities
        # ----------------------------
        data_dict["image"] = image_tensor
        data_dict["motion"] = motion_tensor                  # [T, W, 4]
        data_dict["pose_deltas"] = torch.tensor(             # [T-1, 3] (debug)
            pose_deltas_step, dtype=torch.float32
        )

        if not video_loading_succeed:
            data_dict["labels"][:] = IGNORE_INDEX

        # Hard stop if motion tokens are missing.
        input_ids = data_dict["input_ids"]
        if torch.is_tensor(input_ids):
            mot_pos = (input_ids == MOTION_TOKEN_INDEX).nonzero(as_tuple=False).squeeze(-1).tolist()
        else:
            mot_pos = [idx for idx, tok in enumerate(input_ids) if tok == MOTION_TOKEN_INDEX]
        if len(mot_pos) == 0:
            raise ValueError("No <motion> tokens found in input_ids; aborting training.")

        global _DATAFLOW_DEBUG_DATASET_PRINTED
        if not _DATAFLOW_DEBUG_DATASET_PRINTED:
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
    """Collate examples for supervised fine-tuning."""

    tokenizer: "transformers.PreTrainedTokenizer"
    data_args: "DataArguments"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = [], []
        images = []
        motions = []

        # ----------------------------
        # 1) Flatten packed samples if needed + collect modalities
        # ----------------------------
        for instance in instances:
            # input_ids / labels (support packed datasets)
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]

            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]

            # images: each instance["image"] is [T, 3, H, W] or None
            if instance.get("image") is not None:
                cur_image = instance["image"]
                assert cur_image.ndim == 4, f"Expected image tensor [T,3,H,W], got {cur_image.shape}"
                if not isinstance(instance["input_ids"], list):
                    images.append(cur_image)  # keep per-sample tensor
                else:
                    # packed case: split into per-frame chunks
                    images.extend(cur_image.chunk(cur_image.size(0), dim=0))
            else:
                images.append([])

            # motions: each instance["motion"] is [T, W, 4] or None
            if instance.get("motion") is not None:
                cur_motion = instance["motion"]
                assert cur_motion.ndim == 3, f"Expected motion tensor [T,W,4], got {cur_motion.shape}"
                if not isinstance(instance["input_ids"], list):
                    motions.append(cur_motion)
                else:
                    motions.extend(cur_motion.chunk(cur_motion.size(0), dim=0))
            else:
                motions.append([])

        # ----------------------------
        # 2) Sanity checks: #placeholders == #tensors before padding/truncation
        # ----------------------------
        for _images, _motions, _input_ids in zip(images, motions, input_ids):
            # image check (existing)
            assert len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item(), (
                "Mismatch image tensors vs <image> placeholders.\n"
                f"len(images)={len(_images)} but #<image>={(_input_ids == IMAGE_TOKEN_INDEX).sum().item()}\n"
                f"decoded={self.tokenizer.decode([x if x != IMAGE_TOKEN_INDEX else 200 for x in _input_ids])}"
            )

            # motion check (new)
            assert len(_motions) == (_input_ids == MOTION_TOKEN_INDEX).sum().item(), (
                "Mismatch motion tensors vs <motion> placeholders.\n"
                f"len(motions)={len(_motions)} but #<motion>={(_input_ids == MOTION_TOKEN_INDEX).sum().item()}\n"
                f"decoded={self.tokenizer.decode([x if x >= 0 else 200 for x in _input_ids])}"
            )

        # ----------------------------
        # 3) Pad sequences + truncate to model_max_length
        # ----------------------------
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

        # ----------------------------
        # 4) Trim images/motions if <image>/<motion> tokens got truncated
        # ----------------------------
        new_images = []
        new_motions = []

        for ix in range(len(input_ids)):
            # how many placeholders remain after truncation?
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            num_motions = (input_ids[ix] == MOTION_TOKEN_INDEX).sum().item()

            cur_images = images[ix][:num_images] if len(images[ix]) > 0 else []
            cur_motions = motions[ix][:num_motions] if len(motions[ix]) > 0 else []

            if len(cur_images) > 0:
                new_images.append(cur_images)
            if len(cur_motions) > 0:
                new_motions.append(cur_motions)

        # ----------------------------
        # 5) Build batch["images"] (existing behavior)
        # ----------------------------
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)  # [N_img_total, 3, H, W]
        else:
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])

        # ----------------------------
        # 6) Build batch["motions"] (new behavior)
        # ----------------------------
        if len(new_motions) > 0:
            batch["motions"] = torch.cat(new_motions, dim=0)  # [N_mot_total, W, 4]
        else:
            W = getattr(self.data_args, "motion_window_size", 10)
            batch["motions"] = torch.zeros(1, W, 4, dtype=torch.float32)

        global _DATAFLOW_DEBUG_COLLATOR_PRINTED
        if not _DATAFLOW_DEBUG_COLLATOR_PRINTED and batch["input_ids"].shape[0] == 1:
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
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
