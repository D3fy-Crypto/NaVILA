# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import warnings
from typing import Optional

import torch
from transformers import PretrainedConfig

from .motion_gru import MotionGRU

DEFAULT_GRU_CKPT = "/home/rithvik/IROS_proj/NaVILA_iros/llava/model/gru_model/motion_gru_infonce.pt"

try:
    from safetensors.torch import load_file as safe_load_file

    _HAS_SAFETENSORS = True
except Exception:
    safe_load_file = None
    _HAS_SAFETENSORS = False


def _resolve_path(model_path_or_name: str, config: PretrainedConfig) -> Optional[str]:
    if model_path_or_name is None:
        return None
    if isinstance(model_path_or_name, str) and model_path_or_name.lower() in (
        "default",
        "infonce",
        "motion_gru_infonce",
    ):
        candidate = os.path.join(os.path.dirname(__file__), "motion_gru_infonce.pt")
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(model_path_or_name):
        return model_path_or_name
    root_path = getattr(config, "_name_or_path", None) or getattr(config, "resume_path", None)
    if root_path:
        candidate = os.path.join(root_path, model_path_or_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_config_from_dir(dir_path: str) -> dict:
    config_path = os.path.join(dir_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        if not _HAS_SAFETENSORS:
            raise ImportError("safetensors is required to load .safetensors checkpoints.")
        return safe_load_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def build_motion_encoder(model_path_or_name: str, config: PretrainedConfig) -> Optional[MotionGRU]:
    if not getattr(config, "motion_encode", True):
        print("[MotionGRU] motion_encode=False -> disabling motion encoder.")
        return None

    if model_path_or_name is None:
        if os.path.exists(DEFAULT_GRU_CKPT):
            print(f"[MotionGRU] Using default checkpoint: {DEFAULT_GRU_CKPT}")
            model_path_or_name = DEFAULT_GRU_CKPT
        else:
            print(f"[MotionGRU] Default checkpoint not found at {DEFAULT_GRU_CKPT}; using random init.")

    resolved_path = _resolve_path(model_path_or_name, config)

    config_dir = None
    if isinstance(model_path_or_name, str):
        if os.path.isdir(model_path_or_name):
            config_dir = model_path_or_name
        else:
            root_path = getattr(config, "_name_or_path", None) or getattr(config, "resume_path", None)
            if root_path:
                candidate = os.path.join(root_path, model_path_or_name)
                if os.path.isdir(candidate):
                    config_dir = candidate
    if config_dir is None and isinstance(resolved_path, str) and os.path.isfile(resolved_path):
        config_dir = os.path.dirname(resolved_path)

    config_overrides = _load_config_from_dir(config_dir) if config_dir else {}

    input_size = config_overrides.get("input_size", getattr(config, "motion_input_size", 4))
    hidden_size = config_overrides.get("hidden_size", getattr(config, "motion_hidden_size", 256))
    num_layers = config_overrides.get("num_layers", getattr(config, "motion_num_layers", 2))
    embedding_dim = config_overrides.get("embedding_dim", getattr(config, "motion_embedding_dim", 128))
    dropout = config_overrides.get("dropout", getattr(config, "motion_dropout", 0.1))

    motion_encoder = MotionGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    if resolved_path is not None:
        if os.path.isdir(resolved_path):
            for fname in ("pytorch_model.bin", "model.safetensors", "motion_gru_infonce.pt"):
                candidate = os.path.join(resolved_path, fname)
                if os.path.exists(candidate):
                    resolved_path = candidate
                    break
        if os.path.isfile(resolved_path):
            try:
                state_dict = _load_state_dict(resolved_path)
                missing, unexpected = motion_encoder.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    msg = f"MotionGRU load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
                    warnings.warn(msg)
                    print(f"[MotionGRU] {msg}")
            except Exception as exc:
                print(f"[MotionGRU] Failed to load checkpoint {resolved_path}: {exc}. Using random init.")
        else:
            warnings.warn(f"MotionGRU checkpoint not found at {resolved_path}; using random init.")
            print(f"[MotionGRU] Checkpoint not found at {resolved_path}; using random init.")
    else:
        if isinstance(model_path_or_name, str):
            warnings.warn(f"MotionGRU checkpoint path '{model_path_or_name}' not found; using random init.")
            print(f"[MotionGRU] Checkpoint path '{model_path_or_name}' not found; using random init.")

    motion_encoder = motion_encoder.to(eval(config.model_dtype))
    if not getattr(config, "tune_motion_gru", False):
        motion_encoder.requires_grad_(False)
    return motion_encoder
