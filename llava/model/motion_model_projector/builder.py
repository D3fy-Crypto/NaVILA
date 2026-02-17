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

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .projector import GridToVisionProjector


def _resolve_path(model_path_or_name: str, config: PretrainedConfig) -> Optional[str]:
    if model_path_or_name is None:
        return None
    if os.path.exists(model_path_or_name):
        return model_path_or_name
    root_path = getattr(config, "_name_or_path", None) or getattr(config, "resume_path", None)
    if root_path:
        candidate = os.path.join(root_path, model_path_or_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_state_dict(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def _build_by_type(model_type: str, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    model_type = model_type.lower()
    if model_type in ("grid2vision", "mlp2x_gelu", "mlp"):
        return GridToVisionProjector(
            embedding_dim=input_dim,
            intermediate_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    if model_type == "linear":
        return nn.Linear(input_dim, output_dim)
    if model_type == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown motion projector type: {model_type}")


def build_motion_projector(model_type_or_path: str, config: PretrainedConfig) -> Optional[nn.Module]:
    if model_type_or_path is None:
        return None

    input_dim = getattr(config, "motion_embedding_dim", 128)
    hidden_dim = getattr(config, "motion_projector_intermediate_dim", 256)
    output_dim = getattr(config, "hidden_size", None)
    if output_dim is None:
        raise ValueError("config.hidden_size must be set before building motion projector.")
    dropout = getattr(config, "motion_projector_dropout", 0.1)

    resolved_path = _resolve_path(model_type_or_path, config)
    if resolved_path is not None and os.path.isdir(resolved_path):
        for fname in ("pytorch_model.bin", "model.safetensors"):
            candidate = os.path.join(resolved_path, fname)
            if os.path.exists(candidate):
                resolved_path = candidate
                break

    if resolved_path is not None and os.path.isfile(resolved_path):
        projector = GridToVisionProjector(
            embedding_dim=input_dim,
            intermediate_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        state_dict = _load_state_dict(resolved_path)
        missing, unexpected = projector.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Motion projector load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
            )
    else:
        projector = _build_by_type(model_type_or_path, input_dim, hidden_dim, output_dim, dropout)
        if resolved_path is not None and os.path.isdir(resolved_path):
            warnings.warn(f"Motion projector directory provided without weights: {resolved_path}")

    projector = projector.to(eval(config.model_dtype))
    if not getattr(config, "tune_motion_projector", False):
        projector.requires_grad_(False)
    return projector
