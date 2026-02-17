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

"""
Motion GRU encoder for navigation embeddings.
Input:  [batch, seq_len, 4] where 4 = (dx_norm, dy_norm, sin(dyaw), cos(dyaw))
Output: [batch, embedding_dim] (L2-normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, motion_sequences: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(motion_sequences)
        _, hidden = self.gru(x)
        last_hidden = hidden[-1]
        embeddings = self.embed_proj(last_hidden)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
