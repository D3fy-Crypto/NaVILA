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
Projects motion embeddings to LLM hidden space.
Input:  [batch, embedding_dim]
Output: [batch, hidden_dim]
"""

import torch
import torch.nn as nn


class GridToVisionProjector(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        intermediate_dim: int = 256,
        output_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
        )

    def forward(self, motion_embeddings: torch.Tensor) -> torch.Tensor:
        #print(f"Input shape for motion proj: {motion_embeddings.shape}")
        output = self.layers(motion_embeddings)
        #print(f"Output shape for motion proj: {output.shape}")
        return output
