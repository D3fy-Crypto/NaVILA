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
Motion GRU for navigation embeddings and projection to vision space.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionGRU(nn.Module):
    """
    GRU encoder for motion sequences (pose deltas).
    
    Input shape: [batch, seq_len, 4] where 4 = (dx_norm, dy_norm, sin(dyaw), cos(dyaw))
    Output shape: [batch, embedding_dim] (last hidden state)
    
    The GRU is pretrained with InfoNCE loss for place consistency.
    """
    
    def __init__(
        self,
        input_size=4,
        hidden_size=256,
        num_layers=2,
        embedding_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # Input projection (match checkpoint naming: input_proj)
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Embedding projection (match checkpoint naming: embed_proj)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim),
        )
    
    def forward(self, motion_sequences):
        """
        Args:
            motion_sequences: [batch, seq_len, 4] motion delta sequences
        
        Returns:
            embeddings: [batch, embedding_dim] L2-normalized place embeddings
        """
        # Project input
        x = self.input_proj(motion_sequences)  # [batch, seq_len, hidden_size]
        
        # GRU forward
        gru_out, hidden = self.gru(x)  # hidden: [num_layers, batch, hidden_size]
        
        # Take last layer hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Project to embedding space
        embeddings = self.embed_proj(last_hidden)  # [batch, embedding_dim]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class GridToVisionProjector(nn.Module):
    """
    Projects motion embeddings to vision space for injection into LLM.
    
    Input shape: [batch, embedding_dim=128]
    Output shape: [batch, hidden_dim=4096]
    """
    
    def __init__(
        self,
        embedding_dim=128,
        intermediate_dim=512,
        output_dim=4096,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # Pre-norm + staged expansion improves stability for large output projections.
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.expand = nn.Linear(embedding_dim, intermediate_dim)
        self.expand_norm = nn.LayerNorm(intermediate_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(intermediate_dim, output_dim)

        # Residual shortcut directly to output space, gated for stable warmup.
        self.residual_proj = nn.Linear(embedding_dim, output_dim, bias=False)
        self.residual_gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, motion_embeddings):
        """
        Args:
            motion_embeddings: [batch, embedding_dim] motion embeddings
        
        Returns:
            motion_tokens: [batch, output_dim] tokens in LLM space
        """
        x = self.input_norm(motion_embeddings)
        h = self.expand(x)
        h = self.act(h)
        h = self.expand_norm(h)
        h = self.dropout(h)
        main_path = self.out_proj(h)

        residual = self.residual_proj(x)
        gate = torch.tanh(self.residual_gate)
        motion_tokens = main_path + gate * residual  # [batch, output_dim]
        return motion_tokens


class MotionEncoderWithProjector(nn.Module):
    """
    Complete motion encoding pipeline:
    raw deltas -> MotionGRU -> GridToVisionProjector
    """
    
    def __init__(
        self,
        gru_ckpt_path=None,
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_embedding_dim=128,
        projector_intermediate_dim=512,
        output_dim=4096,
        freeze_gru=True,
        dropout=0.1,
    ):
        super().__init__()
        
        # Initialize MotionGRU
        self.gru = MotionGRU(
            input_size=4,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            embedding_dim=gru_embedding_dim,
            dropout=dropout,
        )
        
        # Load pretrained GRU if provided
        if gru_ckpt_path is not None and os.path.exists(gru_ckpt_path):
            print(f"Loading pretrained MotionGRU from {gru_ckpt_path}")
            checkpoint = torch.load(gru_ckpt_path, map_location="cpu", weights_only=False)
            # Handle both direct state_dict and wrapped state_dict
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Load state dict with prefix handling
            self.gru.load_state_dict(state_dict)
        
        # Freeze GRU if requested
        if freeze_gru:
            for param in self.gru.parameters():
                param.requires_grad = False
        
        # Initialize GridToVisionProjector
        self.projector = GridToVisionProjector(
            embedding_dim=gru_embedding_dim,
            intermediate_dim=projector_intermediate_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
    
    def forward(self, motion_deltas):
        """
        Args:
            motion_deltas: [batch, seq_len, 4] raw or normalized motion sequences
        
        Returns:
            motion_tokens: [batch, output_dim] tokens ready for LLM injection
        """
        # Encode motion to embeddings
        embeddings = self.gru(motion_deltas)  # [batch, embedding_dim]
        
        # Project to vision space
        motion_tokens = self.projector(embeddings)  # [batch, output_dim]
        
        return motion_tokens
    
    def get_trainable_params(self):
        """Returns iterator over trainable parameters (only projector)."""
        return self.projector.parameters()
    
    def get_frozen_params(self):
        """Returns iterator over frozen parameters (GRU)."""
        return self.gru.parameters()
