#!/usr/bin/env python3
# Copyright (c) 2024 Shaojie Li (shaojieli.nlp@gmail.com)
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


from typing import Optional, Tuple

import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layes: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layes = num_hidden_layes
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",  # valid means no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_size)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_size, self.embed_size)
        self.v_proj = nn.Linear(self.embed_size, self.embed_size)
        self.q_proj = nn.Linear(self.embed_size, self.embed_size)
        self.out_proj = nn.Linear(self.embed_size, self.embed_size)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [B, num_patch, embedding_size] --> [B, num_patch, num_heads, head_dim]
        q = q.reshape((q.shape[0], q.shape[1], self.num_heads, self.head_dim))
        k = k.reshape((k.shape[0], k.shape[1], self.num_heads, self.head_dim))
        v = v.reshape((v.shape[0], v.shape[1], self.num_heads, self.head_dim))

        # [B, num_patch, num_heads, head_dim] --> [B, num_heads, num_patch, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # [B, num_heads, num_patch, head_dim] @ [B, num_heads, head_dim, num_patch] --> [B, num_heads, num_patch, num_patch]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)

        # [B, num_heads, num_patch, num_patch] @ [B, num_heads, num_patch, head_dim] --> [B, num_heads, num_patch, head_dim]
        weighted_avg = attn @ v
        # [B, num_heads, num_patch, head_dim] --> [B, num_patch, num_heads, head_dim]
        weighted_avg = weighted_avg.permute(0, 2, 1, 3)
        # [B, num_patch, num_heads, head_dim] --> [B, num_patch, embedding_size]
        weighted_avg = weighted_avg.reshape(
            (weighted_avg.shape[0], weighted_avg.shape[1], self.embed_size)
        )
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        weighted_avg = self.out_proj(weighted_avg)

        return weighted_avg, attn


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_size = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, num_patch, embedding_size]
        residual = hidden_states
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        hidden_states = self.layer_norm1(hidden_states)
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        hidden_states, _ = self.self_attn(hidden_states)
        # [B, num_patch, embedding_size]
        hidden_states = hidden_states + residual
        # [B, num_patch, embedding_size]
        residual = hidden_states
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        hidden_states = self.layer_norm2(hidden_states)
        # [B, num_patch, embedding_size] --> [B, num_patch, embedding_size]
        hidden_states = self.mlp(hidden_states)
        # [B, num_patch, embedding_size]
        hidden_states = hidden_states + residual

        return hidden_states


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, num_patch, embedding_size] --> [B, num_patch, intermediate_size]
        hidden_states = self.fc1(hidden_states)
        # [B, num_patch, intermediate_size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [B, num_patch, intermediate_size] --> [B, num_patch, embedding_size]
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipVisionTransformer(nn.module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_size = config.hidden_size
        self.embeddings = SiglipVisionEmbedding(config)
        self.encoder = SiglipEncoder(config)
        # Why LayerNorm, Because Covariate Shift
        self.post_layernorm = nn.LayerNorm(embed_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor):
        # [B, C, H, W] -> [B, num_patch, embedding_size]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        # [B, C, H, W] -> [B, num_patch, embedding_size]
        return self.vision_model(pixel_values)
