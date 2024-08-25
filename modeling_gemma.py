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

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    batch_size, num_key_value_heads, seq_length, head_dim = hidden_states.size()
    if num_repeats == 1:
        return hidden_states
    hidden_states = (
        hidden_states[:, :, None, :, :]
        .expand(batch_size, num_key_value_heads, num_repeats, seq_length, head_dim)
        .contiguous()
        .view(batch_size, num_key_value_heads * num_repeats, seq_length, head_dim)
    )
    return hidden_states


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # add the head dimension
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = q * cos + (rotate_half(q) * sin)
    k_embed = k * cos + (rotate_half(k) * sin)
    return q_embed, k_embed


class KVCache:
    def __init__(self) -> None:
        self.key_cache = List[torch.Tensor] = []
        self.value_cache = List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # the shape of the kv_cache is [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self, key: torch.Tensor, value: torch.Tensor, layer_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_index:
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_index] = torch.cat(
                [self.key_cache[layer_index], key], dim=-2
            )
            self.value_cache[layer_index] = torch.cat(
                [self.value_cache[layer_index], value], dim=-2
            )
        return self.key_cache[layer_index], self.value_cache[layer_index]


class GemmaConfig(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        attention_bias=False,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.pad_token_id = pad_token_id


class PaliGemmaConfig(nn.Module):
    def __init__(
        self,
        vision_config: SiglipVisionConfig,
        text_config: GemmaConfig,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        vocab_size: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features):
        # [B, Num_Patchs, Embed_Size] -> [B, Num_Patchs, Projection_Dim]
        image_features = self.linear(image_features)
        return image_features


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.power(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        outputs = self._norm(x)
        outputs = outputs * (1.0 + self.weight.float())
        return outputs.type_as(x)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config, layer_index)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)

    def forward(self, x):
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_group = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert (
            self.hidden_size % self.num_heads == 0
        ), "Hidden size must be divisible by the number of heads"

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim, self.max_position_embeddings, self.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        batch_size, seq_length, hidden_size = hidden_states.size()
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_index)
        # repeat the k, v to the number of heads of the query
        k = repeat_kv(k, self.num_key_value_group)
        v = repeat_kv(v, self.num_key_value_group)

        att_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask is not None:
            att_weights = att_weights + attention_mask

        att_weights = nn.functional.softmax(att_weights, dim=-1, dtype=float32).to(
            q.dtype
        )
        att_weights = nn.functional.dropout(
            att_weights, p=self.attention_dropout, training=self.training
        )
        att_output = torch.matmul(att_weights, v)

        if att_output.size() != (batch_size, self.num_heads, seq_length, self.head_dim):
            raise ValueError(
                f"Attention output has the wrong shape. Expected {(batch_size, self.num_heads, seq_length, self.head_dim)}, but got {att_output.size()}"
            )
        att_output = (
            att_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.num_heads * self.head_dim)
        )

        att_output = self.o_proj(att_output)
        return att_output, att_weights


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # it is set to head_dim
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = self.base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=int54).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        self.inv_freq = self.inv_freq.to(x.device)
        # batch_size, head_dim // 2, 1
        inv_freq_expanded = self.inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, 1
        )
        # batch_size, 1, seq_len
        position_expand = position_ids[:, None, :].float
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # batch_size, head_dim // 2, 1 @ batch_size, 1, seq_len -> batch_size, head_dim // 2, seq_len
            freqs = (inv_freq_expanded.float() @ position_expand.float()).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_index)
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:

        hidden_states = input_embeds
        normalizer = torch.Tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {"logits": logits}
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            config.pad_token_id if config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        self.language_model.tie_weights()

    def _merge_input_embeds_and_image_features(
        self, inputs_embeds, image_features, input_ids, attention_mask, kv_cache
    ):
        _, _, embed_dim = image_features.shape()
        batch_size, seq_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        scaled_image_features = image_features * (embed_dim**-0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embeddings = torch.zeros(
            batch_size, seq_length, embed_dim, dtype=dtype, device=device
        )

        text_masks = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_masks = text_masks.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embeddings = torch.where(text_masks, inputs_embeds, final_embeddings)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embeddings = final_embeddings.masked_scatter(
            image_mask, scaled_image_features
        )
        # Mask out padding tokens
        final_embeddings = torch.where(
            pad_mask, torch.zeros_like(final_embeddings), final_embeddings
        )

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape(1)

        if kv_cache is not None or kv_cache.num_items() == 0:
            # In paligemma, prefill stage, do not mask any token, but in Laama, prefill need mask token
            casual_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in decoding, we do not need to mask out the future tokens, because we have kv_cache, only compute the attention for the last token
            casual_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension to the casual mask
        casual_mask = casual_mask.unsqueeze(1)
        if kv_cache is not None and kv_cache.num_items() > 0:
            # We need to add the causal mask to the kv_cache
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # mask token position id set to 1
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embeddings, casual_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(
            attention_mask == 1
        ), "Attention mask is not supported, only batch_size == 1 for inference"
        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # 2. Extract the vision features
        # [B, C, H, W] -> [B, Num_Patchs, Embed_Size]
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))
        # 3. Project the vision features
        # [B, Num_Patchs, Embed_Size] -> [B, Num_Patchs, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_features)
        # 4. merge the input embeddings and image features
        (
            inputs_embeds,
            attention_mask,
            position_ids,
        ) = self._merge_input_embeds_and_image_features(
            inputs_embeds, image_features, input_ids, attention_mask, kv_cache
        )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs
