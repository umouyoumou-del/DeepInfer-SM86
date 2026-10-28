"""Paged-KV adapter for the official DeepSeek-V2 MLA attention.

The existing paged scheduler was written for a GQA model whose attention
consumes a per-layer paged cache object directly.  DeepSeek-V2 uses MLA with
different key and value head dimensions, so this adapter:

1. writes the current MLA key/value states into the generic paged pool;
2. materializes the full dense K/V for the current batch step;
3. returns the dense tensors to the unmodified official attention code.

This is a compatibility bridge, not a FlashMLA kernel.  It makes the MoE
runtime consume the scheduler's block table and KV budget, while keeping the
official attention math/eager path.
"""

from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache


class DeepSeekPagedCache(Cache):
    """A ``Cache`` view over a list of ``PagedLayerKVCache`` objects."""

    is_compileable = False

    def __init__(
        self,
        views: list[Any],
        positions: torch.Tensor,
        *,
        past_length: int,
        max_length: int,
        key_dim: int,
        value_dim: int,
        head_dim: int,
        native_attention: bool = True,
    ) -> None:
        super().__init__()
        if not views:
            raise ValueError("at least one paged layer view is required")
        if len(views) != len({view.layer_index for view in views}):
            raise ValueError("paged views must have unique layer indices")
        self.views = views
        self.positions = positions
        self.past_length = int(past_length)
        self.max_length = int(max_length)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.head_dim = int(head_dim)
        self.native_attention = bool(native_attention)

    def get_usable_length(
        self, new_seq_length: int, layer_idx: int | None = 0
    ) -> int:
        return self.past_length

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return self.past_length

    def get_max_cache_shape(self) -> int:
        return self.max_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= len(self.views):
            raise IndexError(f"layer index {layer_idx} is out of range")
        view = self.views[layer_idx]
        key_padded = F.pad(
            key_states, (0, self.head_dim - key_states.shape[-1])
        )
        value_padded = F.pad(
            value_states, (0, self.head_dim - value_states.shape[-1])
        )
        view.update(key_padded, value_padded, self.positions)
        full_key, full_value, _ = view.materialize()
        return (
            full_key[..., : self.key_dim],
            full_value[..., : self.value_dim],
        )

    def write(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> None:
        view = self.views[layer_idx]
        key_padded = F.pad(
            key_states, (0, self.head_dim - key_states.shape[-1])
        )
        value_padded = F.pad(
            value_states, (0, self.head_dim - value_states.shape[-1])
        )
        view.update(key_padded, value_padded, positions)

    def attend(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Paged MLA attention without materializing full dense K/V."""

        view = self.views[layer_idx]
        batch, heads, query_length, q_dim = query_states.shape
        if q_dim != self.key_dim:
            raise ValueError(
                f"query head dim {q_dim} does not match key dim {self.key_dim}"
            )
        valid_lengths = view.seq_lens + int(view.new_tokens)
        if self.native_attention and os.environ.get(
            "DEEPINFER_DISABLE_CUDA_PAGED_MLA"
        ) != "1":
            from deepinfer.moe.paged_mla_kernel import (
                cuda_paged_mla_supported,
                paged_mla_attend,
                paged_mla_attend_cta,
                paged_mla_attend_scalar,
            )
            from deepinfer.moe.paged_mla_quant import (
                cuda_paged_mla_quant_supported,
                paged_mla_attend_quantized,
            )

            if cuda_paged_mla_quant_supported(
                query_states, view.key_pool, view.value_pool
            ):
                return paged_mla_attend_quantized(
                    query_states,
                    view.key_pool,
                    view.value_pool,
                    view.block_table,
                    valid_lengths,
                    positions,
                    block_size=int(view.block_size),
                    max_blocks=int(view.block_table.shape[1]),
                    scale=float(scale),
                    key_scale_pool=view.key_scale_pool,
                    value_scale_pool=view.value_scale_pool,
                    key_dim=self.key_dim,
                    value_dim=self.value_dim,
                )
            if cuda_paged_mla_supported(
                query_states, view.key_pool, view.value_pool
            ):
                variant = os.environ.get("DEEPINFER_MLA_KERNEL", "scalar")
                attend_kernel = {
                    "scalar": paged_mla_attend_scalar,
                    "half2": paged_mla_attend,
                    "cta": paged_mla_attend_cta,
                }.get(variant, paged_mla_attend_scalar)
                return attend_kernel(
                    query_states,
                    view.key_pool,
                    view.value_pool,
                    view.block_table,
                    valid_lengths,
                    positions,
                    block_size=int(view.block_size),
                    max_blocks=int(view.block_table.shape[1]),
                    scale=float(scale),
                )
        block_size = int(view.block_size)
        max_length = int(view.max_length)
        max_blocks = (max_length + block_size - 1) // block_size
        query_positions = positions.to(query_states.device, dtype=torch.long)
        output = torch.zeros(
            (batch, heads, query_length, self.value_dim),
            dtype=torch.float32,
            device=query_states.device,
        )
        running_max = torch.full(
            (batch, heads, query_length),
            float("-inf"),
            dtype=torch.float32,
            device=query_states.device,
        )
        running_sum = torch.zeros_like(running_max)
        q = query_states.float()
        for block_index in range(max_blocks):
            key_positions = torch.arange(
                block_index * block_size,
                (block_index + 1) * block_size,
                device=query_states.device,
                dtype=torch.long,
            )
            block_ids = view.block_table[:, block_index].to(torch.long)
            safe_ids = block_ids.clamp_min(0)
            keys = view.key_pool[safe_ids].float()[..., : self.key_dim]
            values = view.value_pool[safe_ids].float()[..., : self.value_dim]
            scores = torch.einsum("bhqd,bhkd->bhqk", q, keys) * float(scale)
            valid = key_positions.view(1, 1, 1, -1) < valid_lengths.view(
                batch, 1, 1, 1
            )
            causal = key_positions.view(1, 1, 1, -1) <= query_positions.view(
                batch, 1, query_length, 1
            )
            mask = valid & causal & (block_ids >= 0).view(batch, 1, 1, 1)
            scores = scores.masked_fill(~mask, float("-inf"))
            block_max = scores.max(dim=-1).values
            has_block = mask.any(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            valid_rows = has_block & torch.isfinite(new_max)
            alpha = torch.where(
                valid_rows,
                torch.exp(running_max - new_max),
                torch.ones_like(running_max),
            )
            probabilities = torch.where(
                valid_rows.unsqueeze(-1),
                torch.exp(scores - new_max.unsqueeze(-1)),
                torch.zeros_like(scores),
            )
            output = output * alpha.unsqueeze(-1) + torch.einsum(
                "bhqk,bhkd->bhqd", probabilities, values
            )
            running_sum = running_sum * alpha + probabilities.sum(dim=-1)
            running_max = torch.where(valid_rows, new_max, running_max)
        output = output / running_sum.clamp_min(1e-20).unsqueeze(-1)
        return output.to(query_states.dtype)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, layer_idx: int):
        view = self.views[layer_idx]
        key, value, _ = view.materialize()
        return key[..., : self.key_dim], value[..., : self.value_dim]

    def to_legacy_cache(self):
        return tuple(self[index] for index in range(len(self.views)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (
        _rotate_half(k) * sin
    )


def _native_paged_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Cache | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Drop-in forward for DeepseekV2Attention with native block reads."""

    if not isinstance(past_key_value, DeepSeekPagedCache) or not (
        past_key_value.native_attention
    ):
        return self._deepinfer_original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
    bsz, q_len, _ = hidden_states.shape
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(
        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(
            bsz,
            q_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(
        kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    total_length = past_key_value.get_usable_length(q_len) + q_len
    cos, sin = self.rotary_emb(value_states, seq_len=total_length)
    q_pe, k_pe = _apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    query_states = k_pe.new_empty(
        bsz, self.num_heads, q_len, self.q_head_dim
    )
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
    key_states = k_pe.new_empty(
        bsz, self.num_heads, q_len, self.q_head_dim
    )
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
    past_key_value.write(
        key_states, value_states, self.layer_idx, position_ids
    )
    attn_output = past_key_value.attend(
        query_states,
        self.layer_idx,
        position_ids,
        scale=self.softmax_scale,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        bsz, q_len, self.num_heads * self.v_head_dim
    )
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def install_native_paged_attention(model: torch.nn.Module) -> int:
    """Install the paged forward on every DeepSeek attention module."""

    installed = 0
    for layer in getattr(getattr(model, "model", model), "layers", []):
        attention = getattr(layer, "self_attn", None)
        if attention is None or getattr(
            attention, "_deepinfer_paged_installed", False
        ):
            continue
        attention._deepinfer_original_forward = attention.forward
        attention.forward = MethodType(_native_paged_attention_forward, attention)
        attention._deepinfer_paged_installed = True
        installed += 1
    return installed


__all__ = [
    "DeepSeekPagedCache",
    "install_native_paged_attention",
]
