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

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, layer_idx: int):
        view = self.views[layer_idx]
        key, value, _ = view.materialize()
        return key[..., : self.key_dim], value[..., : self.value_dim]

    def to_legacy_cache(self):
        return tuple(self[index] for index in range(len(self.views)))


__all__ = ["DeepSeekPagedCache"]
