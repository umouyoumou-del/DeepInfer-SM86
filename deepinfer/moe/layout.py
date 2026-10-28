"""Layer-layout helpers for DeepSeek-V2 dense/MoE hybrid checkpoints."""

from __future__ import annotations

from typing import Any


def routed_expert_count(config: Any) -> int:
    value = getattr(config, "n_routed_experts", None)
    if value is None:
        return 0
    return max(0, int(value))


def is_moe_layer(config: Any, layer_idx: int) -> bool:
    """Mirror ``DeepseekV2DecoderLayer``'s dense/MoE decision."""

    num_experts = routed_expert_count(config)
    if num_experts == 0:
        return False
    first_dense = int(getattr(config, "first_k_dense_replace", 0) or 0)
    frequency = max(1, int(getattr(config, "moe_layer_freq", 1) or 1))
    return layer_idx >= first_dense and layer_idx % frequency == 0


def split_layer_indices(config: Any) -> tuple[list[int], list[int]]:
    """Return ``(dense_layer_indices, moe_layer_indices)``.

    A config with ``n_routed_experts`` missing/zero is a fully dense model.
    ``first_k_dense_replace`` controls the leading dense prefix and
    ``moe_layer_freq`` supports interspersed dense layers.
    """

    num_layers = int(getattr(config, "num_hidden_layers", 0))
    dense: list[int] = []
    moe: list[int] = []
    for layer_idx in range(num_layers):
        if is_moe_layer(config, layer_idx):
            moe.append(layer_idx)
        else:
            dense.append(layer_idx)
    return dense, moe
