from __future__ import annotations

import pytest
import torch

from deepinfer.engine.kv_cache import paged_kv_block_bytes
from deepinfer.engine.memory_budget import plan_kv_blocks


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))


def test_cpu_memory_budget_uses_requested_or_required_blocks() -> None:
    model = _Model()
    requested = plan_kv_blocks(
        model,
        device="cpu",
        block_bytes=1024,
        block_size=16,
        max_cache=64,
        max_num_seqs=2,
        prefix_cache_blocks=4,
        requested_blocks=12,
    )
    assert requested.num_blocks == 12
    assert requested.capacity_tokens == 192
    assert requested.model_parameter_bytes == 32

    automatic = plan_kv_blocks(
        model,
        device="cpu",
        block_bytes=1024,
        block_size=16,
        max_cache=64,
        max_num_seqs=2,
        prefix_cache_blocks=4,
    )
    # ceil(64/16) * 2 + 4 prefix blocks
    assert automatic.num_blocks == 12

    with pytest.raises(ValueError):
        plan_kv_blocks(
            model,
            device="cpu",
            block_bytes=1024,
            block_size=16,
            max_cache=64,
            max_num_seqs=2,
            prefix_cache_blocks=0,
            requested_blocks=0,
        )


def test_fp8_block_size_is_half_of_fp16_and_smaller_than_int8() -> None:
    common = {
        "num_layers": 2,
        "num_kv_heads": 2,
        "block_size": 4,
        "head_dim": 8,
        "compute_dtype": torch.float16,
    }
    fp16 = paged_kv_block_bytes(**common, kv_cache_dtype="fp16")
    fp8 = paged_kv_block_bytes(**common, kv_cache_dtype="fp8")
    int8 = paged_kv_block_bytes(**common, kv_cache_dtype="int8")
    assert fp8 * 2 == fp16
    assert fp8 < int8 < fp16
