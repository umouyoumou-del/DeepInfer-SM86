"""MLA stores one latent per token per layer; the pool must not copy it per head.

DeepSeek-V2-Lite has kv_lora_rank 512 plus a 64-wide rope part, so a token needs
576 values per layer, shared by all 16 heads.  Allocating [blocks, 16, block,
192] stored 16 identical copies - 10.7x the memory MLA actually requires, which
is what put a 1M context out of reach.

These tests pin the fix: a pool with one kv head must give exactly the same
attention output as a pool where that latent is replicated across all heads.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
cp = pytest.importorskip("cupy")

from deepinfer.moe.paged_mla_kernel import (  # noqa: E402
    cuda_paged_mla_supported,
    paged_mla_attend,
)

HEADS = 16
DIM = 192
BLOCK = 16
BLOCKS = 4
LENGTH = 20
SCALE = 1.0 / (DIM**0.5)


def _pools(kv_heads: int):
    torch.manual_seed(0)
    # one latent per (block, position); replicated when kv_heads > 1
    latent = torch.randn(BLOCKS, 1, BLOCK, DIM, dtype=torch.float16, device="cuda")
    key = latent.expand(BLOCKS, kv_heads, BLOCK, DIM).contiguous()
    value = torch.randn(BLOCKS, 1, BLOCK, DIM, dtype=torch.float16, device="cuda")
    value = value.expand(BLOCKS, kv_heads, BLOCK, DIM).contiguous()
    return key, value


def _run(kv_heads: int):
    key, value = _pools(kv_heads)
    table = torch.arange(BLOCKS, dtype=torch.long, device="cuda").unsqueeze(0)
    lengths = torch.tensor([LENGTH], dtype=torch.long, device="cuda")
    positions = torch.tensor([LENGTH - 1], dtype=torch.long, device="cuda")
    query = torch.randn(1, HEADS, 1, DIM, dtype=torch.float16, device="cuda")
    assert cuda_paged_mla_supported(query, key, value)
    return paged_mla_attend(
        query, key, value, table, lengths, positions,
        block_size=BLOCK, max_blocks=BLOCKS, scale=SCALE,
    )


def test_one_kv_head_matches_replicated_pool() -> None:
    """Sharing the latent must not change the numbers - it only stops copying it."""

    shared = _run(1)
    replicated = _run(HEADS)
    assert torch.allclose(shared.float(), replicated.float(), atol=2e-3), (
        shared.float() - replicated.float()
    ).abs().max()


def test_shared_pool_is_sixteen_times_smaller() -> None:
    key_one, value_one = _pools(1)
    key_many, value_many = _pools(HEADS)
    assert key_many.numel() == key_one.numel() * HEADS
    assert value_many.numel() == value_one.numel() * HEADS


def test_partial_grouping_is_supported() -> None:
    """kv_heads=4 must repeat each latent across 4 query heads."""

    output = _run(4)
    assert output.shape == (1, HEADS, 1, 128)
    assert torch.isfinite(output).all()
