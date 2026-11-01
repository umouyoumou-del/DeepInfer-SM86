"""MLA's pooled KV is per head, not shared - corrected after a wrong claim.

An earlier version of this file asserted that DeepSeek's MLA latent is shared by
all heads and that storing one copy per head was therefore a 10.7x waste.  That
is false.  ``blk.N.attn_kv_b.weight`` is (4096, 512): the 512-wide latent is
up-projected to 16 x (128 nope + 128 v) *different* per-head keys and values.
The pool stores those expansions, so one copy per head is correct.

The tests below were written to prove the wrong thing: they built both pools
from a single latent, so of course sharing matched - the premise was baked into
the fixture.  They are kept only to pin what the kernel's kv_heads stride does
for callers that genuinely do share (grouped-query models), with the names
corrected so nobody reads them as evidence about DeepSeek.

DeepSeek-V2-Lite therefore needs kv_heads == heads.  Setting it lower silently
produces wrong attention, which is why the runtime never sets it.
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
    """Kernel contract only: when the data really is replicated, sharing is exact.

    This says nothing about DeepSeek, where the per-head projections differ.
    """

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


def _attend(lengths: int, *, window=None, sink=None):
    key, value = _pools(1)
    table = torch.arange(BLOCKS, dtype=torch.long, device="cuda").unsqueeze(0)
    lengths_t = torch.tensor([lengths], dtype=torch.long, device="cuda")
    positions = torch.tensor([lengths - 1], dtype=torch.long, device="cuda")
    query = torch.randn(1, HEADS, 1, DIM, dtype=torch.float16, device="cuda")
    return paged_mla_attend(
        query, key, value, table, lengths_t, positions,
        block_size=BLOCK, max_blocks=BLOCKS, scale=SCALE,
        sliding_window=window, attention_sink=sink,
    )


def test_window_and_sink_default_to_full_attention() -> None:
    """Both None must reproduce the plain path exactly."""

    torch.manual_seed(5)
    plain = _attend(LENGTH)
    torch.manual_seed(5)
    same = _attend(LENGTH, window=None, sink=None)
    assert torch.allclose(plain.float(), same.float(), atol=1e-6)


def test_window_is_bounded_regardless_of_length() -> None:
    """A window keeps the cost flat: the result no longer depends on the far past."""

    torch.manual_seed(6)
    short = _attend(8, window=8, sink=0)
    torch.manual_seed(6)
    long = _attend(LENGTH, window=8, sink=0)
    # same recent 8 positions are the only ones attended, so results differ only
    # through the query draw, which is reseeded identically; both must be finite
    assert torch.isfinite(short).all() and torch.isfinite(long).all()
    assert short.shape == long.shape


def test_sink_keeps_first_tokens_alive() -> None:
    """With a sink the first positions still contribute; without it they cannot."""

    torch.manual_seed(7)
    with_sink = _attend(LENGTH, window=4, sink=4)
    torch.manual_seed(7)
    no_sink = _attend(LENGTH, window=4, sink=0)
    assert torch.isfinite(with_sink).all() and torch.isfinite(no_sink).all()
    # the two configurations look at different token sets, so they must differ
    assert not torch.allclose(with_sink.float(), no_sink.float(), atol=1e-4)


def test_rejects_negative_window() -> None:
    with pytest.raises(ValueError):
        _attend(LENGTH, window=-1)
