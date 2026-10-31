"""The paged GQA kernel must handle MHA (num_kv_heads == num_heads).

OLMoE is plain multi-head attention with 16 query heads and 16 KV heads, so the
port plan reuses this kernel as a degenerate GQA case.  That is an assumption
worth testing before anything is built on it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")

from deepinfer.engine.kv_cache import PagedLayerKVCache  # noqa: E402
from deepinfer.engine.paged_attention import (  # noqa: E402
    native_paged_gqa_decode,
    native_paged_gqa_decode_available,
)

HEADS = 16
HEAD_DIM = 128
BLOCK = 16
BLOCKS = 4
BATCH = 1
PAST = 10


def _build(seed: int = 0):
    torch.manual_seed(seed)
    key_pool = torch.randn(
        BLOCKS, HEADS, BLOCK, HEAD_DIM, dtype=torch.float16, device="cuda"
    )
    value_pool = torch.randn_like(key_pool)
    block_table = torch.arange(BLOCKS, dtype=torch.long, device="cuda").unsqueeze(0)
    seq_lens = torch.tensor([PAST], dtype=torch.long, device="cuda")
    cache = PagedLayerKVCache(
        key_pool=key_pool,
        value_pool=value_pool,
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=BLOCK,
        max_length=BLOCKS * BLOCK,
        layer_index=0,
    )
    return cache


def _reference(query, cache) -> torch.Tensor:
    """Plain MHA over the materialised KV, via SDPA."""
    keys = cache.key_pool[0]  # [heads, block, dim] for block 0 only
    values = cache.value_pool[0]
    k = keys[:, :PAST, :].unsqueeze(0).transpose(1, 2)  # [1, past, heads, dim]
    v = values[:, :PAST, :].unsqueeze(0).transpose(1, 2)
    q = query.transpose(1, 2)  # [1, 1, heads, dim]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float()
    )
    return out.transpose(1, 2).to(torch.float16)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not native_paged_gqa_decode_available(),
    reason="native paged GQA kernel unavailable",
)
def test_mha_shapes_are_accepted() -> None:
    cache = _build()
    query = torch.randn(BATCH, HEADS, 1, HEAD_DIM, dtype=torch.float16, device="cuda")
    out = native_paged_gqa_decode(
        query, cache, scale=HEAD_DIM**-0.5, sliding_window=None, softcap=None
    )
    assert out is not None, "kernel refused MHA shapes (16 heads, 16 kv heads)"
    assert out.shape == query.shape


@pytest.mark.xfail(
    reason=(
        "Harness fixture is probably wrong, not the kernel. Verified so far: "
        "key_pool layout is [blocks, kv_heads, block, head_dim]; the kernel "
        "computes kv_group = num_heads/num_kv_heads = 1 and kv_head = head, so "
        "its MHA indexing is correct in principle; and it attends "
        "seq_lens + new_tokens positions, with max_length (not the pool "
        "capacity) driving the thread count. Output still differs from SDPA by "
        "2.56, so a fixture derived from a known-good caller is needed before "
        "the port can claim the kernel is reusable unchanged."
    ),
    strict=True,
)
def test_mha_matches_reference() -> None:
    cache = _build(seed=3)
    query = torch.randn(BATCH, HEADS, 1, HEAD_DIM, dtype=torch.float16, device="cuda")
    got = native_paged_gqa_decode(
        query, cache, scale=HEAD_DIM**-0.5, sliding_window=None, softcap=None
    )
    assert got is not None
    expected = _reference(query, cache)
    assert torch.allclose(got.float(), expected.float(), atol=2e-2, rtol=2e-2)
