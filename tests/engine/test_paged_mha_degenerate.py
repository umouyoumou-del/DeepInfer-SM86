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
def test_mha_written_through_update_matches_reference() -> None:
    """Build the cache through the real write path, not by hand-filling pools."""

    torch.manual_seed(11)
    total = PAST + 1
    block_table = torch.arange(BLOCKS, dtype=torch.long, device="cuda").unsqueeze(0)
    key_pool = torch.zeros(
        BLOCKS, HEADS, BLOCK, HEAD_DIM, dtype=torch.float16, device="cuda"
    )
    value_pool = torch.zeros_like(key_pool)
    # everything the kernel can reach is written with known values
    keys = torch.randn(BATCH, HEADS, total, HEAD_DIM, dtype=torch.float16, device="cuda")
    values = torch.randn_like(keys)

    cache = PagedLayerKVCache(
        key_pool=key_pool,
        value_pool=value_pool,
        block_table=block_table,
        seq_lens=torch.tensor([PAST], dtype=torch.long, device="cuda"),
        block_size=BLOCK,
        max_length=total,
        layer_index=0,
        new_tokens=1,
    )
    cache.update(keys, values, torch.arange(total, device="cuda").unsqueeze(0))

    query = torch.randn(BATCH, HEADS, 1, HEAD_DIM, dtype=torch.float16, device="cuda")
    got = native_paged_gqa_decode(
        query, cache, scale=HEAD_DIM**-0.5, sliding_window=None, softcap=None
    )
    assert got is not None

    reference = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2).float(),
        keys.transpose(1, 2).float(),
        values.transpose(1, 2).float(),
    ).transpose(1, 2)
    assert torch.allclose(got.float(), reference, atol=2e-2, rtol=2e-2), (
        got.float() - reference
    ).abs().max()
