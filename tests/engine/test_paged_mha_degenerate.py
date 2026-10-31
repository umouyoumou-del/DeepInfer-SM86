"""The paged GQA kernel must handle MHA (num_kv_heads == num_heads).

OLMoE is plain multi-head attention with 16 query heads and 16 KV heads, so the
port plan wants to reuse this kernel as a degenerate GQA case.  That is an
assumption worth testing before anything is built on it.

The fixture mirrors scripts/validate_sm86_kernels.py: tokens are written through
PagedKVCacheManager and the reference is taken from the view's own
materialize(), so nothing here hand-fills pool memory.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cupy")

from deepinfer.engine.kv_cache import PagedKVCacheManager  # noqa: E402
from deepinfer.engine.paged_attention import (  # noqa: E402
    native_paged_gqa_decode,
    native_paged_gqa_decode_available,
)

HEAD_DIM = 32
BLOCK = 4
PAST = 6


def _run(heads: int, kv_heads: int):
    torch.manual_seed(20261030)
    manager = PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=kv_heads,
        head_dim=HEAD_DIM,
        block_size=BLOCK,
        num_blocks=16,
        dtype=torch.float16,
        device="cuda",
    )
    block_ids = [manager.allocate_tokens(PAST + 1)]
    key = torch.randn(1, kv_heads, PAST, HEAD_DIM, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    manager.write_tokens(0, block_ids[0], key, value)

    views = manager.make_views(
        block_ids, [PAST], max_length=PAST + 1, new_tokens=1
    )
    new_key = torch.randn(1, kv_heads, 1, HEAD_DIM, device="cuda", dtype=torch.float16)
    new_value = torch.randn_like(new_key)
    views[0].update(
        new_key, new_value, torch.tensor([PAST], device="cuda").unsqueeze(1)
    )

    query = torch.randn(1, heads, 1, HEAD_DIM, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(HEAD_DIM)
    output = native_paged_gqa_decode(
        query, views[0], scale=scale, sliding_window=None, softcap=None
    )
    assert output is not None

    materialised_key, materialised_value, _valid = views[0].materialize()
    repeat = heads // kv_heads
    expanded_key = (
        materialised_key[:, :, None, :, :]
        .expand(1, kv_heads, repeat, materialised_key.shape[2], HEAD_DIM)
        .reshape(1, heads, materialised_key.shape[2], HEAD_DIM)
    )
    expanded_value = (
        materialised_value[:, :, None, :, :]
        .expand(1, kv_heads, repeat, materialised_value.shape[2], HEAD_DIM)
        .reshape(1, heads, materialised_value.shape[2], HEAD_DIM)
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.float(), expanded_key.float(), expanded_value.float()
    )
    return output.float(), reference


@pytest.mark.skipif(
    not torch.cuda.is_available() or not native_paged_gqa_decode_available(),
    reason="native paged GQA kernel unavailable",
)
def test_gqa_baseline_matches() -> None:
    """Sanity check the fixture itself against the known-working GQA path."""

    got, reference = _run(heads=8, kv_heads=2)
    assert torch.allclose(got, reference, atol=2e-2, rtol=2e-2), (
        got - reference
    ).abs().max()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not native_paged_gqa_decode_available(),
    reason="native paged GQA kernel unavailable",
)
def test_mha_matches_reference() -> None:
    got, reference = _run(heads=8, kv_heads=8)
    assert torch.allclose(got, reference, atol=2e-2, rtol=2e-2), (
        got - reference
    ).abs().max()
