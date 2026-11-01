"""Block retrieval: bounds never under-rate a block, and keeping everything is exact."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    decode_attention,
)
from deepinfer.moe.latent_retrieval import (
    BlockBounds,
    RetrievalConfig,
    select_segments,
    sparse_decode_attention,
)


def _filled_cache(n: int, device: str, seed: int = 0) -> LatentKVCache:
    torch.manual_seed(seed)
    cache = LatentKVCache(num_layers=1, capacity=n + 8, kv_dtype="fp16", device=device)
    cache.write(0, 0, torch.randn(n, 512, device=device).half(), torch.randn(n, 64, device=device).half())
    return cache


def test_bounds_never_under_rate_a_block() -> None:
    n, block = 300, 32
    cache = _filled_cache(n, "cpu")
    bounds = BlockBounds(num_layers=1, capacity=n + 8, block=block, device="cpu")
    bounds.update(cache, 0, n)
    latent, pe = cache.tile(0, 0, n)
    rows = torch.cat((latent, pe), dim=-1).float()
    q = torch.randn(576)
    scores = rows @ q
    upper = q.clamp_min(0) @ bounds.hi[0].float().t() + q.clamp_max(0) @ bounds.lo[0].float().t()
    for b in range(-(-n // block)):
        assert float(upper[b]) >= float(scores[b * block : (b + 1) * block].max()) - 1e-3


def test_the_block_holding_the_match_is_selected() -> None:
    n, block = 4096, 64
    cache = _filled_cache(n, "cpu", seed=1)
    q_abs = torch.randn(4, 512)
    # plant rows that agree with head 2 in block 30, far from sink and window
    planted = (q_abs[2] / q_abs[2].norm() * 40).half()
    latent = cache.tile(0, 30 * block, 30 * block + 4)[0]
    cache.write(0, 30 * block + 2, planted.unsqueeze(0), torch.zeros(1, 64).half())
    del latent
    bounds = BlockBounds(num_layers=1, capacity=n + 8, block=block, device="cpu")
    bounds.update(cache, 0, n)
    plain = DualChunkPositions.plain()
    config = RetrievalConfig(block=block, top_blocks=4, window=256, sink=64)
    segments = select_segments(
        bounds, 0, q_abs.half(), {plain.INTRA: torch.zeros(4, 64).half()},
        [(0, n, plain.INTRA)], config,
    )
    covered = {slot for start, end, _ in segments for slot in range(start, end)}
    assert 30 * block + 2 in covered
    assert 0 in covered and n - 1 in covered  # sink and window are always kept
    assert len(covered) <= 64 + 256 + 4 * block


def test_an_inserted_fact_is_not_kept_by_the_window() -> None:
    """Newest slots, old position: it must earn its place by score."""

    n, block = 4096, 64
    cache = _filled_cache(n, "cpu", seed=3)
    bounds = BlockBounds(num_layers=1, capacity=n + 8, block=block, device="cpu")
    bounds.update(cache, 0, n)
    plain = DualChunkPositions.plain()
    config = RetrievalConfig(block=block, top_blocks=0, window=256, sink=64)
    # slots 0..4032 are the sequence; 4032..4096 were inserted at position 1000
    extents = [(0, 4032, 0), (4032, 4096, 1000)]
    segments = select_segments(
        bounds, 0, torch.randn(4, 512).half(), {plain.INTRA: torch.zeros(4, 64).half()},
        [(0, n, plain.INTRA)], config, extents=extents, query_position=4031,
    )
    covered = {slot for start, end, _ in segments for slot in range(start, end)}
    assert 4031 in covered and 0 in covered
    assert not covered & set(range(4032, 4096))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_keeping_every_block_matches_full_attention() -> None:
    pytest.importorskip("triton")

    class _Attention:
        num_heads = 16
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        softmax_scale = 192**-0.5

        class kv_b_proj:
            weight = (torch.randn(16 * 256, 512, device="cuda") * 0.05).half()

    weights = MLAWeights.from_attention(_Attention)
    rope = LatentRope(1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)))
    positions = DualChunkPositions(trained=1024, chunk=768)
    n = 2500
    cache = _filled_cache(n, "cuda", seed=2)
    bounds = BlockBounds(num_layers=1, capacity=n + 8, block=64, device="cuda")
    bounds.update(cache, 0, n)
    q_nope = torch.randn(16, 128, device="cuda").half()
    q_pe = torch.randn(16, 64, device="cuda").half()
    from deepinfer.moe.latent_mla import _decode_regions

    regions = _decode_regions(n - 1, positions)
    full = decode_attention(weights, cache, 0, q_nope, q_pe, n - 1, positions=positions, rope=rope)
    everything = RetrievalConfig(block=64, top_blocks=10_000, window=64, sink=64)
    sparse = sparse_decode_attention(
        weights, cache, bounds, 0, q_nope, q_pe, n - 1, regions,
        positions=positions, rope=rope, config=everything,
    )
    assert torch.allclose(sparse.float(), full.float(), atol=2e-3, rtol=2e-3)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_repositioned_runs_sit_right_before_the_window() -> None:
    """Direct definition: picked runs re-rotated to virtual positions in front
    of the window, in their original order; everything else where it was."""

    pytest.importorskip("triton")
    from deepinfer.moe.latent_mla import _decode_regions

    torch.manual_seed(4)
    heads = 16
    kv_b = (torch.randn(heads * 256, 512, device="cuda") * 0.05).half()

    class _Attention:
        num_heads = heads
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        softmax_scale = 192**-0.5

        class kv_b_proj:
            weight = kv_b

    weights = MLAWeights.from_attention(_Attention)
    rope = LatentRope(1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)))
    plain = DualChunkPositions.plain()
    n = 3000
    latent = torch.randn(n, 512, device="cuda").half()
    pe_raw = torch.randn(n, 64, device="cuda").half()
    cache = LatentKVCache(num_layers=1, capacity=n + 8, kv_dtype="fp16", device="cuda")
    cache.write(0, 0, latent, rope.apply(pe_raw, torch.arange(n, device="cuda")))
    bounds = BlockBounds(num_layers=1, capacity=n + 8, block=64, device="cuda")
    bounds.update(cache, 0, n)
    config = RetrievalConfig(block=64, top_blocks=3, window=256, sink=64, reposition=True)
    q_nope = torch.randn(heads, 128, device="cuda").half()
    q_pe_raw = torch.randn(heads, 64, device="cuda").half()
    query = n - 1
    regions = _decode_regions(query, plain)
    ours = sparse_decode_attention(
        weights, cache, bounds, 0, q_nope, q_pe_raw, query, regions,
        positions=plain, rope=rope, config=config, extents=[(0, n, 0)],
    )
    q_abs = weights.absorb_query(q_nope)
    rotated_q = rope.apply(q_pe_raw.unsqueeze(0), torch.tensor([query], device="cuda"))[0]
    chosen = select_segments(
        bounds, 0, q_abs, {plain.INTRA: rotated_q}, regions, config,
        extents=[(0, n, 0)], query_position=query, with_kind=True,
    )
    picked = sorted((a, b) for a, b, _, kind in chosen if kind)
    assert picked, "the test needs at least one retrieved run"
    virtual = {}
    cursor = query - config.window + 1 - sum(b - a for a, b in picked)
    for a, b in picked:
        for offset, slot in enumerate(range(a, b)):
            virtual[slot] = cursor + offset
        cursor += b - a
    slots = [slot for a, b, _, _ in chosen for slot in range(a, b)]
    places = torch.tensor([virtual.get(slot, slot) for slot in slots], device="cuda")
    keys = rope.apply(pe_raw[slots], places).float()
    scores = q_abs.float() @ latent[slots].float().t() + rotated_q.float() @ keys.t()
    weights_ = torch.softmax(scores * weights.softmax_scale, dim=-1)
    expected = weights.project_output(weights_ @ latent[slots].float())
    assert torch.allclose(ours.float(), expected.float(), atol=3e-3, rtol=3e-3)
