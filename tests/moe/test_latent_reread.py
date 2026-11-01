"""Locate + re-read: block mass is exact attention mass, span picking is what it says."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
)
from deepinfer.moe.latent_reread import block_mass, pick_spans


class _Attention:
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    softmax_scale = 192**-0.5

    def __init__(self) -> None:
        self.kv_b_proj = torch.nn.Linear(512, 4 * 256, bias=False)
        torch.nn.init.normal_(self.kv_b_proj.weight, std=0.05)


@pytest.mark.parametrize(
    "positions",
    [DualChunkPositions.plain(), DualChunkPositions(trained=128, chunk=96)],
    ids=["plain", "dca"],
)
def test_block_mass_is_the_softmax_mass_summed_per_block(positions) -> None:
    torch.manual_seed(0)
    attention = _Attention()
    weights = MLAWeights.from_attention(attention)
    rope = LatentRope(1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)))
    n, block = 300, 16
    latent = torch.randn(n, 512)
    pe_raw = torch.randn(n, 64)
    cache = LatentKVCache(num_layers=1, capacity=n, kv_dtype="fp16", device="cpu")
    cache.write(0, 0, latent.half(), rope.apply(pe_raw, positions.key_positions(torch.arange(n))))
    q_nope = torch.randn(3, 4, 128)
    q_pe_raw = torch.randn(3, 4, 64)
    start = n  # three questions tokens right after the cache
    regions = positions.history_regions(start)
    mass = block_mass(
        weights, cache, 0, q_nope, q_pe_raw, start, regions,
        positions=positions, rope=rope, block=block, blocks=-(-n // block), tile=64,
    )
    # naive: every (query, head) softmax over all cached slots, summed per block
    stored_latent, stored_pe = cache.tile(0, 0, n)
    expected = torch.zeros(-(-n // block))
    for row in range(3):
        position = torch.tensor([start + row])
        for head in range(4):
            q_abs = q_nope[row, head].float() @ weights.w_uk[head].float()
            scores = stored_latent.float() @ q_abs
            for k0, k1, relation in regions:
                q_pe = rope.apply(
                    q_pe_raw[row, head].view(1, 64), positions.query_positions(position, relation)
                )[0]
                scores[k0:k1] += stored_pe[k0:k1].float() @ q_pe
            p = torch.softmax(scores * weights.softmax_scale, dim=0)
            expected.index_add_(0, torch.arange(n) // block, p)
    assert torch.allclose(mass, expected, atol=2e-2, rtol=2e-2)
    assert float(mass.sum()) == pytest.approx(12.0, rel=1e-3)  # 3 queries x 4 heads


def test_pick_spans_skips_the_sink_and_merges_neighbours() -> None:
    mass = torch.zeros(20)
    mass[0] = 100.0  # the sink always looks heavy
    mass[5] = 3.0
    mass[6] = 2.0
    mass[15] = 1.0
    spans = pick_spans(mass, block=10, top=3, neighbors=1, length=195, skip_before=10)
    # blocks 5 and 6 widen to 4..7 and merge; block 15 widens to 14..16; length clips
    assert spans == [(40, 80), (140, 170)]
    assert pick_spans(mass, block=10, top=1, neighbors=0, length=200, skip_before=10) == [(50, 60)]
