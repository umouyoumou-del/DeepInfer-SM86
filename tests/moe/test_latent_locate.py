"""One-pass locate (Triton) against the PyTorch passes it replaces."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe import latent_locate
from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    extent_regions,
    prefill_attention,
)
from deepinfer.moe.latent_reread import block_mass

needs_triton = pytest.mark.skipif(not latent_locate.available(), reason="needs CUDA and Triton")


class _Attention:
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    softmax_scale = 192**-0.5

    def __init__(self, device) -> None:
        self.kv_b_proj = torch.nn.Linear(512, 4 * 256, bias=False, device=device, dtype=torch.float16)
        torch.nn.init.normal_(self.kv_b_proj.weight, std=0.05)


def test_aligned_needs_block_boundaries_below_the_cached_length() -> None:
    assert latent_locate.aligned([(0, 128, 0), (128, 200, 1)], 200)
    assert latent_locate.aligned([(0, 128, 0), (128, 300, 1)], 200)  # clipped at the length
    assert not latent_locate.aligned([(0, 100, 0), (100, 200, 1)], 200)
    assert latent_locate.aligned([(0, 128, 0), (130, 300, 1)], 128)  # past the length: not scored


@needs_triton
@pytest.mark.parametrize("kv_dtype", ["fp8", "fp16"])
@pytest.mark.parametrize("reduce", ["max", "sum"])
def test_locate_attention_matches_prefill_attention_and_block_mass(kv_dtype, reduce) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    weights = MLAWeights.from_attention(_Attention(device))
    rope = LatentRope(1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64)))
    positions = DualChunkPositions(trained=1024, chunk=768)  # several relations over 3000 slots
    length, extra, n = 3000, 37, 24  # 37 slots past the scored length: an earlier block of the query
    stored = length + extra
    # 1010 slots on the GPU: the block holding slot 1010 straddles the GPU / host boundary
    cache = LatentKVCache(num_layers=1, capacity=stored + n, gpu_tokens=1010, kv_dtype=kv_dtype, device=device)
    latent = torch.randn(stored, 512, device=device, dtype=torch.float16)
    pe = rope.apply(torch.randn(stored, 64, device=device), positions.key_positions(torch.arange(stored, device=device)))
    cache.write(0, 0, latent, pe)
    start = stored
    q_nope = torch.randn(n, 4, 128, device=device, dtype=torch.float16)
    q_pe_raw = torch.randn(n, 4, 64, device=device, dtype=torch.float16)
    own_latent = torch.randn(n, 512, device=device, dtype=torch.float16)
    own_pe = rope.apply(
        torch.randn(n, 64, device=device), positions.key_positions(torch.arange(start, start + n, device=device))
    ).half()
    regions = extent_regions([(0, stored, 0)], start, positions)
    assert len({relation for _, _, relation in regions}) >= 2
    blocks = -(-length // 64)

    expected_out = prefill_attention(
        weights, cache, 0, q_nope, q_pe_raw, own_latent, own_pe, start,
        positions=positions, rope=rope, regions=regions,
    )
    cached = [(a, min(b, length), r) for a, b, r in regions if a < length]
    expected_mass = block_mass(
        weights, cache, 0, q_nope, q_pe_raw, start, cached,
        positions=positions, rope=rope, block=64, blocks=blocks, reduce=reduce,
    )
    # 512-slot tiles: one on the GPU, one straddling the boundary, the rest staged from host memory;
    # fp8 tiles dequantized first or by the kernel, 32 or 64 keys a step
    for dequantize, block_n in ((True, 32), (False, 64), (False, 32)):
        out, mass = latent_locate.locate_attention(
            weights, cache, 0, q_nope, q_pe_raw, own_latent, own_pe, start, regions,
            positions=positions, rope=rope, length=length, blocks=blocks, reduce=reduce,
            tile=512, dequantize=dequantize, block_n=block_n, block_m=16 if block_n == 64 else 32,
            num_warps=4 if block_n == 64 else 8,
        )
        assert out.shape == expected_out.shape and out.dtype == expected_out.dtype
        out_error = float((out.float() - expected_out.float()).abs().max())
        assert torch.allclose(out.float(), expected_out.float(), atol=5e-3, rtol=5e-2), out_error
        mass_error = float(((mass - expected_mass).abs() / expected_mass.abs().clamp_min(1e-6)).max())
        assert torch.allclose(mass, expected_mass, atol=1e-5, rtol=2e-2), mass_error
