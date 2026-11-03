"""Latent MLA: position mapping, RoPE layout, fp8 groups, cache tiers, attention math.

The attention tests compare against a direct per-(query, key) definition:
score = q_nope . k_nope + rot(q_pe, q_pos(relation(i, j))) . rot(k_pe, k_pos(j)),
which is what the region bookkeeping is supposed to implement.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe import latent_mla
from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    decode_attention,
    decode_attention_reference,
    dequantize_fp8_groups,
    extent_regions,
    prefill_attention,
    quantize_fp8_groups,
)

CUDA = torch.cuda.is_available()


def _flash_available() -> bool:
    if not CUDA:
        return False
    try:
        q = torch.zeros(1, 1, 16, 64, device="cuda", dtype=torch.float16)
        torch.ops.aten._scaled_dot_product_flash_attention(q, q, q, 0.0, True, False)
        return True
    except Exception:  # noqa: BLE001
        return False


FLASH = _flash_available()


def _relation(positions: DualChunkPositions, i: int, j: int) -> int:
    ci, cj = positions.chunk_index(i), positions.chunk_index(j)
    if ci == cj:
        return positions.INTRA
    if ci == cj + 1:
        return positions.SUCCESSIVE
    return positions.INTER


def _distance(positions: DualChunkPositions, i: int, j: int) -> int:
    rel = _relation(positions, i, j)
    q = positions.query_positions(torch.tensor([i]), rel)
    k = positions.key_positions(torch.tensor([j]))
    return int(q - k)


# ------------------------------------------------------------------ positions


def test_plain_positions_are_the_identity() -> None:
    plain = DualChunkPositions.plain()
    idx = torch.tensor([0, 5, 1 << 21])
    assert torch.equal(plain.key_positions(idx), idx)
    # every key is in the query's own (single, huge) chunk
    assert torch.equal(plain.query_positions(idx, plain.INTRA), idx)
    assert plain.history_regions(1 << 21) == [(0, 1 << 21, plain.INTRA)]


def test_dual_chunk_is_exact_below_the_trained_length() -> None:
    positions = DualChunkPositions(trained=64, chunk=48)
    for i in range(64):
        for j in range(i + 1):
            assert _distance(positions, i, j) == i - j


def test_dual_chunk_keeps_every_distance_inside_the_trained_range() -> None:
    positions = DualChunkPositions(trained=64, chunk=48)
    for i in range(64, 300):
        for j in range(i + 1):
            distance = _distance(positions, i, j)
            assert 0 <= distance < 64
            if positions.chunk_index(i) == positions.chunk_index(j):
                assert distance == i - j


def test_far_caps_old_distances_and_only_moves_queries() -> None:
    near = DualChunkPositions(trained=64, chunk=48, far=50)
    default = DualChunkPositions(trained=64, chunk=48)
    for i in range(96, 300):
        for j in range(i + 1):
            distance = _distance(near, i, j)
            assert 0 <= distance <= 50
            if near.chunk_index(i) == near.chunk_index(j):
                assert distance == i - j
    idx = torch.arange(500)
    assert torch.equal(near.key_positions(idx), default.key_positions(idx))
    with pytest.raises(ValueError):
        DualChunkPositions(trained=64, chunk=48, far=40)


def test_history_regions_tile_the_prefix_once() -> None:
    positions = DualChunkPositions(trained=64, chunk=48)
    for start in range(0, 300, 7):
        covered = []
        for k0, k1, relation in positions.history_regions(start):
            assert k0 < k1
            for j in range(k0, k1):
                assert _relation(positions, start, j) == relation
            covered.extend(range(k0, k1))
        assert covered == list(range(start))


def test_split_block_cuts_at_chunk_boundaries() -> None:
    positions = DualChunkPositions(trained=64, chunk=48)
    pieces = positions.split_block(40, 70)
    assert pieces == [(40, 8), (48, 48), (96, 14)]
    for start, length in pieces:
        assert positions.chunk_index(start) == positions.chunk_index(start + length - 1)


def test_for_trained_length_uses_three_quarter_chunks() -> None:
    positions = DualChunkPositions.for_trained_length(131072)
    assert positions.chunk == 98304
    with pytest.raises(ValueError):
        DualChunkPositions(trained=10, chunk=11)


def test_extent_regions_follow_logical_positions() -> None:
    positions = DualChunkPositions(trained=64, chunk=48)
    intra, successive, inter = positions.INTRA, positions.SUCCESSIVE, positions.INTER
    # slots 0..100 hold logical 0..100; slots 100..110 hold a segment inserted at logical 30
    extents = [(0, 100, 0), (100, 110, 30)]
    regions = extent_regions(extents, 100, positions)  # query in chunk 2
    assert regions == [
        (0, 48, inter),
        (48, 96, successive),
        (96, 100, intra),
        (100, 110, inter),  # chunk of its logical position, not of its slot
    ]
    # an insert straddling a chunk boundary is cut where its logical chunk changes
    assert extent_regions([(500, 520, 40)], 60, positions) == [
        (500, 508, successive),
        (508, 520, intra),
    ]
    with pytest.raises(ValueError):
        extent_regions([(0, 10, 200)], 100, positions)


# ----------------------------------------------------------------------- rope


def _official_rotate(x, cos, sin):
    """modeling_deepseek.apply_rotary_pos_emb for one tensor [b, h, s, d]."""

    b, h, s, d = x.shape
    x = x.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def test_rotation_matches_the_official_layout() -> None:
    torch.manual_seed(0)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    rope = LatentRope(inv_freq)
    positions = torch.tensor([0, 1, 7, 4095])
    x = torch.randn(4, 3, 64)  # [n, heads, 64]
    ours = rope.apply(x, positions)
    # exact angles: the official code forms them in fp32, which is the
    # imprecision this class removes, not part of the layout being checked
    freqs = torch.outer(positions.double(), inv_freq.double())
    emb = torch.cat((freqs, freqs), dim=-1)
    official = _official_rotate(
        x.permute(1, 0, 2).unsqueeze(0),
        emb.cos().float()[None, None],
        emb.sin().float()[None, None],
    )[0].permute(1, 0, 2)
    assert torch.allclose(ours, official, atol=1e-5)


def test_far_positions_keep_their_angles() -> None:
    # fp32 angles at position 2M are off by ~0.06 rad; fp64 ones are not
    inv_freq = torch.tensor([1.0, 0.75] + [0.5] * 30)
    rope = LatentRope(inv_freq)
    cos, sin = rope.cos_sin(torch.tensor([2_000_003]))
    exact = 2_000_003 * 0.75
    assert float(cos[0, 1]) == pytest.approx(math.cos(exact), abs=1e-6)
    assert float(sin[0, 1]) == pytest.approx(math.sin(exact), abs=1e-6)


# ------------------------------------------------------------------------ fp8


def test_fp8_groups_round_trip_within_e4m3_precision() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 512) * torch.logspace(-2, 2, 64).unsqueeze(1)
    values, scale = quantize_fp8_groups(x)
    assert values.dtype == torch.float8_e4m3fn and scale.shape == (64, 4)
    back = dequantize_fp8_groups(values, scale, torch.float32)
    groups = x.reshape(64, 4, 128).abs().amax(-1, keepdim=True)
    error = (back - x).abs().reshape(64, 4, 128)
    # three mantissa bits: half an ulp at the top of the range is 16/448 of it
    assert bool((error <= groups * (16 / 448) + 1e-6).all())


# ---------------------------------------------------------------------- cache


def test_cache_round_trip_on_cpu() -> None:
    cache = LatentKVCache(num_layers=2, capacity=100, kv_dtype="fp16", device="cpu")
    latent = torch.randn(70, 512).half()
    pe = torch.randn(70, 64).half()
    cache.write(1, 0, latent[:40], pe[:40])
    cache.write(1, 40, latent[40:], pe[40:])
    got_latent, got_pe = cache.tile(1, 10, 65)
    assert torch.equal(got_latent, latent[10:65]) and torch.equal(got_pe, pe[10:65])
    assert cache.length == 70
    assert cache.bytes_per_token(27, "fp16") == 27 * (512 * 2 + 64 * 2)
    assert cache.bytes_per_token(27, "fp8") == 27 * (512 + 16 + 64 * 2)
    with pytest.raises(ValueError):
        cache.write(0, 90, latent[:20], pe[:20])


@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8"])
def test_cache_save_load_round_trip(tmp_path, kv_dtype: str) -> None:
    torch.manual_seed(0)
    cache = LatentKVCache(num_layers=2, capacity=50, kv_dtype=kv_dtype, device="cpu")
    latent = torch.randn(40, 512).half()
    pe = torch.randn(40, 64).half()
    cache.write(0, 0, latent, pe)
    cache.write(1, 0, latent * 2, pe * 2)
    cache.save(str(tmp_path), meta={"document": "abc"})
    loaded, meta = LatentKVCache.load(str(tmp_path), capacity=60, device="cpu")
    assert meta == {"document": "abc"}
    assert loaded.length == 40 and loaded.capacity == 60 and loaded.kv_dtype == kv_dtype
    for layer in (0, 1):
        for got, want in zip(loaded.tile(layer, 0, 40), cache.tile(layer, 0, 40), strict=True):
            assert torch.equal(got, want)


@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA, reason="CUDA"))])
def test_cache_save_spills_layers_to_a_second_directory(tmp_path, monkeypatch, kv_dtype: str, device: str) -> None:
    import json

    from deepinfer.moe import latent_mla

    torch.manual_seed(0)
    # on CUDA 25 of the 40 rows sit on the GPU and the rest in pinned host memory
    gpu_tokens = 25 if device == "cuda" else None
    cache = LatentKVCache(num_layers=3, capacity=50, gpu_tokens=gpu_tokens, kv_dtype=kv_dtype, device=device)
    for layer in range(3):
        latent = torch.randn(40, 512, device=device).half() * (layer + 1)
        cache.write(layer, 0, latent, torch.randn(40, 64, device=device).half())
    primary, spare = tmp_path / "a", tmp_path / "b"
    latent_file = 40 * 512 * (1 if kv_dtype == "fp8" else 2)
    room = {str(primary): 2 * latent_file, str(spare): 1 << 40}  # the first disk takes two latent layers

    def free_bytes(directory: str) -> int:
        used = sum(path.stat().st_size for path in tmp_path.glob(f"{directory[-1]}/*.bin"))
        return room[directory] - used

    monkeypatch.setattr(latent_mla, "_free_bytes", free_bytes)
    cache.save(str(primary), meta={"document": "abc"}, spill=[str(spare)], margin=0)
    info = json.loads((primary / "cache.json").read_text(encoding="utf-8"))
    assert info["files"]["latent"] == [0, 0, 1] and info["files"]["pe"] == [1, 1, 1]
    loaded, meta = LatentKVCache.load(str(primary), capacity=60, gpu_tokens=gpu_tokens, device=device)
    assert meta == {"document": "abc"} and loaded.length == 40 and loaded.capacity == 60
    for layer in range(3):
        for got, want in zip(loaded.tile(layer, 0, 40), cache.tile(layer, 0, 40), strict=True):
            assert torch.equal(got, want)


@pytest.mark.skipif(not CUDA, reason="pinned host tier needs CUDA")
@pytest.mark.parametrize("kv_dtype", ["fp16", "fp8"])
def test_host_tier_reads_back_what_was_written(kv_dtype: str) -> None:
    torch.manual_seed(0)
    cache = LatentKVCache(
        num_layers=1, capacity=300, gpu_tokens=128, kv_dtype=kv_dtype, device="cuda"
    )
    latent = torch.randn(300, 512, device="cuda").half()
    pe = torch.randn(300, 64, device="cuda").half()
    cache.write(0, 0, latent[:200], pe[:200])  # straddles the boundary
    cache.write(0, 200, latent[200:], pe[200:])
    got_latent, got_pe = cache.tile(0, 100, 260)
    assert got_latent.device.type == "cuda"
    assert torch.equal(got_pe, pe[100:260])
    tolerance = 0.0 if kv_dtype == "fp16" else 0.05 * float(latent.abs().max())
    assert float((got_latent.float() - latent[100:260].float()).abs().max()) <= tolerance


@pytest.mark.skipif(not CUDA, reason="pinning needs CUDA")
def test_host_tier_is_pinned_at_its_exact_size() -> None:
    import gc

    cache = LatentKVCache(num_layers=2, capacity=300, gpu_tokens=128, kv_dtype="fp8", device="cuda")
    for host, row_bytes in ((cache.host_latent, 512), (cache.host_pe, 64 * 2), (cache.host_scale, 4 * 4)):
        assert host.is_pinned()
        # torch's own pinned allocator would round each array up to a power of two
        assert host.untyped_storage().nbytes() == 2 * 172 * row_bytes
    del cache, host
    gc.collect()  # unregistering on the way out must leave CUDA usable
    again = latent_mla.pinned_empty((4, 3), torch.float32)
    again.copy_(torch.ones(4, 3, device="cuda"), non_blocking=True)
    torch.cuda.synchronize()
    assert again.is_pinned() and torch.equal(again, torch.ones(4, 3))


# ------------------------------------------------------------------ attention


class _Attention:
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    softmax_scale = 192**-0.5

    def __init__(self, device: str, heads: int = 4) -> None:
        self.num_heads = heads
        self.kv_b_proj = torch.nn.Linear(512, heads * 256, bias=False).to(device).half()
        torch.nn.init.normal_(self.kv_b_proj.weight, std=0.05)


def _naive(attention, latent, pe_raw, q_nope, q_pe_raw, query_positions, positions, rope):
    """fp32 per-(i, j) attention over expanded per-head K/V."""

    heads = attention.num_heads
    kv = (latent.float() @ attention.kv_b_proj.weight.float().t()).view(-1, heads, 256)
    k_nope, v = kv[..., :128], kv[..., 128:]
    n = latent.shape[0]
    k_pe = rope.apply(pe_raw.float(), positions.key_positions(torch.arange(n, device=latent.device)))
    outs = []
    for row, i in enumerate(query_positions):
        keys = torch.arange(i + 1, device=latent.device)
        scores = torch.einsum("hd,jhd->hj", q_nope[row].float(), k_nope[: i + 1])
        rope_scores = torch.empty_like(scores)
        for j in range(i + 1):
            rel = _relation(positions, i, j)
            q_pe = rope.apply(
                q_pe_raw[row : row + 1].float(),
                positions.query_positions(torch.tensor([i], device=latent.device), rel),
            )[0]
            rope_scores[:, j] = q_pe @ k_pe[j]
        weights = torch.softmax((scores + rope_scores) * attention.softmax_scale, dim=-1)
        outs.append(torch.einsum("hj,jhd->hd", weights, v[keys]))
    return torch.stack(outs)


def _setup(device: str, n: int, seed: int = 0, heads: int = 4):
    torch.manual_seed(seed)
    attention = _Attention(device, heads)
    weights = MLAWeights.from_attention(attention)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    rope = LatentRope(inv_freq)
    latent = torch.randn(n, 512, device=device).half()
    pe_raw = torch.randn(n, 64, device=device).half()
    return attention, weights, rope, latent, pe_raw


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(
    "kernel", [pytest.param("flash", marks=pytest.mark.skipif(not FLASH, reason="no FlashAttention here")), "efficient"]
)
@pytest.mark.parametrize(
    "positions",
    [DualChunkPositions.plain(), DualChunkPositions(trained=48, chunk=36)],
    ids=["plain", "dca"],
)
def test_prefill_attention_matches_the_definition(positions: DualChunkPositions, kernel: str, monkeypatch) -> None:
    monkeypatch.setattr(latent_mla, "_FLASH", kernel == "flash")
    n = 120
    attention, weights, rope, latent, pe_raw = _setup("cuda", n)
    cache = LatentKVCache(num_layers=1, capacity=n, kv_dtype="fp16", device="cuda")
    pe = rope.apply(pe_raw, positions.key_positions(torch.arange(n, device="cuda")))
    q_nope = torch.randn(n, 4, 128, device="cuda").half()
    q_pe_raw = torch.randn(n, 4, 64, device="cuda").half()
    outputs = []
    for start, length in positions.split_block(0, n):
        for offset in range(0, length, 16):  # small blocks: history + tiles + merge
            s, e = start + offset, min(start + offset + 16, start + length)
            cache.write(0, s, latent[s:e], pe[s:e])
            outputs.append(
                prefill_attention(
                    weights, cache, 0, q_nope[s:e], q_pe_raw[s:e], latent[s:e], pe[s:e], s,
                    positions=positions, rope=rope, tile_tokens=24,
                )
            )
    ours = torch.cat(outputs).float()
    expected = _naive(attention, latent, pe_raw, q_nope, q_pe_raw, list(range(n)), positions, rope)
    assert torch.allclose(ours, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(
    "positions",
    [DualChunkPositions.plain(), DualChunkPositions(trained=48, chunk=36)],
    ids=["plain", "dca"],
)
@pytest.mark.parametrize("fused", [False, True], ids=["reference", "fused"])
def test_decode_attention_matches_the_definition(positions, fused: bool) -> None:
    if fused:
        pytest.importorskip("triton")
    n = 150
    attention, weights, rope, latent, pe_raw = _setup("cuda", n, seed=1)
    cache = LatentKVCache(
        num_layers=1, capacity=n, gpu_tokens=100, kv_dtype="fp16", device="cuda"
    )
    pe = rope.apply(pe_raw, positions.key_positions(torch.arange(n, device="cuda")))
    cache.write(0, 0, latent, pe)
    q_nope = torch.randn(1, 4, 128, device="cuda").half()
    q_pe_raw = torch.randn(1, 4, 64, device="cuda").half()
    run = decode_attention if fused else decode_attention_reference
    ours = run(weights, cache, 0, q_nope[0], q_pe_raw[0], n - 1, positions=positions, rope=rope)
    expected = _naive(attention, latent, pe_raw, q_nope, q_pe_raw, [n - 1], positions, rope)[0]
    assert torch.allclose(ours.float(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("fused", [False, True], ids=["reference", "fused"])
def test_one_shared_latent_gives_the_per_head_attention_at_the_model_shape(fused: bool) -> None:
    """DeepSeek-V2-Lite's shape: 16 heads, kv_b (4096, 512) like blk.N.attn_kv_b.

    Every head's K and V are a different projection of the latent, yet caching the one
    512 + 64 latent per token (not 16 per-head copies) gives the per-head attention: the
    up-projections fold into the query and the output instead of into the cache.
    """
    if fused:
        pytest.importorskip("triton")
    n, heads = 150, 16
    attention, weights, rope, latent, pe_raw = _setup("cuda", n, seed=2, heads=heads)
    assert tuple(attention.kv_b_proj.weight.shape) == (4096, 512)
    positions = DualChunkPositions.plain()
    cache = LatentKVCache(num_layers=1, capacity=n, kv_dtype="fp16", device="cuda")
    pe = rope.apply(pe_raw, positions.key_positions(torch.arange(n, device="cuda")))
    cache.write(0, 0, latent, pe)
    q_nope = torch.randn(1, heads, 128, device="cuda").half()
    q_pe_raw = torch.randn(1, heads, 64, device="cuda").half()
    run = decode_attention if fused else decode_attention_reference
    ours = run(weights, cache, 0, q_nope[0], q_pe_raw[0], n - 1, positions=positions, rope=rope)
    expected = _naive(attention, latent, pe_raw, q_nope, q_pe_raw, [n - 1], positions, rope)[0]
    assert ours.shape == (heads, 128)
    assert torch.allclose(ours.float(), expected, atol=1e-2, rtol=1e-2)
