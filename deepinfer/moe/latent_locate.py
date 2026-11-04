"""Locate in one pass: a query's attention over the latent cache and its per-block mass together.

``locate`` runs a question at the end of the cached sequence with full attention
and scores every block of 64 slots by the question's attention mass on it.  Done
with stock PyTorch that costs three passes over the cache per layer - the
question's own attention (latent tiles expanded to 16 heads for flash attention),
then two block-mass passes (log-sum-exp, then the mass) that each write and read
an fp32 score matrix of (question tokens x heads) x keys.  At 2M slots with a
question plus a fed-back answer (274 tokens) that was 62 s per locate, 51 of them
in the block-mass passes; most of the cache sits in host memory, so every pass
also crosses PCIe.

Here one Triton kernel reads each key once: absorbed scores ``q_abs . c + q_pe .
k_pe`` for a tile of query rows (token x head) against 32 keys at a time give
each 64-slot block's log-mass per row, and the same probabilities update a
flash-attention accumulator over the latent.  The block log-masses are
normalized afterwards by each row's log-sum-exp over the cached slots, exactly
what the two-pass version computes.  Same math, fp32 accumulation; only the
rounding differs.  At 2M slots: 17.4 -> 2.5 s for a 17-token question, 62.8 ->
13.4 s with a 256-token answer fed back (4080 SUPER, 0.93M slots in VRAM).
"""

from __future__ import annotations

import torch

from deepinfer.moe.latent_mla import (
    DualChunkPositions,
    LatentKVCache,
    LatentRope,
    MLAWeights,
    dequantize_fp8_groups,
    merge_partials,
)
from deepinfer.moe.triton_fp8 import native_fp8_available

try:  # Triton is optional; locate falls back to the PyTorch passes
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  # pragma: no cover - depends on the platform
    triton = None
    tl = None

BLOCK = 64


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _fp8e4m3_bits_to_fp32(bits):
        """Decode one E4M3 byte without requiring an FP8-capable GPU."""

        u = bits.to(tl.uint8)
        sign = tl.where((u & 0x80) != 0, -1.0, 1.0)
        exp = ((u >> 3) & 0x0F).to(tl.float32)
        mant = (u & 0x07).to(tl.float32)
        normal = (1.0 + mant * 0.125) * tl.exp2(exp - 7.0)
        subnormal = (mant * 0.125) * (1.0 / 64.0)
        value = tl.where(exp == 0.0, subnormal, normal)
        value = tl.where((exp == 15.0) & (mant == 7.0), 448.0, value)
        return tl.where(sign < 0.0, -value, value)

    @triton.jit
    def _locate_kernel(
        Q,
        QP,
        LAT,
        SC,
        PE,
        SPLIT_START,
        SPLIT_END,
        SPLIT_REL,
        SPLIT_BLOCK,
        O_ACC,
        O_M,
        O_L,
        LBM,
        rows,
        sm_scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        DR: tl.constexpr,
        IS_FP8: tl.constexpr,
        SOFTWARE_FP8: tl.constexpr,
        WITH_LBM: tl.constexpr,
    ):
        """Rows ``[tile * BLOCK_M, ..)`` of the query against one split of keys.

        Keys are visited ``BLOCK_N`` (64 or 32) at a time; a locate block's
        log-mass ``max + log(sum exp(s - max))`` per row comes from the same
        scores as the flash update (with 32 keys a step, two steps are combined
        before it is stored).  The latent's four groups of 128 are dotted
        separately so an fp8 group scale multiplies a partial score rather than
        every element.
        """

        tile = tl.program_id(0)
        split = tl.program_id(1)
        start = tl.load(SPLIT_START + split)
        end = tl.load(SPLIT_END + split)
        relation = tl.load(SPLIT_REL + split)
        block0 = tl.load(SPLIT_BLOCK + split)
        offs_m = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = offs_m < rows
        offs_c = tl.arange(0, 128)
        offs_r = tl.arange(0, DR)
        q_row = Q + offs_m[:, None].to(tl.int64) * 512 + offs_c[None, :]
        q0 = tl.load(q_row, mask=row_ok[:, None], other=0.0)
        q1 = tl.load(q_row + 128, mask=row_ok[:, None], other=0.0)
        q2 = tl.load(q_row + 256, mask=row_ok[:, None], other=0.0)
        q3 = tl.load(q_row + 384, mask=row_ok[:, None], other=0.0)
        qp = tl.load(
            QP + (relation.to(tl.int64) * rows + offs_m[:, None]) * DR + offs_r[None, :],
            mask=row_ok[:, None],
            other=0.0,
        )
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        a0 = tl.zeros([BLOCK_M, 128], tl.float32)
        a1 = tl.zeros([BLOCK_M, 128], tl.float32)
        a2 = tl.zeros([BLOCK_M, 128], tl.float32)
        a3 = tl.zeros([BLOCK_M, 128], tl.float32)
        run_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        run_sum = tl.zeros([BLOCK_M], tl.float32)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            valid = offs_n < end
            keys = offs_n.to(tl.int64)
            k_row = LAT + keys[:, None] * 512 + offs_c[None, :]
            if IS_FP8 and SOFTWARE_FP8:
                k0 = _fp8e4m3_bits_to_fp32(tl.load(k_row, mask=valid[:, None], other=0)).to(tl.float16)
                k1 = _fp8e4m3_bits_to_fp32(tl.load(k_row + 128, mask=valid[:, None], other=0)).to(tl.float16)
                k2 = _fp8e4m3_bits_to_fp32(tl.load(k_row + 256, mask=valid[:, None], other=0)).to(tl.float16)
                k3 = _fp8e4m3_bits_to_fp32(tl.load(k_row + 384, mask=valid[:, None], other=0)).to(tl.float16)
            else:
                k0 = tl.load(k_row, mask=valid[:, None], other=0.0).to(tl.float16)
                k1 = tl.load(k_row + 128, mask=valid[:, None], other=0.0).to(tl.float16)
                k2 = tl.load(k_row + 256, mask=valid[:, None], other=0.0).to(tl.float16)
                k3 = tl.load(k_row + 384, mask=valid[:, None], other=0.0).to(tl.float16)
            kp = tl.load(PE + keys[:, None] * DR + offs_r[None, :], mask=valid[:, None], other=0.0)
            if IS_FP8:
                s0 = tl.load(SC + keys * 4, mask=valid, other=0.0)
                s1 = tl.load(SC + keys * 4 + 1, mask=valid, other=0.0)
                s2 = tl.load(SC + keys * 4 + 2, mask=valid, other=0.0)
                s3 = tl.load(SC + keys * 4 + 3, mask=valid, other=0.0)
                s = tl.dot(q0, tl.trans(k0)) * s0[None, :]
                s += tl.dot(q1, tl.trans(k1)) * s1[None, :]
                s += tl.dot(q2, tl.trans(k2)) * s2[None, :]
                s += tl.dot(q3, tl.trans(k3)) * s3[None, :]
            else:
                s = tl.dot(q0, tl.trans(k0))
                s += tl.dot(q1, tl.trans(k1))
                s += tl.dot(q2, tl.trans(k2))
                s += tl.dot(q3, tl.trans(k3))
            s += tl.dot(qp, tl.trans(kp))
            s = tl.where(valid[None, :], s * sm_scale, float("-inf"))
            block_max = tl.max(s, 1)
            p = tl.exp(s - block_max[:, None])
            block_sum = tl.sum(p, 1)
            if WITH_LBM:
                block = block0 + (n0 - start) // 64
                if BLOCK_N == 64:
                    tl.store(
                        LBM + block.to(tl.int64) * rows + offs_m,
                        block_max + tl.log(block_sum),
                        mask=row_ok,
                    )
                else:
                    second = ((n0 - start) // BLOCK_N) % 2 == 1
                    both = tl.maximum(run_max, block_max)
                    run_sum = tl.where(
                        second,
                        run_sum * tl.exp(run_max - both) + block_sum * tl.exp(block_max - both),
                        block_sum,
                    )
                    run_max = tl.where(second, both, block_max)
                    done = second | (n0 + BLOCK_N >= end)
                    tl.store(
                        LBM + block.to(tl.int64) * rows + offs_m,
                        run_max + tl.log(run_sum),
                        mask=row_ok & done,
                    )
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(block_max - m_new)
            p = p * beta[:, None]
            l_i = l_i * alpha + block_sum * beta
            if IS_FP8:
                a0 = a0 * alpha[:, None] + tl.dot((p * s0[None, :]).to(tl.float16), k0)
                a1 = a1 * alpha[:, None] + tl.dot((p * s1[None, :]).to(tl.float16), k1)
                a2 = a2 * alpha[:, None] + tl.dot((p * s2[None, :]).to(tl.float16), k2)
                a3 = a3 * alpha[:, None] + tl.dot((p * s3[None, :]).to(tl.float16), k3)
            else:
                ph = p.to(tl.float16)
                a0 = a0 * alpha[:, None] + tl.dot(ph, k0)
                a1 = a1 * alpha[:, None] + tl.dot(ph, k1)
                a2 = a2 * alpha[:, None] + tl.dot(ph, k2)
                a3 = a3 * alpha[:, None] + tl.dot(ph, k3)
            m_i = m_new
        base = split.to(tl.int64) * rows
        out = O_ACC + (base + offs_m[:, None]) * 512 + offs_c[None, :]
        tl.store(out, a0, mask=row_ok[:, None])
        tl.store(out + 128, a1, mask=row_ok[:, None])
        tl.store(out + 256, a2, mask=row_ok[:, None])
        tl.store(out + 384, a3, mask=row_ok[:, None])
        tl.store(O_M + base + offs_m, m_i, mask=row_ok)
        tl.store(O_L + base + offs_m, l_i, mask=row_ok)


def _splits(
    ranges: list[tuple[int, int, int, int]], *, row_tiles: int, target_programs: int, device
) -> torch.Tensor:
    """``(start, end, relation, block)`` work items, ``[4, S]`` int32.

    ``ranges`` are ``(local start, local end, relation, global start)``; a split
    starts on a block boundary of the global slots, so ``block`` - the global
    block index of its first key - also numbers every later step of it.
    """

    total = sum(end - start for start, end, _, _ in ranges)
    wanted = max(1, target_programs // max(1, row_tiles))
    size = max(4 * BLOCK, -(-total // wanted))
    size = -(-size // BLOCK) * BLOCK
    table: list[list[int]] = [[], [], [], []]
    for start, end, relation, first in ranges:
        for lo in range(start, end, size):
            table[0].append(lo)
            table[1].append(min(lo + size, end))
            table[2].append(relation)
            table[3].append((first + lo - start) // BLOCK)
    return torch.tensor(table, dtype=torch.int32, device=device)


def _partial(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    latent: torch.Tensor,
    scale: torch.Tensor | None,
    pe: torch.Tensor,
    ranges: list[tuple[int, int, int, int]],
    lbm: torch.Tensor | None,
    *,
    sm_scale: float,
    block_m: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
    target_programs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized attention ``[M, 512]`` fp32 and log-sum-exp ``[M]`` of the rows over ``ranges``."""

    rows = q_abs.shape[0]
    row_tiles = -(-rows // block_m)
    table = _splits(ranges, row_tiles=row_tiles, target_programs=target_programs, device=q_abs.device)
    count = table.shape[1]
    acc = torch.empty((count, rows, 512), dtype=torch.float32, device=q_abs.device)
    m = torch.empty((count, rows), dtype=torch.float32, device=q_abs.device)
    l = torch.empty((count, rows), dtype=torch.float32, device=q_abs.device)
    is_fp8 = latent.dtype == torch.float8_e4m3fn
    software_fp8 = bool(is_fp8 and not native_fp8_available(latent.device))
    latent_arg = latent.view(torch.uint8) if software_fp8 else latent
    _locate_kernel[(row_tiles, count)](
        q_abs, q_pe, latent_arg, scale if is_fp8 else latent, pe,
        table[0], table[1], table[2], table[3],
        acc, m, l, lbm if lbm is not None else acc,
        rows, float(sm_scale),
        BLOCK_M=block_m, BLOCK_N=block_n, DR=q_pe.shape[-1], IS_FP8=is_fp8,
        SOFTWARE_FP8=software_fp8, WITH_LBM=lbm is not None,
        num_warps=num_warps, num_stages=num_stages,
    )
    top = m.max(dim=0).values
    weight = torch.exp(m - top)
    total = (l * weight).sum(dim=0)
    out = torch.einsum("srd,sr->rd", acc, weight) / total.unsqueeze(-1)
    return out, top + torch.log(total)


_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _COPY_STREAMS.get(device)
    if stream is None:
        stream = _COPY_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


def _tiles(cache: LatentKVCache, layer: int, length: int, tile: int, *, dequantize: bool):
    """``(t0, t1, latent, scale or None, pe)`` for ``[0, length)`` in block-aligned tiles, on the device.

    GPU-resident tiles are views.  A tile that reaches past ``gpu_tokens`` is
    staged into one of two buffers on a side stream while the previous tile is
    being used (its GPU-resident head, if it straddles the boundary, is copied
    in too); the current stream waits for it.  Each yielded tile must be fully
    consumed - kernels launched - before the next is requested.

    The two buffers come from the caching allocator on the current stream, which
    hands out memory freed by kernels that may still be queued there; the side
    stream therefore waits for the current stream once before its first copy
    (without that the copies overwrote live temporaries - NaN at 2M).
    """

    device = cache.device
    tile = max(BLOCK, tile // BLOCK * BLOCK)
    bounds = [(t0, min(t0 + tile, length)) for t0 in range(0, length, tile)]
    compute = torch.cuda.current_stream(device)
    copy = _copy_stream(device)
    fp8 = cache.kv_dtype == "fp8"
    free: list[torch.cuda.Event | None] = [None, None]
    ready: dict[int, torch.cuda.Event] = {}

    def resident(index: int) -> bool:
        return bounds[index][1] <= cache.gpu_tokens

    slots = []
    if not all(resident(index) for index in range(len(bounds))):
        rows = min(tile, length)
        slots = [
            (
                torch.empty((rows, cache.gpu_latent.shape[-1]), dtype=cache.gpu_latent.dtype, device=device),
                torch.empty((rows, cache.gpu_scale.shape[-1]), dtype=torch.float32, device=device) if fp8 else None,
                torch.empty((rows, cache.gpu_pe.shape[-1]), dtype=torch.float16, device=device),
            )
            for _ in range(2)
        ]
        copy.wait_stream(compute)

    def stage(index: int) -> None:
        if index >= len(bounds) or resident(index):
            return
        t0, t1 = bounds[index]
        slot = index % 2
        latent, scale, pe = slots[slot]
        with torch.cuda.stream(copy):
            if free[slot] is not None:
                copy.wait_event(free[slot])
            for where, lo, hi, offset in cache._segments(t0, t1):
                if where == "gpu":
                    source = (cache.gpu_latent, cache.gpu_scale, cache.gpu_pe)
                else:
                    source = (cache.host_latent, cache.host_scale, cache.host_pe)
                count = hi - lo
                latent[lo - t0 : hi - t0].copy_(source[0][layer, offset : offset + count], non_blocking=True)
                pe[lo - t0 : hi - t0].copy_(source[2][layer, offset : offset + count], non_blocking=True)
                if fp8:
                    scale[lo - t0 : hi - t0].copy_(source[1][layer, offset : offset + count], non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy)
        ready[index] = event

    try:
        stage(0)
        for index, (t0, t1) in enumerate(bounds):
            stage(index + 1)
            if resident(index):
                latent = cache.gpu_latent[layer, t0:t1]
                scale = cache.gpu_scale[layer, t0:t1] if fp8 else None
                pe = cache.gpu_pe[layer, t0:t1]
            else:
                compute.wait_event(ready.pop(index))
                buffer_latent, buffer_scale, buffer_pe = slots[index % 2]
                latent = buffer_latent[: t1 - t0]
                scale = None if buffer_scale is None else buffer_scale[: t1 - t0]
                pe = buffer_pe[: t1 - t0]
            if fp8 and dequantize:
                latent, scale = dequantize_fp8_groups(latent, scale), None
            yield t0, t1, latent, scale, pe
            if not resident(index):
                used = torch.cuda.Event()
                used.record(compute)
                free[index % 2] = used
    finally:
        # a copy still in flight (the consumer stopped early) must land before the buffers are reused
        if slots:
            compute.wait_stream(copy)


def aligned(regions: list[tuple[int, int, int]], length: int, block: int = BLOCK) -> bool:
    """Whether every cached region starts on a block boundary and ends on one (or at ``length``)."""

    for start, end, _ in regions:
        if start >= length:
            continue
        if start % block or (min(end, length) % block and min(end, length) != length):
            return False
    return True


@torch.inference_mode()
def locate_attention(
    weights: MLAWeights,
    cache: LatentKVCache,
    layer: int,
    q_nope: torch.Tensor,
    q_pe_raw: torch.Tensor,
    latent: torch.Tensor,
    pe: torch.Tensor,
    start: int,
    regions: list[tuple[int, int, int]],
    *,
    positions: DualChunkPositions,
    rope: LatentRope,
    length: int,
    blocks: int,
    reduce: str = "max",
    tile: int = 131072,
    dequantize: bool = True,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 8,
    num_stages: int = 1,
    target_programs: int = 640,
    keep_tokens: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(attended [n, H, v_dim] fp16, block mass [blocks])`` for a query block at logical ``start``.

    The attention is what :func:`~deepinfer.moe.latent_mla.prefill_attention`
    returns for the same arguments (history ``regions`` plus the block's own
    keys, causal); the mass is what ``latent_reread.block_mass`` returns for the
    regions below ``length`` (each (token, head) row normalized over those
    slots), summed or maxed over rows per block of 64.  Regions below ``length``
    must be block-aligned (:func:`aligned`).  ``keep_tokens`` limits the
    (token, head) rows that count towards the mass; all of them attend.

    The cache is read in tiles of ``tile`` slots: GPU-resident ones in place,
    the rest copied over PCIe on a side stream one tile ahead; an fp8 tile is
    dequantized to fp16 once (``dequantize``) rather than by every row tile of
    the kernel.
    """

    n, heads, _ = q_nope.shape
    device = q_nope.device
    rows = n * heads
    q_abs = torch.bmm(q_nope.transpose(0, 1).to(weights.w_uk.dtype), weights.w_uk)
    q_abs = q_abs.transpose(0, 1).reshape(rows, -1).to(torch.float16).contiguous()
    absolute = torch.arange(start, start + n, device=device)
    relations = sorted({relation for _, _, relation in regions} | {positions.INTRA})
    index_of = {relation: index for index, relation in enumerate(relations)}
    q_pe = torch.stack([
        rope.apply(q_pe_raw, positions.query_positions(absolute, relation)).reshape(rows, -1)
        for relation in relations
    ]).to(torch.float16).contiguous()
    sm_scale = weights.softmax_scale
    options = {
        "sm_scale": sm_scale, "block_m": block_m, "block_n": block_n, "num_warps": num_warps,
        "num_stages": num_stages,
        "target_programs": target_programs,
    }

    lbm = torch.full((blocks, rows), float("-inf"), dtype=torch.float32, device=device)
    cached = [(a, min(b, length), r) for a, b, r in regions if a < length]
    extra = [(max(a, length), b, r) for a, b, r in regions if b > length]
    acc = acc_lse = None
    for t0, t1, tile_latent, tile_scale, tile_pe in _tiles(cache, layer, length, tile, dequantize=dequantize):
        ranges = [
            (max(a, t0) - t0, min(b, t1) - t0, index_of[r], max(a, t0))
            for a, b, r in cached if a < t1 and b > t0
        ]
        if ranges:
            out, lse = _partial(q_abs, q_pe, tile_latent, tile_scale, tile_pe, ranges, lbm, **options)
            acc, acc_lse = merge_partials(acc, acc_lse, out, lse)
    cached_lse = acc_lse
    # keys past the cached sequence: an earlier block of this same query
    for a, b, r in extra:
        tile_latent, tile_scale, tile_pe = cache.raw_tile(layer, a, b)
        out, lse = _partial(
            q_abs, q_pe, tile_latent, tile_scale, tile_pe, [(0, b - a, index_of[r], 0)], None, **options
        )
        acc, acc_lse = merge_partials(acc, acc_lse, out, lse)
    # the block's own keys, causal, in the same chunk
    own_scores = (q_abs.float().view(n, heads, -1) @ latent.float().t()) + (
        q_pe[index_of[positions.INTRA]].float().view(n, heads, -1) @ pe.float().t()
    )
    causal = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    own_scores = (own_scores * sm_scale).masked_fill(~causal.unsqueeze(1), float("-inf"))
    own_lse = torch.logsumexp(own_scores, dim=-1)
    own_out = torch.exp(own_scores - own_lse.unsqueeze(-1)) @ latent.float()
    acc, acc_lse = merge_partials(acc, acc_lse, own_out.reshape(rows, -1), own_lse.reshape(rows))
    attended = torch.bmm(
        acc.view(n, heads, -1).transpose(0, 1).to(weights.w_uv.dtype), weights.w_uv.transpose(1, 2)
    ).transpose(0, 1)
    if cached_lse is None or (keep_tokens is not None and not keep_tokens):
        return attended.to(q_nope.dtype), torch.zeros(blocks, dtype=torch.float32, device=device)
    if keep_tokens is not None:
        keep = torch.tensor(
            [token * heads + head for token in keep_tokens for head in range(heads)], dtype=torch.long, device=device
        )
        lbm, cached_lse = lbm.index_select(1, keep), cached_lse.index_select(0, keep)
    lbm.sub_(cached_lse.unsqueeze(0))
    mass = lbm.amax(dim=1).exp_() if reduce == "max" else lbm.exp_().sum(dim=1)
    return attended.to(q_nope.dtype), mass
