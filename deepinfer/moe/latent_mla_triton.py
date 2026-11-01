"""Fused absorbed-MLA decode attention over the latent cache (Triton).

One query token, 16 heads, all sharing the cached latent: per key the kernel
reads 512 latent values (fp16, or fp8 plus four fp32 scales) and 64 rotary
values once and uses them for every head.  Scores are ``q_abs . c + q_pe . k_pe``
(both dots on tensor cores, M = 16 heads) and the value is the latent itself,
so the output is the attention-weighted latent; the caller applies W_UV.

The key range is split across programs (flash-decoding); each writes a partial
(accumulator, running max, running sum) and :func:`merge_splits` combines them.
A split never crosses a position-relation boundary, so dual chunk attention
just selects which rotated query the split uses.
"""

from __future__ import annotations

import torch

try:  # Triton is optional; callers fall back to the PyTorch reference
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  # pragma: no cover - depends on the platform
    triton = None
    tl = None


def available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _decode_kernel(
        Q,
        QP,
        LAT,
        SC,
        PE,
        SPLIT_START,
        SPLIT_END,
        SPLIT_REL,
        O_ACC,
        O_M,
        O_L,
        sm_scale,
        H: tl.constexpr,
        D: tl.constexpr,
        DR: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_FP8: tl.constexpr,
    ):
        split = tl.program_id(0)
        start = tl.load(SPLIT_START + split)
        end = tl.load(SPLIT_END + split)
        relation = tl.load(SPLIT_REL + split)
        offs_h = tl.arange(0, H)
        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, DR)
        q = tl.load(Q + offs_h[:, None] * D + offs_d[None, :])
        qp = tl.load(QP + relation * H * DR + offs_h[:, None] * DR + offs_r[None, :])
        m_i = tl.full([H], float("-inf"), tl.float32)
        l_i = tl.zeros([H], tl.float32)
        acc = tl.zeros([H, D], tl.float32)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            valid = offs_n < end
            k = tl.load(
                LAT + offs_n[:, None].to(tl.int64) * D + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            if IS_FP8:
                scale = tl.load(
                    SC + offs_n[:, None].to(tl.int64) * (D // 128) + offs_d[None, :] // 128,
                    mask=valid[:, None],
                    other=0.0,
                )
                k = (k.to(tl.float32) * scale).to(tl.float16)
            kp = tl.load(
                PE + offs_n[:, None].to(tl.int64) * DR + offs_r[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) + tl.dot(qp, tl.trans(kp))
            s = tl.where(valid[None, :], s * sm_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), k)
            m_i = m_new
        base = split.to(tl.int64) * H
        tl.store(O_ACC + (base + offs_h[:, None]) * D + offs_d[None, :], acc)
        tl.store(O_M + base + offs_h, m_i)
        tl.store(O_L + base + offs_h, l_i)

    @triton.jit
    def _decode_kernel_fp8(
        Q,
        QP,
        LAT,
        SC,
        PE,
        SPLIT_START,
        SPLIT_END,
        SPLIT_REL,
        O_ACC,
        O_M,
        O_L,
        sm_scale,
        H: tl.constexpr,
        DR: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """fp8 latent, 512 = 4 groups of 128 with one scale each.

        The scale multiplies the group's partial *score* (and the probabilities
        that weight the group's values) instead of each of its 128 elements, so
        it is one load per key and group rather than a gather per element.
        """

        split = tl.program_id(0)
        start = tl.load(SPLIT_START + split)
        end = tl.load(SPLIT_END + split)
        relation = tl.load(SPLIT_REL + split)
        offs_h = tl.arange(0, H)
        offs_c = tl.arange(0, 128)
        offs_r = tl.arange(0, DR)
        q_row = Q + offs_h[:, None] * 512 + offs_c[None, :]
        q0 = tl.load(q_row)
        q1 = tl.load(q_row + 128)
        q2 = tl.load(q_row + 256)
        q3 = tl.load(q_row + 384)
        qp = tl.load(QP + relation * H * DR + offs_h[:, None] * DR + offs_r[None, :])
        m_i = tl.full([H], float("-inf"), tl.float32)
        l_i = tl.zeros([H], tl.float32)
        a0 = tl.zeros([H, 128], tl.float32)
        a1 = tl.zeros([H, 128], tl.float32)
        a2 = tl.zeros([H, 128], tl.float32)
        a3 = tl.zeros([H, 128], tl.float32)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            valid = offs_n < end
            rows = offs_n.to(tl.int64)
            k_row = LAT + rows[:, None] * 512 + offs_c[None, :]
            k0 = tl.load(k_row, mask=valid[:, None], other=0.0).to(tl.float16)
            k1 = tl.load(k_row + 128, mask=valid[:, None], other=0.0).to(tl.float16)
            k2 = tl.load(k_row + 256, mask=valid[:, None], other=0.0).to(tl.float16)
            k3 = tl.load(k_row + 384, mask=valid[:, None], other=0.0).to(tl.float16)
            s0 = tl.load(SC + rows * 4, mask=valid, other=0.0)
            s1 = tl.load(SC + rows * 4 + 1, mask=valid, other=0.0)
            s2 = tl.load(SC + rows * 4 + 2, mask=valid, other=0.0)
            s3 = tl.load(SC + rows * 4 + 3, mask=valid, other=0.0)
            kp = tl.load(PE + rows[:, None] * DR + offs_r[None, :], mask=valid[:, None], other=0.0)
            s = tl.dot(q0, tl.trans(k0)) * s0[None, :]
            s += tl.dot(q1, tl.trans(k1)) * s1[None, :]
            s += tl.dot(q2, tl.trans(k2)) * s2[None, :]
            s += tl.dot(q3, tl.trans(k3)) * s3[None, :]
            s += tl.dot(qp, tl.trans(kp))
            s = tl.where(valid[None, :], s * sm_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            a0 = a0 * alpha[:, None] + tl.dot((p * s0[None, :]).to(tl.float16), k0)
            a1 = a1 * alpha[:, None] + tl.dot((p * s1[None, :]).to(tl.float16), k1)
            a2 = a2 * alpha[:, None] + tl.dot((p * s2[None, :]).to(tl.float16), k2)
            a3 = a3 * alpha[:, None] + tl.dot((p * s3[None, :]).to(tl.float16), k3)
            m_i = m_new
        base = split.to(tl.int64) * H
        out = O_ACC + (base + offs_h[:, None]) * 512 + offs_c[None, :]
        tl.store(out, a0)
        tl.store(out + 128, a1)
        tl.store(out + 256, a2)
        tl.store(out + 384, a3)
        tl.store(O_M + base + offs_h, m_i)
        tl.store(O_L + base + offs_h, l_i)


def plan_splits(
    segments: list[tuple[int, int, int]], *, target_programs: int = 320, minimum: int = 256
) -> tuple[list[int], list[int], list[int]]:
    """Cut ``(start, end, relation)`` key ranges into kernel work items."""

    total = sum(end - start for start, end, _ in segments)
    size = max(minimum, -(-total // max(1, target_programs)))
    size = -(-size // 64) * 64
    starts, ends, relations = [], [], []
    for start, end, relation in segments:
        for lo in range(start, end, size):
            starts.append(lo)
            ends.append(min(lo + size, end))
            relations.append(relation)
    return starts, ends, relations


_SPLIT_TABLES: dict[tuple, torch.Tensor] = {}


def _split_table(segments: tuple, device) -> torch.Tensor:
    """``[3, splits]`` int32 (start, end, relation) on the device, memoized.

    Every layer of a decode step attends to the same segments; building and
    uploading the table once per step instead of once per layer saves 26 host
    round trips per token.
    """

    key = (segments, str(device))
    table = _SPLIT_TABLES.get(key)
    if table is None:
        starts, ends, relations = plan_splits(list(segments))
        table = torch.tensor([starts, ends, relations], dtype=torch.int32, device=device)
        if len(_SPLIT_TABLES) > 64:
            _SPLIT_TABLES.clear()
        _SPLIT_TABLES[key] = table
    return table


def merge_splits(acc: torch.Tensor, m: torch.Tensor, l: torch.Tensor):
    """``acc [S,H,D]``, ``m/l [S,H]`` -> normalized ``[H,D]`` and log-sum-exp ``[H]``."""

    top = m.max(dim=0).values
    weight = torch.exp(m - top)
    total = (l * weight).sum(dim=0)
    out = (acc * weight.unsqueeze(-1)).sum(dim=0) / total.unsqueeze(-1)
    return out, top + torch.log(total)


def decode_partial(
    q_abs: torch.Tensor,
    q_pe: torch.Tensor,
    latent: torch.Tensor,
    scale: torch.Tensor | None,
    pe: torch.Tensor,
    segments: list[tuple[int, int, int]],
    *,
    sm_scale: float,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    grouped_fp8: bool = True,
):
    """Attention of one query over ``segments`` of a latent array.

    ``q_abs [H, 512]`` fp16, ``q_pe [R, H, 64]`` fp16 (one per relation),
    ``latent [N, 512]`` fp16 or float8_e4m3fn, ``scale [N, 4]`` fp32 for fp8,
    ``pe [N, 64]`` fp16; segment offsets index rows of these arrays.  Returns
    the normalized ``[H, 512]`` fp32 latent output and its log-sum-exp ``[H]``.
    """

    if not segments:
        raise ValueError("no keys to attend to")
    heads, width = q_abs.shape
    rope = q_pe.shape[-1]
    padded = max(16, 1 << (heads - 1).bit_length())  # tl.dot needs M >= 16
    if padded != heads:
        q_abs = torch.nn.functional.pad(q_abs, (0, 0, 0, padded - heads))
        q_pe = torch.nn.functional.pad(q_pe, (0, 0, 0, padded - heads))
        out, lse = decode_partial(
            q_abs, q_pe, latent, scale, pe, segments, sm_scale=sm_scale,
            block_n=block_n, num_warps=num_warps, num_stages=num_stages,
            grouped_fp8=grouped_fp8,
        )
        return out[:heads], lse[:heads]
    device = q_abs.device
    meta = _split_table(tuple(segments), device)
    count = meta.shape[1]
    acc = torch.empty((count, heads, width), dtype=torch.float32, device=device)
    m = torch.empty((count, heads), dtype=torch.float32, device=device)
    l = torch.empty((count, heads), dtype=torch.float32, device=device)
    is_fp8 = latent.dtype == torch.float8_e4m3fn
    if is_fp8 and width == 512 and scale is not None and grouped_fp8:
        _decode_kernel_fp8[(count,)](
            q_abs.contiguous(), q_pe.contiguous(), latent, scale, pe,
            meta[0], meta[1], meta[2], acc, m, l, float(sm_scale),
            H=heads, DR=rope, BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages,
        )
        return merge_splits(acc, m, l)
    _decode_kernel[(count,)](
        q_abs.contiguous(),
        q_pe.contiguous(),
        latent,
        scale if is_fp8 else latent,
        pe,
        meta[0],
        meta[1],
        meta[2],
        acc,
        m,
        l,
        float(sm_scale),
        H=heads,
        D=width,
        DR=rope,
        BLOCK_N=block_n,
        IS_FP8=is_fp8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return merge_splits(acc, m, l)
