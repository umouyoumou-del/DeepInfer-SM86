"""Warp-per-row expert GEMVs against independent NumPy dequantisation.

The references decode whole blocks the way ggml's dequantize_row_q4_K /
dequantize_row_q6_K do, sharing nothing with the CUDA lane mapping, so a wrong
nibble, scale or quarter shows up as a numeric mismatch.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.packed_gemv import (  # noqa: E402
    decode_gemv_supported,
    down_gemv_combine,
    gate_up_gemv,
)

pytestmark = pytest.mark.gpu

_BLOCK = {"Q4_K": 144, "Q6_K": 210}


def _require() -> None:
    if not decode_gemv_supported("Q4_K", "Q6_K", k_size=2048, top_k=1):
        pytest.skip("needs CUDA and CuPy")


def _q4_k_blocks(blocks: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q4_K over ``[n, 144]`` blocks -> ``[n, 256]``."""

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc = blocks[:, 4:16].astype(np.int32)
    qs = blocks[:, 16:144].astype(np.int32)
    # get_scale_min_k4 for sub-blocks 0..7
    scale = np.concatenate([sc[:, 0:4] & 63, (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4)], axis=1)
    minimum = np.concatenate([sc[:, 4:8] & 63, (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)], axis=1)
    out = np.empty((len(blocks), 256), np.float32)
    for pair in range(4):
        q = qs[:, 32 * pair:32 * pair + 32]
        lo, hi = 2 * pair, 2 * pair + 1
        out[:, 64 * pair:64 * pair + 32] = d * scale[:, lo:lo + 1] * (q & 0xF) - dmin * minimum[:, lo:lo + 1]
        out[:, 64 * pair + 32:64 * pair + 64] = d * scale[:, hi:hi + 1] * (q >> 4) - dmin * minimum[:, hi:hi + 1]
    return out


def _q6_k_blocks(blocks: np.ndarray) -> np.ndarray:
    """ggml dequantize_row_q6_K over ``[n, 210]`` blocks -> ``[n, 256]``."""

    ql = blocks[:, 0:128].astype(np.int32)
    qh = blocks[:, 128:192].astype(np.int32)
    scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)
    out = np.empty((len(blocks), 256), np.float32)
    l = np.arange(32)
    is_ = l // 16
    for half in (0, 1):
        qlb, qhb = ql[:, half * 64:half * 64 + 64], qh[:, half * 32:half * 32 + 32]
        scb, base = scales[:, half * 8:half * 8 + 8], half * 128
        out[:, base + l] = d * scb[:, is_] * (((qlb[:, l] & 0xF) | ((qhb[:, l] & 3) << 4)) - 32)
        out[:, base + 32 + l] = d * scb[:, is_ + 2] * (((qlb[:, l + 32] & 0xF) | (((qhb[:, l] >> 2) & 3) << 4)) - 32)
        out[:, base + 64 + l] = d * scb[:, is_ + 4] * (((qlb[:, l] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)) - 32)
        out[:, base + 96 + l] = d * scb[:, is_ + 6] * (((qlb[:, l + 32] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)) - 32)
    return out


def _experts(qtype: str, experts: int, rows: int, k: int, seed: int, *, reference: bool = True):
    """Random packed experts, their pointer table and the float64 reference weights."""

    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, size=(experts * rows * (k // 256), _BLOCK[qtype]), dtype=np.uint8)
    if qtype == "Q4_K":
        blocks[:, 0:2] = np.frombuffer(struct.pack("<e", 1e-4), dtype=np.uint8)
        blocks[:, 2:4] = np.frombuffer(struct.pack("<e", 5e-5), dtype=np.uint8)
        decode = _q4_k_blocks
    else:
        blocks[:, 208:210] = np.frombuffer(struct.pack("<e", 1e-4), dtype=np.uint8)
        decode = _q6_k_blocks
    if reference:
        reference = decode(blocks).reshape(experts, rows, k).astype(np.float64)
    storage = torch.from_numpy(blocks.reshape(experts, rows, -1).copy()).cuda()
    stride = storage[0].numel()
    table = torch.tensor([storage.data_ptr() + e * stride for e in range(experts)],
                         dtype=torch.int64, device="cuda")
    return storage, table, reference


def _row(k: int, qtype: str) -> int:
    return k // 256 * _BLOCK[qtype]


@pytest.mark.parametrize("gate_qtype,up_qtype", [("Q4_K", "Q4_K"), ("Q6_K", "Q4_K"), ("Q4_K", "Q6_K")])
def test_gate_up_matches_reference(gate_qtype: str, up_qtype: str) -> None:
    _require()
    k, rows = 2048, 64
    _gs, gate_table, gate_ref = _experts(gate_qtype, 6, rows, k, 1)
    _us, up_table, up_ref = _experts(up_qtype, 6, rows, k, 2)
    x = torch.randn(k, generator=torch.Generator().manual_seed(3)).half().cuda()
    ids = torch.tensor([4, 1, 5], dtype=torch.int64, device="cuda")
    got = gate_up_gemv(x, ids, gate_table, up_table, out_features=rows, k_size=k,
                       gate_qtype=gate_qtype, up_qtype=up_qtype,
                       gate_row_bytes=_row(k, gate_qtype), up_row_bytes=_row(k, up_qtype))
    xr = x.double().cpu().numpy()
    for slot, expert in enumerate(ids.tolist()):
        gate = gate_ref[expert] @ xr
        want = gate / (1 + np.exp(-gate)) * (up_ref[expert] @ xr)
        np.testing.assert_allclose(got[slot].double().cpu().numpy(), want,
                                   rtol=2e-3, atol=2e-3 * np.abs(want).max())


@pytest.mark.parametrize("qtype", ["Q4_K", "Q6_K"])
def test_down_combine_matches_reference(qtype: str) -> None:
    _require()
    k, rows = 1024, 128
    _ds, table, ref = _experts(qtype, 6, rows, k, 4)
    generator = torch.Generator().manual_seed(5)
    act = torch.randn(3, k, generator=generator).half().cuda()
    weights = torch.rand(3, generator=generator).cuda()
    ids = torch.tensor([2, 0, 5], dtype=torch.int64, device="cuda")
    got = down_gemv_combine(act, ids, weights, table, out_features=rows, k_size=k,
                            down_qtype=qtype, down_row_bytes=_row(k, qtype))
    a, w = act.double().cpu().numpy(), weights.double().cpu().numpy()
    want = sum(w[s] * (ref[e] @ a[s]) for s, e in enumerate(ids.tolist()))
    np.testing.assert_allclose(got.double().cpu().numpy(), want,
                               rtol=2e-3, atol=2e-3 * np.abs(want).max())


def test_moe_forward_packed_matches_pointer_grouped_path() -> None:
    """The new kernels must agree with the per-element kernels they replace."""

    _require()
    from deepinfer.moe.olmoe_packed import moe_forward_packed
    from deepinfer.moe.olmoe_router import route_topk
    from deepinfer.moe.ptr_grouped_moe import (
        fused_down_gemm_ptr_grouped,
        fused_gate_up_gemm_ptr_grouped,
    )

    hidden, inter, experts, top_k = 2048, 1024, 8, 4
    gate = _experts("Q4_K", experts, inter, hidden, 7, reference=False)[:2]
    up = _experts("Q4_K", experts, inter, hidden, 8, reference=False)[:2]
    down = _experts("Q6_K", experts, hidden, inter, 9, reference=False)[:2]
    packed = {"gate": (*gate, "Q4_K"), "up": (*up, "Q4_K"), "down": (*down, "Q6_K")}
    generator = torch.Generator().manual_seed(10)
    x = (torch.randn(1, hidden, generator=generator) * 0.5).half().cuda()
    router = torch.randn(hidden, experts, generator=generator).cuda()

    got = moe_forward_packed(x, router, packed, top_k=top_k, gate_out=inter,
                             down_out=hidden, gate_k=hidden, down_k=inter)

    ids, weights = route_topk(x, router, top_k=top_k)
    counts = torch.ones(top_k, dtype=torch.int32, device="cuda")
    act = fused_gate_up_gemm_ptr_grouped(
        x.repeat(top_k, 1).unsqueeze(1), gate[1][ids[0]], up[1][ids[0]], counts,
        n_experts=top_k, max_count=1, out_features=inter, k_size=hidden,
        gate_qtype="Q4_K", up_qtype="Q4_K",
        gate_row_bytes=_row(hidden, "Q4_K"), up_row_bytes=_row(hidden, "Q4_K"))
    out = fused_down_gemm_ptr_grouped(
        act, down[1][ids[0]], counts, n_experts=top_k, max_count=1, out_features=hidden,
        k_size=inter, down_qtype="Q6_K", down_row_bytes=_row(inter, "Q6_K"))
    want = (out.reshape(top_k, hidden) * weights[0].unsqueeze(-1)).sum(0, keepdim=True)
    torch.testing.assert_close(got.float(), want.float(), rtol=1e-2,
                               atol=1e-2 * float(want.abs().max()))
