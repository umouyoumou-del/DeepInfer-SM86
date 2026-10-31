"""dp_q6_k_value must agree with a reference Q6_K block dequant."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cp = pytest.importorskip("cupy")

from deepinfer.moe.fused_moe import _FUSED_MOE_SOURCE, _QTYPE_CODES  # noqa: E402

_BLOCK = 210
_QK = 256


def _reference_block(block: np.ndarray) -> np.ndarray:
    ql = block[0:128].astype(np.int32)
    qh = block[128:192].astype(np.int32)
    scales = np.frombuffer(block[192:208].tobytes(), dtype=np.int8).astype(np.float32)
    d = np.frombuffer(block[208:210].tobytes(), dtype=np.float16).astype(np.float32)[0]
    out = np.zeros(_QK, dtype=np.float32)
    for half in (0, 1):
        qlb = ql[half * 64 : half * 64 + 64]
        qhb = qh[half * 32 : half * 32 + 32]
        scb = scales[half * 8 : half * 8 + 8]
        for l in range(32):
            is_ = l // 16
            q1 = ((qlb[l] & 0xF) | (((qhb[l] >> 0) & 3) << 4)) - 32
            q2 = ((qlb[l + 32] & 0xF) | (((qhb[l] >> 2) & 3) << 4)) - 32
            q3 = ((qlb[l] >> 4) | (((qhb[l] >> 4) & 3) << 4)) - 32
            q4 = ((qlb[l + 32] >> 4) | (((qhb[l] >> 6) & 3) << 4)) - 32
            base = half * 128
            out[base + l + 0] = d * scb[is_ + 0] * q1
            out[base + l + 32] = d * scb[is_ + 2] * q2
            out[base + l + 64] = d * scb[is_ + 4] * q3
            out[base + l + 96] = d * scb[is_ + 6] * q4
    return out


def _probe_kernel():
    source = _FUSED_MOE_SOURCE + r"""
extern "C" __global__ void dp_probe_q6_k(
    const unsigned char* row, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = dp_q6_k_value(row, i); }
}
"""
    return cp.RawKernel(source, "dp_probe_q6_k", options=("--std=c++14",))


def test_q6_k_registered() -> None:
    assert _QTYPE_CODES["Q6_K"] == 3


def test_q6_k_matches_reference() -> None:
    rng = np.random.default_rng(0)
    blocks = 3
    raw = rng.integers(0, 256, size=(blocks, _BLOCK), dtype=np.uint8)
    for index in range(blocks):
        d = np.float16(rng.uniform(0.001, 0.02))
        raw[index, 208:210] = np.frombuffer(d.tobytes(), dtype=np.uint8)

    expected = np.concatenate([_reference_block(raw[i]) for i in range(blocks)])

    kernel = _probe_kernel()
    flat = np.ascontiguousarray(raw).reshape(-1)
    device_in = cp.asarray(flat)
    device_out = cp.empty(expected.size, dtype=cp.float32)
    total = int(expected.size)
    kernel(((total + 255) // 256,), (256,), (device_in, device_out, np.int32(total)))
    got = cp.asnumpy(device_out)

    assert np.allclose(got, expected, rtol=1e-3, atol=1e-4), np.abs(got - expected).max()
