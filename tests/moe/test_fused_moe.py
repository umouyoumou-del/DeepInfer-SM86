from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from deepinfer.moe.fused_moe import (
    fused_down_gemm,
    fused_gate_up_gemm,
    fused_moe_supported,
)
from deepinfer.moe.gpu_quant import dequantize_matrix

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not fused_moe_supported():
        pytest.skip("fused MoE requires CUDA and CuPy")


def _packed_blocks(
    block_count: int,
    block_bytes: int,
    *,
    scale: float,
    min_scale: float | None = None,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(
        0, 256, size=(block_count, block_bytes), dtype=np.uint8
    )
    values[:, 0:2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
    if min_scale is not None:
        values[:, 2:4] = np.frombuffer(
            struct.pack("<e", min_scale), dtype=np.uint8
        )
    return values


def _q4_k_matrix(out_features: int, k_size: int, *, seed: int) -> torch.Tensor:
    k_blocks = k_size // 256
    rows = _packed_blocks(
        out_features * k_blocks,
        144,
        scale=1e-4,
        min_scale=5e-5,
        seed=seed,
    ).reshape(out_features, k_blocks * 144)
    return torch.from_numpy(rows)


def _q5_q8_matrix(
    out_features: int, k_size: int, block_bytes: int, *, seed: int
) -> torch.Tensor:
    k_blocks = k_size // 32
    rows = _packed_blocks(
        out_features * k_blocks,
        block_bytes,
        scale=1e-3,
        seed=seed,
    ).reshape(out_features, k_blocks * block_bytes)
    return torch.from_numpy(rows)


@pytest.mark.parametrize("n_tokens", [1, 3])
def test_fused_gate_up_gemm_matches_dequantized_reference(n_tokens: int) -> None:
    _require_cuda()
    torch.manual_seed(7)
    out_features = 128
    k_size = 2048
    gate_cpu = _q4_k_matrix(out_features, k_size, seed=11)
    up_cpu = _q4_k_matrix(out_features, k_size, seed=12)
    gate_gpu = gate_cpu.cuda()
    up_gpu = up_cpu.cuda()
    x = torch.randn(n_tokens, k_size, device="cuda", dtype=torch.float16)

    gate_weight = dequantize_matrix(
        gate_cpu,
        "Q4_K",
        out_features=out_features,
        in_features=k_size,
        dtype=torch.float32,
    ).cuda()
    up_weight = dequantize_matrix(
        up_cpu,
        "Q4_K",
        out_features=out_features,
        in_features=k_size,
        dtype=torch.float32,
    ).cuda()
    reference = (
        torch.nn.functional.silu(x.float() @ gate_weight.T)
        * (x.float() @ up_weight.T)
    ).half()
    actual = fused_gate_up_gemm(
        x,
        gate_gpu,
        up_gpu,
        n_tokens=n_tokens,
        out_features=out_features,
        k_size=k_size,
        gate_qtype="Q4_K",
        up_qtype="Q4_K",
        gate_row_bytes=k_size // 256 * 144,
        up_row_bytes=k_size // 256 * 144,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(("qtype", "block_bytes"), [("Q5_0", 22), ("Q8_0", 34)])
def test_fused_down_gemm_matches_dequantized_reference(
    qtype: str, block_bytes: int
) -> None:
    _require_cuda()
    torch.manual_seed(9)
    out_features = 128
    k_size = 1408
    n_tokens = 4
    down_cpu = _q5_q8_matrix(
        out_features, k_size, block_bytes, seed=21 if qtype == "Q5_0" else 22
    )
    down_gpu = down_cpu.cuda()
    x = torch.randn(n_tokens, k_size, device="cuda", dtype=torch.float16)
    down_weight = dequantize_matrix(
        down_cpu,
        qtype,
        out_features=out_features,
        in_features=k_size,
        dtype=torch.float32,
    ).cuda()
    reference = (x.float() @ down_weight.T).half()
    actual = fused_down_gemm(
        x,
        down_gpu,
        n_tokens=n_tokens,
        out_features=out_features,
        k_size=k_size,
        down_qtype=qtype,
        down_row_bytes=k_size // 32 * block_bytes,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
