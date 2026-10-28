from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from deepinfer.moe.fused_moe import fused_moe_supported
from deepinfer.moe.gpu_quant import dequantize_matrix
from deepinfer.moe.ptr_grouped_moe import (
    fused_down_gemm_ptr_grouped,
    fused_gate_up_gemm_ptr_grouped,
)

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not fused_moe_supported():
        pytest.skip("pointer grouped MoE requires CUDA and CuPy")


def _blocks(count: int, block_bytes: int, *, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    values[:, 0:2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
    return values


def _q4_k(n: int, m: int, k: int, seed: int) -> torch.Tensor:
    blocks = k // 256
    values = _blocks(n * m * blocks, 144, scale=1e-4, seed=seed)
    values[:, 2:4] = np.frombuffer(struct.pack("<e", 5e-5), dtype=np.uint8)
    return torch.from_numpy(values.reshape(n, m, blocks * 144)).cuda()


def _q8_0(n: int, m: int, k: int, seed: int) -> torch.Tensor:
    blocks = k // 32
    values = _blocks(n * m * blocks, 34, scale=1e-3, seed=seed)
    return torch.from_numpy(values.reshape(n, m, blocks * 34)).cuda()


def test_pointer_grouped_gate_up_matches_reference() -> None:
    _require_cuda()
    torch.manual_seed(6)
    n_experts, out_features, k_size, max_count = 3, 64, 2048, 3
    counts = [1, 2, 3]
    gate = _q4_k(n_experts, out_features, k_size, 61)
    up = _q4_k(n_experts, out_features, k_size, 62)
    x = torch.randn(n_experts, max_count, k_size, device="cuda", dtype=torch.float16)
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device="cuda")
    gate_ptrs = torch.tensor(
        [gate[i].data_ptr() for i in range(n_experts)],
        dtype=torch.int64,
        device="cuda",
    )
    up_ptrs = torch.tensor(
        [up[i].data_ptr() for i in range(n_experts)],
        dtype=torch.int64,
        device="cuda",
    )
    actual = fused_gate_up_gemm_ptr_grouped(
        x,
        gate_ptrs,
        up_ptrs,
        counts_gpu,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        gate_qtype="Q4_K",
        up_qtype="Q4_K",
        gate_row_bytes=k_size // 256 * 144,
        up_row_bytes=k_size // 256 * 144,
    )
    torch.cuda.synchronize()
    for expert, count in enumerate(counts):
        gate_weight = dequantize_matrix(
            gate[expert].cpu(),
            "Q4_K",
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        up_weight = dequantize_matrix(
            up[expert].cpu(),
            "Q4_K",
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        reference = (
            torch.nn.functional.silu(x[expert, :count].float() @ gate_weight.T)
            * (x[expert, :count].float() @ up_weight.T)
        ).half()
        torch.testing.assert_close(actual[expert, :count], reference, rtol=2e-2, atol=2e-2)


def test_pointer_grouped_down_matches_reference() -> None:
    _require_cuda()
    torch.manual_seed(8)
    n_experts, out_features, k_size, max_count = 3, 64, 1408, 3
    counts = [3, 1, 2]
    down = _q8_0(n_experts, out_features, k_size, 81)
    x = torch.randn(n_experts, max_count, k_size, device="cuda", dtype=torch.float16)
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device="cuda")
    down_ptrs = torch.tensor(
        [down[i].data_ptr() for i in range(n_experts)],
        dtype=torch.int64,
        device="cuda",
    )
    actual = fused_down_gemm_ptr_grouped(
        x,
        down_ptrs,
        counts_gpu,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        down_qtype="Q8_0",
        down_row_bytes=k_size // 32 * 34,
    )
    torch.cuda.synchronize()
    for expert, count in enumerate(counts):
        down_weight = dequantize_matrix(
            down[expert].cpu(),
            "Q8_0",
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        reference = (x[expert, :count].float() @ down_weight.T).half()
        torch.testing.assert_close(actual[expert, :count], reference, rtol=2e-2, atol=2e-2)
