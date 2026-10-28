from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from deepinfer.moe.fused_moe import fused_moe_supported
from deepinfer.moe.gpu_quant import dequantize_matrix
from deepinfer.moe.grouped_moe import (
    fused_down_gemm_grouped,
    fused_gate_up_gemm_grouped,
)

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not fused_moe_supported():
        pytest.skip("grouped fused MoE requires CUDA and CuPy")


def _blocks(
    count: int,
    block_bytes: int,
    *,
    scale: float,
    min_scale: float | None = None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 256, size=(count, block_bytes), dtype=np.uint8)
    values[:, 0:2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
    if min_scale is not None:
        values[:, 2:4] = np.frombuffer(
            struct.pack("<e", min_scale), dtype=np.uint8
        )
    return values


def _q4_k_grouped(
    n_experts: int, out_features: int, k_size: int, seed: int
) -> torch.Tensor:
    blocks = k_size // 256
    rows = _blocks(
        n_experts * out_features * blocks,
        144,
        scale=1e-4,
        min_scale=5e-5,
        seed=seed,
    )
    return torch.from_numpy(
        rows.reshape(n_experts, out_features, blocks * 144)
    )


def _q5_q8_grouped(
    n_experts: int,
    out_features: int,
    k_size: int,
    block_bytes: int,
    seed: int,
) -> torch.Tensor:
    blocks = k_size // 32
    rows = _blocks(
        n_experts * out_features * blocks,
        block_bytes,
        scale=1e-3,
        seed=seed,
    )
    return torch.from_numpy(
        rows.reshape(n_experts, out_features, blocks * block_bytes)
    )


def test_grouped_gate_up_matches_per_expert_reference() -> None:
    _require_cuda()
    torch.manual_seed(3)
    n_experts = 3
    out_features = 128
    k_size = 2048
    max_count = 3
    counts = [1, 2, 3]
    gate_cpu = _q4_k_grouped(n_experts, out_features, k_size, seed=31)
    up_cpu = _q4_k_grouped(n_experts, out_features, k_size, seed=32)
    x = torch.randn(
        n_experts, max_count, k_size, device="cuda", dtype=torch.float16
    )
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device="cuda")
    actual = fused_gate_up_gemm_grouped(
        x,
        gate_cpu.reshape(n_experts, -1).cuda(),
        up_cpu.reshape(n_experts, -1).cuda(),
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
            gate_cpu[expert],
            "Q4_K",
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        up_weight = dequantize_matrix(
            up_cpu[expert],
            "Q4_K",
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        reference = (
            torch.nn.functional.silu(x[expert, :count].float() @ gate_weight.T)
            * (x[expert, :count].float() @ up_weight.T)
        ).half()
        torch.testing.assert_close(
            actual[expert, :count], reference, rtol=2e-2, atol=2e-2
        )


@pytest.mark.parametrize(("qtype", "block_bytes"), [("Q5_0", 22), ("Q8_0", 34)])
def test_grouped_down_matches_per_expert_reference(
    qtype: str, block_bytes: int
) -> None:
    _require_cuda()
    torch.manual_seed(4)
    n_experts = 3
    out_features = 128
    k_size = 1408
    max_count = 3
    counts = [3, 1, 2]
    down_cpu = _q5_q8_grouped(
        n_experts,
        out_features,
        k_size,
        block_bytes,
        seed=41 if qtype == "Q5_0" else 42,
    )
    x = torch.randn(
        n_experts, max_count, k_size, device="cuda", dtype=torch.float16
    )
    counts_gpu = torch.tensor(counts, dtype=torch.int32, device="cuda")
    actual = fused_down_gemm_grouped(
        x,
        down_cpu.reshape(n_experts, -1).cuda(),
        counts_gpu,
        n_experts=n_experts,
        max_count=max_count,
        out_features=out_features,
        k_size=k_size,
        down_qtype=qtype,
        down_row_bytes=k_size // 32 * block_bytes,
    )
    torch.cuda.synchronize()
    for expert, count in enumerate(counts):
        down_weight = dequantize_matrix(
            down_cpu[expert],
            qtype,
            out_features=out_features,
            in_features=k_size,
            dtype=torch.float32,
        ).cuda()
        reference = (x[expert, :count].float() @ down_weight.T).half()
        torch.testing.assert_close(
            actual[expert, :count], reference, rtol=2e-2, atol=2e-2
        )
