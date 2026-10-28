from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from deepinfer.moe import cpu_avx2
from deepinfer.moe.gpu_quant import (
    dequantize_q4_k,
    dequantize_q5_0,
    dequantize_q8_0,
)


def _packed(rows: int, blocks: int, block_bytes: int, scale: float) -> torch.Tensor:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 256, size=(rows * blocks, block_bytes), dtype=np.uint8)
    values[:, 0:2] = np.frombuffer(struct.pack("<e", scale), dtype=np.uint8)
    if block_bytes == 144:
        values[:, 2:4] = np.frombuffer(
            struct.pack("<e", scale / 2), dtype=np.uint8
        )
    return torch.from_numpy(values.reshape(rows, blocks * block_bytes))


@pytest.mark.parametrize(
    ("quant_type", "block_size", "block_bytes", "reference"),
    [
        ("Q4_K", 256, 144, dequantize_q4_k),
        ("Q5_0", 32, 22, dequantize_q5_0),
        ("Q8_0", 32, 34, dequantize_q8_0),
    ],
)
def test_cpu_avx2_dequant_matches_reference(
    quant_type: str,
    block_size: int,
    block_bytes: int,
    reference,
) -> None:
    if not cpu_avx2.available():
        pytest.skip("g++ AVX2 compiler is unavailable")
    rows = 3
    blocks = 2
    in_features = block_size * blocks
    packed = _packed(rows, blocks, block_bytes, 0.03125)
    actual = cpu_avx2.dequantize(
        packed, quant_type, out_features=rows, in_features=in_features
    )
    expected = reference(
        packed,
        out_features=rows,
        in_features=in_features,
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("down_quant_type", ["Q5_0", "Q8_0"])
def test_cpu_avx2_fused_expert_matches_dense(down_quant_type: str) -> None:
    if not cpu_avx2.available():
        pytest.skip("g++ AVX2 compiler is unavailable")
    rows, in_features, tokens = 32, 256, 2
    down_rows, down_in_features = 4, 32
    gate = _packed(rows, 1, 144, 1e-3)
    up = _packed(rows, 1, 144, 1e-3)
    block_bytes = 22 if down_quant_type == "Q5_0" else 34
    down = _packed(down_rows, 1, block_bytes, 1e-3)
    x = torch.randn(tokens, in_features)
    actual = cpu_avx2.fused_expert_chunk(
        gate,
        up,
        down,
        x,
        gate_rows=rows,
        gate_in_features=in_features,
        down_rows=down_rows,
        down_in_features=down_in_features,
        down_quant_type=down_quant_type,
        tokens=tokens,
    )
    gate_weight = dequantize_q4_k(
        gate, out_features=rows, in_features=in_features, dtype=torch.float32
    )
    up_weight = dequantize_q4_k(
        up, out_features=rows, in_features=in_features, dtype=torch.float32
    )
    down_reference = (
        dequantize_q5_0
        if down_quant_type == "Q5_0"
        else dequantize_q8_0
    )
    down_weight = down_reference(
        down,
        out_features=down_rows,
        in_features=down_in_features,
        dtype=torch.float32,
    )
    hidden = torch.nn.functional.silu(x @ gate_weight.T) * (x @ up_weight.T)
    expected = hidden @ down_weight.T
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
