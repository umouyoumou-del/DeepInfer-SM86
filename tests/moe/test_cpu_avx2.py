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
