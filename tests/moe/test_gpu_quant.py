from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from deepinfer.moe.gpu_quant import dequantize_matrix

pytestmark = pytest.mark.gpu


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused GGUF dequantization tests")
    pytest.importorskip("cupy")


def _pack_half(value: float) -> bytes:
    return struct.pack("<e", value)


@pytest.mark.parametrize(
    ("quant_type", "block_bytes", "block_values"),
    [("Q4_K", 144, 256), ("Q5_0", 22, 32), ("Q8_0", 34, 32)],
)
def test_cuda_dequant_matches_gguf_reference(
    quant_type: str, block_bytes: int, block_values: int
) -> None:
    _require_cuda()
    from gguf import GGMLQuantizationType, dequantize

    rng = np.random.default_rng(123)
    block_count = 64
    blocks = rng.integers(0, 256, size=(block_count, block_bytes), dtype=np.uint8)
    blocks[:, 0:2] = np.frombuffer(_pack_half(0.01), dtype=np.uint8)
    if quant_type in ("Q4_K",):
        blocks[:, 2:4] = np.frombuffer(_pack_half(0.001), dtype=np.uint8)
    reference = dequantize(
        blocks,
        {
            "Q4_K": GGMLQuantizationType.Q4_K,
            "Q5_0": GGMLQuantizationType.Q5_0,
            "Q8_0": GGMLQuantizationType.Q8_0,
        }[quant_type],
    )
    packed = torch.from_numpy(blocks.copy()).cuda()
    dequantized = dequantize_matrix(
        packed,
        quant_type,
        out_features=block_count,
        in_features=block_values,
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    assert dequantized.shape == (block_count, block_values)
    np.testing.assert_allclose(
        dequantized.float().cpu().numpy(),
        reference.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    repeated = dequantize_matrix(
        packed,
        quant_type,
        out_features=block_count,
        in_features=block_values,
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    assert torch.equal(dequantized, repeated)
