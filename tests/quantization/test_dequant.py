from __future__ import annotations

import json

import pytest
import torch

from deepinfer.quantization.dequant import dequantize_awq, dequantize_gptq
from deepinfer.quantization.formats import (
    UnsupportedQuantFormatError,
    detect_source_format,
)
from deepinfer.quantization.pack import (
    pack_symmetric_int4,
    unpack_symmetric_int4,
)
from tests.quantization.helpers import pack_awq_weight, pack_gptq_weight


def test_gptq_dequant_with_act_order() -> None:
    torch.manual_seed(11)
    weight = torch.randn(16, 32)
    # A non-contiguous group assignment exercises GPTQ desc_act/g_idx handling.
    g_idx = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3] * 4,
        dtype=torch.long,
    )
    qweight, scales, qzeros, packed_g_idx = pack_gptq_weight(
        weight, group_size=8, g_idx=g_idx
    )
    actual = dequantize_gptq(
        qweight,
        scales,
        qzeros,
        bits=4,
        group_size=8,
        g_idx=packed_g_idx,
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), weight.flatten(), dim=0
    )
    assert cosine.item() > 0.99
    assert (actual - weight).abs().mean().item() < 0.1


def test_awq_gemm_dequant_ordering() -> None:
    torch.manual_seed(12)
    weight = torch.randn(16, 32)
    qweight, scales, qzeros = pack_awq_weight(weight, group_size=8)
    actual = dequantize_awq(
        qweight,
        scales,
        qzeros,
        bits=4,
        group_size=8,
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), weight.flatten(), dim=0
    )
    assert cosine.item() > 0.99
    assert (actual - weight).abs().mean().item() < 0.1


def test_native_w4a16_pack_roundtrip() -> None:
    torch.manual_seed(13)
    weight = torch.randn(8, 64)
    packed, scales = pack_symmetric_int4(weight, group_size=32)
    actual = unpack_symmetric_int4(packed, scales, group_size=32)
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten(), weight.flatten(), dim=0
    )
    assert cosine.item() > 0.99


def test_format_detection_rejects_unsupported_awq_layout(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "quantization_config": {
                    "quant_method": "awq",
                    "version": "exllama",
                    "bits": 4,
                    "group_size": 128,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedQuantFormatError, match="AWQ layout"):
        detect_source_format(tmp_path)
