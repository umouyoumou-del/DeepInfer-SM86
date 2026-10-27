"""GPU-only correctness test for the SM86 W4A16 Qwen2 linear backend."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_w4a16_linear_matches_dense_prefill_and_decode() -> None:
    pytest.importorskip("cupy")
    from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear

    torch.manual_seed(20261025)
    in_features = 256
    out_features = 128
    dense = torch.nn.Linear(
        in_features, out_features, bias=True, device="cuda", dtype=torch.float16
    )
    quantized = W4A16Linear(
        in_features,
        out_features,
        True,
        group_size=128,
        dtype=torch.float16,
        device="cuda",
    )
    quantized.load_from_linear(dense)

    with torch.no_grad():
        decode_input = torch.randn(
            1, 1, in_features, device="cuda", dtype=torch.float16
        )
        prefill_input = torch.randn(
            1, 7, in_features, device="cuda", dtype=torch.float16
        )
        decode_ref = dense(decode_input)
        prefill_ref = dense(prefill_input)
        decode_out = quantized(decode_input)
        prefill_out = quantized(prefill_input)

    for reference, actual in ((decode_ref, decode_out), (prefill_ref, prefill_out)):
        cosine = torch.nn.functional.cosine_similarity(
            reference.flatten().float(),
            actual.flatten().float(),
            dim=0,
        )
        assert cosine.item() > 0.985
        assert torch.mean(torch.abs(reference.float() - actual.float())).item() < 0.05


def test_w4a16_tiled_prefill_matches_dequantized_reference() -> None:
    pytest.importorskip("cupy")
    from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear

    torch.manual_seed(20261102)
    in_features = 256
    out_features = 128
    dense = torch.nn.Linear(
        in_features, out_features, bias=True, device="cuda", dtype=torch.float16
    )
    quantized = W4A16Linear(
        in_features,
        out_features,
        True,
        group_size=128,
        dtype=torch.float16,
        device="cuda",
    )
    quantized.load_from_linear(dense)
    inputs = torch.randn(37, in_features, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        actual = quantized(inputs)
        reference = torch.nn.functional.linear(
            inputs,
            quantized._dequantize().to(torch.float16),
            quantized.bias,
        )

    assert quantized.last_prefill_backend == "tiled_w4a16_gemm"
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten().float(),
        actual.flatten().float(),
        dim=0,
    )
    assert cosine.item() > 0.9999
    assert (reference - actual).abs().mean().item() < 0.001


def test_w4a16_batched_gemv_matches_single_row_decode() -> None:
    pytest.importorskip("cupy")
    from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear

    torch.manual_seed(20261103)
    quantized = W4A16Linear(
        256,
        128,
        True,
        group_size=128,
        dtype=torch.float16,
        device="cuda",
    )
    source = torch.nn.Linear(
        256, 128, bias=True, device="cuda", dtype=torch.float16
    )
    quantized.load_from_linear(source)
    inputs = torch.randn(4, 256, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        actual = quantized(inputs)
        reference = torch.cat(
            [quantized._forward_gemv(inputs[row : row + 1]) for row in range(4)],
            dim=0,
        )

    assert quantized.last_prefill_backend == "w4a16_batched_gemv"
    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)


def test_w3a16_tiled_prefill_matches_dequantized_reference() -> None:
    pytest.importorskip("cupy")
    from deepinfer.model_executor.models.qwen2_mixed_sm86 import W3A16Linear

    torch.manual_seed(20261104)
    in_features = 256
    out_features = 128
    group_size = 128
    row_bytes = in_features * 3 // 8
    quantized = W3A16Linear(
        in_features,
        out_features,
        True,
        group_size=group_size,
        dtype=torch.float16,
        device="cuda",
    )
    packed = torch.randint(
        0,
        256,
        (out_features * row_bytes + 1,),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.rand(
        out_features,
        in_features // group_size,
        dtype=torch.float16,
        device="cuda",
    ) * 0.05 + 0.01
    bias = torch.randn(out_features, dtype=torch.float16, device="cuda")
    quantized.load_from_packed(packed, scales, bias)
    inputs = torch.randn(37, in_features, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        actual = quantized(inputs)
        reference = torch.nn.functional.linear(
            inputs,
            quantized._dequantize().to(torch.float16),
            quantized.bias,
        )

    assert quantized.last_prefill_backend == "tiled_w3a16_gemm"
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten().float(),
        actual.flatten().float(),
        dim=0,
    )
    assert cosine.item() > 0.9999
    assert (reference - actual).abs().mean().item() < 0.001


def test_w3a16_batched_gemv_matches_single_row_decode() -> None:
    pytest.importorskip("cupy")
    from deepinfer.model_executor.models.qwen2_mixed_sm86 import W3A16Linear

    torch.manual_seed(20261105)
    in_features = 256
    out_features = 128
    group_size = 128
    row_bytes = in_features * 3 // 8
    quantized = W3A16Linear(
        in_features,
        out_features,
        True,
        group_size=group_size,
        dtype=torch.float16,
        device="cuda",
    )
    packed = torch.randint(
        0,
        256,
        (out_features * row_bytes + 1,),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.rand(
        out_features,
        in_features // group_size,
        dtype=torch.float16,
        device="cuda",
    ) * 0.05 + 0.01
    bias = torch.randn(out_features, dtype=torch.float16, device="cuda")
    quantized.load_from_packed(packed, scales, bias)
    inputs = torch.randn(4, in_features, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        actual = quantized._forward_batched_gemv(inputs)
        reference = torch.cat(
            [quantized._forward_gemv(inputs[row : row + 1]) for row in range(4)],
            dim=0,
        )

    torch.testing.assert_close(actual, reference, atol=0.0, rtol=0.0)
