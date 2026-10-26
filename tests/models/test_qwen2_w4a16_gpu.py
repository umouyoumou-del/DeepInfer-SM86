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
