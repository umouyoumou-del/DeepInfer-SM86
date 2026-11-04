"""The Triton FP8 decode must work on SM80 through its software decode path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from deepinfer.moe import latent_mla_triton as fused
from deepinfer.moe.latent_mla import dequantize_fp8_groups


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_decode_software_fallback_matches_torch(monkeypatch) -> None:
    monkeypatch.setenv("DEEPINFER_TRITON_FP8", "0")
    torch.manual_seed(0)
    n = 128
    latent = (torch.randn(n, 512, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scale = torch.rand(n, 4, device="cuda") * 0.01 + 0.01
    q_abs = (torch.randn(16, 512, device="cuda", dtype=torch.float16) * 0.1)
    q_pe = (torch.randn(1, 16, 64, device="cuda", dtype=torch.float16) * 0.1)
    pe = (torch.randn(n, 64, device="cuda", dtype=torch.float16) * 0.1)

    out, lse = fused.decode_partial(
        q_abs, q_pe, latent, scale, pe, [(0, n, 0)], sm_scale=1.0,
        block_n=32, num_warps=4, num_stages=1,
    )

    dequantized = dequantize_fp8_groups(latent, scale).float()
    scores = q_abs.float() @ dequantized.T + q_pe[0].float() @ pe.float().T
    expected = torch.softmax(scores, dim=-1) @ dequantized
    expected_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
