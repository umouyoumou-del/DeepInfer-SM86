"""QK-norm used by OLMoE, checked against a naive reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_attention import (  # noqa: E402
    per_head_rms_norm,
    reference_per_head_rms_norm,
)

HEADS = 16
DIM = 128


def test_per_head_rms_norm_matches_reference() -> None:
    torch.manual_seed(0)
    tensor = torch.randn(5, HEADS * DIM, dtype=torch.float32)
    weight = torch.randn(HEADS * DIM, dtype=torch.float32)

    got = per_head_rms_norm(tensor, weight, num_heads=HEADS, head_dim=DIM)
    expected = reference_per_head_rms_norm(
        tensor, weight, num_heads=HEADS, head_dim=DIM
    )
    assert torch.allclose(got, expected, atol=1e-5)


def test_heads_are_normalised_independently() -> None:
    tensor = torch.zeros(1, HEADS * DIM)
    tensor[0, :DIM] = 3.0
    tensor[0, DIM : 2 * DIM] = 7.0
    weight = torch.ones(HEADS * DIM)
    out = per_head_rms_norm(tensor, weight, num_heads=HEADS, head_dim=DIM)
    # each head is scaled to unit RMS regardless of the other heads' magnitude
    assert torch.allclose(
        out[0, :DIM].pow(2).mean().sqrt(), torch.tensor(1.0), atol=1e-4
    )
    assert torch.allclose(
        out[0, DIM : 2 * DIM].pow(2).mean().sqrt(), torch.tensor(1.0), atol=1e-4
    )


def test_rejects_wrong_width() -> None:
    tensor = torch.randn(1, HEADS * DIM)
    with pytest.raises(ValueError):
        per_head_rms_norm(
            tensor, torch.ones(HEADS * DIM - 1), num_heads=HEADS, head_dim=DIM
        )
    with pytest.raises(ValueError):
        per_head_rms_norm(tensor, torch.ones(HEADS * DIM), num_heads=8, head_dim=DIM)


def test_preserves_dtype() -> None:
    tensor = torch.randn(2, HEADS * DIM, dtype=torch.float16)
    weight = torch.randn(HEADS * DIM, dtype=torch.float16)
    out = per_head_rms_norm(tensor, weight, num_heads=HEADS, head_dim=DIM)
    assert out.dtype == torch.float16
