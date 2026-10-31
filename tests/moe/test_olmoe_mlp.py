"""OLMoE's MoE block, checked against an independent formulation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.moe.olmoe_mlp import moe_forward, reference_moe_forward  # noqa: E402

HIDDEN = 16
INTER = 8
EXPERTS = 8
TOP_K = 3


def _weights(seed: int = 0, dtype=torch.float32):
    torch.manual_seed(seed)
    return (
        torch.randn(HIDDEN, EXPERTS, dtype=dtype) * 0.3,
        torch.randn(EXPERTS, INTER, HIDDEN, dtype=dtype) * 0.2,
        torch.randn(EXPERTS, INTER, HIDDEN, dtype=dtype) * 0.2,
        torch.randn(EXPERTS, HIDDEN, INTER, dtype=dtype) * 0.2,
    )


@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_matches_reference(activation: str) -> None:
    gate_weight, expert_gate, expert_up, expert_down = _weights()
    hidden = torch.randn(4, HIDDEN)
    got = moe_forward(
        hidden, gate_weight, expert_gate, expert_up, expert_down,
        top_k=TOP_K, activation=activation,
    )
    expected = reference_moe_forward(
        hidden, gate_weight, expert_gate, expert_up, expert_down,
        top_k=TOP_K, activation=activation,
    )
    assert torch.allclose(got, expected, atol=1e-4)


def test_no_shared_expert_term() -> None:
    """Zeroing the selected experts must zero the output - nothing else runs."""

    gate_weight, expert_gate, expert_up, expert_down = _weights(seed=2)
    hidden = torch.randn(2, HIDDEN)
    zeroed = moe_forward(
        hidden,
        gate_weight,
        torch.zeros_like(expert_gate),
        torch.zeros_like(expert_up),
        torch.zeros_like(expert_down),
        top_k=TOP_K,
    )
    assert torch.equal(zeroed, torch.zeros_like(zeroed))


def test_single_expert_equals_dense_mlp() -> None:
    """With top_k == 1 the block must equal that expert's SwiGLU, scaled."""

    gate_weight, expert_gate, expert_up, expert_down = _weights(seed=5)
    hidden = torch.randn(1, HIDDEN)
    got = moe_forward(
        hidden, gate_weight, expert_gate, expert_up, expert_down, top_k=1
    )
    logits = hidden.float() @ gate_weight.float()
    probabilities = torch.softmax(logits, dim=-1)
    weight, expert = probabilities.max(dim=-1)
    gate = expert_gate[int(expert)].float() @ hidden.float()[0]
    up = expert_up[int(expert)].float() @ hidden.float()[0]
    expected = weight * (expert_down[int(expert)].float() @ (gate * torch.sigmoid(gate) * up))
    assert torch.allclose(got[0], expected, atol=1e-5)


def test_rejects_bad_inputs() -> None:
    gate_weight, expert_gate, expert_up, expert_down = _weights()
    hidden = torch.randn(1, HIDDEN)
    with pytest.raises(ValueError):
        moe_forward(
            hidden, gate_weight, expert_gate, expert_up, expert_down,
            top_k=TOP_K, activation="relu",
        )
    with pytest.raises(ValueError):
        moe_forward(
            hidden, gate_weight, expert_gate, expert_up, expert_down[:-1],
            top_k=TOP_K,
        )
