"""OLMoE's MoE block: router -> top-k experts -> weighted sum, no shared expert.

DeepSeek-V2 adds a shared expert that every token passes through; OLMoE has
none, so the block is exactly the weighted sum of the selected experts.  This
module implements the reference (unquantised) form so the wiring can be checked
against it before any packed-quantised path is involved.
"""

from __future__ import annotations

import torch

from deepinfer.moe.olmoe_router import route_topk


def moe_forward(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    *,
    top_k: int,
    activation: str = "silu",
) -> torch.Tensor:
    """SwiGLU experts, weighted and summed.  No shared-expert term.

    ``expert_gate`` / ``expert_up`` are ``[experts, intermediate, hidden]`` and
    ``expert_down`` is ``[experts, hidden, intermediate]``.
    """

    if activation not in ("silu", "gelu"):
        raise ValueError(f"unsupported activation {activation!r}")
    if expert_gate.shape[0] != expert_up.shape[0] or expert_gate.shape[0] != expert_down.shape[0]:
        raise ValueError("expert tensors disagree on the expert count")

    expert_ids, weights = route_topk(hidden, gate_weight, top_k=top_k)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    hidden_f32 = hidden.float()

    for token in range(hidden.shape[0]):
        for slot in range(top_k):
            index = int(expert_ids[token, slot])
            weight = weights[token, slot]
            gate = hidden_f32[token] @ expert_gate[index].float().t()
            up = hidden_f32[token] @ expert_up[index].float().t()
            if activation == "silu":
                act = gate * torch.sigmoid(gate)
            else:
                act = torch.nn.functional.gelu(gate)
            down = (act * up) @ expert_down[index].float().t()
            output[token] += weight * down
    return output.to(hidden.dtype)


def reference_moe_forward(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    *,
    top_k: int,
    activation: str = "silu",
) -> torch.Tensor:
    """Deliberately independent formulation used by the tests.

    Selects experts with an explicit sort instead of ``route_topk``'s ``topk``,
    and accumulates by scattering into an expert-major buffer.
    """

    logits = hidden.float() @ gate_weight.float()
    probabilities = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

    output = torch.zeros(hidden.shape[0], hidden.shape[1], dtype=torch.float32)
    for token in range(hidden.shape[0]):
        order = sorted(
            range(probabilities.shape[1]), key=lambda e: -probabilities[token, e]
        )[:top_k]
        accumulated = torch.zeros(hidden.shape[1], dtype=torch.float32)
        for expert in order:
            weight = probabilities[token, expert]
            gate = expert_gate[expert].float() @ hidden.float()[token]
            up = expert_up[expert].float() @ hidden.float()[token]
            if activation == "silu":
                activations = gate / (1.0 + torch.exp(-gate))
            else:
                activations = torch.nn.functional.gelu(gate)
            accumulated = accumulated + weight * (
                expert_down[expert].float() @ (activations * up)
            )
        output[token] = accumulated
    return output.to(hidden.dtype)
