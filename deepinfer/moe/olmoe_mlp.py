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


def moe_forward_batched(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    *,
    top_k: int,
    activation: str = "silu",
) -> torch.Tensor:
    """Same maths as ``moe_forward`` but with no per-expert Python loop.

    Gathers the selected experts into one batch and uses ``bmm``, so a token
    costs a fixed handful of kernel launches instead of ``top_k`` matmul triples.
    Must agree with ``moe_forward``; a test pins that.
    """

    if activation not in ("silu", "gelu"):
        raise ValueError(f"unsupported activation {activation!r}")

    expert_ids, weights = route_topk(hidden, gate_weight, top_k=top_k)
    tokens = hidden.shape[0]
    hidden_f32 = hidden.float()

    # Keep the matmuls in the weights' own dtype.  Widening to fp32 first
    # doubles the gathered traffic and gives up tensor cores; fp16 bmm still
    # accumulates in fp32 internally.
    compute = expert_gate.dtype
    gate = expert_gate[expert_ids]           # [tokens, top_k, inter, hidden]
    up = expert_up[expert_ids]
    down = expert_down[expert_ids]
    x = hidden.to(compute).unsqueeze(1).expand(tokens, top_k, hidden.shape[1])
    flat_x = x.reshape(-1, x.shape[2]).unsqueeze(-1)

    gate_out = torch.bmm(
        gate.reshape(-1, gate.shape[2], gate.shape[3]), flat_x
    ).reshape(tokens, top_k, gate.shape[2])
    up_out = torch.bmm(
        up.reshape(-1, up.shape[2], up.shape[3]), flat_x
    ).reshape(tokens, top_k, up.shape[2])
    if activation == "silu":
        act = torch.nn.functional.silu(gate_out)
    else:
        act = torch.nn.functional.gelu(gate_out)
    combined = act * up_out
    down_out = torch.bmm(
        down.reshape(-1, down.shape[2], down.shape[3]),
        combined.reshape(-1, combined.shape[2]).unsqueeze(-1),
    ).reshape(tokens, top_k, hidden.shape[1])
    result = (down_out * weights.to(compute).unsqueeze(-1)).sum(dim=1)
    return result.to(hidden.dtype)


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
