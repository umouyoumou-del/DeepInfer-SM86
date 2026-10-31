"""OLMoE's router: softmax over all experts, then top-k, without renorm.

DeepSeek-V2 uses group-limited routing on top of a sigmoid gate; OLMoE uses the
simpler scheme from ``OlmoeTopKRouter``: a dense F32 ``ffn_gate_inp`` producing
``[tokens, num_experts]`` logits, a softmax across *all* experts, then ``topk``.
The selected weights are deliberately left un-normalised, so they sum to well
under 1 - that is the reference behaviour, not an oversight.
"""

from __future__ import annotations

import torch


def route_topk(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(expert_ids, weights)`` of shape ``[tokens, top_k]``."""

    if hidden.dim() != 2:
        raise ValueError(f"expected 2-D hidden states, got {hidden.shape}")
    if gate_weight.shape[0] != hidden.shape[1] or gate_weight.shape[1] < top_k:
        raise ValueError(
            "gate weight shape %s does not match hidden %s / top_k %d"
            % (tuple(gate_weight.shape), tuple(hidden.shape), top_k)
        )
    logits = torch.matmul(hidden.float(), gate_weight.float())
    probabilities = torch.softmax(logits, dim=-1)
    weights, expert_ids = torch.topk(probabilities, top_k, dim=-1)
    return expert_ids, weights


def reference_route_topk(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token loop, used by the tests."""

    rows = hidden.shape[0]
    ids = torch.empty(rows, top_k, dtype=torch.long)
    values = torch.empty(rows, top_k, dtype=torch.float32)
    for row in range(rows):
        logits = [
            sum(
                float(hidden[row, index]) * float(gate_weight[index, expert])
                for index in range(hidden.shape[1])
            )
            for expert in range(gate_weight.shape[1])
        ]
        highest = max(logits)
        total = sum(pow(2.718281828459045, value - highest) for value in logits)
        probabilities = [
            pow(2.718281828459045, value - highest) / total for value in logits
        ]
        paired = sorted(enumerate(probabilities), key=lambda item: -item[1])[:top_k]
        for slot, (expert, probability) in enumerate(paired):
            ids[row, slot] = expert
            values[row, slot] = probability
    return ids, values
