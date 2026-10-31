"""Full OLMoE forward: embeddings, decoder stack, final norm, LM head.

The layer itself lives in ``olmoe_layer.py``; this module is the assembly step
and is validated end to end against ``transformers.OlmoeForCausalLM``.
"""

from __future__ import annotations

import torch

from deepinfer.moe.olmoe_layer import decoder_layer_forward, rms_norm


def causal_mask(tokens: int, *, device, dtype) -> torch.Tensor:
    mask = torch.full((1, 1, tokens, tokens), float("-inf"), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def forward(
    input_ids: torch.Tensor,
    weights: dict[str, torch.Tensor],
    layer_weights: list[dict[str, torch.Tensor]],
    *,
    num_heads: int,
    eps: float,
    rope_base: float,
    top_k: int,
    tie_word_embeddings: bool = False,
    cache=None,
) -> torch.Tensor:
    """Return logits ``[batch, tokens, vocab]``."""

    if len(layer_weights) == 0:
        raise ValueError("at least one decoder layer is required")
    hidden = torch.nn.functional.embedding(input_ids, weights["embed_tokens"])
    mask = causal_mask(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)
    for index, layer in enumerate(layer_weights):
        hidden = decoder_layer_forward(
            hidden,
            layer,
            num_heads=num_heads,
            eps=eps,
            rope_base=rope_base,
            top_k=top_k,
            attention_mask=None if cache is not None else mask,
            cache=cache,
            layer_index=index,
        )
    hidden = rms_norm(hidden, weights["final_norm"], eps)
    if tie_word_embeddings:
        return hidden @ weights["embed_tokens"].t()
    return hidden @ weights["lm_head"].t()
