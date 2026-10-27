"""Mistral-family source model.

Mistral uses the same standard GQA + RMSNorm + RoPE + gated-MLP decoder shape
as Llama.  Its distinguishing execution feature is sliding-window attention,
which is normalized into ``ModelSpec`` and implemented by the generic
attention mask builder rather than by a family-specific model class.
"""

from __future__ import annotations

from deepinfer.model_executor.models.llama import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    """Mistral-family decoder driven by the normalized ModelSpec."""

    source_backend = "mistral_reference_sdpa"


__all__ = ["MistralForCausalLM"]
