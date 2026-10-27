"""Gemma-family source model.

Gemma shares the standard decoder skeleton but differs in offset RMSNorm,
embedding normalisation and GeGLU.  Those differences are normalized into
ModelSpec and selected by the generic decoder rather than by a copied model
implementation.
"""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM


class GemmaForCausalLM(Qwen2ForCausalLM):
    source_backend = "gemma_reference_sdpa"


__all__ = ["GemmaForCausalLM"]
