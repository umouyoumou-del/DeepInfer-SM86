"""Gemma 2 source model with sandwich norms, softcaps and alternating SWA."""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM


class Gemma2ForCausalLM(Qwen2ForCausalLM):
    source_backend = "gemma2_reference_sdpa"


__all__ = ["Gemma2ForCausalLM"]
