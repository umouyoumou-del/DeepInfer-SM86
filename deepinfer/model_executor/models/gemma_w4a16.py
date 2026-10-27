"""SM86 W4A16 backend for Gemma-family checkpoints."""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2_w4a16 import Qwen2W4A16ForCausalLM


class GemmaW4A16ForCausalLM(Qwen2W4A16ForCausalLM):
    source_backend = "gemma_w4a16_sm86_model_executor"


__all__ = ["GemmaW4A16ForCausalLM"]
