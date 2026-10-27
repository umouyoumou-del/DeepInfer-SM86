"""SM86 W4A16 backend for Gemma 2 checkpoints."""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2_w4a16 import Qwen2W4A16ForCausalLM


class Gemma2W4A16ForCausalLM(Qwen2W4A16ForCausalLM):
    source_backend = "gemma2_w4a16_sm86_model_executor"


__all__ = ["Gemma2W4A16ForCausalLM"]
