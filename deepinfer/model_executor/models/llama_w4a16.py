"""SM86 W4A16 backend for the Llama-family source model."""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2_w4a16 import Qwen2W4A16ForCausalLM


class LlamaW4A16ForCausalLM(Qwen2W4A16ForCausalLM):
    """Llama W4A16 model; the generic loader consumes a layered checkpoint."""

    source_backend = "llama_w4a16_sm86_model_executor"


__all__ = ["LlamaW4A16ForCausalLM"]
