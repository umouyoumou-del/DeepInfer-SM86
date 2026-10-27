"""SM86 W4A16 backend for the Mistral-family source model."""

from __future__ import annotations

from deepinfer.model_executor.models.llama_w4a16 import LlamaW4A16ForCausalLM


class MistralW4A16ForCausalLM(LlamaW4A16ForCausalLM):
    """Mistral W4A16 model using the layered checkpoint loader."""

    source_backend = "mistral_w4a16_sm86_model_executor"


__all__ = ["MistralW4A16ForCausalLM"]
