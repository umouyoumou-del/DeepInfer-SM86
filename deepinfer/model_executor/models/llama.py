"""Llama-family source model.

The generic Qwen2 decoder implementation is already a standard GQA +
RMSNorm + RoPE + SwiGLU decoder.  Llama differs mainly in its adapter-produced
model spec: no q/k/v bias, standard RoPE by default, and Llama weight names.
The class exists so the registry and execution layer do not special-case Qwen.
"""

from __future__ import annotations

from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM


class LlamaForCausalLM(Qwen2ForCausalLM):
    """Llama-family source model driven by :class:`deepinfer.model_spec.ModelSpec`."""

    source_backend = "llama_reference_sdpa"


__all__ = ["LlamaForCausalLM"]
