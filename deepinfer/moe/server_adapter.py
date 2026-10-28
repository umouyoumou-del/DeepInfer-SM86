"""Server-facing adapter for the DeepSeek-V2-Lite MoE runtime.

The paged scheduler only needs a small runtime surface (model, tokenizer,
device, forward_logits and _sample).  The HTTP server additionally expects
model_id, health(), last_stats and stream_generate(), which the native
DeepSeek runtime does not expose directly.  This adapter keeps those
responsibilities out of the core inference code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from apps.runtime import GenerationStats
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime


class DeepSeekServerRuntime:
    """Thin OpenAI-server facade over :class:`DeepSeekV2LiteGGUFRuntime`."""

    backend = "deepseek"
    architecture = "deepseek_v2_lite_moe"
    use_graph = False

    def __init__(
        self,
        runtime: DeepSeekV2LiteGGUFRuntime,
        *,
        model_id: str = "deepseek-v2-lite-q4km",
        checkpoint: Path | str | None = None,
    ) -> None:
        self.runtime = runtime
        self.model_id = str(model_id)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.device = runtime.device
        self.max_cache = runtime.max_cache
        self.vocab_size = int(getattr(runtime.model.config, "vocab_size", 0))
        self.last_stats = GenerationStats(0, 0)
        self.last_generated_ids: list[int] = []
        self.last_timings: dict[str, float] = {}

    def _sample(self, *args, **kwargs):
        return self.runtime._sample(*args, **kwargs)

    def forward_logits(self, *args, **kwargs):
        return self.runtime.forward_logits(*args, **kwargs)

    def generate(
        self,
        messages: list[dict[str, str]] | None = None,
        *,
        prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        result = self.runtime.generate(
            messages,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        self.last_stats = GenerationStats(
            result.prompt_tokens, result.completion_tokens
        )
        self.last_generated_ids = list(result.token_ids)
        self.last_timings = {
            "prefill_seconds": round(result.prefill_seconds, 6),
            "decode_seconds": round(result.decode_seconds, 6),
            "decode_tokens_per_second": round(
                result.decode_tokens_per_second, 3
            ),
        }
        return result.text

    def stream_generate(
        self,
        messages: list[dict[str, str]] | None = None,
        *,
        prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Direct-runtime fallback stream.

        The paged scheduler provides true token streaming.  When the server is
        started without a scheduler, generate once and expose deterministic
        text slices so the OpenAI SSE contract still works.
        """

        text = self.generate(
            messages,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        for start in range(0, len(text), 8):
            yield text[start : start + 8]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.model_id,
            "backend": self.backend,
            "architecture": self.architecture,
            "checkpoint": str(self.checkpoint or ""),
            "max_cache": self.max_cache,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0),
            "use_graph": False,
            "last_timings": self.last_timings,
            "fused_moe": bool(self.runtime.fused_moe_enabled),
            "paged_overlap_supported": bool(
                getattr(self.runtime, "supports_prefill_decode_overlap", False)
            ),
        }

    def close(self) -> None:
        self.runtime.close()


__all__ = ["DeepSeekServerRuntime"]
