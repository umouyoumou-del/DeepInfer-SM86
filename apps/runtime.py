"""Reusable inference runtime for the DeepInfer SM86 OpenAI-compatible gateway."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qwen7b_mixed_gpu_harness import (
    _install_deepinfer_import_stubs,
    _load_module,
)

DEFAULT_CHECKPOINT = Path(r"D:\Qwen7B-Mixed-SM86")
DEFAULT_MAX_CACHE = 256


@dataclass
class GenerationStats:
    prompt_tokens: int
    completion_tokens: int


def _load_backend_modules() -> tuple[Any, Any, Any]:
    _install_deepinfer_import_stubs()
    _load_module("deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py")
    qwen2_module = _load_module(
        "deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py"
    )
    w4a16_module = _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )
    mixed_module = _load_module(
        "deepinfer.model_executor.models.qwen2_mixed_sm86",
        "model_executor/models/qwen2_mixed_sm86.py",
    )
    return qwen2_module, w4a16_module, mixed_module


class QwenSm86Runtime:
    """Single-request, resident Qwen2.5 runtime with static-cache CUDA Graph."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        max_cache: int = DEFAULT_MAX_CACHE,
        device: str = "cuda",
        use_graph: bool = True,
        backend: str = "mixed",
        model_id: str | None = None,
    ) -> None:
        if max_cache > 256:
            raise ValueError(
                "max_cache > 256 is not supported by the current SM86 GQA kernel"
            )
        self.checkpoint = Path(checkpoint)
        self.backend = backend
        self.model_id = model_id or (
            "deepinfer-sm86-3b-w4a16"
            if backend == "w4a16"
            else "deepinfer-sm86-7b-mixed"
        )
        self.max_cache = max_cache
        self.device = torch.device(device)
        self.use_graph = use_graph
        self.lock = threading.Lock()
        self.last_stats = GenerationStats(0, 0)
        self.last_timings: dict[str, float] = {}
        self.last_generated_ids: list[int] = []

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint),
            local_files_only=True,
            trust_remote_code=True,
        )
        qwen2_module, w4a16_module, mixed_module = _load_backend_modules()
        self.qwen2_module = qwen2_module
        if backend == "w4a16":
            self.model = w4a16_module.Qwen2W4A16ForCausalLM.from_checkpoint(
                self.checkpoint,
                device=device,
                group_size=128,
            )
        elif backend == "mixed":
            self.model = mixed_module.Qwen2MixedForCausalLM.from_checkpoint(
                self.checkpoint,
                device=device,
                group_size=128,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")
        self.model.eval()
        self._build_static_cache()
        self.graph = None
        if self.use_graph:
            self._capture_graph()

    def _build_static_cache(self) -> None:
        self.static_cache = []
        for _ in range(int(self.model.cfg.num_hidden_layers)):
            self.static_cache.append(
                self.qwen2_module.Qwen2StaticLayerKVCache(
                    key=torch.zeros(
                        (
                            1,
                            int(self.model.cfg.num_key_value_heads),
                            self.max_cache,
                            int(self.model.cfg.head_dim),
                        ),
                        dtype=torch.float16,
                        device=self.device,
                    ),
                    value=torch.zeros(
                        (
                            1,
                            int(self.model.cfg.num_key_value_heads),
                            self.max_cache,
                            int(self.model.cfg.head_dim),
                        ),
                        dtype=torch.float16,
                        device=self.device,
                    ),
                )
            )
        self.static_token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_position = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.static_mask = torch.zeros(
            (1, self.max_cache), dtype=torch.bool, device=self.device
        )

    def _reset_static_cache(self) -> None:
        for cache in self.static_cache:
            cache.key.zero_()
            cache.value.zero_()
        self.static_mask.zero_()
        self.static_position.zero_()

    def _capture_graph(self) -> None:
        dummy = self.tokenizer.eos_token_id
        if dummy is None:
            dummy = 0
        self.static_token.fill_(dummy)
        self.static_position.fill_(0)
        self.static_mask.zero_()
        self.static_mask[:, 0] = True
        with torch.no_grad():
            self.model.forward_logits(
                self.static_token,
                self.static_position,
                kv_cache=self.static_cache,
                attention_mask=self.static_mask,
            )
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.graph_logits = self.model.forward_logits(
                self.static_token,
                self.static_position,
                kv_cache=self.static_cache,
                attention_mask=self.static_mask,
            )
        torch.cuda.synchronize()
        # Capture records custom kernels; the first replay executes them.
        self.graph.replay()
        torch.cuda.synchronize()
        self._reset_static_cache()

    def _sample(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        flat = logits.float().reshape(-1)
        if temperature <= 0:
            return int(flat.argmax(dim=-1).item())
        flat = flat / temperature
        if top_k > 0:
            values, indices = torch.topk(flat, min(top_k, flat.numel()))
            filtered = torch.full_like(flat, float("-inf"))
            filtered[indices] = values
            flat = filtered
        if 0 < top_p < 1:
            sorted_logits, sorted_indices = torch.sort(flat, descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            keep = cumulative <= top_p
            keep[0] = True
            flat = torch.full_like(flat, float("-inf"))
            flat[sorted_indices[keep]] = sorted_logits[keep]
        probs = torch.softmax(flat, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _copy_prefill_cache(
        self, dynamic_cache: Sequence[Any], prompt_length: int
    ) -> None:
        for dynamic, static in zip(dynamic_cache, self.static_cache, strict=False):
            if dynamic is None or dynamic.key is None or dynamic.value is None:
                raise RuntimeError("prefill did not populate every KV cache layer")
            static.key.zero_()
            static.value.zero_()
            static.key[:, :, :prompt_length, :].copy_(dynamic.key)
            static.value[:, :, :prompt_length, :].copy_(dynamic.value)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> str:
        for _ in self.stream_generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            pass
        return self.tokenizer.decode(self.last_generated_ids, skip_special_tokens=True)

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Iterator[str]:
        with self.lock:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
            prompt_length = int(input_ids.shape[1])
            max_new_tokens = min(max_new_tokens, self.max_cache - prompt_length)
            print(
                f"[runtime] prompt_tokens={prompt_length} max_new={max_new_tokens}",
                flush=True,
            )
            if max_new_tokens <= 0:
                raise ValueError(
                    f"prompt({prompt_length}) leaves no room in max_cache={self.max_cache}"
                )

            positions = torch.arange(prompt_length, device=self.device).unsqueeze(0)
            dynamic_cache = self.model.new_kv_cache()
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model.forward_logits(
                    input_ids, positions, kv_cache=dynamic_cache
                )
                current_token = self._sample(
                    logits[:, -1],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            prefill_seconds = time.perf_counter() - prefill_start
            self._copy_prefill_cache(dynamic_cache, prompt_length)
            decode_start = time.perf_counter()

            generated_ids: list[int] = []
            sent_text = ""
            completion_tokens = 0
            eos_token_id = self.tokenizer.eos_token_id
            for step in range(max_new_tokens):
                generated_ids.append(current_token)
                completion_tokens += 1
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                # Byte-level BPE can end a prefix in the middle of a UTF-8
                # character. Hold that unstable tail until the next token
                # completes it; never emit U+FFFD or duplicate the prefix.
                safe_text = text.split(chr(0xFFFD), 1)[0]
                if safe_text.startswith(sent_text):
                    delta = safe_text[len(sent_text) :]
                    sent_text = safe_text
                    if delta:
                        yield delta
                if current_token == eos_token_id or completion_tokens >= max_new_tokens:
                    break
                self.static_token.fill_(current_token)
                next_position = prompt_length + step
                self.static_position.fill_(next_position)
                self.static_mask.zero_()
                self.static_mask[:, : next_position + 1] = True
                if self.use_graph:
                    with torch.no_grad():
                        self.graph.replay()
                        torch.cuda.synchronize()
                        next_logits = self.graph_logits
                else:
                    with torch.no_grad():
                        next_logits = self.model.forward_logits(
                            self.static_token,
                            self.static_position,
                            kv_cache=self.static_cache,
                            attention_mask=self.static_mask,
                        )
                current_token = self._sample(
                    next_logits[:, -1],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            decode_seconds = time.perf_counter() - decode_start
            self.last_stats = GenerationStats(prompt_length, completion_tokens)
            self.last_generated_ids = generated_ids.copy()
            self.last_timings = {
                "prefill_seconds": round(prefill_seconds, 6),
                "decode_seconds": round(decode_seconds, 6),
                "decode_tokens_per_second": round(completion_tokens / decode_seconds, 3)
                if decode_seconds > 0
                else 0.0,
            }
            print(f"[runtime] {self.last_timings}", flush=True)
            self._reset_static_cache()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_id": self.model_id,
            "backend": self.backend,
            "checkpoint": str(self.checkpoint),
            "max_cache": self.max_cache,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0),
            "use_graph": self.use_graph,
            "last_timings": self.last_timings,
        }
