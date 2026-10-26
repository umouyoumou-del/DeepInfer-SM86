"""Continuous dynamic batching scheduler for the SM86 model executor.

The scheduler keeps a fixed set of batch slots.  Requests are prefetched one at
a time (prefill currently uses the existing dynamic-cache path), then active
requests decode together through the batched W4/W3 GEMV and SDPA attention
path.  This is an eager scheduler; CUDA Graph is intentionally not used when
batch size can vary.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import torch

from apps.runtime import QwenSm86Runtime


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass
class ScheduledRequest:
    messages: list[dict[str, str]]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    future: Future[GenerationResult] = field(default_factory=Future)
    stream_queue: queue.Queue[str | None] | None = None
    slot: int = -1
    prompt_length: int = 0
    generated_ids: list[int] = field(default_factory=list)
    sent_text: str = ""
    finished: bool = False


class DynamicBatchScheduler:
    """Continuous batching over a fixed slot table."""

    def __init__(self, runtime: QwenSm86Runtime, *, max_batch_size: int = 2) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.max_batch_size = max_batch_size
        self.max_cache = runtime.max_cache
        self.slots: list[ScheduledRequest | None] = [None] * max_batch_size
        self.batch_cache = self._make_cache()
        self.static_token = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=runtime.device
        )
        self.static_position = torch.zeros(
            (max_batch_size, 1), dtype=torch.long, device=runtime.device
        )
        self.static_mask = torch.zeros(
            (max_batch_size, self.max_cache), dtype=torch.bool, device=runtime.device
        )
        self.graph = None
        self.graph_logits = None
        self._capture_graph()
        self.requests: queue.Queue[ScheduledRequest] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="sm86-batch-scheduler", daemon=True
        )
        self.thread.start()

    def _make_cache(self) -> list[Any]:
        cache = []
        cfg = self.model.cfg
        for _ in range(int(cfg.num_hidden_layers)):
            cache.append(
                self.runtime.qwen2_module.Qwen2StaticLayerKVCache(
                    key=torch.zeros(
                        (
                            self.max_batch_size,
                            int(cfg.num_key_value_heads),
                            self.max_cache,
                            int(cfg.head_dim),
                        ),
                        dtype=torch.float16,
                        device=self.runtime.device,
                    ),
                    value=torch.zeros(
                        (
                            self.max_batch_size,
                            int(cfg.num_key_value_heads),
                            self.max_cache,
                            int(cfg.head_dim),
                        ),
                        dtype=torch.float16,
                        device=self.runtime.device,
                    ),
                )
            )
        return cache

    def _capture_graph(self) -> None:
        self.static_token.fill_(int(self.tokenizer.eos_token_id or 0))
        self.static_position.zero_()
        self.static_mask.zero_()
        self.static_mask[:, 0] = True
        try:
            with torch.no_grad():
                self.model.forward_logits(
                    self.static_token,
                    self.static_position,
                    kv_cache=self.batch_cache,
                    attention_mask=self.static_mask,
                )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                self.graph_logits = self.model.forward_logits(
                    self.static_token,
                    self.static_position,
                    kv_cache=self.batch_cache,
                    attention_mask=self.static_mask,
                )
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.graph = graph
            for slot in range(self.max_batch_size):
                self._zero_slot(slot)
            print("batch CUDA Graph captured", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"batch CUDA Graph unavailable, eager fallback: {exc}", flush=True)
            self.graph = None
            for slot in range(self.max_batch_size):
                self._zero_slot(slot)

    def submit(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
    ) -> ScheduledRequest:
        request = ScheduledRequest(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream_queue=queue.Queue() if stream else None,
        )
        self.requests.put(request)
        return request

    def _free_slot(self) -> int | None:
        for index, request in enumerate(self.slots):
            if request is None:
                return index
        return None

    def _zero_slot(self, slot: int) -> None:
        for cache in self.batch_cache:
            cache.key[slot].zero_()
            cache.value[slot].zero_()

    def _prefill(self, request: ScheduledRequest) -> None:
        prompt_text = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.runtime.device)
        prompt_length = int(input_ids.shape[1])
        max_new = min(request.max_new_tokens, self.max_cache - prompt_length)
        if max_new <= 0:
            raise ValueError(
                f"prompt({prompt_length}) leaves no room in max_cache={self.max_cache}"
            )
        positions = torch.arange(prompt_length, device=self.runtime.device).unsqueeze(0)
        dynamic_cache = self.model.new_kv_cache()
        with torch.inference_mode():
            logits = self.model.forward_logits(
                input_ids, positions, kv_cache=dynamic_cache
            )
            first = self.runtime._sample(
                logits[:, -1],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
        slot = self._free_slot()
        if slot is None:
            raise RuntimeError("no free batch slot")
        self._zero_slot(slot)
        for batch, source in zip(self.batch_cache, dynamic_cache, strict=False):
            if source is None or source.key is None or source.value is None:
                raise RuntimeError("prefill did not populate all KV layers")
            batch.key[slot, :, :prompt_length, :].copy_(source.key[0])
            batch.value[slot, :, :prompt_length, :].copy_(source.value[0])
        request.slot = slot
        request.prompt_length = prompt_length
        request.max_new_tokens = max_new
        request.generated_ids = [first]
        self.slots[slot] = request
        self._emit(request)
        if self._is_finished(request, first):
            self._finish(request)

    def _is_finished(self, request: ScheduledRequest, token: int) -> bool:
        return (
            token == self.tokenizer.eos_token_id
            or len(request.generated_ids) >= request.max_new_tokens
        )

    def _emit(self, request: ScheduledRequest) -> None:
        text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        safe_text = text.split(chr(0xFFFD), 1)[0]
        if safe_text.startswith(request.sent_text):
            delta = safe_text[len(request.sent_text) :]
            request.sent_text = safe_text
            if delta and request.stream_queue is not None:
                request.stream_queue.put(delta)

    def _finish(self, request: ScheduledRequest) -> None:
        if request.finished:
            return
        request.finished = True
        text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        result = GenerationResult(
            text=text,
            prompt_tokens=request.prompt_length,
            completion_tokens=len(request.generated_ids),
            finish_reason="stop",
        )
        request.future.set_result(result)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        if request.slot >= 0:
            self._zero_slot(request.slot)
            self.slots[request.slot] = None
            request.slot = -1

    def _fail(self, request: ScheduledRequest, exc: BaseException) -> None:
        request.finished = True
        request.future.set_exception(exc)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        if request.slot >= 0:
            self._zero_slot(request.slot)
            self.slots[request.slot] = None
            request.slot = -1

    def _admit(self, request: ScheduledRequest) -> None:
        try:
            self._prefill(request)
        except Exception as exc:  # noqa: BLE001
            self._fail(request, exc)

    def _decode_step(self) -> None:
        active = [request for request in self.slots if request is not None]
        if not active:
            return
        batch = self.max_batch_size
        eos = int(self.tokenizer.eos_token_id or 0)
        input_ids = torch.full(
            (batch, 1), eos, dtype=torch.long, device=self.runtime.device
        )
        positions = torch.zeros(
            (batch, 1), dtype=torch.long, device=self.runtime.device
        )
        mask = torch.zeros(
            (batch, self.max_cache), dtype=torch.bool, device=self.runtime.device
        )
        mask[:, 0] = True  # free slots must not create an all -inf softmax
        for slot, request in enumerate(self.slots):
            if request is None:
                continue
            token = request.generated_ids[-1]
            position = request.prompt_length + len(request.generated_ids) - 1
            input_ids[slot, 0] = token
            positions[slot, 0] = position
            mask[slot, : position + 1] = True
        if self.graph is not None:
            self.static_token.copy_(input_ids)
            self.static_position.copy_(positions)
            self.static_mask.copy_(mask)
            with torch.no_grad():
                self.graph.replay()
                torch.cuda.synchronize()
                logits = self.graph_logits
        else:
            with torch.inference_mode():
                logits = self.model.forward_logits(
                    input_ids,
                    positions,
                    kv_cache=self.batch_cache,
                    attention_mask=mask,
                )
                torch.cuda.synchronize()
        for slot, request in enumerate(self.slots):
            if request is None:
                continue
            next_token = self.runtime._sample(
                logits[slot, -1],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated_ids.append(next_token)
            self._emit(request)
            if self._is_finished(request, next_token):
                self._finish(request)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            admitted = 0
            while self._free_slot() is not None:
                try:
                    request = self.requests.get_nowait()
                except queue.Empty:
                    break
                self._admit(request)
                admitted += 1
            if any(request is not None for request in self.slots):
                self._decode_step()
            elif not admitted:
                time.sleep(0.01)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def status(self) -> dict[str, Any]:
        return {
            "max_batch_size": self.max_batch_size,
            "graph_captured": self.graph is not None,
            "active": sum(request is not None for request in self.slots),
            "queued": self.requests.qsize(),
            "slots": [
                None if request is None else request.prompt_length
                for request in self.slots
            ],
        }


__all__ = ["DynamicBatchScheduler", "GenerationResult", "ScheduledRequest"]
