"""Continuous-batching scheduler backed by a paged/elastic KV cache.

Compared with :mod:`apps.scheduler`, this scheduler does not reserve a dense
``max_cache`` slot per active request.  A request owns only the blocks its
current prompt/output needs, completed sequences release blocks immediately,
and identical full prompt blocks can be shared through the prefix cache.

The attention reference path is currently PyTorch SDPA rather than the SM86
batch=1 native GQA kernel.  That trade is deliberate: it enables dynamic batch
sizes, ragged sequence lengths, paged block tables and sliding-window masking.
The native dense-slot scheduler remains available for latency-sensitive,
single-request use.
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
from deepinfer.engine.kv_cache import (
    CacheOutOfMemoryError,
    PagedKVCacheManager,
)
from deepinfer.engine.metrics import EngineMetrics


class GenerationCancelledError(RuntimeError):
    """Raised through a request future when a client cancels generation."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prefix_cached_tokens: int = 0


@dataclass
class PagedScheduledRequest:
    request_id: int
    messages: list[dict[str, str]]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    future: Future[GenerationResult] = field(default_factory=Future)
    stream_queue: queue.Queue[str | None] | None = None
    prompt_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0
    generated_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    prefix_cached_tokens: int = 0
    sent_text: str = ""
    finished: bool = False
    cancelled: bool = False
    finish_reason: str = ""
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    completion_time: float | None = None


class PagedBatchScheduler:
    """Continuous batching with block-level KV allocation and prefix reuse."""

    def __init__(
        self,
        runtime: QwenSm86Runtime,
        *,
        max_num_seqs: int = 4,
        block_size: int = 16,
        num_gpu_blocks: int | None = None,
        prefix_cache_blocks: int | None = None,
        max_queue_size: int = 64,
        metrics: EngineMetrics | None = None,
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.max_num_seqs = int(max_num_seqs)
        self.block_size = int(block_size)
        self.max_cache = int(runtime.max_cache)
        self.max_model_len = min(
            self.max_cache,
            int(getattr(self.model.cfg, "max_position_embeddings", self.max_cache)),
        )
        cfg = self.model.cfg
        blocks_per_seq = (self.max_cache + self.block_size - 1) // self.block_size
        if prefix_cache_blocks is None:
            prefix_cache_blocks = min(16, blocks_per_seq)
        self.prefix_cache_blocks = max(0, int(prefix_cache_blocks))
        if num_gpu_blocks is None:
            num_gpu_blocks = (
                self.max_num_seqs * blocks_per_seq + self.prefix_cache_blocks
            )
        if int(num_gpu_blocks) < 1:
            raise ValueError("num_gpu_blocks must be >= 1")
        self.manager = PagedKVCacheManager(
            num_layers=int(cfg.num_hidden_layers),
            num_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.head_dim),
            block_size=self.block_size,
            num_blocks=int(num_gpu_blocks),
            dtype=next(self.model.parameters()).dtype,
            device=runtime.device,
            prefix_cache_blocks=self.prefix_cache_blocks,
        )
        self.metrics = metrics or EngineMetrics()
        self.requests: queue.Queue[PagedScheduledRequest] = queue.Queue(
            maxsize=max_queue_size
        )
        self.running: list[PagedScheduledRequest] = []
        self.stop_event = threading.Event()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name="sm86-paged-scheduler", daemon=True
        )
        self._refresh_gauges()
        self.thread.start()

    def submit(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
    ) -> PagedScheduledRequest:
        with self._request_lock:
            self._request_counter += 1
            request_id = self._request_counter
        request = PagedScheduledRequest(
            request_id=request_id,
            messages=messages,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            stream_queue=queue.Queue() if stream else None,
        )
        try:
            self.requests.put_nowait(request)
        except queue.Full as exc:
            raise RuntimeError(
                f"engine request queue is full ({self.requests.maxsize})"
            ) from exc
        self.metrics.inc("requests_total")
        self._refresh_gauges()
        return request

    def cancel(self, request: PagedScheduledRequest) -> None:
        """Best-effort cancellation from an HTTP worker thread."""
        request.cancelled = True

    def _tokenize(self, request: PagedScheduledRequest) -> list[int]:
        prompt_text = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[0]) != 1:
            raise ValueError("only single-sequence chat prompts are supported")
        return [int(token) for token in input_ids[0].tolist()]

    def _emit(self, request: PagedScheduledRequest) -> None:
        text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        safe_text = text.split(chr(0xFFFD), 1)[0]
        if safe_text.startswith(request.sent_text):
            delta = safe_text[len(request.sent_text) :]
            request.sent_text = safe_text
            if delta and request.stream_queue is not None:
                request.stream_queue.put(delta)

    def _is_finished(self, request: PagedScheduledRequest, token: int) -> bool:
        if token == self.tokenizer.eos_token_id:
            request.finish_reason = "stop"
            return True
        if len(request.generated_ids) >= request.max_new_tokens:
            request.finish_reason = "length"
            return True
        return False

    def _release(self, request: PagedScheduledRequest) -> None:
        if request.block_ids:
            try:
                self.manager.allocator.free(request.block_ids)
            finally:
                request.block_ids = []
        self._refresh_gauges()

    def _finish(self, request: PagedScheduledRequest) -> None:
        if request.finished:
            return
        request.finished = True
        request.completion_time = time.perf_counter()
        self._release(request)
        if request in self.running:
            self.running.remove(request)
        text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        result = GenerationResult(
            text=text,
            prompt_tokens=request.prompt_length,
            completion_tokens=len(request.generated_ids),
            finish_reason=request.finish_reason or "stop",
            prefix_cached_tokens=request.prefix_cached_tokens,
        )
        if not request.future.done():
            request.future.set_result(result)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        self.metrics.inc("requests_finished_total")
        self.metrics.set_gauge("running_requests", len(self.running))

    def _fail(
        self,
        request: PagedScheduledRequest,
        exc: BaseException,
        *,
        release_blocks: list[int] | None = None,
    ) -> None:
        if request.finished:
            return
        request.finished = True
        request.completion_time = time.perf_counter()
        if release_blocks:
            self.manager.allocator.free(release_blocks)
        self._release(request)
        if request in self.running:
            self.running.remove(request)
        if request.cancelled and isinstance(exc, GenerationCancelledError):
            if not request.future.done():
                request.future.cancel()
        elif not request.future.done():
            request.future.set_exception(exc)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        self.metrics.inc("requests_failed_total")
        self.metrics.set_gauge("running_requests", len(self.running))

    def _seed_prefill_cache(
        self,
        shared_blocks: list[int],
        prefix_length: int,
    ) -> list[Any]:
        caches = self.model.new_kv_cache()
        if prefix_length <= 0:
            return caches
        for layer_index in range(len(caches)):
            key, value = self.manager.read_blocks(
                layer_index, shared_blocks, prefix_length
            )
            if key is None or value is None:
                raise RuntimeError("prefix cache did not return K/V tensors")
            caches[layer_index] = self.runtime.qwen2_module.Qwen2LayerKVCache(
                key=key, value=value
            )
        return caches

    def _admit(self, request: PagedScheduledRequest) -> bool:
        if request.cancelled:
            self._fail(request, GenerationCancelledError("request cancelled"))
            return False
        shared_blocks: list[int] = []
        new_blocks: list[int] = []
        try:
            prompt_ids = self._tokenize(request)
            prompt_length = len(prompt_ids)
            if prompt_length < 1:
                raise ValueError("prompt tokenized to an empty sequence")
            if prompt_length > self.max_model_len:
                raise ValueError(
                    f"prompt has {prompt_length} tokens, engine limit is {self.max_model_len}"
                )
            max_new_tokens = min(
                request.max_new_tokens,
                self.max_model_len - prompt_length,
            )
            if max_new_tokens < 1:
                raise ValueError("prompt leaves no room for at least one output token")

            request.prompt_ids = prompt_ids
            request.prompt_length = prompt_length
            request.max_new_tokens = max_new_tokens

            hashes = self.manager.prefix_cache.block_hashes(
                prompt_ids, self.block_size
            )
            max_reusable_blocks = (prompt_length - 1) // self.block_size
            if self.prefix_cache_blocks:
                shared_blocks = self.manager.prefix_cache.match(
                    hashes, max_blocks=max_reusable_blocks
                )
            prefix_length = len(shared_blocks) * self.block_size
            suffix_length = prompt_length - prefix_length
            suffix_blocks = (suffix_length + self.block_size - 1) // self.block_size
            if self.manager.allocator.available_blocks < suffix_blocks:
                raise CacheOutOfMemoryError(
                    suffix_blocks, self.manager.allocator.available_blocks
                )
            new_blocks = self.manager.allocator.allocate(suffix_blocks)
            request.block_ids = shared_blocks + new_blocks
            request.prefix_cached_tokens = prefix_length
            if prefix_length:
                self.metrics.inc("prefix_reused_tokens_total", prefix_length)

            input_ids = torch.tensor(
                [prompt_ids[prefix_length:]],
                dtype=torch.long,
                device=self.runtime.device,
            )
            positions = torch.arange(
                prefix_length,
                prompt_length,
                dtype=torch.long,
                device=self.runtime.device,
            ).unsqueeze(0)
            dynamic_cache = self._seed_prefill_cache(shared_blocks, prefix_length)
            prefill_start = time.perf_counter()
            with torch.inference_mode():
                logits = self.model.forward_logits(
                    input_ids, positions, kv_cache=dynamic_cache
                )
                first_token = self.runtime._sample(
                    logits[:, -1],
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
            if request.cancelled:
                raise GenerationCancelledError("request cancelled during prefill")
            for layer_index, source in enumerate(dynamic_cache):
                if source is None or source.key is None or source.value is None:
                    raise RuntimeError("prefill did not populate every KV layer")
                self.manager.write_tokens(
                    layer_index,
                    new_blocks,
                    source.key,
                    source.value,
                    source_start=prefix_length,
                    length=suffix_length,
                )
            self.metrics.observe("prefill_seconds", time.perf_counter() - prefill_start)

            full_blocks = prompt_length // self.block_size
            new_full_blocks = max(0, full_blocks - len(shared_blocks))
            if self.prefix_cache_blocks and new_full_blocks:
                self.manager.prefix_cache.register(
                    hashes[len(shared_blocks) : full_blocks],
                    new_blocks[:new_full_blocks],
                )

            request.cached_tokens = prompt_length
            request.generated_ids = [first_token]
            request.first_token_time = time.perf_counter()
            self.running.append(request)
            self.metrics.inc("prompt_tokens_total", prompt_length)
            self.metrics.inc("generated_tokens_total")
            self.metrics.set_gauge("running_requests", len(self.running))
            self._emit(request)
            if self._is_finished(request, first_token):
                self._finish(request)
            return True
        except BaseException as exc:  # noqa: BLE001
            # Prefix matches are already retained; release both ownership sets.
            owned = shared_blocks + new_blocks
            if request.block_ids and request.block_ids != owned:
                owned = request.block_ids
            request.block_ids = owned
            self._fail(request, exc)
            return False

    def _ensure_decode_blocks(self, request: PagedScheduledRequest) -> None:
        required = (request.cached_tokens + 1 + self.block_size - 1) // self.block_size
        missing = required - len(request.block_ids)
        if missing <= 0:
            return
        request.block_ids.extend(self.manager.allocator.allocate(missing))

    def _decode_step(self) -> None:
        active = [request for request in list(self.running) if not request.finished]
        if not active:
            return
        ready: list[PagedScheduledRequest] = []
        for request in active:
            if request.cancelled:
                self._fail(request, GenerationCancelledError("request cancelled"))
                continue
            try:
                self._ensure_decode_blocks(request)
            except CacheOutOfMemoryError as exc:
                self._fail(request, exc)
                continue
            ready.append(request)
        if not ready:
            return

        max_length = max(request.cached_tokens + 1 for request in ready)
        block_tables = [request.block_ids for request in ready]
        seq_lens = [request.cached_tokens for request in ready]
        views = self.manager.make_views(
            block_tables,
            seq_lens,
            max_length=max_length,
        )
        input_ids = torch.tensor(
            [[request.generated_ids[-1]] for request in ready],
            dtype=torch.long,
            device=self.runtime.device,
        )
        positions = torch.tensor(
            [[request.cached_tokens] for request in ready],
            dtype=torch.long,
            device=self.runtime.device,
        )
        decode_start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model.forward_logits(
                input_ids,
                positions,
                kv_cache=views,
            )
            if self.runtime.device.type == "cuda":
                torch.cuda.synchronize()
        self.metrics.observe("decode_step_seconds", time.perf_counter() - decode_start)
        self.metrics.inc("decode_steps_total")

        for row, request in enumerate(ready):
            if request.finished:
                continue
            request.cached_tokens += 1
            next_token = self.runtime._sample(
                logits[row, -1],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated_ids.append(next_token)
            self.metrics.inc("generated_tokens_total")
            self._emit(request)
            if self._is_finished(request, next_token):
                self._finish(request)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            did_work = False
            while (
                len(self.running) < self.max_num_seqs
                and not self.requests.empty()
            ):
                try:
                    request = self.requests.get_nowait()
                except queue.Empty:
                    break
                did_work = True
                self._admit(request)
            if self.running:
                did_work = True
                try:
                    self._decode_step()
                except BaseException as exc:  # noqa: BLE001
                    for request in list(self.running):
                        self._fail(request, exc)
            if not did_work:
                time.sleep(0.01)
            self._refresh_gauges()

    def _refresh_gauges(self) -> None:
        allocator = self.manager.allocator
        self.metrics.set_gauge("running_requests", len(self.running))
        self.metrics.set_gauge("queued_requests", self.requests.qsize())
        self.metrics.set_gauge("kv_cache_blocks_used", allocator.used_blocks)
        self.metrics.set_gauge("kv_cache_blocks_free", allocator.available_blocks)
        self.metrics.set_gauge("kv_cache_blocks_total", allocator.num_blocks)

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        for request in list(self.running):
            self._fail(request, GenerationCancelledError("engine stopped"))
        while True:
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            self._fail(request, GenerationCancelledError("engine stopped"))

    def status(self) -> dict[str, Any]:
        self._refresh_gauges()
        return {
            "mode": "paged_continuous",
            "max_num_seqs": self.max_num_seqs,
            "block_size": self.block_size,
            "max_model_len": self.max_model_len,
            "active": len(self.running),
            "queued": self.requests.qsize(),
            "batched_decode": True,
            "paged_kv_cache": True,
            "prefix_cache": self.manager.prefix_cache.stats(),
            "kv_cache": self.manager.stats(),
            "requests": [
                {
                    "id": request.request_id,
                    "prompt_tokens": request.prompt_length,
                    "generated_tokens": len(request.generated_ids),
                    "prefix_cached_tokens": request.prefix_cached_tokens,
                    "blocks": len(request.block_ids),
                }
                for request in self.running
            ],
        }


__all__ = [
    "GenerationCancelledError",
    "GenerationResult",
    "PagedBatchScheduler",
    "PagedScheduledRequest",
]
