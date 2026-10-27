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
from enum import Enum
from typing import Any

import torch

from apps.runtime import QwenSm86Runtime
from deepinfer.engine.kv_cache import (
    CacheOutOfMemoryError,
    PagedKVCacheManager,
    paged_kv_block_bytes,
)
from deepinfer.engine.memory_budget import plan_kv_blocks
from deepinfer.engine.metrics import EngineMetrics


class GenerationCancelledError(RuntimeError):
    """Raised through a request future when a client cancels generation."""


class RequestState(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prefix_cached_tokens: int = 0
    ttft_seconds: float = 0.0
    tpot_seconds: float = 0.0
    stop_sequence: str | None = None
    inter_token_seconds: tuple[float, ...] = ()


@dataclass
class PagedScheduledRequest:
    request_id: int
    messages: list[dict[str, str]] | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    future: Future[GenerationResult] = field(default_factory=Future)
    stream_queue: queue.Queue[str | None] | None = None
    prompt: str | None = None
    stop: tuple[str, ...] = ()
    tenant: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    prompt_length: int = 0
    prefill_done: int = 0
    prompt_hashes: list[bytes] = field(default_factory=list)
    shared_block_count: int = 0
    generated_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    cached_tokens: int = 0
    prefix_cached_tokens: int = 0
    sent_text: str = ""
    final_text: str | None = None
    state: RequestState = RequestState.PREFILL
    finished: bool = False
    cancelled: bool = False
    finish_reason: str = ""
    stop_sequence: str | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    completion_time: float | None = None
    token_times: list[float] = field(default_factory=list)


class PagedBatchScheduler:
    """Continuous batching with block-level KV allocation and prefix reuse."""

    def __init__(
        self,
        runtime: QwenSm86Runtime,
        *,
        max_num_seqs: int = 4,
        block_size: int = 16,
        prefill_chunk_tokens: int = 64,
        kv_cache_dtype: str = "fp16",
        num_gpu_blocks: int | None = None,
        prefix_cache_blocks: int | None = None,
        gpu_memory_utilization: float = 0.80,
        reserve_memory_mb: int = 512,
        kv_budget_mb: int | None = None,
        max_queue_size: int = 64,
        metrics: EngineMetrics | None = None,
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if prefill_chunk_tokens < 1:
            raise ValueError("prefill_chunk_tokens must be >= 1")
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.max_num_seqs = int(max_num_seqs)
        self.block_size = int(block_size)
        self.prefill_chunk_tokens = int(prefill_chunk_tokens)
        self.kv_cache_dtype = str(kv_cache_dtype)
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
        self.metrics = metrics or EngineMetrics()
        dtype = next(self.model.parameters()).dtype
        block_bytes = paged_kv_block_bytes(
            num_layers=int(cfg.num_hidden_layers),
            num_kv_heads=int(cfg.num_key_value_heads),
            block_size=self.block_size,
            head_dim=int(cfg.head_dim),
            compute_dtype=dtype,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        self.memory_budget = plan_kv_blocks(
            self.model,
            device=runtime.device,
            block_bytes=block_bytes,
            block_size=self.block_size,
            max_cache=self.max_cache,
            max_num_seqs=self.max_num_seqs,
            prefix_cache_blocks=self.prefix_cache_blocks,
            requested_blocks=num_gpu_blocks,
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_memory_mb=reserve_memory_mb,
            kv_budget_mb=kv_budget_mb,
        )
        self.manager = PagedKVCacheManager(
            num_layers=int(cfg.num_hidden_layers),
            num_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.head_dim),
            block_size=self.block_size,
            num_blocks=self.memory_budget.num_blocks,
            dtype=dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            device=runtime.device,
            prefix_cache_blocks=self.prefix_cache_blocks,
        )
        self.requests: queue.Queue[PagedScheduledRequest] = queue.Queue(
            maxsize=max_queue_size
        )
        self.running: list[PagedScheduledRequest] = []
        self.stop_event = threading.Event()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self._prefill_cursor = 0
        self.thread = threading.Thread(
            target=self._run, name="sm86-paged-scheduler", daemon=True
        )
        self._refresh_gauges()
        self.thread.start()

    def _submit_request(
        self,
        *,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        stream: bool,
        stop: tuple[str, ...] | list[str] | str | None,
        tenant: str,
    ) -> PagedScheduledRequest:
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if stop is None:
            stop_sequences: tuple[str, ...] = ()
        elif isinstance(stop, str):
            stop_sequences = (stop,)
        else:
            stop_sequences = tuple(str(value) for value in stop if str(value))
        with self._request_lock:
            self._request_counter += 1
            request_id = self._request_counter
        request = PagedScheduledRequest(
            request_id=request_id,
            messages=messages,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            stream_queue=queue.Queue() if stream else None,
            stop=stop_sequences,
            tenant=str(tenant or ""),
        )
        try:
            self.requests.put_nowait(request)
        except queue.Full as exc:
            self.metrics.inc("request_rejected_total")
            raise RuntimeError(
                f"engine request queue is full ({self.requests.maxsize})"
            ) from exc
        self.metrics.inc("requests_total")
        self._refresh_gauges()
        return request

    def submit(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
        stop: tuple[str, ...] | list[str] | str | None = None,
        tenant: str = "",
    ) -> PagedScheduledRequest:
        return self._submit_request(
            messages=messages,
            prompt=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            stop=stop,
            tenant=tenant,
        )

    def submit_completion(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        stream: bool = False,
        stop: tuple[str, ...] | list[str] | str | None = None,
        tenant: str = "",
    ) -> PagedScheduledRequest:
        return self._submit_request(
            messages=None,
            prompt=str(prompt),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=stream,
            stop=stop,
            tenant=tenant,
        )

    def cancel(self, request: PagedScheduledRequest) -> None:
        """Best-effort cancellation from an HTTP worker thread."""
        request.cancelled = True

    def _tokenize(self, request: PagedScheduledRequest) -> list[int]:
        if request.prompt is not None:
            prompt_text = request.prompt
        else:
            if not request.messages:
                raise ValueError("chat request has no messages")
            prompt_text = self.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[0]) != 1:
            raise ValueError("only single-sequence prompts are supported")
        return [int(token) for token in input_ids[0].tolist()]

    def _find_stop_sequence(
        self, request: PagedScheduledRequest, text: str
    ) -> tuple[str, str | None]:
        best_index: int | None = None
        best_stop: str | None = None
        for stop in request.stop:
            index = text.find(stop)
            if index >= 0 and (best_index is None or index < best_index):
                best_index = index
                best_stop = stop
        if best_index is None:
            return text, None
        return text[:best_index], best_stop

    def _emit(self, request: PagedScheduledRequest, *, final: bool = False) -> None:
        if final and request.final_text is not None:
            full_text = request.final_text
        else:
            full_text = self.tokenizer.decode(
                request.generated_ids, skip_special_tokens=True
            )
        safe_text = full_text.split(chr(0xFFFD), 1)[0]
        if not final and request.stop:
            holdback = min(len(safe_text), max(len(stop) for stop in request.stop) - 1)
            visible = safe_text[:-holdback] if holdback > 0 else safe_text
        else:
            visible = safe_text
        if visible.startswith(request.sent_text):
            delta = visible[len(request.sent_text) :]
            request.sent_text = visible
            if delta and request.stream_queue is not None:
                request.stream_queue.put(delta)

    def _completion_finish_reason(
        self, request: PagedScheduledRequest, token: int
    ) -> str | None:
        if token == self.tokenizer.eos_token_id:
            return "stop"
        if len(request.generated_ids) >= request.max_new_tokens:
            return "length"
        return None

    def _accept_token(
        self,
        request: PagedScheduledRequest,
        token: int,
        *,
        decode_token: bool,
    ) -> None:
        request.generated_ids.append(token)
        request.token_times.append(time.perf_counter())
        self.metrics.inc("generated_tokens_total")
        if decode_token:
            self.metrics.inc("decode_tokens_total")
        text = self.tokenizer.decode(request.generated_ids, skip_special_tokens=True)
        final_text, matched_stop = self._find_stop_sequence(request, text)
        finish_reason = self._completion_finish_reason(request, token)
        if matched_stop is not None:
            request.final_text = final_text
            request.finish_reason = "stop"
            request.stop_sequence = matched_stop
            self.metrics.inc("stop_sequence_total")
            self._finish(request)
        elif finish_reason is not None:
            request.final_text = final_text
            request.finish_reason = finish_reason
            self._finish(request)
        else:
            self._emit(request)

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
        request.state = RequestState.FINISHED
        request.completion_time = time.perf_counter()
        self._release(request)
        if request in self.running:
            self.running.remove(request)
        if request.final_text is None:
            text = self.tokenizer.decode(
                request.generated_ids, skip_special_tokens=True
            )
            request.final_text, matched_stop = self._find_stop_sequence(
                request, text
            )
            if matched_stop is not None:
                request.finish_reason = "stop"
                request.stop_sequence = matched_stop
        self._emit(request, final=True)
        completion_tokens = len(request.generated_ids)
        ttft = (
            request.first_token_time - request.arrival_time
            if request.first_token_time is not None
            else 0.0
        )
        inter_token_seconds = tuple(
            request.token_times[index] - request.token_times[index - 1]
            for index in range(1, len(request.token_times))
        )
        tpot = (
            sum(inter_token_seconds) / len(inter_token_seconds)
            if inter_token_seconds
            else 0.0
        )
        result = GenerationResult(
            text=request.final_text or "",
            prompt_tokens=request.prompt_length,
            completion_tokens=completion_tokens,
            finish_reason=request.finish_reason or "stop",
            prefix_cached_tokens=request.prefix_cached_tokens,
            ttft_seconds=round(ttft, 6),
            tpot_seconds=round(tpot, 6),
            stop_sequence=request.stop_sequence,
            inter_token_seconds=tuple(
                round(value, 6) for value in inter_token_seconds
            ),
        )
        if not request.future.done():
            request.future.set_result(result)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        self.metrics.inc("requests_finished_total")
        for interval in inter_token_seconds:
            self.metrics.observe("tpot_seconds", interval)
        self._refresh_gauges()

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
        request.state = RequestState.FINISHED
        request.completion_time = time.perf_counter()
        if release_blocks:
            self.manager.allocator.free(release_blocks)
        self._release(request)
        if request in self.running:
            self.running.remove(request)
        if isinstance(exc, CacheOutOfMemoryError):
            self.metrics.inc("kv_cache_oom_total")
        elif self._is_oom_exception(exc):
            self.metrics.inc("runtime_oom_total")
        if request.cancelled and isinstance(exc, GenerationCancelledError):
            if not request.future.done():
                request.future.cancel()
        elif not request.future.done():
            request.future.set_exception(exc)
        if request.stream_queue is not None:
            request.stream_queue.put(None)
        self.metrics.inc("requests_failed_total")
        self._refresh_gauges()

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
            request.generated_ids = []
            request.token_times = []
            request.cached_tokens = 0
            request.state = RequestState.PREFILL

            hashes = self.manager.prefix_cache.block_hashes(
                prompt_ids, self.block_size, namespace=request.tenant
            )
            request.prompt_hashes = hashes
            max_reusable_blocks = (prompt_length - 1) // self.block_size
            if self.prefix_cache_blocks:
                shared_blocks = self.manager.prefix_cache.match(
                    hashes, max_blocks=max_reusable_blocks
                )
            prefix_length = len(shared_blocks) * self.block_size
            # Reserve every physical block needed for prompt + output up front.
            # This turns most decode-time OOMs into explicit admission failures.
            capacity_tokens = prompt_length + max_new_tokens - prefix_length
            new_block_count = (
                capacity_tokens + self.block_size - 1
            ) // self.block_size
            if self.manager.allocator.available_blocks < new_block_count:
                self.metrics.inc("request_rejected_total")
                raise CacheOutOfMemoryError(
                    new_block_count, self.manager.allocator.available_blocks
                )
            new_blocks = self.manager.allocator.allocate(new_block_count)
            request.block_ids = shared_blocks + new_blocks
            request.shared_block_count = len(shared_blocks)
            request.prefix_cached_tokens = prefix_length
            request.prefill_done = prefix_length
            if prefix_length:
                self.metrics.inc("prefix_reused_tokens_total", prefix_length)
            self.running.append(request)
            self.metrics.set_gauge("running_requests", len(self.running))
            self._refresh_gauges()
            return True
        except BaseException as exc:  # noqa: BLE001
            owned = shared_blocks + new_blocks
            if request.block_ids and request.block_ids != owned:
                owned = request.block_ids
            request.block_ids = owned
            self._fail(request, exc)
            return False

    @staticmethod
    def _is_oom_exception(exc: BaseException) -> bool:
        return isinstance(exc, torch.cuda.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )

    def _maybe_clear_cuda_oom(self, exc: BaseException) -> None:
        if self.runtime.device.type == "cuda" and self._is_oom_exception(exc):
            torch.cuda.empty_cache()

    def _prefill_step(self) -> bool:
        prefill_requests = [
            item
            for item in self.running
            if not item.finished and item.state == RequestState.PREFILL
        ]
        if not prefill_requests:
            return False
        request = prefill_requests[self._prefill_cursor % len(prefill_requests)]
        self._prefill_cursor += 1
        if request.cancelled:
            self._fail(request, GenerationCancelledError("request cancelled"))
            return True
        if request.prefill_start_time is None:
            request.prefill_start_time = time.perf_counter()
        start = request.prefill_done
        end = min(
            request.prompt_length,
            start + self.prefill_chunk_tokens,
        )
        chunk_length = end - start
        if chunk_length < 1:
            self._fail(request, RuntimeError("invalid empty prefill chunk"))
            return True
        input_ids = torch.tensor(
            [request.prompt_ids[start:end]],
            dtype=torch.long,
            device=self.runtime.device,
        )
        positions = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=self.runtime.device,
        ).unsqueeze(0)
        views = self.manager.make_views(
            [request.block_ids],
            [start],
            max_length=end,
            new_tokens=chunk_length,
        )
        prefill_start = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = self.model.forward_logits(
                    input_ids,
                    positions,
                    kv_cache=views,
                )
                if self.runtime.device.type == "cuda":
                    torch.cuda.synchronize()
        except BaseException as exc:  # noqa: BLE001
            self._maybe_clear_cuda_oom(exc)
            self._fail(request, exc)
            return True
        if request.cancelled:
            self._fail(request, GenerationCancelledError("request cancelled"))
            return True
        elapsed = time.perf_counter() - prefill_start
        self.metrics.observe("prefill_chunk_seconds", elapsed)
        self.metrics.inc("prefill_steps_total")
        self.metrics.inc("prefill_chunk_tokens_total", chunk_length)
        request.prefill_done = end
        if end < request.prompt_length:
            return True

        first_token = self.runtime._sample(
            logits[:, -1],
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        full_blocks = request.prompt_length // self.block_size
        new_full_blocks = max(0, full_blocks - request.shared_block_count)
        if self.prefix_cache_blocks and new_full_blocks:
            first_new_block = request.shared_block_count
            self.manager.prefix_cache.register(
                request.prompt_hashes[
                    request.shared_block_count : full_blocks
                ],
                request.block_ids[
                    first_new_block : first_new_block + new_full_blocks
                ],
            )
        request.cached_tokens = request.prompt_length
        request.state = RequestState.DECODE
        request.first_token_time = time.perf_counter()
        ttft = request.first_token_time - request.arrival_time
        self.metrics.inc("prompt_tokens_total", request.prompt_length)
        self.metrics.observe("ttft_seconds", ttft)
        self._accept_token(request, first_token, decode_token=False)
        self._refresh_gauges()
        return True

    def _ensure_decode_blocks(self, request: PagedScheduledRequest) -> None:
        required = (request.cached_tokens + 1 + self.block_size - 1) // self.block_size
        missing = required - len(request.block_ids)
        if missing <= 0:
            return
        request.block_ids.extend(self.manager.allocator.allocate(missing))

    def _decode_step(self) -> bool:
        active = [
            request
            for request in list(self.running)
            if not request.finished and request.state == RequestState.DECODE
        ]
        if not active:
            return False
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
            return True

        max_length = max(request.cached_tokens + 1 for request in ready)
        block_tables = [request.block_ids for request in ready]
        seq_lens = [request.cached_tokens for request in ready]
        views = self.manager.make_views(
            block_tables,
            seq_lens,
            max_length=max_length,
            new_tokens=1,
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
        try:
            with torch.inference_mode():
                logits = self.model.forward_logits(
                    input_ids,
                    positions,
                    kv_cache=views,
                )
                if self.runtime.device.type == "cuda":
                    torch.cuda.synchronize()
        except BaseException as exc:  # noqa: BLE001
            self._maybe_clear_cuda_oom(exc)
            for request in ready:
                self._fail(request, exc)
            return True
        self.metrics.observe(
            "decode_step_seconds", time.perf_counter() - decode_start
        )
        self.metrics.inc("decode_steps_total")
        self.metrics.inc("decode_tokens_total", len(ready))
        previous_batch = int(self.metrics.get("max_decode_batch_size", 0))
        self.metrics.set_gauge(
            "max_decode_batch_size", max(previous_batch, len(ready))
        )

        for row, request in enumerate(ready):
            if request.finished:
                continue
            if request.cancelled:
                self._fail(request, GenerationCancelledError("request cancelled"))
                continue
            request.cached_tokens += 1
            try:
                next_token = self.runtime._sample(
                    logits[row, -1],
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                )
            except BaseException as exc:  # noqa: BLE001
                self._fail(request, exc)
                continue
            self._accept_token(request, next_token, decode_token=True)
        return True

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
                    decode_worked = self._decode_step()
                    if decode_worked and any(
                        request.state == RequestState.PREFILL
                        for request in self.running
                    ):
                        self.metrics.inc("prefill_decode_overlap_steps")
                    did_work = decode_worked or did_work
                except BaseException as exc:  # noqa: BLE001
                    self._maybe_clear_cuda_oom(exc)
                    for request in list(self.running):
                        if request.state == RequestState.DECODE:
                            self._fail(request, exc)
                try:
                    did_work = self._prefill_step() or did_work
                except BaseException as exc:  # noqa: BLE001
                    self._maybe_clear_cuda_oom(exc)
                    for request in list(self.running):
                        if request.state == RequestState.PREFILL:
                            self._fail(request, exc)
            if not did_work:
                time.sleep(0.01)
            self._refresh_gauges()

    def _refresh_gauges(self) -> None:
        allocator = self.manager.allocator
        running_snapshot = list(self.running)
        prefill_requests = sum(
            request.state == RequestState.PREFILL for request in running_snapshot
        )
        decode_requests = sum(
            request.state == RequestState.DECODE for request in running_snapshot
        )
        self.metrics.set_gauge("running_requests", len(self.running))
        self.metrics.set_gauge("queued_requests", self.requests.qsize())
        self.metrics.set_gauge("prefill_requests", prefill_requests)
        self.metrics.set_gauge("decode_requests", decode_requests)
        self.metrics.set_gauge("kv_cache_blocks_used", allocator.used_blocks)
        self.metrics.set_gauge("kv_cache_blocks_free", allocator.available_blocks)
        self.metrics.set_gauge("kv_cache_blocks_total", allocator.num_blocks)
        self.metrics.set_gauge(
            "kv_cache_usage_ratio",
            allocator.used_blocks / allocator.num_blocks
            if allocator.num_blocks
            else 0.0,
        )

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
        running_snapshot = list(self.running)
        prefill_requests = sum(
            request.state == RequestState.PREFILL for request in running_snapshot
        )
        decode_requests = sum(
            request.state == RequestState.DECODE for request in running_snapshot
        )
        return {
            "mode": "paged_continuous",
            "max_num_seqs": self.max_num_seqs,
            "block_size": self.block_size,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_model_len": self.max_model_len,
            "active": len(self.running),
            "prefill": prefill_requests,
            "decode": decode_requests,
            "queued": self.requests.qsize(),
            "chunked_prefill": True,
            "batched_decode": True,
            "paged_kv_cache": True,
            "native_paged_attention": bool(
                getattr(self.runtime.model, "native_paged_attention_kernel", False)
            ),
            "native_batched_w4_gemv": bool(
                getattr(self.runtime.model, "native_batched_w4_gemv", False)
            ),
            "memory_budget": self.memory_budget.as_dict(),
            "prefix_cache": self.manager.prefix_cache.stats(),
            "kv_cache": self.manager.stats(),
            "requests": [
                {
                    "id": request.request_id,
                    "state": request.state.value,
                    "tenant": request.tenant,
                    "prompt_tokens": request.prompt_length,
                    "prefill_done": request.prefill_done,
                    "generated_tokens": len(request.generated_ids),
                    "prefix_cached_tokens": request.prefix_cached_tokens,
                    "blocks": len(request.block_ids),
                }
                for request in running_snapshot
            ],
        }


__all__ = [
    "GenerationCancelledError",
    "GenerationResult",
    "PagedBatchScheduler",
    "PagedScheduledRequest",
]
