"""Paged-KV continuous batching for the ECS expert-parallel runtime.

The dense slot engine in :mod:`deepinfer.moe.continuous_batching` keeps every
active row in one ``DynamicCache``.  A new admission has to re-pack every live
row and shift it to the right whenever the newcomer is longer, and every
completion has to gather the surviving rows again.  Measured on the two-GPU
ECS host, that padding/selection traffic -- not the expert transport -- is what
keeps aggregate throughput around 4.2 tok/s while fixed-size batch 8 reaches
25.9 tok/s.

This module removes the copies.  Each request owns a list of physical KV blocks
in a :class:`~deepinfer.engine.kv_cache.PagedKVCacheManager`:

* admission reserves only the blocks the prompt actually needs;
* a decode step builds one block table for the active rows and lets the native
  paged MLA kernel read straight out of the pool;
* completion returns the blocks to the allocator and nothing is re-indexed.

Rows therefore never move.  The block table is the only per-step allocation and
it is ``[batch, max_blocks]`` int64 -- a few hundred bytes.

Two prefill strategies are available:

``prefill_mode="batched"`` (default)
    Queue up to the free slot count of requests, prefill them in one
    left-padded dense forward (this is what keeps remote expert requests at
    ~391 instead of ~478), then copy each row's valid suffix into its own
    blocks once.  Requires the runtime's dense ``model.model(...)`` entry
    point.

``prefill_mode="paged"``
    Prefill each admitted request on its own through
    ``runtime.forward_logits`` with ``seq_lens=0``.  No dense KV is ever built,
    at the cost of one prefill forward per request.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from transformers.cache_utils import DynamicCache

from deepinfer.engine.kv_cache import (
    CacheOutOfMemoryError,
    PagedKVCacheManager,
    PrefixCache,
)
from deepinfer.moe.continuous_batching import build_generation_result


class SlotOutOfBlocks(RuntimeError):
    """Raised when a request cannot reserve the blocks it needs."""

    def __init__(self, needed: int, available: int) -> None:
        self.needed = int(needed)
        self.available = int(available)
        super().__init__(
            f"paged KV slot allocation failed: needs {self.needed} blocks, "
            f"{self.available} available"
        )


@dataclass
class PagedRequest:
    """One in-flight request and the blocks it owns."""

    messages: list[dict[str, str]]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    future: Future | None = None
    prompt_tokens: int = 0
    generated: list[int] = field(default_factory=list)
    cache_len: int = 0
    block_ids: list[int] = field(default_factory=list)
    prefill_seconds: float = 0.0
    prefill_kind: str = ""
    cached_tokens: int = 0
    reused_blocks: int = 0
    decode_started_at: float = 0.0
    decode_seconds: float = 0.0
    finished: bool = False


def paged_dims(runtime: Any) -> tuple[int, int, int, int, int]:
    """Return ``(layers, heads, key_dim, value_dim, head_dim)`` for MLA."""

    config = runtime.config
    layers = int(getattr(config, "num_hidden_layers"))
    heads = int(getattr(config, "num_attention_heads"))
    key_dim = int(getattr(config, "qk_nope_head_dim")) + int(
        getattr(config, "qk_rope_head_dim")
    )
    value_dim = int(getattr(config, "v_head_dim"))
    return layers, heads, key_dim, value_dim, max(key_dim, value_dim)


class PagedSlotAllocator:
    """Bounded block allocator that tracks one block list per request.

    ``max_length`` is the per-request cache ceiling; a request may grow up to
    ``ceil(max_length / block_size)`` blocks and no further.  Growth is
    incremental: a request that arrives with 9 prompt tokens and decodes 40
    more only pays for the blocks it crosses into.
    """

    def __init__(
        self,
        backend: Any,
        *,
        block_size: int,
        max_length: int,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        self.backend = backend
        self.block_size = int(block_size)
        self.max_length = int(max_length)
        self.peak_blocks = 0
        self.allocations = 0
        self.releases = 0
        self.growth_events = 0
        self.rejected = 0

    def required_blocks(self, tokens: int) -> int:
        tokens = max(0, int(tokens))
        return (tokens + self.block_size - 1) // self.block_size

    def capacity_blocks(self) -> int:
        return self.required_blocks(self.max_length)

    def reserve(self, request: PagedRequest, tokens: int) -> int:
        """Grow ``request.block_ids`` so ``tokens`` positions fit.

        Returns the number of newly allocated blocks (0 when the request
        already has enough).  Raises :class:`SlotOutOfBlocks` when the request
        would exceed ``max_length`` or the pool is exhausted.
        """

        tokens = int(tokens)
        if tokens > self.max_length:
            self.rejected += 1
            raise SlotOutOfBlocks(self.capacity_blocks(), len(request.block_ids))
        want = self.required_blocks(tokens)
        have = len(request.block_ids)
        if want <= have:
            return 0
        delta = want - have
        try:
            blocks = self.backend.allocate(delta)
        except CacheOutOfMemoryError as exc:  # pragma: no cover - device path
            self.rejected += 1
            raise SlotOutOfBlocks(delta, getattr(exc, "available", 0)) from exc
        request.block_ids.extend(int(block) for block in blocks)
        self.allocations += 1
        self.growth_events += 1
        self.peak_blocks = max(self.peak_blocks, self.backend.used_blocks)
        return delta

    def release(self, request: PagedRequest) -> int:
        if not request.block_ids:
            return 0
        released = len(request.block_ids)
        self.backend.free(list(request.block_ids))
        request.block_ids = []
        self.releases += 1
        return released

    def stats(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "max_length": self.max_length,
            "capacity_blocks_per_request": self.capacity_blocks(),
            "num_blocks": self.backend.num_blocks,
            "used_blocks": self.backend.used_blocks,
            "free_blocks": self.backend.available_blocks,
            "peak_blocks": self.peak_blocks,
            "allocations": self.allocations,
            "releases": self.releases,
            "growth_events": self.growth_events,
            "rejected": self.rejected,
        }


def _dense_row(
    cache: DynamicCache, layer: int, row: int, valid: int, *, padded_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the ``valid`` right-aligned tokens of one row out of a dense cache."""

    offset = int(padded_len) - int(valid)
    key = cache.key_cache[layer][row : row + 1, :, offset:, :]
    value = cache.value_cache[layer][row : row + 1, :, offset:, :]
    return key, value


def write_dense_rows_to_paged(
    manager: PagedKVCacheManager,
    *,
    runtime: Any,
    rows: Sequence[tuple[PagedRequest, DynamicCache, int]],
    padded_len: int,
    dims: tuple[int, int, int, int, int] | None = None,
) -> None:
    """Copy per-request rows of a dense batch cache into paged blocks.

    ``rows`` holds ``(request, single-row cache, valid_length)``.  The valid
    tokens of each row are right-aligned in the dense cache and are written to
    positions ``0 .. valid_length-1`` of that request's own blocks, so every
    paged row is zero-based and contiguous.

    Rows are grouped by length and written one group per layer.  Admission
    batches usually share a prompt length, so this turns
    ``requests x layers`` scatter launches into ``groups x layers``.
    """

    if not rows:
        return
    from deepinfer.moe.paged_adapter import DeepSeekPagedCache

    layers, _, key_dim, value_dim, head_dim = dims or paged_dims(runtime)
    device = manager.device
    groups: dict[int, list[tuple[PagedRequest, DynamicCache]]] = {}
    for request, cache, valid in rows:
        if int(valid) <= 0:
            continue
        groups.setdefault(int(valid), []).append((request, cache))
    for valid, entries in groups.items():
        positions = torch.arange(valid, dtype=torch.long, device=device)
        views = manager.make_views(
            [list(request.block_ids) for request, _ in entries],
            [0] * len(entries),
            max_length=valid,
            new_tokens=valid,
        )
        paged = DeepSeekPagedCache(
            views,
            positions.unsqueeze(0).expand(len(entries), -1),
            past_length=0,
            max_length=valid,
            key_dim=key_dim,
            value_dim=value_dim,
            head_dim=head_dim,
            native_attention=False,
        )
        for layer in range(layers):
            keys, values = [], []
            for _, cache in entries:
                key, value = _dense_row(
                    cache, layer, 0, valid, padded_len=padded_len
                )
                keys.append(key)
                values.append(value)
            paged.write(
                torch.cat(keys, dim=0).to(device),
                torch.cat(values, dim=0).to(device),
                layer,
                positions.unsqueeze(0).expand(len(entries), -1),
            )


class PagedContinuousBatchEngine:
    """Continuous batching over a paged KV pool.

    Same external contract as
    :class:`~deepinfer.moe.continuous_batching.ContinuousBatchEngine`:
    :meth:`submit` returns a ``Future`` that resolves to a
    :class:`~deepinfer.moe.deepseek_v2_runtime.GenerationResult`.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        max_batch_size: int = 4,
        block_size: int = 16,
        num_blocks: int | None = None,
        kv_memory_utilization: float = 0.30,
        max_cache: int | None = None,
        device: int = 0,
        prefill_mode: str = "batched",
        kv_cache_dtype: str = "fp16",
        pool_device: str | torch.device | None = None,
        prefix_cache_blocks: int = 0,
        prefix_namespace: str = "",
        admission_grace_ms: float = 0.0,
    ) -> None:
        if prefill_mode not in {"batched", "paged"}:
            raise ValueError("prefill_mode must be 'batched' or 'paged'")
        self.runtime = runtime
        self.max_batch_size = max(1, int(max_batch_size))
        self.device = int(device)
        self.prefill_mode = prefill_mode
        self.dims = paged_dims(runtime)
        layers, heads, _, _, head_dim = self.dims
        self.max_cache = int(max_cache if max_cache is not None else runtime.max_cache)
        if num_blocks is None:
            num_blocks = self._plan_blocks(
                layers=layers,
                heads=heads,
                head_dim=head_dim,
                block_size=block_size,
                utilization=kv_memory_utilization,
            )
        self.manager = PagedKVCacheManager(
            num_layers=layers,
            num_kv_heads=heads,
            head_dim=head_dim,
            block_size=int(block_size),
            num_blocks=int(num_blocks),
            dtype=torch.float16,
            kv_cache_dtype=kv_cache_dtype,
            prefix_cache_blocks=max(0, int(prefix_cache_blocks)),
            device=(
                torch.device(pool_device)
                if pool_device is not None
                else torch.device(f"cuda:{self.device}")
            ),
        )
        self.allocator = PagedSlotAllocator(
            self.manager.allocator,
            block_size=int(block_size),
            max_length=self.max_cache,
        )
        self.prefix_namespace = str(prefix_namespace)
        self.admission_grace_ms = max(0.0, float(admission_grace_ms))
        self._prefix_hits = 0
        self._prefix_reused_tokens = 0
        self.queue: queue.Queue[PagedRequest | None] = queue.Queue()
        self._active: list[PagedRequest] = []
        self._stop = threading.Event()
        self._completed = 0
        self._failed = 0
        self._last_error: str | None = None
        self._decode_steps = 0
        self._admissions = 0
        self._prefill_forwards = 0
        self._prefill_rows = 0
        self._max_active_seen = 0
        self._thread = threading.Thread(
            target=self._run, name="ecs-paged-continuous-batch", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ setup

    def _plan_blocks(
        self,
        *,
        layers: int,
        heads: int,
        head_dim: int,
        block_size: int,
        utilization: float,
    ) -> int:
        """Size the pool from the memory left after the model is resident."""

        per_block = (
            2  # key and value pools
            * int(layers)
            * int(heads)
            * int(block_size)
            * int(head_dim)
            * 2  # fp16
        )
        need = self.max_batch_size * (
            (self.max_cache + int(block_size) - 1) // int(block_size)
        )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return max(1, need)
        budget = int(float(free_bytes) * float(utilization))
        blocks = budget // max(1, per_block)
        return max(1, int(min(blocks, max(need, 1))))

    # ------------------------------------------------------------- submission

    def submit(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Future:
        future: Future = Future()
        self.queue.put(
            PagedRequest(
                messages=messages,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                future=future,
            )
        )
        return future

    # ---------------------------------------------------------------- prefill

    def _prefill_batched(
        self, requests: Sequence[PagedRequest]
    ) -> list[PagedRequest]:
        """One left-padded dense forward, then a one-time write into paged KV."""

        runtime = self.runtime
        texts = [
            runtime.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            for request in requests
        ]
        old_padding_side = runtime.tokenizer.padding_side
        runtime.tokenizer.padding_side = "left"
        try:
            encoded = runtime.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            runtime.tokenizer.padding_side = old_padding_side
        input_ids = encoded["input_ids"].to(torch.long)
        attention_mask = encoded["attention_mask"].to(torch.long)
        prompt_lengths = attention_mask.sum(dim=1).to(torch.long)
        attention_mask = attention_mask.to(runtime.device)
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        started = time.perf_counter()
        hidden = runtime._embed_tokens(input_ids)
        outputs = runtime.model.model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        logits = runtime._logits(outputs.last_hidden_state)
        elapsed = time.perf_counter() - started
        dense_cache = outputs.past_key_values
        padded_len = int(dense_cache.get_seq_length())
        eos = int(runtime.tokenizer.eos_token_id)
        rows: list[tuple[PagedRequest, DynamicCache, int]] = []
        for index, request in enumerate(requests):
            valid = int(prompt_lengths[index].item())
            request.prompt_tokens = valid
            self.allocator.reserve(request, valid + 1)
            rows.append(
                (request, _single_row_cache(dense_cache, index), valid)
            )
            token = runtime._sample(
                logits[index, 0],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated = [token]
            request.cache_len = valid
            request.prefill_seconds = elapsed
            request.prefill_kind = "batched"
            request.finished = (
                token == eos or len(request.generated) >= request.max_new_tokens
            )
        write_dense_rows_to_paged(
            self.manager,
            runtime=runtime,
            rows=rows,
            padded_len=padded_len,
            dims=self.dims,
        )
        self._prefill_forwards += 1
        self._prefill_rows += len(requests)
        return list(requests)

    def _prefill_paged(self, requests: Sequence[PagedRequest]) -> list[PagedRequest]:
        """Prefill each request into the paged pool, reusing cached prefixes.

        With a prefix cache enabled the longest cached block-aligned prefix is
        matched, its blocks are retained, and only the remaining suffix is
        forwarded -- at the position it belongs to, not from zero.
        """

        runtime = self.runtime
        eos = int(runtime.tokenizer.eos_token_id)
        admitted: list[PagedRequest] = []
        for request in requests:
            text = runtime.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
            encoded = runtime.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(torch.long)
            length = int(input_ids.shape[1])
            token_ids = [int(token) for token in input_ids[0].tolist()]
            start_position = self._match_prefix(request, token_ids, length)
            self.allocator.reserve(request, length + 1)
            request.prompt_tokens = length
            request.cached_tokens = start_position
            request.prefill_kind = "paged"
            started = time.perf_counter()
            if start_position >= length:
                # The whole prompt is cached; score the last token in place.
                start_position = max(0, length - 1)
                suffix = input_ids[:, start_position:]
                views = self._make_views(
                    [request],
                    [start_position],
                    max_length=length,
                    new_tokens=1,
                )
                positions = torch.full(
                    (1, 1), start_position, dtype=torch.long, device=runtime.device
                )
            else:
                suffix = input_ids[:, start_position:]
                new_tokens = length - start_position
                views = self._make_views(
                    [request],
                    [start_position],
                    max_length=length,
                    new_tokens=new_tokens,
                )
                positions = torch.arange(
                    start_position, length, dtype=torch.long, device=runtime.device
                ).unsqueeze(0)
            logits = runtime.forward_logits(
                suffix.to(runtime.device), positions, kv_cache=views
            )
            elapsed = time.perf_counter() - started
            token = runtime._sample(
                logits[0, -1],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated = [token]
            request.cache_len = length
            request.prefill_seconds = elapsed
            request.finished = (
                token == eos or len(request.generated) >= request.max_new_tokens
            )
            self._register_prefix(request, token_ids)
            admitted.append(request)
            self._prefill_rows += 1
        self._prefill_forwards += 1
        return admitted

    def _match_prefix(
        self, request: PagedRequest, token_ids: list[int], length: int
    ) -> int:
        """Retain the longest cached prefix and return its token length."""

        cache = self.manager.prefix_cache
        if cache.max_blocks <= 0 or length <= 0:
            return 0
        hashes = PrefixCache.block_hashes(
            token_ids,
            self.manager.block_size,
            namespace=self.prefix_namespace,
        )
        if not hashes:
            return 0
        matched = cache.match(hashes, max_blocks=max(0, cache.max_blocks))
        if not matched:
            return 0
        start = len(matched) * self.manager.block_size
        if start >= length and len(matched) > 1:
            # Never hand back the final block: the engine still has to write
            # this row's tail, and cached blocks must stay immutable.
            cache.allocator.free(matched[-1:])
            matched = matched[:-1]
            start = len(matched) * self.manager.block_size
        if start <= 0:
            return 0
        request.block_ids = [int(block) for block in matched]
        request.reused_blocks = len(matched)
        self._prefix_hits += 1
        self._prefix_reused_tokens += start
        return start

    def _register_prefix(self, request: PagedRequest, token_ids: list[int]) -> None:
        """Publish this request's complete prompt blocks to the prefix cache."""

        cache = self.manager.prefix_cache
        if cache.max_blocks <= 0:
            return
        hashes = PrefixCache.block_hashes(
            token_ids,
            self.manager.block_size,
            namespace=self.prefix_namespace,
        )
        if not hashes:
            return
        blocks = request.block_ids[: len(hashes)]
        if len(blocks) != len(hashes):
            return
        cache.register(hashes, blocks)

    # ----------------------------------------------------------------- decode

    def _make_views(
        self,
        requests: Sequence[PagedRequest],
        seq_lens: Sequence[int],
        *,
        max_length: int,
        new_tokens: int = 1,
    ) -> list[Any]:
        return self.manager.make_views(
            [list(request.block_ids) for request in requests],
            [int(length) for length in seq_lens],
            max_length=int(max_length),
            new_tokens=int(new_tokens),
        )

    def _decode_step(self) -> None:
        if not self._active:
            return
        runtime = self.runtime
        max_length = max(
            int(request.cache_len) + 1 for request in self._active
        )
        views = self._make_views(
            self._active,
            [request.cache_len for request in self._active],
            max_length=max_length,
        )
        input_ids = torch.tensor(
            [[request.generated[-1] if request.generated else 0]
             for request in self._active],
            dtype=torch.long,
            device=runtime.device,
        )
        positions = torch.tensor(
            [[int(request.cache_len)] for request in self._active],
            dtype=torch.long,
            device=runtime.device,
        )
        started = time.perf_counter()
        logits = runtime.forward_logits(input_ids, positions, kv_cache=views)
        logits = logits[:, 0] if logits.dim() == 3 else logits
        self._decode_steps += 1
        eos = int(runtime.tokenizer.eos_token_id)
        finished: list[PagedRequest] = []
        for index, request in enumerate(self._active):
            token = runtime._sample(
                logits[index],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated.append(int(token))
            request.cache_len += 1
            if request.decode_started_at <= 0.0:
                request.decode_started_at = started
            request.decode_seconds = time.perf_counter() - request.decode_started_at
            if (
                int(token) == eos
                or len(request.generated) >= request.max_new_tokens
            ):
                request.finished = True
                finished.append(request)
        if finished:
            self._retire(finished)

    def _retire(self, requests: Sequence[PagedRequest]) -> None:
        done = {id(request) for request in requests}
        keep: list[PagedRequest] = []
        for request in self._active:
            if id(request) in done:
                self.allocator.release(request)
                self._finish(request)
            else:
                keep.append(request)
        self._active = keep

    # --------------------------------------------------------------- lifecycle

    def _collect_pending(self, max_count: int, *, idle: bool) -> list[PagedRequest]:
        """Drain the queue, optionally waiting out an admission grace window.

        When rows are already resident this must stay non-blocking: a decode
        step is often faster than the arrival interval, and waiting for the
        next request there would stall the whole batch.  Only an idle engine
        blocks for work, and only an idle engine applies the grace window --
        it coalesces a burst into one prefill instead of paying the full
        prefill cost per request.
        """

        if max_count <= 0 or self._stop.is_set():
            return []
        pending: list[PagedRequest] = []
        try:
            first = self.queue.get(timeout=0.5) if idle else self.queue.get_nowait()
        except queue.Empty:
            return []
        if first is None:
            self._stop.set()
            return []
        pending.append(first)
        grace = self.admission_grace_ms / 1000.0 if idle else 0.0
        if grace <= 0.0:
            while len(pending) < max_count:
                try:
                    request = self.queue.get_nowait()
                except queue.Empty:
                    break
                if request is None:
                    self._stop.set()
                    break
                pending.append(request)
            return pending
        deadline = time.perf_counter() + grace
        while len(pending) < max_count and not self._stop.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break
            try:
                request = self.queue.get(timeout=remaining)
            except queue.Empty:
                break
            if request is None:
                self._stop.set()
                break
            pending.append(request)
        return pending

    def _admit(self, pending: Sequence[PagedRequest]) -> int:
        if not pending:
            return 0
        try:
            admitted = (
                self._prefill_batched(pending)
                if self.prefill_mode == "batched"
                else self._prefill_paged(pending)
            )
        except SlotOutOfBlocks as exc:
            for request in pending:
                self.allocator.release(request)
                self._fail(request, exc)
            raise
        except BaseException:
            for request in pending:
                self.allocator.release(request)
            raise
        finally:
            for _ in pending:
                self.queue.task_done()
        ready = [request for request in admitted if not request.finished]
        early = [request for request in admitted if request.finished]
        for request in early:
            self.allocator.release(request)
            self._finish(request)
        self._active.extend(ready)
        self._admissions += len(admitted)
        self._max_active_seen = max(self._max_active_seen, len(self._active))
        return len(admitted)

    def _fail(self, request: PagedRequest, exc: BaseException) -> None:
        self._failed += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        if request.future is not None and not request.future.done():
            request.future.set_exception(exc)

    def _finish(self, request: PagedRequest) -> None:
        result = build_generation_result(self.runtime, request)
        if request.future is not None and not request.future.done():
            request.future.set_result(result)
        self._completed += 1

    def _run(self) -> None:
        try:
            torch.cuda.set_device(self.device)
            try:
                import cupy

                cupy.cuda.Device(self.device).use()
            except Exception:
                pass
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                # Decode first: an arrival must never delay a step whose rows
                # are already resident.
                if self._active:
                    self._decode_step()
                pending = self._collect_pending(
                    self.max_batch_size - len(self._active),
                    idle=not self._active,
                )
                if pending:
                    self._admit(pending)
                elif not self._active:
                    time.sleep(0.005)
            except Exception as exc:  # noqa: BLE001
                self._failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                for request in self._active:
                    self.allocator.release(request)
                    self._fail(request, exc)
                self._active = []
        while self._active:
            request = self._active.pop()
            self.allocator.release(request)
            if request.future is not None and not request.future.done():
                request.future.set_exception(
                    RuntimeError("paged continuous batch service closed")
                )

    def status(self) -> dict[str, Any]:
        return {
            "engine": "paged",
            "prefill_mode": self.prefill_mode,
            "max_batch_size": self.max_batch_size,
            "max_cache": self.max_cache,
            "active": len(self._active),
            "queue_depth": int(self.queue.qsize()),
            "completed": self._completed,
            "failed": self._failed,
            "last_error": self._last_error,
            "admissions": self._admissions,
            "prefill_forwards": self._prefill_forwards,
            "prefill_rows": self._prefill_rows,
            "decode_steps": self._decode_steps,
            "max_active_seen": self._max_active_seen,
            "kv": self.allocator.stats(),
            "prefix_cache": {
                **self.manager.prefix_cache.stats(),
                "hits_this_engine": self._prefix_hits,
                "reused_tokens": self._prefix_reused_tokens,
            },
        }

    def close(self, timeout: float = 30.0) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


def _single_row_cache(cache: DynamicCache, row: int) -> DynamicCache:
    single = DynamicCache()
    single.key_cache = [tensor[row : row + 1] for tensor in cache.key_cache]
    single.value_cache = [tensor[row : row + 1] for tensor in cache.value_cache]
    single._seen_tokens = int(cache.get_seq_length())
    return single


__all__ = [
    "PagedContinuousBatchEngine",
    "PagedRequest",
    "PagedSlotAllocator",
    "SlotOutOfBlocks",
    "paged_dims",
    "write_dense_rows_to_paged",
]
