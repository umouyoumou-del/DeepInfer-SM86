"""Continuous batching for the NCCL expert-parallel runtime.

The coordinator uses a fixed number of slots.  Completed slots are removed at
decode-step boundaries and newly queued requests are prefilled into the free
slots while other rows continue decoding.  KV tensors are left-padded inside
each slot so sequences with different prompt lengths can share a batch.
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

from deepinfer.moe.deepseek_v2_runtime import GenerationResult


@dataclass
class ContinuousRequest:
    messages: list[dict[str, str]]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    future: Future | None = None
    prompt_tokens: int = 0
    generated: list[int] = field(default_factory=list)
    cache_len: int = 0
    prefill_seconds: float = 0.0
    decode_started_at: float = 0.0
    decode_seconds: float = 0.0
    finished: bool = False


def _cache_tensor_parts(cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return list(cache.key_cache), list(cache.value_cache)


def pack_caches(
    caches: Sequence[DynamicCache],
    *,
    target_len: int | None = None,
) -> DynamicCache:
    """Pack single-row caches into a left-padded batch cache."""

    if not caches:
        raise ValueError("at least one cache is required")
    keys, values = _cache_tensor_parts(caches[0])
    layer_count = len(keys)
    if target_len is None:
        target_len = max(int(cache.get_seq_length()) for cache in caches)
    target_len = max(1, int(target_len))
    packed_keys: list[torch.Tensor] = []
    packed_values: list[torch.Tensor] = []
    for layer in range(layer_count):
        first_key = keys[layer]
        first_value = values[layer]
        shape = list(first_key.shape)
        shape[0] = len(caches)
        shape[-2] = target_len
        key = first_key.new_zeros(tuple(shape))
        value = first_value.new_zeros(
            (len(caches), *first_value.shape[1:-2], target_len, first_value.shape[-1])
        )
        for index, cache in enumerate(caches):
            source_key = cache.key_cache[layer]
            source_value = cache.value_cache[layer]
            length = int(source_key.shape[-2])
            offset = target_len - length
            key[index, :, offset:, :] = source_key[0]
            value[index, :, offset:, :] = source_value[0]
        packed_keys.append(key)
        packed_values.append(value)
    packed = DynamicCache()
    packed.key_cache = packed_keys
    packed.value_cache = packed_values
    packed._seen_tokens = target_len
    return packed


def append_caches(
    active: DynamicCache,
    new_caches: Sequence[DynamicCache],
) -> DynamicCache:
    """Append new prefill rows to an existing batch cache."""

    if not new_caches:
        return active
    active_len = int(active.get_seq_length())
    new_len = max(int(cache.get_seq_length()) for cache in new_caches)
    target_len = max(active_len, new_len)
    keys, values = _cache_tensor_parts(active)
    packed_keys: list[torch.Tensor] = []
    packed_values: list[torch.Tensor] = []
    for layer in range(len(keys)):
        active_key = keys[layer]
        active_value = values[layer]
        batch = int(active_key.shape[0]) + len(new_caches)
        key = active_key.new_zeros(
            (batch, *active_key.shape[1:-2], target_len, active_key.shape[-1])
        )
        value = active_value.new_zeros(
            (batch, *active_value.shape[1:-2], target_len, active_value.shape[-1])
        )
        # Shift existing rows to the right when the new prompt is longer.
        active_offset = target_len - active_len
        key[: active_key.shape[0], :, active_offset:, :] = active_key
        value[: active_value.shape[0], :, active_offset:, :] = active_value
        for new_index, cache in enumerate(new_caches):
            source_key = cache.key_cache[layer]
            source_value = cache.value_cache[layer]
            length = int(source_key.shape[-2])
            offset = target_len - length
            row = active_key.shape[0] + new_index
            key[row, :, offset:, :] = source_key[0]
            value[row, :, offset:, :] = source_value[0]
        packed_keys.append(key)
        packed_values.append(value)
    packed = DynamicCache()
    packed.key_cache = packed_keys
    packed.value_cache = packed_values
    packed._seen_tokens = target_len
    return packed


def select_caches(cache: DynamicCache, indices: Sequence[int]) -> DynamicCache:
    if not indices:
        raise ValueError("at least one cache row is required")
    index = torch.tensor(indices, dtype=torch.long, device=cache.key_cache[0].device)
    selected = DynamicCache()
    selected.key_cache = [tensor.index_select(0, index) for tensor in cache.key_cache]
    selected.value_cache = [tensor.index_select(0, index) for tensor in cache.value_cache]
    selected._seen_tokens = int(cache.get_seq_length())
    return selected


class ContinuousBatchEngine:
    """A bounded continuous-batch scheduler over one runtime."""

    def __init__(
        self,
        runtime: Any,
        *,
        max_batch_size: int = 4,
        device: int = 0,
    ) -> None:
        self.runtime = runtime
        self.max_batch_size = max(1, int(max_batch_size))
        self.device = int(device)
        self.queue: queue.Queue[ContinuousRequest | None] = queue.Queue()
        self._active: list[ContinuousRequest] = []
        self._active_cache: DynamicCache | None = None
        self._stop = threading.Event()
        self._completed = 0
        self._failed = 0
        self._last_error: str | None = None
        self._last_latency = 0.0
        self._thread = threading.Thread(
            target=self._run, name="ecs-continuous-batch", daemon=True
        )
        self._thread.start()

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
        request = ContinuousRequest(
            messages=messages,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            future=future,
        )
        self.queue.put(request)
        return future

    def _prefill_batch(
        self, requests: Sequence[ContinuousRequest]
    ) -> list[tuple[ContinuousRequest, DynamicCache]]:
        """Prefill all queued requests in one model forward."""

        if not requests:
            return []
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
        batch_cache = outputs.past_key_values
        results: list[tuple[ContinuousRequest, DynamicCache]] = []
        eos = int(runtime.tokenizer.eos_token_id)
        for index, request in enumerate(requests):
            token = runtime._sample(
                logits[index, 0],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.prompt_tokens = int(prompt_lengths[index].item())
            request.generated = [token]
            request.cache_len = request.prompt_tokens
            request.prefill_seconds = elapsed
            request.finished = (
                token == eos or len(request.generated) >= request.max_new_tokens
            )
            results.append((request, select_caches(batch_cache, [index])))
        return results


    def _admit(self, max_count: int) -> list[ContinuousRequest]:
        pending: list[ContinuousRequest] = []
        for _ in range(max_count):
            try:
                request = self.queue.get_nowait()
            except queue.Empty:
                break
            if request is None:
                break
            pending.append(request)
        if not pending:
            return []
        prefetched = self._prefill_batch(pending)
        admitted = [request for request, _ in prefetched]
        caches = [cache for _, cache in prefetched]
        for _ in pending:
            self.queue.task_done()
        if not admitted:
            return []
        if self._active_cache is None:
            self._active_cache = pack_caches(caches)
        else:
            self._active_cache = append_caches(self._active_cache, caches)
        self._active.extend(admitted)
        finished = [request for request in admitted if request.finished]
        for request in finished:
            self._finish(request)
        if finished:
            keep = [index for index, request in enumerate(self._active) if not request.finished]
            if keep:
                self._active_cache = select_caches(self._active_cache, keep)
                self._active = [self._active[index] for index in keep]
            else:
                self._active_cache = None
                self._active = []
        return admitted

    def _finish(self, request: ContinuousRequest) -> None:
        runtime = self.runtime
        eos = int(runtime.tokenizer.eos_token_id)
        output_ids = [token for token in request.generated if token != eos]
        text = runtime.tokenizer.decode(output_ids, skip_special_tokens=True)
        total_decode_tokens = max(0, len(request.generated) - 1)
        decode_seconds = max(0.0, request.decode_seconds)
        aggregate = (
            total_decode_tokens / decode_seconds if decode_seconds > 0 else 0.0
        )
        result = GenerationResult(
            text=text,
            prompt_tokens=request.prompt_tokens,
            completion_tokens=len(request.generated),
            prefill_seconds=request.prefill_seconds,
            decode_seconds=decode_seconds,
            decode_tokens_per_second=aggregate,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            gpu_free_bytes_after_load=runtime.gpu_free_bytes_after_load,
            shared_experts_device=runtime.shared_experts_device,
            lm_head_device=runtime.lm_head_device,
            token_ids=list(request.generated),
            timings={
                **(
                    dict(runtime.expert_store.timings)
                    if runtime.expert_store is not None
                    else {}
                ),
                "cpu_moe_seconds": float(runtime.cpu_moe_seconds),
                "expert_device_chunks": float(runtime.expert_device_chunks),
                "remote_expert_chunks": float(runtime.remote_expert_chunks),
                **{
                    f"expert_device_{device}_experts": float(count)
                    for device, count in runtime.expert_device_counts.items()
                },
            },
            dense_layer_indices=list(runtime.dense_layer_indices),
            moe_layer_indices=list(runtime.moe_layer_indices),
            dense_mlp_devices=dict(runtime.dense_mlp_devices),
            fused_moe_enabled=runtime.fused_moe_enabled,
            fused_moe_chunks=runtime.fused_moe_chunks,
            fallback_moe_chunks=runtime.fallback_moe_chunks,
            cpu_moe_layers=len(runtime.cpu_moe_layer_indices),
            cpu_moe_chunks=runtime.cpu_moe_chunks,
            cpu_experts_per_layer=runtime.requested_cpu_experts_per_layer,
            cpu_expert_chunks=runtime.cpu_expert_chunks,
            ecs=(
                runtime.ecs_controller.snapshot()
                if runtime.ecs_controller is not None
                else None
            ),
        )
        if request.future is not None:
            request.future.set_result(result)
        self._completed += 1

    def _decode_one(self) -> None:
        if not self._active or self._active_cache is None:
            return
        runtime = self.runtime
        cache = self._active_cache
        cache_len = int(cache.get_seq_length())
        batch_size = len(self._active)
        mask = torch.zeros(
            (batch_size, cache_len + 1),
            dtype=torch.long,
            device=runtime.device,
        )
        current_tokens = []
        positions = []
        for index, request in enumerate(self._active):
            valid = min(cache_len, int(request.cache_len))
            mask[index, cache_len - valid :] = 1
            current_tokens.append(
                [request.generated[-1] if request.generated else 0]
            )
            positions.append([valid])
        current = torch.tensor(current_tokens, dtype=torch.long, device=runtime.device)
        position_ids = torch.tensor(
            positions, dtype=torch.long, device=runtime.device
        )
        decode_started = time.perf_counter()
        hidden = runtime._embed_tokens(current)
        outputs = runtime.model.model(
            inputs_embeds=hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        self._active_cache = outputs.past_key_values
        logits = runtime._logits(outputs.last_hidden_state)
        finish_count = 0
        for index, request in enumerate(self._active):
            token = runtime._sample(
                logits[index, 0],
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
            )
            request.generated.append(token)
            request.cache_len += 1
            if request.decode_started_at <= 0.0:
                request.decode_started_at = decode_started
            request.decode_seconds = time.perf_counter() - request.decode_started_at
            if (
                token == int(runtime.tokenizer.eos_token_id)
                or len(request.generated) >= request.max_new_tokens
            ):
                request.finished = True
                finish_count += 1
        if finish_count:
            keep = [
                index
                for index, request in enumerate(self._active)
                if not request.finished
            ]
            for request in self._active:
                if request.finished:
                    self._finish(request)
            if keep:
                self._active_cache = select_caches(self._active_cache, keep)
                self._active = [self._active[index] for index in keep]
            else:
                self._active_cache = None
                self._active = []

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
                if not self._active:
                    first = self.queue.get(timeout=0.5)
                    if first is None:
                        break
                    self.queue.put(first)
                self._admit(self.max_batch_size - len(self._active))
                if self._active:
                    self._decode_one()
                else:
                    time.sleep(0.01)
            except Exception as exc:
                self._failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                while self._active:
                    request = self._active.pop()
                    if request.future is not None and not request.future.done():
                        request.future.set_exception(exc)
                self._active_cache = None
        while self._active:
            request = self._active.pop()
            if request.future is not None and not request.future.done():
                request.future.set_exception(
                    RuntimeError("continuous batch service closed")
                )

    def status(self) -> dict[str, Any]:
        return {
            "max_batch_size": self.max_batch_size,
            "active": len(self._active),
            "queue_depth": int(self.queue.qsize()),
            "completed": self._completed,
            "failed": self._failed,
            "last_error": self._last_error,
        }

    def close(self, timeout: float = 30.0) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


__all__ = [
    "ContinuousBatchEngine",
    "ContinuousRequest",
    "append_caches",
    "pack_caches",
    "select_caches",
]
