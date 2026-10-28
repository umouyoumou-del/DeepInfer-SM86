"""Native DeepSeek-V2-Lite MoE execution with extreme CPU/GPU offload.

Design constraints
-------------------
The target machine is a 4 GB RTX 3050 Laptop GPU with 16 GB of host RAM.
DeepSeek-V2-Lite has ~15.7 B parameters; even Q4_K_M is a 9.65 GiB file.
Loading the checkpoint into anonymous host memory or materializing all 1,664
routed experts on the GPU is therefore not an option.

This runtime keeps:

* attention, norms, router and the first dense MLP resident on the GPU;
* routed expert weights packed in the GGUF file / OS page cache;
* only the experts selected by the router present on the GPU at a time;
* shared experts and the LM head on the GPU when the post-attention free
  memory budget allows, otherwise on the CPU.

The hot path reads packed Q4_K/Q5_0/Q8_0 expert slices, copies those packed
bytes over PCIe, dequantizes them with SM86 CUDA kernels, and executes grouped
batched matrix multiplications.  Nothing in the expert path is ever
materialized for all 64 experts or all 27 layers at once.
"""

from __future__ import annotations

import concurrent.futures
import gc
import os
import threading
from collections import OrderedDict
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from deepinfer.logger import init_logger
from deepinfer.moe.fused_moe import fused_moe_supported
from deepinfer.moe.grouped_moe import (
    fused_down_gemm_grouped,
    fused_gate_up_gemm_grouped,
)
from deepinfer.moe.ptr_grouped_moe import (
    fused_down_gemm_ptr_grouped,
    fused_gate_up_gemm_ptr_grouped,
)
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.gpu_quant import dequantize_matrix
from deepinfer.moe.layout import split_layer_indices

logger = init_logger(__name__)

_MIB = 1024**2


@dataclass(frozen=True, slots=True)
class ExpertTensorSpec:
    name: str
    in_features: int
    out_features: int
    expert_count: int
    quant_type: str
    bytes_per_expert: int
    row_bytes: int


@dataclass(slots=True)
class StagedExpertBatch:
    specs: tuple[ExpertTensorSpec, ...]
    packed: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    gate_ready: torch.cuda.Event | None
    up_ready: torch.cuda.Event | None
    down_ready: torch.cuda.Event | None
    expert_ids: tuple[int, ...]
    keepalive: tuple[Any, ...] = ()

    @staticmethod
    def _wait(event: torch.cuda.Event | None, tensor: torch.Tensor) -> None:
        if event is not None:
            stream = torch.cuda.current_stream()
            stream.wait_event(event)
            tensor.record_stream(stream)

    def wait_for_gate_up(self) -> None:
        self._wait(self.gate_ready, self.packed[0])
        self._wait(self.up_ready, self.packed[1])

    def wait_for_down(self) -> None:
        self._wait(self.down_ready, self.packed[2])


@dataclass(slots=True)
class PointerExpertBatch:
    specs: tuple[ExpertTensorSpec, ...]
    pointers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    entries: tuple[tuple[torch.Tensor, ...], ...]
    keepalive: tuple[Any, ...] = ()


@dataclass(slots=True)
class ExpertPrefetch:
    layer_idx: int
    target_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    slot: int
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    futures: list[Any]

    def wait(self) -> None:
        for future in self.futures:
            future.result()


@dataclass(slots=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode_tokens_per_second: float
    peak_gpu_allocated_bytes: int
    peak_gpu_reserved_bytes: int
    gpu_free_bytes_after_load: int
    shared_experts_device: str
    lm_head_device: str
    token_ids: list[int]
    timings: dict[str, float]
    dense_layer_indices: list[int]
    moe_layer_indices: list[int]
    dense_mlp_devices: dict[int, str]
    fused_moe_enabled: bool
    fused_moe_chunks: int
    fallback_moe_chunks: int
    cpu_moe_layers: int
    cpu_moe_chunks: int
    cpu_experts_per_layer: int
    cpu_expert_chunks: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated_bytes,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved_bytes,
            "gpu_free_bytes_after_load": self.gpu_free_bytes_after_load,
            "shared_experts_device": self.shared_experts_device,
            "lm_head_device": self.lm_head_device,
            "token_ids": self.token_ids,
            "timings": self.timings,
            "dense_layer_indices": self.dense_layer_indices,
            "moe_layer_indices": self.moe_layer_indices,
            "dense_mlp_devices": self.dense_mlp_devices,
            "fused_moe_enabled": self.fused_moe_enabled,
            "fused_moe_chunks": self.fused_moe_chunks,
            "fallback_moe_chunks": self.fallback_moe_chunks,
            "cpu_moe_layers": self.cpu_moe_layers,
            "cpu_moe_chunks": self.cpu_moe_chunks,
            "cpu_experts_per_layer": self.cpu_experts_per_layer,
            "cpu_expert_chunks": self.cpu_expert_chunks,
        }


def _expert_spec(index: GGUFIndex, layer_idx: int, projection: str) -> ExpertTensorSpec:
    name = f"blk.{layer_idx}.ffn_{projection}_exps.weight"
    info = index.tensor(name)
    if len(info.dims) != 3:
        raise ValueError(f"expected a 3-D expert tensor for {name}, got {info.dims}")
    in_features, out_features, expert_count = info.dims
    row_bytes = int(in_features) // info.block_size * info.type_size
    return ExpertTensorSpec(
        name=name,
        in_features=int(in_features),
        out_features=int(out_features),
        expert_count=int(expert_count),
        quant_type=info.qtype.name,
        bytes_per_expert=info.nbytes // int(expert_count),
        row_bytes=row_bytes,
    )


class GGUFExpertStore:
    """Packed routed-expert reader and GPU dequantizer.

    The store owns one file handle and only allocates memory proportional to
    the number of experts selected for one layer.  The caller is responsible
    for deleting returned GPU tensors before the next layer if memory pressure
    matters.
    """

    def __init__(
        self,
        index: GGUFIndex,
        *,
        layer_indices: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        max_gpu_spill_mb: int = 512,
        parallel_read_workers: int = 8,
        expert_cache_mb: int = 0,
        gpu_expert_cache_mb: int = 0,
        predictive_cache_mb: int = 256,
    ) -> None:
        self.index = index
        self.device = device
        self.dtype = dtype
        self.max_gpu_spill_bytes = int(max_gpu_spill_mb) * _MIB
        self.parallel_read_workers = max(1, int(parallel_read_workers))
        self._cache_limit_bytes = max(0, int(expert_cache_mb)) * _MIB
        self._layer_count = max(1, len(set(layer_indices)))
        self._cache_limit_per_layer = self._cache_limit_bytes // self._layer_count
        self._packed_cache: dict[int, OrderedDict[tuple[int, str, int], torch.Tensor]] = {}
        self._packed_cache_bytes = 0
        self._packed_cache_bytes_by_layer: dict[int, int] = {}
        self._expert_access_counts: dict[tuple[str, int], int] = {}
        self._gpu_cache_limit_bytes = max(0, int(gpu_expert_cache_mb)) * _MIB
        self._gpu_cache_per_layer = (
            self._gpu_cache_limit_bytes // self._layer_count
        )
        self._gpu_expert_cache: dict[
            int, OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = {}
        self._gpu_cache_bytes = 0
        self._gpu_cache_bytes_by_layer: dict[int, int] = {}
        self._gpu_expert_access_counts: dict[tuple[int, int], int] = {}
        self._gpu_cache_active = True
        self._specs: dict[tuple[int, str], ExpertTensorSpec] = {}
        for layer_idx in layer_indices:
            for projection in ("gate", "up", "down"):
                self._specs[(layer_idx, projection)] = _expert_spec(
                    index, layer_idx, projection
                )
        self._file = index.path.open("rb")
        self._read_thread_local = threading.local()
        self._read_handles: list[Any] = []
        self._read_handles_lock = threading.Lock()
        self._read_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.parallel_read_workers,
            thread_name_prefix="deepinfer-moe-read",
        )
        self._pinned_buffers: dict[tuple[str, int], torch.Tensor] = {}
        self._pinned_copy_events: dict[tuple[str, int], torch.cuda.Event] = {}
        self._pinned_slot_count = 2
        self._pinned_slot = 0
        self._copy_streams = (
            [torch.cuda.Stream(device=device) for _ in range(self._pinned_slot_count)]
            if device.type == "cuda"
            else []
        )
        self._prefetch_slot_count = 2
        self._prefetch_slot = 0
        self._prefetch_buffers: dict[tuple[str, int], torch.Tensor] = {}
        self._prefetch_events: dict[tuple[str, int], torch.cuda.Event] = {}
        self._predictive_cache_limit = max(
            0, int(predictive_cache_mb)
        ) * _MIB
        self._predictive_cache: OrderedDict[tuple[int, str, int], torch.Tensor] = OrderedDict()
        self._predictive_cache_bytes = 0
        self._predictive_pending: set[tuple[int, str, int]] = set()
        self._predictive_lock = threading.RLock()
        self._overlap_down_copy = (
            bool(self._copy_streams)
            and os.environ.get("DEEPINFER_DISABLE_EXPERT_COPY_STREAM") != "1"
        )
        self.timings: dict[str, float] = {
            "expert_read_seconds": 0.0,
            "expert_h2d_seconds": 0.0,
            "expert_dequant_launch_seconds": 0.0,
            "expert_load_calls": 0.0,
            "expert_cache_hits": 0.0,
            "expert_cache_misses": 0.0,
            "expert_cache_bytes": 0.0,
            "expert_cache_evictions": 0.0,
            "gpu_cache_hits": 0.0,
            "gpu_cache_misses": 0.0,
            "gpu_cache_bytes": 0.0,
            "gpu_cache_evictions": 0.0,
            "predictive_cache_hits": 0.0,
            "predictive_cache_misses": 0.0,
            "predictive_prefetch_reads": 0.0,
        }

    def supports_fused_moe(self) -> bool:
        supported_types = {"Q4_K", "Q5_0", "Q8_0"}
        return all(
            spec.quant_type in supported_types and spec.in_features <= 4096
            for spec in self._specs.values()
        )

    def close(self) -> None:
        self._read_executor.shutdown(wait=True, cancel_futures=True)
        self._packed_cache.clear()
        self._packed_cache_bytes = 0
        self._packed_cache_bytes_by_layer.clear()
        self._gpu_expert_cache.clear()
        self._gpu_cache_bytes = 0
        self._gpu_cache_bytes_by_layer.clear()
        with self._read_handles_lock:
            handles = list(self._read_handles)
            self._read_handles.clear()
        for handle in handles:
            handle.close()
        try:
            self._file.close()
        finally:
            self.index.close()

    def _reader_file(self):
        handle = getattr(self._read_thread_local, "file", None)
        if handle is None:
            handle = self.index.path.open("rb")
            self._read_thread_local.file = handle
            with self._read_handles_lock:
                self._read_handles.append(handle)
        return handle

    def gpu_cache_configured(self) -> bool:
        return self._gpu_cache_per_layer > 0

    def gpu_cache_enabled(self) -> bool:
        return self._gpu_cache_active and self.gpu_cache_configured()

    def set_gpu_cache_active(self, active: bool) -> None:
        self._gpu_cache_active = bool(active)

    def _gpu_cache_get(
        self, layer_idx: int, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.gpu_cache_enabled():
            return None
        layer_cache = self._gpu_expert_cache.get(layer_idx)
        if layer_cache is None:
            return None
        entry = layer_cache.get(expert_id)
        if entry is None:
            return None
        layer_cache.move_to_end(expert_id)
        self.timings["gpu_cache_hits"] += 1.0
        return entry

    def _gpu_cache_put(
        self,
        layer_idx: int,
        expert_id: int,
        entry: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> bool:
        if not self.gpu_cache_enabled():
            return False
        size = sum(tensor.numel() for tensor in entry)
        if size > self._gpu_cache_per_layer:
            return False
        layer_cache = self._gpu_expert_cache.setdefault(layer_idx, OrderedDict())
        old = layer_cache.pop(expert_id, None)
        if old is not None:
            old_size = sum(tensor.numel() for tensor in old)
            self._gpu_cache_bytes -= old_size
            self._gpu_cache_bytes_by_layer[layer_idx] -= old_size
        while (
            layer_cache
            and self._gpu_cache_bytes_by_layer.get(layer_idx, 0) + size
            > self._gpu_cache_per_layer
        ):
            _, evicted = layer_cache.popitem(last=False)
            evicted_size = sum(tensor.numel() for tensor in evicted)
            self._gpu_cache_bytes -= evicted_size
            self._gpu_cache_bytes_by_layer[layer_idx] -= evicted_size
            self.timings["gpu_cache_evictions"] += 1.0
        layer_cache[expert_id] = entry
        self._gpu_cache_bytes += size
        self._gpu_cache_bytes_by_layer[layer_idx] = (
            self._gpu_cache_bytes_by_layer.get(layer_idx, 0) + size
        )
        self.timings["gpu_cache_bytes"] = float(self._gpu_cache_bytes)
        return True

    def clear_gpu_cache(self) -> None:
        self._gpu_expert_cache.clear()
        self._gpu_cache_bytes = 0
        self._gpu_cache_bytes_by_layer.clear()

    def load_pointer_experts(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
    ) -> PointerExpertBatch:
        """Load experts and return per-expert GPU pointers.

        Cached experts bypass file reads and H2D. Missing experts are staged
        through the normal double-buffered packed path, then exposed as views
        or cloned into the GPU cache.
        """

        if not expert_ids:
            raise ValueError("at least one expert must be selected")
        if not self.gpu_cache_enabled():
            raise RuntimeError("GPU expert cache is disabled")
        ids = tuple(int(expert_id) for expert_id in expert_ids)
        entries: list[tuple[torch.Tensor, ...] | None] = [None] * len(ids)
        missing: list[tuple[int, int]] = []
        for position, expert_id in enumerate(ids):
            cached = self._gpu_cache_get(layer_idx, expert_id)
            if cached is not None:
                entries[position] = cached
            else:
                self.timings["gpu_cache_misses"] += 1.0
                missing.append((position, expert_id))

        keepalive: tuple[Any, ...] = ()
        if missing:
            batch = self.load_packed_experts(
                layer_idx, [expert_id for _, expert_id in missing]
            )
            batch.wait_for_gate_up()
            batch.wait_for_down()
            sizes = tuple(spec.bytes_per_expert for spec in batch.specs)
            for missing_position, (original_position, expert_id) in enumerate(missing):
                views = tuple(
                    batch.packed[projection][
                        missing_position * sizes[projection] :
                        (missing_position + 1) * sizes[projection]
                    ]
                    for projection in range(3)
                )
                key = (layer_idx, expert_id)
                access_count = self._gpu_expert_access_counts.get(key, 0) + 1
                self._gpu_expert_access_counts[key] = access_count
                if access_count >= 2:
                    cached_entry = tuple(view.clone() for view in views)
                    if self._gpu_cache_put(layer_idx, expert_id, cached_entry):
                        views = cached_entry
                entries[original_position] = views
            keepalive = (batch,)

        resolved = tuple(entry for entry in entries if entry is not None)
        if len(resolved) != len(ids):
            raise RuntimeError("internal error: expert entry was not resolved")
        pointers = tuple(
            torch.tensor(
                [entry[projection].data_ptr() for entry in resolved],
                dtype=torch.int64,
                device=self.device,
            )
            for projection in range(3)
        )
        specs = tuple(
            self._specs[(layer_idx, projection)]
            for projection in ("gate", "up", "down")
        )
        return PointerExpertBatch(
            specs=specs,
            pointers=pointers,
            entries=resolved,
            keepalive=keepalive,
        )

    def _cache_get(
        self, layer_idx: int, key: tuple[str, int]
    ) -> torch.Tensor | None:
        layer_cache = self._packed_cache.get(layer_idx)
        if layer_cache is None:
            return None
        value = layer_cache.get(key)
        if value is None:
            return None
        layer_cache.move_to_end(key)
        self.timings["expert_cache_hits"] += 1.0
        return value

    def _cache_put(
        self, layer_idx: int, key: tuple[str, int], value: torch.Tensor
    ) -> None:
        if self._cache_limit_per_layer <= 0:
            return
        layer_cache = self._packed_cache.setdefault(layer_idx, OrderedDict())
        old = layer_cache.pop(key, None)
        if old is not None:
            self._packed_cache_bytes -= old.numel()
            self._packed_cache_bytes_by_layer[layer_idx] -= old.numel()
        size = value.numel()
        while (
            layer_cache
            and self._packed_cache_bytes_by_layer.get(layer_idx, 0) + size
            > self._cache_limit_per_layer
        ):
            _, evicted = layer_cache.popitem(last=False)
            self._packed_cache_bytes -= evicted.numel()
            self._packed_cache_bytes_by_layer[layer_idx] -= evicted.numel()
            self.timings["expert_cache_evictions"] += 1.0
        if size > self._cache_limit_per_layer:
            return
        layer_cache[key] = value
        self._packed_cache_bytes += size
        self._packed_cache_bytes_by_layer[layer_idx] = (
            self._packed_cache_bytes_by_layer.get(layer_idx, 0) + size
        )
        self.timings["expert_cache_bytes"] = float(self._packed_cache_bytes)

    def _predictive_cache_get(
        self, key: tuple[int, str, int]
    ) -> torch.Tensor | None:
        with self._predictive_lock:
            value = self._predictive_cache.get(key)
            if value is None:
                return None
            self._predictive_cache.move_to_end(key)
            self.timings["predictive_cache_hits"] += 1.0
            return value

    def _predictive_cache_put(
        self, key: tuple[int, str, int], value: torch.Tensor
    ) -> None:
        with self._predictive_lock:
            self._predictive_pending.discard(key)
            if self._predictive_cache_limit <= 0:
                return
            size = value.numel()
            old = self._predictive_cache.pop(key, None)
            if old is not None:
                self._predictive_cache_bytes -= old.numel()
            while (
                self._predictive_cache
                and self._predictive_cache_bytes + size
                > self._predictive_cache_limit
            ):
                _, evicted = self._predictive_cache.popitem(last=False)
                self._predictive_cache_bytes -= evicted.numel()
            if size > self._predictive_cache_limit:
                return
            self._predictive_cache[key] = value
            self._predictive_cache_bytes += size

    def prefetch_predictive(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None:
        """Warm packed bytes for predicted next-layer experts in background."""

        if self._predictive_cache_limit <= 0 or not expert_ids:
            return
        for projection in ("gate", "up", "down"):
            spec = self._specs[(layer_idx, projection)]
            for expert_id in expert_ids:
                key = (int(layer_idx), spec.name, int(expert_id))
                with self._predictive_lock:
                    if (
                        key in self._predictive_cache
                        or key in self._predictive_pending
                    ):
                        continue
                    self._predictive_pending.add(key)
                data = torch.empty(
                    spec.bytes_per_expert, dtype=torch.uint8
                )
                future = self._read_executor.submit(
                    self._read_expert_slice,
                    spec,
                    int(expert_id),
                    memoryview(data.numpy()),
                )

                def done(completed, cache_key=key, tensor=data) -> None:
                    try:
                        completed.result()
                    except Exception:
                        with self._predictive_lock:
                            self._predictive_pending.discard(cache_key)
                        return
                    self.timings["predictive_prefetch_reads"] += 1.0
                    self._predictive_cache_put(cache_key, tensor)

                future.add_done_callback(done)

    def _cache_candidate(self, key: tuple[str, int]) -> bool:
        if self._cache_limit_per_layer <= 0:
            return False
        count = self._expert_access_counts.get(key, 0) + 1
        self._expert_access_counts[key] = count
        # First touch is often a one-off router choice.  Admit on the second
        # touch to avoid filling a small cache with random expert churn.
        return count >= 2

    def _read_expert_slice(
        self,
        spec: ExpertTensorSpec,
        expert_id: int,
        destination: memoryview,
    ) -> None:
        base_offset = self.index.data_start + self.index.tensor(spec.name).offset
        handle = self._reader_file()
        handle.seek(base_offset + int(expert_id) * spec.bytes_per_expert)
        written = handle.readinto(destination)
        if written != spec.bytes_per_expert:
            raise OSError(
                f"short read for {spec.name} expert {expert_id}: "
                f"wanted {spec.bytes_per_expert}, got {written}"
            )

    def prefetch_experts(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        *,
        skip_cached: bool = False,
    ) -> ExpertPrefetch:
        """Start background reads for one expert chunk into pinned buffers.

        ``skip_cached`` leaves experts whose three projections are already in
        the predictive cache out of the read set; the consumer combines the
        remaining prefetched projections with that cache.
        """

        if not expert_ids:
            raise ValueError("at least one expert must be selected")
        target_ids = tuple(int(expert_id) for expert_id in expert_ids)
        projections = ("gate", "up", "down")
        specs = tuple(
            self._specs[(layer_idx, projection)] for projection in projections
        )
        source_ids = target_ids
        if skip_cached:
            source_ids = tuple(
                expert_id
                for expert_id in target_ids
                if not all(
                    self._predictive_cache_get(
                        (int(layer_idx), spec.name, expert_id)
                    )
                    is not None
                    for spec in specs
                )
            )
        slot = self._prefetch_slot % self._prefetch_slot_count
        self._prefetch_slot += 1
        buffers: list[torch.Tensor] = []
        for projection, spec in zip(projections, specs, strict=True):
            total_bytes = spec.bytes_per_expert * len(source_ids)
            key = (projection, slot)
            previous_copy = self._prefetch_events.get(key)
            if previous_copy is not None:
                previous_copy.synchronize()
            buffer = self._prefetch_buffers.get(key)
            if buffer is None or buffer.numel() < total_bytes:
                buffer = torch.empty(
                    total_bytes,
                    dtype=torch.uint8,
                    pin_memory=self.device.type == "cuda",
                )
                self._prefetch_buffers[key] = buffer
            buffers.append(buffer[:total_bytes])

        futures: list[Any] = []
        for spec, buffer in zip(specs, buffers, strict=True):
            view = memoryview(buffer.numpy())
            size = spec.bytes_per_expert
            for position, expert_id in enumerate(source_ids):
                destination = view[position * size : (position + 1) * size]
                futures.append(
                    self._read_executor.submit(
                        self._read_expert_slice,
                        spec,
                        expert_id,
                        destination,
                    )
                )
        self.timings["expert_prefetch_calls"] = (
            self.timings.get("expert_prefetch_calls", 0.0) + 1.0
        )
        return ExpertPrefetch(
            layer_idx=int(layer_idx),
            target_ids=target_ids,
            source_ids=source_ids,
            slot=slot,
            buffers=(buffers[0], buffers[1], buffers[2]),
            futures=futures,
        )

    def _stage_experts(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        *,
        device: torch.device | None = None,
        async_gate_up: bool = False,
        prefetched: ExpertPrefetch | None = None,
    ) -> StagedExpertBatch:
        """Read packed expert bytes and stage them on the target device.

        The returned down projection is not guaranteed to be ready yet when
        the down copy runs on a separate CUDA stream.  Call
        :meth:`StagedExpertBatch.wait_for_down` before consuming it.
        """

        if not expert_ids:
            raise ValueError("at least one expert must be selected")
        target_device = self.device if device is None else device
        expert_id_tuple = tuple(int(expert_id) for expert_id in expert_ids)
        expert_count = len(expert_id_tuple)
        projections = ("gate", "up", "down")
        specs = tuple(
            self._specs[(layer_idx, projection)] for projection in projections
        )
        use_cuda = target_device.type == "cuda"
        if prefetched is not None:
            if (
                prefetched.layer_idx != layer_idx
                or prefetched.target_ids != expert_id_tuple
            ):
                raise ValueError(
                    "prefetched expert chunk does not match request"
                )
            started = time.perf_counter()
            prefetched.wait()
            if prefetched.source_ids == expert_id_tuple:
                slot = prefetched.slot
                packed_cpu_tensors = list(prefetched.buffers)
            else:
                # Predictive mode can cover only a subset.  Build an actual-
                # order pinned staging buffer from prefetched/cache bytes and
                # read only the remaining experts.
                slot = self._pinned_slot % self._pinned_slot_count
                self._pinned_slot += 1
                packed_cpu_tensors = []
                source_positions = {
                    expert_id: position
                    for position, expert_id in enumerate(prefetched.source_ids)
                }
                read_futures = []
                cache_misses: list[tuple[tuple[str, int], memoryview]] = []
                for projection_index, (projection, spec) in enumerate(
                    zip(projections, specs, strict=True)
                ):
                    total_bytes = spec.bytes_per_expert * expert_count
                    key = (projection, slot)
                    previous_copy = self._pinned_copy_events.get(key)
                    if previous_copy is not None:
                        previous_copy.synchronize()
                    buffer = self._pinned_buffers.get(key)
                    if buffer is None or buffer.numel() < total_bytes:
                        buffer = torch.empty(
                            total_bytes,
                            dtype=torch.uint8,
                            pin_memory=use_cuda,
                        )
                        self._pinned_buffers[key] = buffer
                    destination_all = buffer[:total_bytes]
                    packed_cpu_tensors.append(destination_all)
                    view = memoryview(destination_all.numpy())
                    size = spec.bytes_per_expert
                    source_view = memoryview(
                        prefetched.buffers[projection_index].numpy()
                    )
                    for position, expert_id in enumerate(expert_id_tuple):
                        destination = view[
                            position * size : (position + 1) * size
                        ]
                        source_position = source_positions.get(expert_id)
                        if source_position is not None:
                            destination[:] = source_view[
                                source_position * size :
                                (source_position + 1) * size
                            ]
                            continue
                        cache_key = (spec.name, expert_id)
                        cached = self._cache_get(layer_idx, cache_key)
                        if cached is not None:
                            destination[:] = memoryview(cached.numpy())
                            continue
                        cached = self._predictive_cache_get(
                            (layer_idx, spec.name, expert_id)
                        )
                        if cached is not None:
                            destination[:] = memoryview(cached.numpy())
                            continue
                        self.timings["expert_cache_misses"] += 1.0
                        cache_misses.append((cache_key, destination))
                        read_futures.append(
                            self._read_executor.submit(
                                self._read_expert_slice,
                                spec,
                                expert_id,
                                destination,
                            )
                        )
                for future in read_futures:
                    future.result()
                for cache_key, destination in cache_misses:
                    if not self._cache_candidate(cache_key):
                        continue
                    data = torch.from_numpy(
                        np.frombuffer(destination, dtype=np.uint8).copy()
                    )
                    self._cache_put(layer_idx, cache_key, data)
            self.timings["expert_read_seconds"] += time.perf_counter() - started
        else:
            slot = self._pinned_slot % self._pinned_slot_count
            self._pinned_slot += 1
            packed_cpu_tensors: list[torch.Tensor] = []
            for projection, spec in zip(projections, specs, strict=True):
                total_bytes = spec.bytes_per_expert * expert_count
                key = (projection, slot)
                previous_copy = self._pinned_copy_events.get(key)
                if previous_copy is not None:
                    # Two slots let the reader prepare chunk N+1 while chunk N
                    # is still being consumed by the GPU.  A slot is only
                    # overwritten after its previous H2D copy completed.
                    previous_copy.synchronize()
                buffer = self._pinned_buffers.get(key)
                if buffer is None or buffer.numel() < total_bytes:
                    buffer = torch.empty(
                        total_bytes,
                        dtype=torch.uint8,
                        pin_memory=use_cuda,
                    )
                    self._pinned_buffers[key] = buffer
                packed_cpu_tensors.append(buffer[:total_bytes])

            started = time.perf_counter()
            read_futures = []
            cache_misses: list[tuple[tuple[str, int], memoryview]] = []
            for spec, buffer in zip(specs, packed_cpu_tensors, strict=True):
                view = memoryview(buffer.numpy())
                size = spec.bytes_per_expert
                for position, expert_id in enumerate(expert_id_tuple):
                    destination = view[position * size : (position + 1) * size]
                    key = (spec.name, expert_id)
                    cached = self._cache_get(layer_idx, key)
                    if cached is not None:
                        destination[:] = memoryview(cached.numpy())
                        continue
                    cached = self._predictive_cache_get(
                        (layer_idx, key[0], key[1])
                    )
                    if cached is not None:
                        destination[:] = memoryview(cached.numpy())
                        continue
                    self.timings["expert_cache_misses"] += 1.0
                    cache_misses.append((key, destination))
                    read_futures.append(
                        self._read_executor.submit(
                            self._read_expert_slice,
                            spec,
                            expert_id,
                            destination,
                        )
                    )
            for future in read_futures:
                future.result()
            for key, destination in cache_misses:
                if not self._cache_candidate(key):
                    continue
                data = torch.from_numpy(
                    np.frombuffer(destination, dtype=np.uint8).copy()
                )
                self._cache_put(layer_idx, key, data)
            self.timings["expert_read_seconds"] += time.perf_counter() - started

        gate_ready = None
        up_ready = None
        down_ready = None
        if use_cuda:
            started = time.perf_counter()
            compute_stream = torch.cuda.current_stream()
            copy_stream = (
                self._copy_streams[slot] if self._overlap_down_copy else None
            )
            if async_gate_up and copy_stream is not None:
                # Prefill pipeline: all three copies are issued on the slot's
                # copy stream so the next expert chunk can be transferred while
                # the current grouped kernels are still running.  The consumer
                # waits for gate/up before compute and for down later.
                with torch.cuda.stream(copy_stream):
                    packed_gate = packed_cpu_tensors[0].to(
                        target_device, non_blocking=True
                    )
                    gate_ready = torch.cuda.Event()
                    gate_ready.record(copy_stream)
                    packed_up = packed_cpu_tensors[1].to(
                        target_device, non_blocking=True
                    )
                    up_ready = torch.cuda.Event()
                    up_ready.record(copy_stream)
                    packed_down = packed_cpu_tensors[2].to(
                        target_device, non_blocking=True
                    )
                    down_ready = torch.cuda.Event()
                    down_ready.record(copy_stream)
            else:
                # Decode stays on the compute stream: on WDDM, moving the
                # small gate/up copies to a second stream adds event overhead
                # without useful overlap for a single expert chunk.
                packed_gate = packed_cpu_tensors[0].to(
                    target_device, non_blocking=True
                )
                gate_ready = torch.cuda.Event()
                gate_ready.record(compute_stream)
                packed_up = packed_cpu_tensors[1].to(
                    target_device, non_blocking=True
                )
                up_ready = torch.cuda.Event()
                up_ready.record(compute_stream)
                if copy_stream is not None:
                    with torch.cuda.stream(copy_stream):
                        packed_down = packed_cpu_tensors[2].to(
                            target_device, non_blocking=True
                        )
                        down_ready = torch.cuda.Event()
                        down_ready.record(copy_stream)
                else:
                    packed_down = packed_cpu_tensors[2].to(
                        target_device, non_blocking=True
                    )
                    down_ready = torch.cuda.Event()
                    down_ready.record(compute_stream)
                # Gate/up are ordered on the compute stream; only down needs a
                # consumer-side event wait.
                gate_ready = None
                up_ready = None
                if copy_stream is None:
                    down_ready = None
            event_store = (
                self._prefetch_events if prefetched is not None else self._pinned_copy_events
            )
            event_store[("gate", slot)] = gate_ready
            event_store[("up", slot)] = up_ready
            event_store[("down", slot)] = down_ready
            self.timings["expert_h2d_seconds"] += time.perf_counter() - started
        else:
            packed_gate, packed_up, packed_down = (
                buffer.clone() for buffer in packed_cpu_tensors
            )

        self.timings["expert_load_calls"] += 1.0
        return StagedExpertBatch(
            specs=specs,
            packed=(packed_gate, packed_up, packed_down),
            gate_ready=gate_ready,
            up_ready=up_ready,
            down_ready=down_ready,
            expert_ids=expert_id_tuple,
            keepalive=(prefetched,) if prefetched is not None else (),
        )

    def load_experts(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gate/up/down as ``(experts, out_features, in_features)``."""

        target_dtype = self.dtype if dtype is None else dtype
        batch = self._stage_experts(layer_idx, expert_ids, device=device)
        expert_count = len(batch.expert_ids)
        tensors: list[torch.Tensor] = []
        try:
            batch.wait_for_gate_up()
            for index in (0, 1):
                spec = batch.specs[index]
                started = time.perf_counter()
                dequantized = dequantize_matrix(
                    batch.packed[index],
                    spec.quant_type,
                    out_features=expert_count * spec.out_features,
                    in_features=spec.in_features,
                    dtype=target_dtype,
                )
                self.timings["expert_dequant_launch_seconds"] += (
                    time.perf_counter() - started
                )
                tensors.append(
                    dequantized.reshape(
                        expert_count, spec.out_features, spec.in_features
                    )
                )
            batch.wait_for_down()
            spec = batch.specs[2]
            started = time.perf_counter()
            dequantized = dequantize_matrix(
                batch.packed[2],
                spec.quant_type,
                out_features=expert_count * spec.out_features,
                in_features=spec.in_features,
                dtype=target_dtype,
            )
            self.timings["expert_dequant_launch_seconds"] += (
                time.perf_counter() - started
            )
            tensors.append(
                dequantized.reshape(
                    expert_count, spec.out_features, spec.in_features
                )
            )
            del batch
            return tensors[0], tensors[1], tensors[2]
        except torch.cuda.OutOfMemoryError:
            for tensor in tensors:
                del tensor
            gc.collect()
            torch.cuda.empty_cache()
            raise

    def load_packed_experts(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        *,
        device: torch.device | None = None,
        async_gate_up: bool = False,
        prefetched: ExpertPrefetch | None = None,
    ) -> StagedExpertBatch:
        """Return staged packed expert bytes without dequantizing them."""

        return self._stage_experts(
            layer_idx,
            expert_ids,
            device=self.device if device is None else device,
            async_gate_up=async_gate_up,
            prefetched=prefetched,
        )


def _set_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    setattr(module, name, nn.Parameter(value, requires_grad=False))


class DeepSeekV2LiteGGUFRuntime:
    """Single-request native runtime for DeepSeek-V2-Lite Q4_K_M GGUF."""

    def __init__(
        self,
        gguf_path: str | Path,
        *,
        meta_dir: str | Path | None = None,
        device: str = "cuda",
        max_cache: int = 256,
        cpu_threads: int = 6,
        expert_chunk_size: int = 8,
        cpu_moe_layers: int = 0,
        cpu_experts_per_layer: int = 0,
        cpu_expert_layers: int = 0,
        cpu_expert_cache_mb: int = 512,
        prefill_pipeline: bool = False,
        prefill_prefetch: bool = True,
        decode_prefetch: bool = True,
        predictive_prefetch: bool = False,
        predictive_cache_mb: int = 256,
        prefill_all_experts: bool = False,
        expert_cache_mb: int = 0,
        gpu_expert_cache_mb: int = 0,
        fused_moe: bool = True,
        shared_experts_device: str = "auto",
        dense_mlp_device: str = "auto",
        lm_head_device: str = "auto",
        max_position_embeddings: int = 2048,
        verbose: bool = True,
    ) -> None:
        self.gguf_path = Path(gguf_path)
        if meta_dir is not None:
            self.meta_dir = Path(meta_dir)
        else:
            candidates = [
                self.gguf_path.parent.parent / "meta",
                self.gguf_path.parent.parent.parent / "meta",
            ]
            self.meta_dir = next(
                (candidate for candidate in candidates if candidate.is_dir()),
                candidates[0],
            )
        self.device = torch.device(device)
        self.max_cache = int(max_cache)
        self.cpu_threads = int(cpu_threads)
        self.expert_chunk_size = max(1, int(expert_chunk_size))
        self.requested_cpu_moe_layers = max(0, int(cpu_moe_layers))
        self.requested_cpu_experts_per_layer = max(
            0, int(cpu_experts_per_layer)
        )
        self.requested_cpu_expert_layers = max(0, int(cpu_expert_layers))
        self.requested_cpu_expert_cache_mb = max(
            0, int(cpu_expert_cache_mb)
        )
        self.prefill_pipeline = bool(prefill_pipeline)
        self.prefill_prefetch = bool(prefill_prefetch)
        self.decode_prefetch = bool(decode_prefetch)
        self.predictive_prefetch = bool(predictive_prefetch)
        self.predictive_cache_mb = max(0, int(predictive_cache_mb))
        self._predictive_cache_limit = self.predictive_cache_mb * _MIB
        self.prefill_all_experts = bool(prefill_all_experts)
        self.requested_fused_moe = bool(fused_moe)
        self.dense_mlp_device = dense_mlp_device
        self.verbose = verbose
        # Expert staging keeps per-slot events and double buffers; the
        # scheduler's two CUDA streams can overlap prefill and decode.  The
        # only shared mutable state is telemetry, guarded by the Python GIL.
        self.supports_prefill_decode_overlap = (
            os.environ.get("DEEPINFER_DISABLE_PREFILL_DECODE_OVERLAP") != "1"
        )
        self.native_paged_attention_enabled = True
        self._closed = False

        if self.device.type != "cuda":
            raise ValueError("DeepSeek-V2-Lite MoE runtime currently requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        # Full-precision accumulation matters for a sparse router: a 1e-3
        # change in a hidden state can swap a top-6 expert on the next token.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False

        self.index = GGUFIndex(self.gguf_path)
        if self.index.tensor_data_bytes > self.index.file_size:
            raise ValueError("GGUF checkpoint is incomplete")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.meta_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.config = AutoConfig.from_pretrained(
            str(self.meta_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        # The official remote-code model eagerly creates RoPE caches up to
        # max_position_embeddings for every attention layer.  The stock 163840
        # value wastes gigabytes of host RAM for no benefit in this runtime.
        self.config.max_position_embeddings = int(max_position_embeddings)
        self.config._attn_implementation = "eager"
        self.config.use_cache = True
        raw_experts = getattr(self.config, "n_routed_experts", None)
        if raw_experts is not None and int(raw_experts) <= 0:
            # The official decoder treats only ``None`` as fully dense.
            self.config.n_routed_experts = None
        self.routed_expert_count = int(
            getattr(self.config, "n_routed_experts", 0) or 0
        )
        self.dense_layer_indices, self.moe_layer_indices = split_layer_indices(
            self.config
        )
        cpu_layer_count = min(
            self.requested_cpu_moe_layers, len(self.moe_layer_indices)
        )
        self.cpu_moe_layer_indices = tuple(
            self.moe_layer_indices[-cpu_layer_count:]
        ) if cpu_layer_count else ()
        self.cpu_moe_layer_set = set(self.cpu_moe_layer_indices)
        if self.requested_cpu_expert_layers:
            expert_layer_count = min(
                self.requested_cpu_expert_layers, len(self.moe_layer_indices)
            )
            self.cpu_expert_layer_set = set(
                self.moe_layer_indices[-expert_layer_count:]
            )
        else:
            self.cpu_expert_layer_set = set(self.moe_layer_indices)
        self.has_moe = bool(self.moe_layer_indices)
        self._next_moe_layer = {
            current: following
            for current, following in zip(
                self.moe_layer_indices, self.moe_layer_indices[1:]
            )
        }
        self._last_token_routes: dict[int, tuple[int, ...]] = {}
        self.dense_mlp_devices: dict[int, str] = {}
        self._validate_gguf_metadata()

        model_cls = get_class_from_dynamic_module(
            "modeling_deepseek.DeepseekV2ForCausalLM",
            str(self.meta_dir),
            local_files_only=True,
        )
        with init_empty_weights():
            self.model = model_cls(self.config)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.shared_experts_device = "n/a" if not self.has_moe else "cpu"
        self.lm_head_device = "cpu"
        self.expert_store = (
            GGUFExpertStore(
                self.index,
                layer_indices=self.moe_layer_indices,
                device=self.device,
                expert_cache_mb=expert_cache_mb,
                gpu_expert_cache_mb=gpu_expert_cache_mb,
                predictive_cache_mb=self.predictive_cache_mb,
            )
            if self.has_moe
            else None
        )
        self.fused_moe_enabled = (
            self.requested_fused_moe
            and fused_moe_supported()
            and self.expert_store is not None
            and self.expert_store.supports_fused_moe()
            and os.environ.get("DEEPINFER_DISABLE_FUSED_MOE") != "1"
        )
        self.fused_moe_chunks = 0
        self.fallback_moe_chunks = 0
        self.cpu_moe_chunks = 0
        self.cpu_moe_seconds = 0.0
        self.cpu_expert_chunks = 0
        self._cpu_expert_weight_cache: OrderedDict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._cpu_expert_weight_bytes = 0
        self._cpu_packed_cache: OrderedDict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._cpu_packed_bytes = 0
        self._cpu_expert_cache_limit = (
            self.requested_cpu_expert_cache_mb * _MIB
        )
        self._moe_route_counts: dict[int, list[int]] = {}
        self.cpu_avx2_available = False
        if (
            self.requested_cpu_experts_per_layer > 0
            or self.requested_cpu_moe_layers > 0
        ):
            from deepinfer.moe import cpu_avx2

            self.cpu_avx2_available = cpu_avx2.available()
        self._load_weights(shared_experts_device, lm_head_device)
        from deepinfer.moe.paged_adapter import install_native_paged_attention

        self.native_paged_attention_layers = install_native_paged_attention(
            self.model
        )
        self.model.native_paged_attention_kernel = bool(
            self.native_paged_attention_layers
        ) and os.environ.get("DEEPINFER_DISABLE_NATIVE_PAGED_MLA") != "1"
        self.gpu_free_bytes_after_load = int(torch.cuda.mem_get_info()[0])
        self._log(
            "loaded: GPU free %.1f MiB, shared=%s, lm_head=%s, "
            "fused_moe=%s, peak allocated %.1f MiB"
            % (
                self.gpu_free_bytes_after_load / _MIB,
                self.shared_experts_device,
                self.lm_head_device,
                self.fused_moe_enabled,
                torch.cuda.max_memory_allocated() / _MIB,
            )
        )

    def _validate_gguf_metadata(self) -> None:
        if str(self.config.model_type) != "deepseek_v2":
            raise ValueError(
                f"expected a DeepSeek-V2 GGUF checkpoint, got model_type="
                f"{self.config.model_type!r}"
            )
        expected = {
            "general.architecture": "deepseek2",
            "deepseek2.block_count": int(self.config.num_hidden_layers),
            "deepseek2.embedding_length": int(self.config.hidden_size),
            "deepseek2.attention.head_count": int(self.config.num_attention_heads),
            "deepseek2.attention.kv_lora_rank": int(self.config.kv_lora_rank),
            "deepseek2.feed_forward_length": int(self.config.intermediate_size),
        }
        if self.has_moe:
            expected.update(
                {
                    "deepseek2.expert_count": int(self.config.n_routed_experts),
                    "deepseek2.expert_used_count": int(
                        self.config.num_experts_per_tok
                    ),
                    "deepseek2.expert_shared_count": int(
                        self.config.n_shared_experts or 0
                    ),
                    "deepseek2.expert_feed_forward_length": int(
                        self.config.moe_intermediate_size
                    ),
                }
            )
        else:
            stored_experts = self.index.metadata.get("deepseek2.expert_count")
            if stored_experts not in (None, 0):
                raise ValueError(
                    "checkpoint contains routed experts but the config describes "
                    f"a dense model (expert_count={stored_experts!r})"
                )
        for key, value in expected.items():
            actual = self.index.metadata.get(key)
            if actual != value:
                raise ValueError(
                    f"GGUF metadata mismatch for {key}: got {actual!r}, "
                    f"expected {value!r}"
                )

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def _read_weight(
        self,
        name: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        array = self.index.read_tensor(name)
        if not array.flags.writeable:
            array = np.array(array, copy=True, order="C")
        else:
            array = np.ascontiguousarray(array)
        tensor = torch.from_numpy(array).to(dtype=dtype)
        del array
        if device.type == "cuda":
            tensor = tensor.to(device)
        return tensor

    def _load_norm(
        self,
        module: nn.Module,
        name: str,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        target = self.device if device is None else device
        _set_parameter(module, "weight", self._read_weight(name, device=target, dtype=dtype))

    def _load_linear(
        self,
        module: nn.Linear,
        name: str,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        target = self.device if device is None else device
        _set_parameter(module, "weight", self._read_weight(name, device=target, dtype=dtype))

    def _choose_optional_device(self, requested: str, *, required_mib: int) -> str:
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError(f"device choice must be auto/cpu/cuda, got {requested!r}")
        if requested == "cpu":
            return "cpu"
        free_bytes = int(torch.cuda.mem_get_info()[0])
        if requested == "cuda":
            if free_bytes < required_mib * _MIB:
                raise RuntimeError(
                    f"requested CUDA placement for a {required_mib} MiB block, "
                    f"but only {free_bytes / _MIB:.1f} MiB is free"
                )
            return "cuda"
        return "cuda" if free_bytes >= required_mib * _MIB else "cpu"

    @staticmethod
    def _module_fp16_bytes(module: nn.Module) -> int:
        # Meta tensors may report float32 element size even though the runtime
        # loads every non-embedding module as FP16.  Budget the actual target
        # dtype instead of trusting the meta tensor's element_size().
        return int(sum(parameter.numel() * 2 for parameter in module.parameters()))

    def _choose_dense_mlp_device(self, mlp: nn.Module, requested: str) -> str:
        if requested not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"dense_mlp_device must be auto/cpu/cuda, got {requested!r}"
            )
        if requested == "cpu":
            return "cpu"
        dense_bytes = self._module_fp16_bytes(mlp)
        required_bytes = dense_bytes + self._unloaded_attention_bytes + 256 * _MIB
        free_bytes = int(torch.cuda.mem_get_info()[0])
        if requested == "cuda":
            if free_bytes < required_bytes:
                raise RuntimeError(
                    "requested CUDA dense-MLP placement needs "
                    f"{required_bytes / _MIB:.1f} MiB including remaining "
                    f"attention weights, but only {free_bytes / _MIB:.1f} MiB "
                    "is free"
                )
            return "cuda"
        return "cuda" if free_bytes >= required_bytes else "cpu"

    def _load_dense_mlp(self, mlp: nn.Module, layer_idx: int) -> None:
        device_name = self._choose_dense_mlp_device(mlp, self.dense_mlp_device)
        if device_name == "cpu":
            cpu_bytes = sum(parameter.numel() * 4 for parameter in mlp.parameters())
            try:
                import psutil

                available = int(psutil.virtual_memory().available)
            except Exception:  # noqa: BLE001
                available = None
            if available is not None and available < cpu_bytes + 512 * _MIB:
                raise MemoryError(
                    f"dense MLP layer {layer_idx} needs about "
                    f"{cpu_bytes / _MIB:.0f} MiB of CPU RAM for FP32 weights, "
                    f"but only {available / _MIB:.0f} MiB is available; free "
                    "other applications or use a smaller dense checkpoint"
                )
            target_device = torch.device("cpu")
            target_dtype = torch.float32
        else:
            target_device = self.device
            target_dtype = torch.float16
        self._load_linear(
            mlp.gate_proj,
            f"blk.{layer_idx}.ffn_gate.weight",
            device=target_device,
            dtype=target_dtype,
        )
        self._load_linear(
            mlp.up_proj,
            f"blk.{layer_idx}.ffn_up.weight",
            device=target_device,
            dtype=target_dtype,
        )
        self._load_linear(
            mlp.down_proj,
            f"blk.{layer_idx}.ffn_down.weight",
            device=target_device,
            dtype=target_dtype,
        )
        if device_name == "cpu":
            mlp.forward = _cpu_mlp_forward.__get__(mlp, type(mlp))
        self.dense_mlp_devices[layer_idx] = device_name

    def _load_weights(self, shared_experts_device: str, lm_head_device: str) -> None:
        model = self.model
        # Embedding lookup is tiny at decode time and is deliberately kept on
        # the host; only the resulting [tokens, hidden] activation crosses PCIe.
        embedding = self._read_weight(
            "token_embd.weight",
            device=torch.device("cpu"),
            dtype=torch.float16,
        )
        _set_parameter(model.model.embed_tokens, "weight", embedding)
        self._log(
            "embedding resident on CPU (%.1f MiB)"
            % (embedding.numel() * embedding.element_size() / _MIB)
        )

        num_layers = int(self.config.num_hidden_layers)
        dense_set = set(self.dense_layer_indices)
        self._unloaded_attention_bytes = sum(
            self._module_fp16_bytes(layer.self_attn)
            for layer in model.model.layers
        )
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            self._load_linear(attn.q_proj, f"blk.{layer_idx}.attn_q.weight")
            self._load_linear(
                attn.kv_a_proj_with_mqa, f"blk.{layer_idx}.attn_kv_a_mqa.weight"
            )
            self._load_linear(attn.kv_b_proj, f"blk.{layer_idx}.attn_kv_b.weight")
            self._load_linear(attn.o_proj, f"blk.{layer_idx}.attn_output.weight")
            self._load_norm(
                attn.kv_a_layernorm, f"blk.{layer_idx}.attn_kv_a_norm.weight"
            )
            self._load_norm(layer.input_layernorm, f"blk.{layer_idx}.attn_norm.weight")
            self._load_norm(
                layer.post_attention_layernorm, f"blk.{layer_idx}.ffn_norm.weight"
            )
            attn.rotary_emb.to(self.device)
            self._unloaded_attention_bytes = max(
                0, self._unloaded_attention_bytes - self._module_fp16_bytes(attn)
            )
            if layer_idx in dense_set:
                self._load_dense_mlp(layer.mlp, layer_idx)
            if layer_idx % 4 == 0:
                self._log(
                    f"loaded layer {layer_idx}: dense={layer_idx in dense_set}"
                )
        self._load_norm(model.model.norm, "output_norm.weight")
        torch.cuda.empty_cache()

        if self.has_moe:
            self.shared_experts_device = self._choose_optional_device(
                shared_experts_device, required_mib=1150
            )
            if self.shared_experts_device == "cuda":
                try:
                    self._load_shared_experts(torch.device("cuda"), torch.float16)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    self._log("shared expert CUDA placement OOMed; falling back to CPU")
                    self.shared_experts_device = "cpu"
                    self._load_shared_experts(torch.device("cpu"), torch.float32)
            else:
                self._load_shared_experts(torch.device("cpu"), torch.float32)
        else:
            self.shared_experts_device = "n/a"

        self.lm_head_device = self._choose_optional_device(
            lm_head_device, required_mib=600
        )
        if self.lm_head_device == "cuda":
            try:
                lm_head = self._read_weight(
                    "output.weight", device=self.device, dtype=torch.float16
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                self._log("LM head CUDA placement OOMed; falling back to CPU")
                self.lm_head_device = "cpu"
                lm_head = self._read_weight(
                    "output.weight", device=torch.device("cpu"), dtype=torch.float32
                )
        else:
            lm_head = self._read_weight(
                "output.weight", device=torch.device("cpu"), dtype=torch.float32
            )
        _set_parameter(model.lm_head, "weight", lm_head)
        self._log(
            "lm_head resident on %s (%.1f MiB)"
            % (
                self.lm_head_device,
                lm_head.numel() * lm_head.element_size() / _MIB,
            )
        )

        for layer_idx in self.moe_layer_indices:
            layer = model.model.layers[layer_idx]
            self._load_router(layer.mlp, layer_idx)
            # The routed expert module parameters remain meta and are never
            # called.  Dropping the list avoids accidental state_dict moves.
            layer.mlp.experts = nn.ModuleList()
            self._install_moe(layer.mlp, layer_idx)
        torch.cuda.empty_cache()
        self._log(
            "layer layout: %d dense %s, %d MoE %s"
            % (
                len(self.dense_layer_indices),
                self.dense_layer_indices,
                len(self.moe_layer_indices),
                self.moe_layer_indices,
            )
        )

    def _load_router(self, mlp: nn.Module, layer_idx: int) -> None:
        router = self._read_weight(
            f"blk.{layer_idx}.ffn_gate_inp.weight",
            device=self.device,
            dtype=torch.float32,
        )
        _set_parameter(mlp.gate, "weight", router)

    def _load_shared_experts(self, device: torch.device, dtype: torch.dtype) -> None:
        for layer_idx in self.moe_layer_indices:
            shared = self.model.model.layers[layer_idx].mlp.shared_experts
            self._load_linear(
                shared.gate_proj,
                f"blk.{layer_idx}.ffn_gate_shexp.weight",
                device=device,
                dtype=dtype,
            )
            self._load_linear(
                shared.up_proj,
                f"blk.{layer_idx}.ffn_up_shexp.weight",
                device=device,
                dtype=dtype,
            )
            self._load_linear(
                shared.down_proj,
                f"blk.{layer_idx}.ffn_down_shexp.weight",
                device=device,
                dtype=dtype,
            )
            if dtype == torch.float32:
                shared.forward = _cpu_mlp_forward.__get__(shared, type(shared))

    def _install_moe(self, mlp: nn.Module, layer_idx: int) -> None:
        def forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            return self._deepinfer_moe_forward(  # type: ignore[attr-defined]
                hidden_states, layer_idx=layer_idx
            )

        mlp._deepinfer_moe_forward = MethodType(  # type: ignore[attr-defined]
            self._streaming_moe_forward, mlp
        )
        mlp.forward = MethodType(forward, mlp)

    def _streaming_moe_forward(
        self,
        mlp: nn.Module,
        hidden_states: torch.Tensor,
        *,
        layer_idx: int,
    ) -> torch.Tensor:
        identity = hidden_states
        original_shape = hidden_states.shape
        topk_idx, topk_weight, _ = mlp.gate(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        token_count, top_k = topk_idx.shape
        expert_ids = torch.unique(topk_idx.reshape(-1), sorted=True)
        expert_count = expert_ids.numel()
        # Materialize the small routing list before the sort/padding setup so
        # decode can start pinned expert reads while the GPU prepares the
        # assignment layout.
        expert_list = expert_ids.cpu().tolist()
        decode_prefetch: ExpertPrefetch | None = None
        if (
            self.decode_prefetch
            and token_count == 1
            and self.fused_moe_enabled
            and self.expert_store is not None
            and not self.expert_store.gpu_cache_enabled()
            and layer_idx not in self.cpu_moe_layer_set
            and self.requested_cpu_experts_per_layer == 0
            and len(expert_list) <= self.expert_chunk_size
        ):
            decode_prefetch = self.expert_store.prefetch_experts(
                layer_idx,
                expert_list,
                skip_cached=self.predictive_prefetch,
            )
        group_index = torch.searchsorted(expert_ids, topk_idx.reshape(-1))
        token_ids = torch.arange(topk_idx.numel(), device=flat.device) // top_k
        assignment_weights = topk_weight.reshape(-1)

        # Sort assignments once by expert.  This replaces a Python loop with
        # one nonzero()/boolean-index sequence per expert, which was a large
        # source of launch and synchronization overhead on the 3050.
        order = torch.argsort(group_index, stable=True)
        sorted_groups = group_index.index_select(0, order)
        sorted_token_ids = token_ids.index_select(0, order)
        sorted_weights = assignment_weights.index_select(0, order)
        counts = torch.bincount(group_index, minlength=expert_count)
        group_offsets = torch.cumsum(counts, dim=0) - counts
        positions_in_group = torch.arange(
            sorted_groups.numel(), device=flat.device
        ) - group_offsets.index_select(0, sorted_groups)
        max_count = 1 if token_count == 1 else int(counts.max().item())
        padded_flat_index = sorted_groups * max_count + positions_in_group
        sorted_tokens = flat.index_select(0, sorted_token_ids)
        padded_flat = torch.zeros(
            (expert_count * max_count, flat.shape[-1]),
            dtype=flat.dtype,
            device=flat.device,
        )
        padded_flat.index_copy_(0, padded_flat_index, sorted_tokens)
        padded_tokens = padded_flat.view(expert_count, max_count, flat.shape[-1])

        counts_list = counts.cpu().tolist()
        cpu_expert_set: set[int] = set()
        if (
            self.requested_cpu_experts_per_layer > 0
            and layer_idx in self.cpu_expert_layer_set
            and layer_idx not in self.cpu_moe_layer_set
        ):
            counts_all = self._moe_route_counts.get(layer_idx)
            if counts_all is None:
                counts_all = [0] * max(
                    self.routed_expert_count, max(expert_list) + 1
                )
                self._moe_route_counts[layer_idx] = counts_all
            for expert_id, count in zip(expert_list, counts_list, strict=True):
                counts_all[expert_id] += int(count)
            cpu_expert_set = set(
                sorted(
                    expert_list,
                    key=lambda expert_id: (counts_all[expert_id], expert_id),
                )[: self.requested_cpu_experts_per_layer]
            )
        if self.predictive_prefetch and token_count == 1:
            next_moe_layer = self._next_moe_layer.get(layer_idx)
            predicted = (
                self._last_token_routes.get(next_moe_layer)
                if next_moe_layer is not None
                else None
            )
            if predicted:
                self.expert_store.prefetch_predictive(
                    next_moe_layer, predicted
                )
            self._last_token_routes[layer_idx] = tuple(
                int(expert_id) for expert_id in expert_list
            )
        assignment_outputs = torch.zeros(
            (topk_idx.numel(), flat.shape[-1]),
            dtype=flat.dtype,
            device=flat.device,
        )
        chunk_size = self.expert_chunk_size
        if (
            self.prefill_all_experts
            and token_count > 1
            and layer_idx not in self.cpu_moe_layer_set
            and self.expert_store is not None
        ):
            packed_bytes = sum(
                self.expert_store._specs[(layer_idx, projection)].bytes_per_expert
                for projection in ("gate", "up", "down")
            ) * len(expert_list)
            free_bytes = int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else 0
            if packed_bytes < max(64 * _MIB, int(free_bytes * 0.45)):
                chunk_size = len(expert_list)
        use_prefetch = (
            self.prefill_prefetch
            and token_count > 1
            and self.fused_moe_enabled
            and self.expert_store is not None
            and not self.expert_store.gpu_cache_enabled()
            and layer_idx not in self.cpu_moe_layer_set
            and not cpu_expert_set
        )
        current_prefetch: ExpertPrefetch | None = decode_prefetch
        if use_prefetch:
            first_end = min(chunk_size, len(expert_list))
            current_prefetch = self.expert_store.prefetch_experts(
                layer_idx, expert_list[:first_end]
            )
        sorted_offset = 0
        for chunk_start in range(0, len(expert_list), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(expert_list))
            next_prefetch: ExpertPrefetch | None = None
            if use_prefetch and chunk_end < len(expert_list):
                next_end = min(chunk_end + chunk_size, len(expert_list))
                next_prefetch = self.expert_store.prefetch_experts(
                    layer_idx, expert_list[chunk_end:next_end]
                )
            chunk_ids = expert_list[chunk_start:chunk_end]
            chunk_tokens = padded_tokens[chunk_start:chunk_end]
            chunk_counts = counts_list[chunk_start:chunk_end]
            chunk_count = sum(chunk_counts)
            local_pad_index = (
                padded_flat_index[sorted_offset : sorted_offset + chunk_count]
                - chunk_start * max_count
            )
            chunk_weights = sorted_weights[sorted_offset : sorted_offset + chunk_count]
            chunk_order = order[sorted_offset : sorted_offset + chunk_count]
            sorted_offset += chunk_count
            try:
                if layer_idx in self.cpu_moe_layer_set:
                    cpu_started = time.perf_counter()
                    expert_output = self._moe_cpu_packed_chunk(
                        layer_idx, chunk_ids, chunk_tokens
                    )
                    self.cpu_moe_seconds += time.perf_counter() - cpu_started
                    self.cpu_moe_chunks += 1
                elif cpu_expert_set:
                    cpu_local = [
                        index
                        for index, expert_id in enumerate(chunk_ids)
                        if expert_id in cpu_expert_set
                    ]
                    gpu_local = [
                        index
                        for index in range(len(chunk_ids))
                        if index not in cpu_local
                    ]
                    expert_output = torch.zeros(
                        (len(chunk_ids), chunk_tokens.shape[1], flat.shape[-1]),
                        dtype=flat.dtype,
                        device=flat.device,
                    )
                    if gpu_local:
                        gpu_index = torch.tensor(
                            gpu_local, dtype=torch.long, device=flat.device
                        )
                        gpu_ids = [chunk_ids[index] for index in gpu_local]
                        gpu_tokens = chunk_tokens.index_select(0, gpu_index)
                        gpu_counts = [chunk_counts[index] for index in gpu_local]
                        if self.fused_moe_enabled:
                            gpu_output = self._fused_expert_chunk(
                                layer_idx,
                                gpu_ids,
                                gpu_tokens,
                                gpu_counts,
                            )
                            self.fused_moe_chunks += 1
                        else:
                            gate_w, up_w, down_w = self.expert_store.load_experts(
                                layer_idx, gpu_ids
                            )
                            gate = torch.bmm(
                                gate_w, gpu_tokens.transpose(1, 2)
                            ).transpose(1, 2)
                            up = torch.bmm(
                                up_w, gpu_tokens.transpose(1, 2)
                            ).transpose(1, 2)
                            hidden = F.silu(gate) * up
                            gpu_output = torch.bmm(
                                down_w, hidden.transpose(1, 2)
                            ).transpose(1, 2)
                            del gate_w, up_w, down_w, gate, up, hidden
                            self.fallback_moe_chunks += 1
                        expert_output.index_copy_(0, gpu_index, gpu_output)
                    if cpu_local:
                        cpu_index = torch.tensor(
                            cpu_local, dtype=torch.long, device=flat.device
                        )
                        cpu_ids = [chunk_ids[index] for index in cpu_local]
                        cpu_tokens = chunk_tokens.index_select(0, cpu_index)
                        cpu_started = time.perf_counter()
                        cpu_output = self._moe_cpu_packed_chunk(
                            layer_idx, cpu_ids, cpu_tokens
                        )
                        self.cpu_moe_seconds += time.perf_counter() - cpu_started
                        self.cpu_expert_chunks += 1
                        expert_output.index_copy_(0, cpu_index, cpu_output)
                elif self.fused_moe_enabled:
                    expert_output = self._fused_expert_chunk(
                        layer_idx,
                        chunk_ids,
                        chunk_tokens,
                        chunk_counts,
                        prefetched=current_prefetch,
                    )
                    self.fused_moe_chunks += 1
                else:
                    gate_w, up_w, down_w = self.expert_store.load_experts(
                        layer_idx, chunk_ids
                    )
                    gate = torch.bmm(
                        gate_w, chunk_tokens.transpose(1, 2)
                    ).transpose(1, 2)
                    up = torch.bmm(
                        up_w, chunk_tokens.transpose(1, 2)
                    ).transpose(1, 2)
                    hidden = F.silu(gate) * up
                    expert_output = torch.bmm(
                        down_w, hidden.transpose(1, 2)
                    ).transpose(1, 2)
                    del gate_w, up_w, down_w, gate, up, hidden
                    self.fallback_moe_chunks += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                self._log(
                    f"layer {layer_idx} expert chunk {chunk_start} OOMed; "
                    "falling back to CPU for this chunk"
                )
                expert_output = self._moe_cpu_packed_chunk(
                    layer_idx, chunk_ids, chunk_tokens
                )
                self.fallback_moe_chunks += 1
            current_prefetch = next_prefetch
            selected = expert_output.reshape(-1, flat.shape[-1]).index_select(
                0, local_pad_index
            )
            contribution = selected.float() * chunk_weights.unsqueeze(-1)
            assignment_outputs.index_copy_(
                0,
                chunk_order,
                contribution.to(assignment_outputs.dtype),
            )
        routed = assignment_outputs.view(token_count, top_k, flat.shape[-1])
        routed = routed.float().sum(dim=1).to(flat.dtype)
        routed = routed.reshape(*original_shape)
        return routed + mlp.shared_experts(identity)

    def _fused_expert_chunk(
        self,
        layer_idx: int,
        chunk_ids: Sequence[int],
        chunk_tokens: torch.Tensor,
        chunk_counts: Sequence[int],
        prefetched: ExpertPrefetch | None = None,
    ) -> torch.Tensor:
        n_experts = len(chunk_ids)
        max_count = chunk_tokens.shape[1]
        counts = torch.tensor(
            [int(count) for count in chunk_counts],
            dtype=torch.int32,
            device=chunk_tokens.device,
        )
        if self.expert_store.gpu_cache_enabled():
            pointer_batch = self.expert_store.load_pointer_experts(
                layer_idx, chunk_ids
            )
            gate_spec, up_spec, down_spec = pointer_batch.specs
            gate_activation = fused_gate_up_gemm_ptr_grouped(
                chunk_tokens,
                pointer_batch.pointers[0],
                pointer_batch.pointers[1],
                counts,
                n_experts=n_experts,
                max_count=max_count,
                out_features=gate_spec.out_features,
                k_size=gate_spec.in_features,
                gate_qtype=gate_spec.quant_type,
                up_qtype=up_spec.quant_type,
                gate_row_bytes=gate_spec.row_bytes,
                up_row_bytes=up_spec.row_bytes,
            )
            return fused_down_gemm_ptr_grouped(
                gate_activation,
                pointer_batch.pointers[2],
                counts,
                n_experts=n_experts,
                max_count=max_count,
                out_features=down_spec.out_features,
                k_size=down_spec.in_features,
                down_qtype=down_spec.quant_type,
                down_row_bytes=down_spec.row_bytes,
            )

        batch = self.expert_store.load_packed_experts(
            layer_idx,
            chunk_ids,
            async_gate_up=self.prefill_pipeline and chunk_tokens.shape[1] > 1,
            prefetched=prefetched,
        )
        gate_spec, up_spec, down_spec = batch.specs
        batch.wait_for_gate_up()
        gate_activation = fused_gate_up_gemm_grouped(
            chunk_tokens,
            batch.packed[0],
            batch.packed[1],
            counts,
            n_experts=n_experts,
            max_count=max_count,
            out_features=gate_spec.out_features,
            k_size=gate_spec.in_features,
            gate_qtype=gate_spec.quant_type,
            up_qtype=up_spec.quant_type,
            gate_row_bytes=gate_spec.row_bytes,
            up_row_bytes=up_spec.row_bytes,
        )
        batch.wait_for_down()
        return fused_down_gemm_grouped(
            gate_activation,
            batch.packed[2],
            counts,
            n_experts=n_experts,
            max_count=max_count,
            out_features=down_spec.out_features,
            k_size=down_spec.in_features,
            down_qtype=down_spec.quant_type,
            down_row_bytes=down_spec.row_bytes,
        )

    def _cpu_expert_weight_get(
        self, layer_idx: int, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        key = (layer_idx, expert_id)
        value = self._cpu_expert_weight_cache.get(key)
        if value is None:
            return None
        self._cpu_expert_weight_cache.move_to_end(key)
        return value

    def _cpu_expert_weight_put(
        self,
        layer_idx: int,
        expert_id: int,
        value: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        key = (layer_idx, expert_id)
        old = self._cpu_expert_weight_cache.pop(key, None)
        if old is not None:
            old_bytes = sum(tensor.numel() * tensor.element_size() for tensor in old)
            self._cpu_expert_weight_bytes -= old_bytes
        size = sum(tensor.numel() * tensor.element_size() for tensor in value)
        if size > self._cpu_expert_cache_limit:
            return
        while (
            self._cpu_expert_weight_cache
            and self._cpu_expert_weight_bytes + size > self._cpu_expert_cache_limit
        ):
            _, evicted = self._cpu_expert_weight_cache.popitem(last=False)
            self._cpu_expert_weight_bytes -= sum(
                tensor.numel() * tensor.element_size() for tensor in evicted
            )
        self._cpu_expert_weight_cache[key] = value
        self._cpu_expert_weight_bytes += size

    def _cpu_packed_get(
        self, layer_idx: int, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        key = (layer_idx, expert_id)
        value = self._cpu_packed_cache.get(key)
        if value is None:
            return None
        self._cpu_packed_cache.move_to_end(key)
        return value

    def _cpu_packed_put(
        self,
        layer_idx: int,
        expert_id: int,
        value: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        key = (layer_idx, expert_id)
        old = self._cpu_packed_cache.pop(key, None)
        if old is not None:
            self._cpu_packed_bytes -= sum(tensor.numel() for tensor in old)
        size = sum(tensor.numel() for tensor in value)
        if size > self._cpu_expert_cache_limit:
            return
        while (
            self._cpu_packed_cache
            and self._cpu_packed_bytes + size > self._cpu_expert_cache_limit
        ):
            _, evicted = self._cpu_packed_cache.popitem(last=False)
            self._cpu_packed_bytes -= sum(tensor.numel() for tensor in evicted)
        self._cpu_packed_cache[key] = value
        self._cpu_packed_bytes += size

    def _load_cpu_packed_expert(
        self, layer_idx: int, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cpu_packed_get(layer_idx, expert_id)
        if cached is not None:
            return cached
        batch = self.expert_store.load_packed_experts(
            layer_idx, [expert_id], device=torch.device("cpu")
        )
        value = tuple(
            batch.packed[index][: batch.specs[index].bytes_per_expert].clone()
            for index in range(3)
        )
        self._cpu_packed_put(layer_idx, expert_id, value)
        return value

    def _load_cpu_expert_weights(
        self, layer_idx: int, expert_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cpu_expert_weight_get(layer_idx, expert_id)
        if cached is not None:
            return cached
        from deepinfer.moe import cpu_avx2

        batch = self.expert_store.load_packed_experts(
            layer_idx, [expert_id], device=torch.device("cpu")
        )
        weights: list[torch.Tensor] = []
        for projection, spec in enumerate(batch.specs):
            packed = batch.packed[projection][: spec.bytes_per_expert]
            weights.append(
                cpu_avx2.dequantize(
                    packed,
                    spec.quant_type,
                    out_features=spec.out_features,
                    in_features=spec.in_features,
                )
            )
        value = (weights[0], weights[1], weights[2])
        self._cpu_expert_weight_put(layer_idx, expert_id, value)
        return value

    def _moe_cpu_packed_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        padded_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if not expert_ids:
            raise ValueError("CPU expert chunk must not be empty")
        if not self.cpu_avx2_available:
            return self._moe_cpu_fallback(layer_idx, expert_ids, padded_tokens)
        from deepinfer.moe import cpu_avx2

        gate_spec = self.expert_store._specs[(layer_idx, "gate")]
        up_spec = self.expert_store._specs[(layer_idx, "up")]
        down_spec = self.expert_store._specs[(layer_idx, "down")]
        fused_supported = (
            padded_tokens.shape[1] <= 64
            and gate_spec.quant_type.upper() == "Q4_K"
            and up_spec.quant_type.upper() == "Q4_K"
            and down_spec.quant_type.upper() in ("Q4_K", "Q5_0", "Q8_0")
            and gate_spec.in_features % 256 == 0
            and down_spec.in_features % 32 == 0
        )
        if fused_supported:
            outputs: list[torch.Tensor] = []
            tokens = padded_tokens.shape[1]
            for position, expert_id in enumerate(expert_ids):
                packed = self._load_cpu_packed_expert(
                    layer_idx, int(expert_id)
                )
                outputs.append(
                    cpu_avx2.fused_expert_chunk(
                        packed[0],
                        packed[1],
                        packed[2],
                        padded_tokens[position].float(),
                        gate_rows=gate_spec.out_features,
                        gate_in_features=gate_spec.in_features,
                        down_rows=down_spec.out_features,
                        down_in_features=down_spec.in_features,
                        down_quant_type=down_spec.quant_type,
                        tokens=tokens,
                    )
                )
            output = torch.stack(outputs, dim=0)
            return output.to(
                padded_tokens.device, dtype=padded_tokens.dtype
            )

        weights = [
            self._load_cpu_expert_weights(layer_idx, int(expert_id))
            for expert_id in expert_ids
        ]
        gate_w = torch.stack([weight[0] for weight in weights], dim=0)
        up_w = torch.stack([weight[1] for weight in weights], dim=0)
        down_w = torch.stack([weight[2] for weight in weights], dim=0)
        padded = padded_tokens.float().cpu()
        gate = torch.bmm(gate_w, padded.transpose(1, 2)).transpose(1, 2)
        up = torch.bmm(up_w, padded.transpose(1, 2)).transpose(1, 2)
        hidden = torch.nn.functional.silu(gate) * up
        output = torch.bmm(down_w, hidden.transpose(1, 2)).transpose(1, 2)
        del gate_w, up_w, down_w, gate, up, hidden
        return output.to(padded_tokens.device, dtype=padded_tokens.dtype)

    def _moe_cpu_fallback(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        padded_tokens: torch.Tensor,
    ) -> torch.Tensor:
        cpu = torch.device("cpu")
        gate_w, up_w, down_w = self.expert_store.load_experts(
            layer_idx,
            expert_ids,
            device=cpu,
            dtype=torch.float32,
        )
        padded = padded_tokens.float().cpu()
        gate = torch.bmm(gate_w, padded.transpose(1, 2)).transpose(1, 2)
        up = torch.bmm(up_w, padded.transpose(1, 2)).transpose(1, 2)
        hidden = F.silu(gate) * up
        output = torch.bmm(down_w, hidden.transpose(1, 2)).transpose(1, 2)
        return output.to(padded_tokens.device, dtype=padded_tokens.dtype)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedded = self.model.model.embed_tokens(input_ids.cpu())
        return embedded.to(self.device)

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        kv_cache: Sequence[Any] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scheduler-compatible forward entry point for paged KV views."""

        if kv_cache is None:
            raise ValueError("forward_logits requires paged kv_cache views")
        return self._paged_forward_logits(input_ids, position_ids, kv_cache)

    @torch.inference_mode()
    def _paged_forward_logits(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Sequence[Any],
    ) -> torch.Tensor:
        """Forward one prefill chunk/decode batch through paged KV views."""

        from deepinfer.moe.paged_adapter import DeepSeekPagedCache

        views = list(kv_cache)
        if not views:
            raise ValueError("paged forward requires at least one layer view")
        batch, query_length = input_ids.shape
        seq_lens = views[0].seq_lens
        past_length = int(seq_lens.max().item())
        max_length = max(int(view.max_length) for view in views)
        total_length = past_length + int(query_length)
        if total_length > max_length:
            raise ValueError(
                f"paged forward length {total_length} exceeds view max {max_length}"
            )
        valid_lengths = (seq_lens + int(views[0].new_tokens)).clamp(
            max=total_length
        )
        positions = torch.arange(total_length, device=self.device)
        attention_mask = (
            positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
        ).to(torch.long)
        key_dim = int(self.config.qk_nope_head_dim) + int(
            self.config.qk_rope_head_dim
        )
        value_dim = int(self.config.v_head_dim)
        cache = DeepSeekPagedCache(
            views,
            position_ids,
            past_length=past_length,
            max_length=total_length,
            key_dim=key_dim,
            value_dim=value_dim,
            head_dim=max(key_dim, value_dim),
            native_attention=(
                views[0].key_pool.dtype == torch.float16
                and getattr(self, "native_paged_attention_enabled", True)
                and os.environ.get("DEEPINFER_DISABLE_NATIVE_PAGED_MLA") != "1"
            ),
        )
        outputs = self.model.model(
            inputs_embeds=self._embed_tokens(input_ids),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return self._all_logits(outputs.last_hidden_state)

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last = hidden_states[:, -1:, :]
        weight = self.model.lm_head.weight
        if weight.device.type == "cuda":
            return F.linear(last.to(dtype=weight.dtype), weight).float()
        return F.linear(last.float().cpu(), weight.float())

    def _all_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.model.lm_head.weight
        if weight.device.type == "cuda":
            return F.linear(hidden_states.to(dtype=weight.dtype), weight).float()
        return F.linear(hidden_states.float().cpu(), weight.float()).to(
            hidden_states.device
        )

    @torch.inference_mode()
    def score_text(self, text: str) -> dict[str, float | int]:
        """Compute next-token perplexity for a raw text string."""

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(torch.long)
        token_count = int(input_ids.shape[1])
        if token_count < 2:
            raise ValueError("score_text requires at least two tokens")
        if token_count > self.max_cache:
            raise ValueError(
                f"score_text length {token_count} exceeds max_cache={self.max_cache}"
            )
        attention_mask = torch.ones(
            (1, token_count), dtype=torch.long, device=self.device
        )
        outputs = self.model.model(
            inputs_embeds=self._embed_tokens(input_ids),
            attention_mask=attention_mask,
            position_ids=torch.arange(
                token_count, device=self.device
            ).unsqueeze(0),
            past_key_values=DynamicCache(),
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        total_nll = 0.0
        chunk = 8
        for start in range(0, token_count - 1, chunk):
            end = min(start + chunk, token_count - 1)
            logits = self._all_logits(hidden[:, start:end, :])
            targets = input_ids[:, start + 1 : end + 1].to(logits.device)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss.item())
        scored = token_count - 1
        return {
            "tokens": token_count,
            "scored_tokens": scored,
            "nll": total_nll,
            "perplexity": float(torch.exp(torch.tensor(total_nll / scored))),
        }

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        logits = logits.float()
        if temperature <= 1e-5:
            return int(torch.argmax(logits, dim=-1).item())
        logits = logits / float(temperature)
        if top_k > 0:
            k = min(int(top_k), logits.numel())
            values, indices = torch.topk(logits, k)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(0, indices, values)
            logits = filtered
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1)
            remove = cumulative - probabilities > float(top_p)
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_indices, sorted_logits)
        probabilities = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]] | None = None,
        *,
        prompt: str | None = None,
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        stop_token_ids: Sequence[int] | None = None,
    ) -> GenerationResult:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if prompt is not None:
            prompt_text = str(prompt)
        elif messages is not None:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            raise ValueError("either messages or prompt must be provided")
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(torch.long)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + max_new_tokens > self.max_cache:
            raise ValueError(
                f"prompt+output ({prompt_tokens + max_new_tokens}) exceeds "
                f"max_cache={self.max_cache}"
            )
        stop_ids = set(int(token) for token in (stop_token_ids or ()))
        stop_ids.add(int(self.tokenizer.eos_token_id))
        eos_token_id = int(self.tokenizer.eos_token_id)

        expert_store = self.expert_store
        if expert_store is not None and expert_store.gpu_cache_configured():
            # Prefill sees many experts per layer and would evict the decode
            # working set.  Free the previous request's cache first, keep
            # prefill on the contiguous path, then start decode with a clean
            # GPU packed cache.
            expert_store.clear_gpu_cache()
            expert_store.set_gpu_cache_active(False)
        torch.cuda.reset_peak_memory_stats()
        cache = DynamicCache()
        attention_mask = torch.ones((1, prompt_tokens), dtype=torch.long, device=self.device)
        prefill_start = time.perf_counter()
        hidden = self._embed_tokens(input_ids)
        outputs = self.model.model(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=torch.arange(prompt_tokens, device=self.device).unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        logits = self._logits(outputs.last_hidden_state)
        next_token = self._sample(
            logits[0],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        prefill_seconds = time.perf_counter() - prefill_start
        if expert_store is not None and expert_store.gpu_cache_configured():
            expert_store.clear_gpu_cache()
            expert_store.set_gpu_cache_active(True)
        generated = [next_token]
        total_tokens = prompt_tokens + 1

        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token == eos_token_id:
                break
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            hidden = self._embed_tokens(token_tensor)
            attention_mask = torch.ones(
                (1, total_tokens), dtype=torch.long, device=self.device
            )
            outputs = self.model.model(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=torch.tensor(
                    [[total_tokens - 1]], dtype=torch.long, device=self.device
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            logits = self._logits(outputs.last_hidden_state)
            next_token = self._sample(
                logits[0],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            generated.append(next_token)
            total_tokens += 1
            if next_token in stop_ids:
                break
        decode_seconds = time.perf_counter() - decode_start
        output_ids = [token for token in generated if token not in stop_ids]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        decode_tokens = max(0, len(generated) - 1)
        decode_tps = decode_tokens / decode_seconds if decode_seconds > 0 else 0.0
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(generated),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode_tokens_per_second=decode_tps,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            gpu_free_bytes_after_load=self.gpu_free_bytes_after_load,
            shared_experts_device=self.shared_experts_device,
            lm_head_device=self.lm_head_device,
            token_ids=list(generated),
            timings=(
                {
                    **(
                        dict(self.expert_store.timings)
                        if self.expert_store
                        else {}
                    ),
                    "cpu_moe_seconds": float(self.cpu_moe_seconds),
                }
            ),
            dense_layer_indices=list(self.dense_layer_indices),
            moe_layer_indices=list(self.moe_layer_indices),
            dense_mlp_devices=dict(self.dense_mlp_devices),
            fused_moe_enabled=self.fused_moe_enabled,
            fused_moe_chunks=self.fused_moe_chunks,
            fallback_moe_chunks=self.fallback_moe_chunks,
            cpu_moe_layers=len(self.cpu_moe_layer_indices),
            cpu_moe_chunks=self.cpu_moe_chunks,
            cpu_experts_per_layer=self.requested_cpu_experts_per_layer,
            cpu_expert_chunks=self.cpu_expert_chunks,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.expert_store is not None:
            self.expert_store.close()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def _cpu_mlp_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """FP32 host MLP used when the shared expert does not fit in VRAM."""

    original_dtype = hidden_states.dtype
    host = hidden_states.float().cpu()
    gate = F.linear(host, self.gate_proj.weight.float())  # type: ignore[attr-defined]
    up = F.linear(host, self.up_proj.weight.float())  # type: ignore[attr-defined]
    hidden = F.silu(gate) * up
    output = F.linear(hidden, self.down_proj.weight.float())  # type: ignore[attr-defined]
    return output.to(hidden_states.device, dtype=original_dtype)
