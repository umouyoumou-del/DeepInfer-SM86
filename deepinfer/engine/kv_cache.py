"""Elastic paged KV cache primitives for the SM86 executor.

The original SM86 backend reserved one dense ``[batch, kv_heads, max_seq,
head_dim]`` tensor per layer.  That layout is simple and CUDA-graph friendly,
but capacity is tied to the maximum batch size and every request pays the full
context cost.  This module adds a block allocator and a paged-attention cache
view while keeping the dense/static cache available as a fast path.

Design notes
------------
* The physical pool is allocated once.  Logical sequences own lists of blocks
  and only consume pool capacity for tokens that actually exist.
* ``block_table`` and ``seq_lens`` are tensors, so the attention path never
  needs a GPU->CPU synchronization to build a mask.
* Full-block prefix sharing is reference counted.  Blocks stored in the prefix
  cache are immutable; blocks containing the prompt tail are never inserted
  into the prefix cache, which avoids copy-on-write for decode.
* The paged attention reference path is intentionally written in PyTorch.  It
  is a correctness/generalization backend; the native SM86 GQA kernel remains
  the low-latency path for the batch=1 dense slot cache.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import torch

_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
_FP8_E5M2 = getattr(torch, "float8_e5m2", None)


def resolve_kv_cache_dtype(
    value: str | torch.dtype, compute_dtype: torch.dtype
) -> torch.dtype:
    """Resolve a user-facing KV cache dtype without silently aliasing FP8.

    ``fp16`` means "the model compute dtype" for backwards compatibility with
    CPU tests and non-FP16 backends.  INT8 and FP8 are explicit storage dtypes.
    """
    if isinstance(value, torch.dtype):
        return value
    name = str(value).lower()
    if name in ("", "auto", "fp16", "float16", "torch.float16"):
        return compute_dtype
    if name in ("int8", "i8", "torch.int8"):
        return torch.int8
    if name in ("fp8", "fp8_e4m3", "e4m3", "torch.float8_e4m3fn"):
        if _FP8_E4M3 is None:
            raise ValueError("torch.float8_e4m3fn is unavailable in this PyTorch build")
        return _FP8_E4M3
    if name in ("fp8_e5m2", "e5m2", "torch.float8_e5m2"):
        if _FP8_E5M2 is None:
            raise ValueError("torch.float8_e5m2 is unavailable in this PyTorch build")
        return _FP8_E5M2
    raise ValueError(
        "unsupported kv_cache_dtype "
        f"{value!r}; choose fp16, int8, fp8 (e4m3), or fp8_e5m2"
    )


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {value for value in (_FP8_E4M3, _FP8_E5M2) if value is not None}


def _storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret FP8 tensors as uint8 for CPU-safe indexing/copy ops."""
    if is_fp8_dtype(tensor.dtype):
        return tensor.view(torch.uint8)
    return tensor


def _restore_storage(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return tensor.view(dtype) if is_fp8_dtype(dtype) else tensor


def _index_select(
    tensor: torch.Tensor, index: torch.Tensor, dim: int
) -> torch.Tensor:
    selected = _storage_view(tensor).index_select(dim, index)
    return _restore_storage(selected, tensor.dtype)


def _index_copy(
    destination: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    dim: int,
) -> None:
    _storage_view(destination).index_copy_(
        dim, index, _storage_view(source)
    )


def paged_kv_block_bytes(
    *,
    num_layers: int,
    num_kv_heads: int,
    block_size: int,
    head_dim: int,
    compute_dtype: torch.dtype,
    kv_cache_dtype: str | torch.dtype = "fp16",
) -> int:
    """Estimate one logical KV block, including INT8 scale sidecars."""
    resolved = resolve_kv_cache_dtype(kv_cache_dtype, compute_dtype)
    data_elements = num_layers * num_kv_heads * block_size * head_dim * 2
    total = data_elements * torch.empty((), dtype=resolved).element_size()
    if resolved == torch.int8:
        # FP16 K and V scales for every head/token in the block.
        scale_elements = num_layers * num_kv_heads * block_size * 2
        total += scale_elements * 2
    return int(total)


class CacheOutOfMemoryError(RuntimeError):
    """Raised when a sequence cannot obtain the KV blocks it needs."""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(
            f"KV cache out of memory: requested {requested} block(s), "
            f"only {available} free"
        )


class BlockAllocator:
    """Reference-counted fixed-capacity block allocator.

    ``allocate`` returns blocks with a reference count of one.  Prefix cache
    entries call :meth:`retain`; both owners call :meth:`free`.  A physical
    block returns to the free list only when the last reference is released.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = int(num_blocks)
        self._free: deque[int] = deque(range(self.num_blocks))
        self._refcounts = [0] * self.num_blocks
        self._lock = threading.RLock()

    @property
    def available_blocks(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def used_blocks(self) -> int:
        with self._lock:
            return self.num_blocks - len(self._free)

    def allocate(self, count: int) -> list[int]:
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return []
        with self._lock:
            if len(self._free) < count:
                raise CacheOutOfMemoryError(count, len(self._free))
            blocks = [self._free.popleft() for _ in range(count)]
            for block in blocks:
                self._refcounts[block] = 1
            return blocks

    def retain(self, blocks: list[int]) -> None:
        with self._lock:
            for block in blocks:
                if not 0 <= block < self.num_blocks:
                    raise ValueError(f"invalid KV block id: {block}")
                if self._refcounts[block] <= 0:
                    raise ValueError(f"cannot retain free KV block {block}")
                self._refcounts[block] += 1

    def free(self, blocks: list[int]) -> None:
        with self._lock:
            seen: set[int] = set()
            for block in blocks:
                if not 0 <= block < self.num_blocks:
                    raise ValueError(f"invalid KV block id: {block}")
                if block in seen:
                    raise ValueError(f"duplicate block release: {block}")
                seen.add(block)
                if self._refcounts[block] <= 0:
                    raise ValueError(f"double free of KV block {block}")
                self._refcounts[block] -= 1
                if self._refcounts[block] == 0:
                    self._free.append(block)

    def owns(self, block: int) -> bool:
        with self._lock:
            return 0 <= block < self.num_blocks and self._refcounts[block] > 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            shared = sum(1 for value in self._refcounts if value > 1)
            peak_ref = max(self._refcounts, default=0)
            return {
                "num_blocks": self.num_blocks,
                "used_blocks": self.num_blocks - len(self._free),
                "free_blocks": len(self._free),
                "shared_blocks": shared,
                "peak_refcount": peak_ref,
            }


class PrefixCache:
    """Bounded LRU cache for immutable, full KV blocks.

    Callers split a token sequence into complete blocks, compute cumulative
    block hashes with :meth:`block_hashes`, then query :meth:`match`.  Matched
    blocks are retained for the caller.  :meth:`register` takes an additional
    cache reference for newly materialized full blocks.
    """

    def __init__(
        self,
        allocator: BlockAllocator,
        *,
        max_blocks: int,
        heat_fn: Any = None,
    ) -> None:
        if max_blocks < 0:
            raise ValueError("max_blocks must be non-negative")
        self.allocator = allocator
        self.max_blocks = int(max_blocks)
        # Optional ``heat_fn(block_id) -> float``.  When supplied, eviction
        # picks the coldest entry instead of the least recently used one.
        # Strict LRU throws away a block precisely when it was last touched,
        # which is the right signal for recency, but the wrong one when a
        # high-heat prefix (a shared system prompt, say) is simply older than
        # a stream of one-off prompts.
        self.heat_fn = heat_fn
        self._entries: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.reused_blocks = 0
        self.evictions = 0

    @staticmethod
    def block_hashes(
        token_ids: list[int] | tuple[int, ...],
        block_size: int,
        *,
        namespace: str = "",
    ) -> list[bytes]:
        """Return one cumulative hash per complete token block.

        ``namespace`` is mixed into the initial hasher state.  The gateway uses
        the OpenAI ``user`` field as a tenant namespace, preventing one tenant
        from probing or reusing another tenant's cached prompt blocks.
        """
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        hasher = hashlib.blake2b(digest_size=16)
        if namespace:
            encoded = namespace.encode("utf-8")
            hasher.update(b"dp-sm86-prefix-v1\x00")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
        hashes: list[bytes] = []
        for start in range(0, len(token_ids) - block_size + 1, block_size):
            for token in token_ids[start : start + block_size]:
                hasher.update(struct.pack("<q", int(token)))
            hashes.append(hasher.digest())
        return hashes

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def match(self, hashes: list[bytes], *, max_blocks: int | None = None) -> list[int]:
        """Retain and return the longest cached prefix of ``hashes``."""
        limit = len(hashes) if max_blocks is None else max(0, int(max_blocks))
        matched: list[int] = []
        with self._lock:
            for digest in hashes[:limit]:
                block = self._entries.get(digest)
                if block is None:
                    self.misses += 1
                    break
                self._entries.move_to_end(digest)
                try:
                    self.allocator.retain([block])
                except ValueError:
                    # The pool was cleared behind the cache's back.  Drop the
                    # stale entry rather than handing out a free block.
                    self._entries.pop(digest, None)
                    self.misses += 1
                    break
                matched.append(block)
            if matched:
                self.hits += 1
                self.reused_blocks += len(matched)
        return matched

    def _pick_eviction(self) -> int:
        """Remove and return the block to evict.

        Without a ``heat_fn`` this is plain LRU.  With one, the coldest block
        wins; ties keep LRU order because the strict ``<`` comparison leaves the
        first (oldest) entry in place.
        """

        if self.heat_fn is None:
            _, evicted = self._entries.popitem(last=False)
            return evicted
        coldest_digest = None
        coldest_block = None
        coldest_heat = None
        for digest, block in self._entries.items():
            try:
                heat = float(self.heat_fn(block))
            except Exception:  # noqa: BLE001 - a bad heat source must not break serving
                heat = 0.0
            if coldest_heat is None or heat < coldest_heat:
                coldest_digest, coldest_block, coldest_heat = digest, block, heat
        self._entries.pop(coldest_digest, None)
        return coldest_block

    def register(self, hashes: list[bytes], blocks: list[int]) -> int:
        """Cache full blocks and retain one prefix-cache reference each."""
        if self.max_blocks == 0:
            return 0
        if len(hashes) != len(blocks):
            raise ValueError("hashes and blocks must have the same length")
        added = 0
        with self._lock:
            for digest, block in zip(hashes, blocks, strict=True):
                if digest in self._entries:
                    continue
                if len(self._entries) >= self.max_blocks:
                    evicted = self._pick_eviction()
                    self.allocator.free([evicted])
                    self.evictions += 1
                self.allocator.retain([block])
                self._entries[digest] = int(block)
                added += 1
        return added

    def clear(self) -> None:
        with self._lock:
            for block in self._entries.values():
                self.allocator.free([block])
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_blocks": self.max_blocks,
                "cached_blocks": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "reused_blocks": self.reused_blocks,
                "evictions": self.evictions,
            }


@dataclass
class PagedLayerKVCache:
    """One layer's view into the paged physical KV pool.

    ``seq_lens`` is the logical length *before* the current step is written.
    For the decode path (one query token per sequence), ``update`` writes the
    token and ``materialize`` exposes ``seq_lens + 1`` valid positions.
    """

    key_pool: torch.Tensor  # [num_blocks, num_kv_heads, block_size, head_dim]
    value_pool: torch.Tensor
    block_table: torch.Tensor  # [batch, max_blocks_per_batch], int64
    seq_lens: torch.Tensor  # [batch], int64
    block_size: int
    max_length: int
    layer_index: int = -1
    new_tokens: int = 1
    key_scale_pool: torch.Tensor | None = None
    value_scale_pool: torch.Tensor | None = None
    compute_dtype: torch.dtype | None = None

    @property
    def is_paged(self) -> bool:
        return True

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_positions: torch.Tensor,
    ) -> None:
        batch, _, token_count, _ = key.shape
        if value.shape != key.shape:
            raise ValueError(
                f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}"
            )
        positions = cache_positions.to(
            device=self.key_pool.device, dtype=torch.long
        )
        if positions.ndim == 1:
            positions = positions.unsqueeze(1)
        if positions.numel() == batch and token_count == 1:
            positions = positions.reshape(batch, 1)
        if tuple(positions.shape) != (batch, token_count):
            raise ValueError(
                "cache_positions must have one position per batch row and query "
                f"token; got {tuple(positions.shape)} for batch={batch}, "
                f"tokens={token_count}"
            )
        batch_indices = torch.arange(batch, device=self.key_pool.device).unsqueeze(1)
        block_indices = positions // self.block_size
        block_ids = self.block_table[batch_indices, block_indices]
        offsets = positions % self.block_size
        # Trust the scheduler invariant: blocks are allocated before building
        # the view.  A negative-id check here would force a GPU->CPU sync once
        # per layer.
        heads = self.key_pool.shape[1]
        head_index = torch.arange(
            heads, device=self.key_pool.device
        ).view(1, 1, heads, 1)
        dim_index = torch.arange(
            self.key_pool.shape[3], device=self.key_pool.device
        ).view(1, 1, 1, -1)
        block_index = block_ids[:, :, None, None]
        offset_index = offsets[:, :, None, None]
        key_values = key.permute(0, 2, 1, 3).contiguous()
        value_values = value.permute(0, 2, 1, 3).contiguous()
        if self.key_pool.dtype == torch.int8:
            if self.key_scale_pool is None or self.value_scale_pool is None:
                raise RuntimeError("INT8 KV cache is missing scale pools")
            key_float = key_values.float()
            value_float = value_values.float()
            key_scale = (key_float.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
            value_scale = (value_float.abs().amax(dim=-1) / 127.0).clamp_min(1e-8)
            key_values = torch.round(key_float / key_scale[..., None]).clamp(
                -127, 127
            ).to(torch.int8)
            value_values = torch.round(value_float / value_scale[..., None]).clamp(
                -127, 127
            ).to(torch.int8)
            scale_head_index = torch.arange(
                heads, device=self.key_pool.device
            ).view(1, 1, heads)
            scale_block_index = block_ids[:, :, None]
            scale_offset_index = offsets[:, :, None]
            self.key_scale_pool[
                scale_block_index, scale_head_index, scale_offset_index
            ] = key_scale.to(self.key_scale_pool.dtype)
            self.value_scale_pool[
                scale_block_index, scale_head_index, scale_offset_index
            ] = value_scale.to(self.value_scale_pool.dtype)
        elif is_fp8_dtype(self.key_pool.dtype):
            key_values = key_values.to(self.key_pool.dtype)
            value_values = value_values.to(self.value_pool.dtype)
        self.key_pool[
            block_index, head_index, offset_index, dim_index
        ] = key_values
        self.value_pool[
            block_index, head_index, offset_index, dim_index
        ] = value_values
        self.new_tokens = int(token_count)

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather blocks into dense K/V plus a validity mask.

        Short sequences are padded to the batch maximum, and the returned bool
        mask tells attention to ignore the padding.
        """
        batch = int(self.block_table.shape[0])
        max_length = int(self.max_length)
        if max_length < 1:
            raise ValueError("max_length must be >= 1")
        num_blocks = (max_length + self.block_size - 1) // self.block_size
        table = self.block_table[:, :num_blocks].to(torch.long)
        safe_table = table.clamp_min(0)
        # [B, N, H, block, D] -> [B, H, N * block, D]
        key = _restore_storage(
            _storage_view(self.key_pool)[safe_table], self.key_pool.dtype
        )
        value = _restore_storage(
            _storage_view(self.value_pool)[safe_table], self.value_pool.dtype
        )
        if key.dtype == torch.int8:
            if self.key_scale_pool is None or self.value_scale_pool is None:
                raise RuntimeError("INT8 KV cache is missing scale pools")
            key_scales = self.key_scale_pool[safe_table].float()[..., None]
            value_scales = self.value_scale_pool[safe_table].float()[..., None]
            key = key.float() * key_scales
            value = value.float() * value_scales
        elif is_fp8_dtype(key.dtype):
            target_dtype = self.compute_dtype or torch.float16
            key = key.to(target_dtype)
            value = value.to(target_dtype)
        key = key.permute(0, 2, 1, 3, 4).reshape(
            batch,
            self.key_pool.shape[1],
            num_blocks * self.block_size,
            self.key_pool.shape[3],
        )[:, :, :max_length, :]
        value = value.permute(0, 2, 1, 3, 4).reshape(
            batch,
            self.value_pool.shape[1],
            num_blocks * self.block_size,
            self.value_pool.shape[3],
        )[:, :, :max_length, :]
        if self.compute_dtype is not None:
            key = key.to(self.compute_dtype)
            value = value.to(self.compute_dtype)

        positions = torch.arange(max_length, device=self.seq_lens.device)
        valid_lengths = (self.seq_lens + int(self.new_tokens)).clamp(
            max=max_length
        )
        valid = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
        return key, value, valid


@dataclass
class CpuKVBlockPayload:
    """CPU copy of one or more full physical KV blocks across all layers."""

    key: torch.Tensor
    value: torch.Tensor
    key_scales: torch.Tensor | None
    value_scales: torch.Tensor | None
    num_blocks: int
    num_layers: int

    @property
    def nbytes(self) -> int:
        total = self.key.numel() * self.key.element_size()
        total += self.value.numel() * self.value.element_size()
        if self.key_scales is not None:
            total += self.key_scales.numel() * self.key_scales.element_size()
        if self.value_scales is not None:
            total += self.value_scales.numel() * self.value_scales.element_size()
        return int(total)


class PagedKVCacheManager:
    """Own the physical pool and create per-layer views for a decode batch."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
        kv_cache_dtype: str | torch.dtype = "fp16",
        device: str | torch.device = "cuda",
        prefix_cache_blocks: int = 0,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_layers < 1 or num_kv_heads < 1 or head_dim < 1:
            raise ValueError("cache dimensions must all be positive")
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.num_blocks = int(num_blocks)
        self.dtype = dtype
        self.kv_cache_dtype = resolve_kv_cache_dtype(
            kv_cache_dtype, compute_dtype=dtype
        )
        self.kv_cache_dtype_name = str(kv_cache_dtype)
        self.device = torch.device(device)
        self.allocator = BlockAllocator(self.num_blocks)
        self.prefix_cache = PrefixCache(
            self.allocator, max_blocks=min(int(prefix_cache_blocks), self.num_blocks)
        )
        shape = (
            self.num_layers,
            self.num_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )
        self.key_pool = torch.zeros(
            shape, dtype=self.kv_cache_dtype, device=self.device
        )
        self.value_pool = torch.zeros_like(self.key_pool)
        if self.kv_cache_dtype == torch.int8:
            scale_shape = (
                self.num_layers,
                self.num_blocks,
                self.num_kv_heads,
                self.block_size,
            )
            self.key_scale_pool = torch.ones(
                scale_shape, dtype=torch.float16, device=self.device
            )
            self.value_scale_pool = torch.ones_like(self.key_scale_pool)
        else:
            self.key_scale_pool = None
            self.value_scale_pool = None

    @property
    def block_bytes(self) -> int:
        return paged_kv_block_bytes(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            block_size=self.block_size,
            head_dim=self.head_dim,
            compute_dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    @property
    def memory_bytes(self) -> int:
        return self.num_blocks * self.block_bytes

    def allocate_tokens(self, num_tokens: int) -> list[int]:
        blocks = (int(num_tokens) + self.block_size - 1) // self.block_size
        return self.allocator.allocate(blocks)

    def read_blocks(
        self,
        layer_index: int,
        block_ids: list[int],
        length: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read a contiguous prefix from immutable full blocks."""
        if length <= 0:
            return None, None
        if not block_ids:
            raise ValueError("non-zero prefix length requires at least one block")
        capacity = len(block_ids) * self.block_size
        if length > capacity:
            raise ValueError(f"prefix length {length} exceeds {capacity} cached tokens")
        block_index = torch.tensor(
            block_ids, dtype=torch.long, device=self.device
        )
        key_blocks = _index_select(self.key_pool[layer_index], block_index, 0)
        value_blocks = _index_select(
            self.value_pool[layer_index], block_index, 0
        )
        key = key_blocks.permute(1, 0, 2, 3).reshape(
            self.num_kv_heads, capacity, self.head_dim
        )[:, :length, :]
        value = value_blocks.permute(1, 0, 2, 3).reshape(
            self.num_kv_heads, capacity, self.head_dim
        )[:, :length, :]
        if key.dtype == torch.int8:
            assert self.key_scale_pool is not None
            assert self.value_scale_pool is not None
            key_scales = self.key_scale_pool[layer_index, block_ids].permute(
                1, 0, 2
            ).reshape(self.num_kv_heads, capacity)[:, :length]
            value_scales = self.value_scale_pool[layer_index, block_ids].permute(
                1, 0, 2
            ).reshape(self.num_kv_heads, capacity)[:, :length]
            key = key.float() * key_scales.float()[..., None]
            value = value.float() * value_scales.float()[..., None]
        return key.to(self.dtype).unsqueeze(0), value.to(self.dtype).unsqueeze(0)

    def write_tokens(
        self,
        layer_index: int,
        block_ids: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        source_start: int = 0,
        length: int | None = None,
    ) -> None:
        """Copy a range of computed K/V into a sequence's physical blocks."""
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key/value must be [batch, kv_heads, tokens, head_dim]")
        if int(key.shape[0]) != 1:
            raise ValueError("prefill writes currently expect batch=1")
        total = int(key.shape[2]) - int(source_start) if length is None else int(length)
        if total < 0 or source_start + total > int(key.shape[2]):
            raise ValueError("invalid source range for paged KV write")
        if total > len(block_ids) * self.block_size:
            raise ValueError("destination blocks are too small for the KV range")
        written = 0
        destination_offset = 0
        while written < total:
            block = block_ids[destination_offset // self.block_size]
            offset = destination_offset % self.block_size
            take = min(self.block_size - offset, total - written)
            src = slice(source_start + written, source_start + written + take)
            dst = slice(offset, offset + take)
            key_segment = key[0, :, src, :]
            value_segment = value[0, :, src, :]
            if self.key_pool.dtype == torch.int8:
                if self.key_scale_pool is None or self.value_scale_pool is None:
                    raise RuntimeError("INT8 KV cache is missing scale pools")
                key_scales = (key_segment.float().abs().amax(dim=-1) / 127.0).clamp_min(
                    1e-8
                )
                value_scales = (
                    value_segment.float().abs().amax(dim=-1) / 127.0
                ).clamp_min(1e-8)
                key_segment = torch.round(
                    key_segment.float() / key_scales[..., None]
                ).clamp(-127, 127).to(torch.int8)
                value_segment = torch.round(
                    value_segment.float() / value_scales[..., None]
                ).clamp(-127, 127).to(torch.int8)
                self.key_scale_pool[layer_index, block, :, dst] = key_scales.to(
                    self.key_scale_pool.dtype
                )
                self.value_scale_pool[layer_index, block, :, dst] = value_scales.to(
                    self.value_scale_pool.dtype
                )
            elif is_fp8_dtype(self.key_pool.dtype):
                key_segment = key_segment.to(self.key_pool.dtype)
                value_segment = value_segment.to(self.value_pool.dtype)
            self.key_pool[layer_index, block, :, dst, :] = key_segment
            self.value_pool[layer_index, block, :, dst, :] = value_segment
            written += take
            destination_offset += take

    def make_layer_view(
        self,
        layer_index: int,
        block_ids: list[list[int]],
        seq_lens: list[int],
        *,
        max_length: int,
        new_tokens: int = 1,
    ) -> PagedLayerKVCache:
        batch = len(block_ids)
        if len(seq_lens) != batch:
            raise ValueError("block_ids and seq_lens must have the same batch size")
        if new_tokens < 1:
            raise ValueError("new_tokens must be >= 1")
        max_blocks = max(1, (int(max_length) + self.block_size - 1) // self.block_size)
        table = torch.full(
            (batch, max_blocks), -1, dtype=torch.long, device=self.device
        )
        for row, blocks in enumerate(block_ids):
            if not blocks:
                continue
            table[row, : min(len(blocks), max_blocks)] = torch.tensor(
                blocks[:max_blocks], dtype=torch.long, device=self.device
            )
        lengths = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        return PagedLayerKVCache(
            key_pool=self.key_pool[layer_index],
            value_pool=self.value_pool[layer_index],
            block_table=table,
            seq_lens=lengths,
            block_size=self.block_size,
            max_length=int(max_length),
            layer_index=layer_index,
            new_tokens=int(new_tokens),
            key_scale_pool=(
                self.key_scale_pool[layer_index]
                if self.key_scale_pool is not None
                else None
            ),
            value_scale_pool=(
                self.value_scale_pool[layer_index]
                if self.value_scale_pool is not None
                else None
            ),
            compute_dtype=self.dtype,
        )

    def make_views(
        self,
        block_ids: list[list[int]],
        seq_lens: list[int],
        *,
        max_length: int,
        new_tokens: int = 1,
    ) -> list[PagedLayerKVCache]:
        """Create one view per layer while sharing block-table tensors.

        Building a fresh block table for every layer would add dozens of tiny
        allocations per decode step.  The table and length tensors are
        immutable during the step, so all layers can safely share them.
        """
        batch = len(block_ids)
        if len(seq_lens) != batch:
            raise ValueError("block_ids and seq_lens must have the same batch size")
        max_blocks = max(1, (int(max_length) + self.block_size - 1) // self.block_size)
        table = torch.full(
            (batch, max_blocks), -1, dtype=torch.long, device=self.device
        )
        if new_tokens < 1:
            raise ValueError("new_tokens must be >= 1")
        for row, blocks in enumerate(block_ids):
            if not blocks:
                continue
            table[row, : min(len(blocks), max_blocks)] = torch.tensor(
                blocks[:max_blocks], dtype=torch.long, device=self.device
            )
        lengths = torch.tensor(seq_lens, dtype=torch.long, device=self.device)
        return [
            PagedLayerKVCache(
                key_pool=self.key_pool[layer],
                value_pool=self.value_pool[layer],
                block_table=table,
                seq_lens=lengths,
                block_size=self.block_size,
                max_length=int(max_length),
                layer_index=layer,
                new_tokens=int(new_tokens),
                key_scale_pool=(
                    self.key_scale_pool[layer]
                    if self.key_scale_pool is not None
                    else None
                ),
                value_scale_pool=(
                    self.value_scale_pool[layer]
                    if self.value_scale_pool is not None
                    else None
                ),
                compute_dtype=self.dtype,
            )
            for layer in range(self.num_layers)
        ]

    def drop_blocks(self, block_ids: list[int]) -> int:
        """Release blocks without copying them anywhere.

        The alternative to offloading.  A full block costs 589824 bytes of GPU
        memory and about 1.46 ms to copy to the host; if the request that owns
        it is old enough to be evicted, paying that copy only to reload it
        later is often worse than letting the request re-prefill.  Nothing is
        written to ``CPU`` here, so the caller must treat the request as having
        no KV state at all.
        """

        if not block_ids:
            return 0
        released = list(block_ids)
        self.allocator.free(released)
        return len(released)

    def offload_blocks(self, block_ids: list[int]) -> CpuKVBlockPayload:
        """Copy physical blocks from the GPU pool to CPU memory.

        The caller keeps ownership of ``block_ids``.  The allocator is not
        touched here; the scheduler frees the GPU blocks only after it has
        accepted the CPU payload.
        """
        if not block_ids:
            empty_key = torch.empty(
                (self.num_layers, 0, self.num_kv_heads, self.block_size, self.head_dim),
                dtype=self.key_pool.dtype,
                device="cpu",
            )
            empty_value = torch.empty_like(empty_key)
            empty_scale = (
                torch.empty(
                    (self.num_layers, 0, self.num_kv_heads, self.block_size),
                    dtype=torch.float16,
                    device="cpu",
                )
                if self.key_scale_pool is not None
                else None
            )
            return CpuKVBlockPayload(
                key=empty_key,
                value=empty_value,
                key_scales=empty_scale,
                value_scales=empty_scale.clone() if empty_scale is not None else None,
                num_blocks=0,
                num_layers=self.num_layers,
            )
        index = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        if int(index.numel()) != len(set(block_ids)):
            raise ValueError("offload block ids must be unique")
        key = _index_select(self.key_pool, index, 1).detach().to("cpu", copy=True)
        value = _index_select(self.value_pool, index, 1).detach().to("cpu", copy=True)
        key_scales = None
        value_scales = None
        if self.key_scale_pool is not None:
            key_scales = (
                self.key_scale_pool.index_select(1, index)
                .detach()
                .to("cpu", copy=True)
            )
        if self.value_scale_pool is not None:
            value_scales = (
                self.value_scale_pool.index_select(1, index)
                .detach()
                .to("cpu", copy=True)
            )
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                key = key.pin_memory()
                value = value.pin_memory()
                if key_scales is not None:
                    key_scales = key_scales.pin_memory()
                if value_scales is not None:
                    value_scales = value_scales.pin_memory()
            except RuntimeError:
                # Pinned memory can be unavailable in restricted containers.
                pass
        return CpuKVBlockPayload(
            key=key,
            value=value,
            key_scales=key_scales,
            value_scales=value_scales,
            num_blocks=len(block_ids),
            num_layers=self.num_layers,
        )

    def reload_blocks(
        self, payload: CpuKVBlockPayload, block_ids: list[int]
    ) -> None:
        """Copy a CPU payload into newly allocated GPU physical blocks."""
        if payload.num_layers != self.num_layers:
            raise ValueError(
                f"payload has {payload.num_layers} layers, pool has {self.num_layers}"
            )
        if payload.num_blocks != len(block_ids):
            raise ValueError(
                f"payload has {payload.num_blocks} blocks, got {len(block_ids)} ids"
            )
        if not block_ids:
            return
        if int(torch.tensor(block_ids).numel()) != len(set(block_ids)):
            raise ValueError("reload block ids must be unique")
        index = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        key = payload.key.to(
            device=self.device,
            dtype=self.key_pool.dtype,
            non_blocking=payload.key.is_pinned(),
        )
        value = payload.value.to(
            device=self.device,
            dtype=self.value_pool.dtype,
            non_blocking=payload.value.is_pinned(),
        )
        with torch.no_grad():
            _index_copy(self.key_pool, index, key, 1)
            _index_copy(self.value_pool, index, value, 1)
            if self.key_scale_pool is not None:
                if payload.key_scales is None or payload.value_scales is None:
                    raise ValueError("INT8 payload is missing scale tensors")
                key_scales = payload.key_scales.to(
                    device=self.device,
                    dtype=self.key_scale_pool.dtype,
                    non_blocking=payload.key_scales.is_pinned(),
                )
                value_scales = payload.value_scales.to(
                    device=self.device,
                    dtype=self.value_scale_pool.dtype,
                    non_blocking=payload.value_scales.is_pinned(),
                )
                self.key_scale_pool.index_copy_(1, index, key_scales)
                self.value_scale_pool.index_copy_(1, index, value_scales)

    def stats(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "kv_cache_dtype": self.kv_cache_dtype_name,
            "resolved_kv_cache_dtype": str(self.kv_cache_dtype),
            "memory_bytes": self.memory_bytes,
            "block_bytes": self.block_bytes,
            "allocator": self.allocator.stats(),
            "prefix_cache": self.prefix_cache.stats(),
        }


__all__ = [
    "BlockAllocator",
    "CacheOutOfMemoryError",
    "CpuKVBlockPayload",
    "PagedKVCacheManager",
    "PagedLayerKVCache",
    "PrefixCache",
    "is_fp8_dtype",
    "paged_kv_block_bytes",
    "resolve_kv_cache_dtype",
]
