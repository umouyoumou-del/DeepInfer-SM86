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

    def __init__(self, allocator: BlockAllocator, *, max_blocks: int) -> None:
        if max_blocks < 0:
            raise ValueError("max_blocks must be non-negative")
        self.allocator = allocator
        self.max_blocks = int(max_blocks)
        self._entries: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.reused_blocks = 0
        self.evictions = 0

    @staticmethod
    def block_hashes(token_ids: list[int] | tuple[int, ...], block_size: int) -> list[bytes]:
        """Return one cumulative hash per *complete* token block."""
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        hasher = hashlib.blake2b(digest_size=16)
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
                    _, evicted = self._entries.popitem(last=False)
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

    @property
    def is_paged(self) -> bool:
        return True

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        cache_positions: torch.Tensor,
    ) -> None:
        batch = int(key.shape[0])
        if key.shape[2] != 1 or value.shape[2] != 1:
            raise NotImplementedError(
                "the paged attention backend currently writes one decode token "
                f"per call, got query length {key.shape[2]}"
            )
        positions = cache_positions.reshape(-1)[:batch].to(
            device=self.key_pool.device, dtype=torch.long
        )
        if positions.numel() != batch:
            raise ValueError("cache_positions must provide one position per batch row")
        batch_indices = torch.arange(batch, device=self.key_pool.device)
        block_indices = positions // self.block_size
        block_ids = self.block_table[batch_indices, block_indices]
        offsets = positions % self.block_size
        # Trust the scheduler invariant: decode blocks are allocated before
        # building the view.  Checking ``(block_ids < 0).any()`` here would
        # force a GPU->CPU sync once per layer and erase the benefit of a
        # tensor-only cache path.
        self.key_pool[block_ids, :, offsets, :] = key[:, :, 0, :]
        self.value_pool[block_ids, :, offsets, :] = value[:, :, 0, :]

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
        key = self.key_pool[safe_table]
        value = self.value_pool[safe_table]
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

        positions = torch.arange(max_length, device=self.seq_lens.device)
        valid_lengths = (self.seq_lens + 1).clamp(max=max_length)
        valid = positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
        return key, value, valid


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
        self.key_pool = torch.zeros(shape, dtype=dtype, device=self.device)
        self.value_pool = torch.zeros_like(self.key_pool)

    @property
    def block_bytes(self) -> int:
        elements = (
            self.num_layers
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
        )
        return elements * self.key_pool.element_size() * 2

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
        key = self.key_pool[layer_index, block_ids].reshape(
            self.num_kv_heads, capacity, self.head_dim
        )[:, :length, :].unsqueeze(0)
        value = self.value_pool[layer_index, block_ids].reshape(
            self.num_kv_heads, capacity, self.head_dim
        )[:, :length, :].unsqueeze(0)
        return key, value

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
            self.key_pool[layer_index, block, :, dst, :] = key[0, :, src, :]
            self.value_pool[layer_index, block, :, dst, :] = value[0, :, src, :]
            written += take
            destination_offset += take

    def make_layer_view(
        self,
        layer_index: int,
        block_ids: list[list[int]],
        seq_lens: list[int],
        *,
        max_length: int,
    ) -> PagedLayerKVCache:
        batch = len(block_ids)
        if len(seq_lens) != batch:
            raise ValueError("block_ids and seq_lens must have the same batch size")
        max_blocks = max(1, (int(max_length) + self.block_size - 1) // self.block_size)
        table = torch.full(
            (batch, max_blocks), -1, dtype=torch.long, device=self.device
        )
        for row, blocks in enumerate(block_ids):
            if not blocks:
                continue
            if len(blocks) > max_blocks:
                raise ValueError(
                    f"row {row} has {len(blocks)} blocks, max_length only needs {max_blocks}"
                )
            table[row, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.long, device=self.device
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
        )

    def make_views(
        self,
        block_ids: list[list[int]],
        seq_lens: list[int],
        *,
        max_length: int,
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
        for row, blocks in enumerate(block_ids):
            if not blocks:
                continue
            if len(blocks) > max_blocks:
                raise ValueError(
                    f"row {row} has {len(blocks)} blocks, max_length only needs {max_blocks}"
                )
            table[row, : len(blocks)] = torch.tensor(
                blocks, dtype=torch.long, device=self.device
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
            )
            for layer in range(self.num_layers)
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "memory_bytes": self.memory_bytes,
            "block_bytes": self.block_bytes,
            "allocator": self.allocator.stats(),
            "prefix_cache": self.prefix_cache.stats(),
        }


__all__ = [
    "BlockAllocator",
    "CacheOutOfMemoryError",
    "PagedKVCacheManager",
    "PagedLayerKVCache",
    "PrefixCache",
]
