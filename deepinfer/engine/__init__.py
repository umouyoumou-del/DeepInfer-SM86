"""Execution-engine primitives layered on the SM86 model executor."""

from deepinfer.engine.kv_cache import (
    BlockAllocator,
    CacheOutOfMemoryError,
    CpuKVBlockPayload,
    PagedKVCacheManager,
    PagedLayerKVCache,
    PrefixCache,
    is_fp8_dtype,
    paged_kv_block_bytes,
    resolve_kv_cache_dtype,
)
from deepinfer.engine.memory_budget import (
    InsufficientKvMemoryError,
    MemoryBudgetReport,
    plan_kv_blocks,
)
from deepinfer.engine.metrics import EngineMetrics

__all__ = [
    "BlockAllocator",
    "CacheOutOfMemoryError",
    "CpuKVBlockPayload",
    "EngineMetrics",
    "InsufficientKvMemoryError",
    "MemoryBudgetReport",
    "PagedKVCacheManager",
    "PagedLayerKVCache",
    "PrefixCache",
    "is_fp8_dtype",
    "paged_kv_block_bytes",
    "plan_kv_blocks",
    "resolve_kv_cache_dtype",
]
