"""Execution-engine primitives layered on the SM86 model executor."""

from deepinfer.engine.kv_cache import (
    BlockAllocator,
    CacheOutOfMemoryError,
    PagedKVCacheManager,
    PagedLayerKVCache,
    PrefixCache,
    paged_kv_block_bytes,
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
    "EngineMetrics",
    "InsufficientKvMemoryError",
    "MemoryBudgetReport",
    "PagedKVCacheManager",
    "PagedLayerKVCache",
    "PrefixCache",
    "paged_kv_block_bytes",
    "plan_kv_blocks",
]
