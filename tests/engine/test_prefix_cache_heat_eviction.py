"""Prefix-cache eviction: LRU by default, coldest-first when heat is supplied."""

from __future__ import annotations

from deepinfer.engine.kv_cache import BlockAllocator, PrefixCache


def _fill(cache: PrefixCache, count: int, *, offset: int = 0) -> None:
    for index in range(offset, offset + count):
        digest = bytes([index]) * 16
        block = cache.allocator.allocate(1)[0]
        cache.register([digest], [block])


def test_default_eviction_is_lru() -> None:
    allocator = BlockAllocator(8)
    cache = PrefixCache(allocator, max_blocks=3)
    _fill(cache, 3)
    first = next(iter(cache._entries.values()))
    _fill(cache, 1, offset=3)  # forces one eviction
    assert first not in cache._entries.values()
    assert cache.evictions == 1


def test_heat_eviction_keeps_the_hot_old_block() -> None:
    allocator = BlockAllocator(8)
    heat = {0: 0.1, 1: 9.0, 2: 0.2}
    cache = PrefixCache(allocator, max_blocks=3, heat_fn=lambda block: heat[block])
    _fill(cache, 3)
    cold_block = 0          # oldest, but also the coldest
    hot_block = 1           # older than 2 yet much hotter
    _fill(cache, 1, offset=3)
    remaining = set(cache._entries.values())
    assert cold_block not in remaining, "the coldest block should go first"
    assert hot_block in remaining, "a hot block must survive despite its age"


def test_heat_source_failure_falls_back_to_zero() -> None:
    allocator = BlockAllocator(8)

    def broken(block: int) -> float:
        if block == 1:
            raise RuntimeError("heat lookup failed")
        return 5.0

    cache = PrefixCache(allocator, max_blocks=2, heat_fn=broken)
    _fill(cache, 2)
    _fill(cache, 1, offset=2)  # must not raise
    assert cache.evictions == 1
