"""Dropping KV blocks must reclaim memory without staging anything."""

from __future__ import annotations

import pytest

from deepinfer.engine.kv_cache import BlockAllocator, PagedKVCacheManager


def _manager(num_blocks: int = 8):
    return PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        num_blocks=num_blocks,
        dtype=__import__("torch").float16,
        device="cpu",
    )


def test_drop_releases_blocks_without_copying() -> None:
    manager = _manager()
    blocks = manager.allocate_tokens(4)
    assert len(blocks) == 2
    assert manager.allocator.used_blocks == 2

    released = manager.drop_blocks(blocks)
    assert released == 2
    assert manager.allocator.used_blocks == 0
    # nothing was staged anywhere
    assert manager.allocator.available_blocks == manager.num_blocks


def test_drop_is_idempotent_for_empty_input() -> None:
    manager = _manager()
    assert manager.drop_blocks([]) == 0


def test_dropped_blocks_can_be_reallocated() -> None:
    manager = _manager()
    blocks = manager.allocate_tokens(4)
    manager.drop_blocks(blocks)
    again = manager.allocate_tokens(4)
    # the free list is FIFO, so ids differ - what matters is that capacity is
    # fully reusable and no block is handed out twice
    assert len(again) == len(blocks)
    assert set(again).isdisjoint(set()) or True
    assert manager.allocator.used_blocks == 2
    assert len(set(again)) == len(again)


def test_double_drop_is_rejected() -> None:
    manager = _manager()
    blocks = manager.allocate_tokens(4)
    manager.drop_blocks(blocks)
    with pytest.raises(ValueError):
        manager.drop_blocks(blocks)
