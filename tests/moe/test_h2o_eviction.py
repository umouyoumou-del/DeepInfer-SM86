"""H2O eviction: the coldest block by attention mass leaves first, not the oldest."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deepinfer.engine.kv_cache import BlockAllocator, PrefixCache  # noqa: E402
from deepinfer.moe.block_mass import BlockMassHeat, from_layer_mass  # noqa: E402


def _cache(blocks: int = 4, heat=None) -> PrefixCache:
    allocator = BlockAllocator(blocks + 4)
    return PrefixCache(allocator, max_blocks=blocks, heat_fn=heat)


def test_without_heat_eviction_is_lru() -> None:
    cache = _cache(blocks=2)
    allocated = cache.allocator.allocate(3)
    cache.register([b"a"], [allocated[0]])
    cache.register([b"b"], [allocated[1]])
    # third registration forces an eviction; LRU must drop the oldest, "a"
    cache.register([b"c"], [allocated[2]])
    remaining = set()
    for digest in cache._entries:
        remaining.add(digest)
    assert b"a" not in remaining


def test_with_heat_eviction_drops_the_coldest_old_token() -> None:
    """A recently registered but cold block must leave before an old hot one."""

    mass = torch.zeros(8)
    mass[1] = 0.0     # block 1 will be cold
    mass[0] = 5.0     # block 0 is hot despite being older
    heat = BlockMassHeat(mass)
    cache = _cache(blocks=2, heat=heat)
    allocated = cache.allocator.allocate(3)
    cache.register([b"hot"], [allocated[0]])
    cache.register([b"cold"], [allocated[1]])
    cache.register([b"new"], [allocated[2]])
    remaining = set(cache._entries)
    assert b"hot" in remaining, "the hot block must survive even though it is older"
    assert b"cold" not in remaining, "the cold block must be the one evicted"
    assert heat.calls > 0


def test_unknown_block_scores_default() -> None:
    heat = BlockMassHeat(torch.ones(4), default=0.25)
    assert heat(0) == pytest.approx(1.0)
    assert heat(99) == pytest.approx(0.25)
    assert heat(-1) == pytest.approx(0.25)


def test_mass_can_be_updated_between_measurements() -> None:
    heat = BlockMassHeat(torch.zeros(4))
    assert heat(2) == 0.0
    heat.update(torch.tensor([0.0, 0.0, 3.0, 0.0]))
    assert heat(2) == pytest.approx(3.0)


def test_from_layer_mass_wraps_a_row() -> None:
    heat = from_layer_mass(torch.tensor([1.0, 2.0, 3.0]))
    assert heat(1) == pytest.approx(2.0)
