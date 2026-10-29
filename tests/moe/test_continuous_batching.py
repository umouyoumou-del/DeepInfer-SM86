from __future__ import annotations

from types import SimpleNamespace

import torch

from deepinfer.moe.continuous_batching import pack_caches, select_caches


def _cache(length: int, value: float):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.key_cache = [torch.full((1, 2, length, 3), value)]
    cache.value_cache = [torch.full((1, 2, length, 3), value)]
    cache._seen_tokens = length
    return cache


def test_continuous_cache_pack_and_select_left_pads() -> None:
    packed = pack_caches([_cache(2, 1.0), _cache(1, 2.0)], target_len=2)
    assert packed.key_cache[0].shape == (2, 2, 2, 3)
    assert torch.all(packed.key_cache[0][0, :, 0, :] == 1.0)
    assert torch.all(packed.key_cache[0][1, :, 0, :] == 0.0)
    assert torch.all(packed.key_cache[0][1, :, 1, :] == 2.0)
    selected = select_caches(packed, [1])
    assert selected.key_cache[0].shape == (1, 2, 2, 3)
    assert torch.all(selected.key_cache[0][0, :, 1, :] == 2.0)
