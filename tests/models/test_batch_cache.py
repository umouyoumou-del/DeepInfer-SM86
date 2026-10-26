from __future__ import annotations

import torch

from deepinfer.model_executor.models.qwen2 import Qwen2StaticLayerKVCache


def test_static_cache_per_batch_positions() -> None:
    cache = Qwen2StaticLayerKVCache(
        key=torch.zeros((2, 2, 8, 4)),
        value=torch.zeros((2, 2, 8, 4)),
    )
    key = torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 4)
    value = key + 100
    positions = torch.tensor([[3], [5]])
    cache.update(key, value, positions)
    assert torch.equal(cache.key[0, :, 3, :], key[0, :, 0, :])
    assert torch.equal(cache.key[1, :, 5, :], key[1, :, 0, :])
    assert torch.equal(cache.value[0, :, 3, :], value[0, :, 0, :])
    assert torch.equal(cache.value[1, :, 5, :], value[1, :, 0, :])
    assert torch.count_nonzero(cache.key[0, :, 5, :]) == 0
