from __future__ import annotations

import pytest
import torch

from deepinfer.engine.kv_cache import PagedKVCacheManager
from deepinfer.moe.paged_adapter import DeepSeekPagedCache

pytestmark = pytest.mark.gpu


def test_deepseek_paged_cache_materializes_mla_shapes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manager = PagedKVCacheManager(
        num_layers=2,
        num_kv_heads=4,
        head_dim=8,
        block_size=4,
        num_blocks=16,
        device="cuda",
    )
    blocks = manager.allocate_tokens(8)
    try:
        views = manager.make_views(
            [blocks],
            [0],
            max_length=2,
            new_tokens=2,
        )
        cache = DeepSeekPagedCache(
            views,
            torch.tensor([[0, 1]], dtype=torch.long, device="cuda"),
            past_length=0,
            max_length=2,
            key_dim=6,
            value_dim=4,
            head_dim=8,
        )
        key = torch.randn(
            (1, 4, 2, 6), dtype=torch.float16, device="cuda"
        )
        value = torch.randn(
            (1, 4, 2, 4), dtype=torch.float16, device="cuda"
        )
        full_key, full_value = cache.update(key, value, layer_idx=0)
        torch.cuda.synchronize()
        assert full_key.shape == (1, 4, 2, 6)
        assert full_value.shape == (1, 4, 2, 4)
        torch.testing.assert_close(full_key, key)
        torch.testing.assert_close(full_value, value)
        assert cache.get_usable_length(2) == 0
        assert cache.get_seq_length() == 0
        assert cache.get_max_cache_shape() == 2
        assert len(cache) == 2
    finally:
        manager.allocator.free(blocks)
