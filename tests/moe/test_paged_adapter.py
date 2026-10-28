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


def test_deepseek_paged_cache_uses_global_block_positions() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(7)
    manager = PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        num_blocks=16,
        device="cuda",
    )
    blocks = manager.allocate_tokens(8)
    try:
        prefill = manager.make_views(
            [blocks[:2]],
            [0],
            max_length=6,
            new_tokens=6,
        )
        key = torch.randn(1, 2, 6, 6, device="cuda", dtype=torch.float16)
        value = torch.randn(1, 2, 6, 4, device="cuda", dtype=torch.float16)
        prefill[0].update(
            torch.nn.functional.pad(key, (0, 2)),
            torch.nn.functional.pad(value, (0, 4)),
            torch.arange(6, device="cuda").unsqueeze(0),
        )
        positions = torch.tensor([[4, 5]], device="cuda")
        views = manager.make_views(
            [blocks[:2]],
            [4],
            max_length=6,
            new_tokens=2,
        )
        cache = DeepSeekPagedCache(
            views,
            positions,
            past_length=4,
            max_length=6,
            key_dim=6,
            value_dim=4,
            head_dim=8,
        )
        query = torch.randn(1, 2, 2, 6, device="cuda", dtype=torch.float16)
        actual = cache.attend(query, 0, positions, scale=6**-0.5)
        dense_key, dense_value, valid = views[0].materialize()
        dense_key = dense_key[..., :6].float()
        dense_value = dense_value[..., :4].float()
        key_positions = torch.arange(6, device="cuda")
        mask = valid.view(1, 1, 1, 6) & (
            key_positions.view(1, 1, 1, 6)
            <= positions.view(1, 1, 2, 1)
        )
        scores = torch.einsum(
            "bhqd,bhkd->bhqk", query.float(), dense_key
        ) * (6**-0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        reference = torch.einsum(
            "bhqk,bhkd->bhqd",
            torch.softmax(scores, dim=-1),
            dense_value,
        )
        torch.testing.assert_close(
            actual.float(), reference, rtol=2e-3, atol=2e-3
        )
    finally:
        manager.allocator.free(blocks)
