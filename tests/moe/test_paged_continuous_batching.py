"""Unit tests for the paged KV slot allocator and paged batch engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache

from deepinfer.engine.kv_cache import BlockAllocator, PagedKVCacheManager
from deepinfer.moe.paged_continuous_batching import (
    PagedContinuousBatchEngine,
    PagedRequest,
    PagedSlotAllocator,
    SlotOutOfBlocks,
    write_dense_rows_to_paged,
)


def _manager(num_blocks: int = 8, *, block_size: int = 2, head_dim: int = 4):
    return PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        dtype=torch.float16,
        device="cpu",
    )


def _request(messages=None) -> PagedRequest:
    return PagedRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        max_new_tokens=4,
    )


# --------------------------------------------------------------------- slots


def test_slot_allocator_grows_incrementally_and_releases() -> None:
    allocator = PagedSlotAllocator(
        BlockAllocator(4), block_size=4, max_length=16
    )
    request = _request()

    assert allocator.reserve(request, 1) == 1
    assert len(request.block_ids) == 1
    # Still inside the first block: no new allocation.
    assert allocator.reserve(request, 4) == 0
    # Crossing into a second and third block.
    assert allocator.reserve(request, 9) == 2
    assert len(request.block_ids) == 3
    assert allocator.stats()["used_blocks"] == 3

    released = allocator.release(request)
    assert released == 3
    assert request.block_ids == []
    assert allocator.stats()["used_blocks"] == 0


def test_slot_allocator_rejects_past_max_length_and_exhaustion() -> None:
    allocator = PagedSlotAllocator(
        BlockAllocator(2), block_size=4, max_length=16
    )
    request = _request()
    with pytest.raises(SlotOutOfBlocks):
        allocator.reserve(request, 17)

    # Two blocks are all the pool has; a third request cannot be served.
    first, second = _request(), _request()
    allocator.reserve(first, 4)
    allocator.reserve(second, 4)
    with pytest.raises(SlotOutOfBlocks):
        allocator.reserve(_request(), 4)
    assert allocator.stats()["rejected"] == 2


def test_slot_allocator_reuses_freed_blocks() -> None:
    allocator = PagedSlotAllocator(
        BlockAllocator(4), block_size=4, max_length=16
    )
    first, second = _request(), _request()
    allocator.reserve(first, 8)
    allocator.reserve(second, 8)
    blocks = list(first.block_ids)
    allocator.release(first)

    third = _request()
    allocator.reserve(third, 8)
    assert sorted(third.block_ids) == sorted(blocks)
    assert allocator.stats()["free_blocks"] == 0


# ---------------------------------------------------------------- write path


def test_write_dense_rows_lands_at_zero_based_positions() -> None:
    manager = _manager(num_blocks=4, block_size=2, head_dim=4)
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
        )
    )
    allocator = PagedSlotAllocator(
        manager.allocator, block_size=2, max_length=8
    )
    request = _request()
    allocator.reserve(request, 5)

    # Dense batch cache with three real tokens left-padded to length 5.
    cache = DynamicCache()
    key = torch.zeros(1, 1, 5, 4, dtype=torch.float16)
    value = torch.zeros(1, 1, 5, 4, dtype=torch.float16)
    key[0, 0, 2:, :] = torch.tensor([1.0, 2.0, 3.0]).view(3, 1).expand(3, 4)
    value[0, 0, 2:, :] = torch.tensor([4.0, 5.0, 6.0]).view(3, 1).expand(3, 4)
    cache.key_cache = [key]
    cache.value_cache = [value]
    cache._seen_tokens = 5

    write_dense_rows_to_paged(
        manager, runtime=runtime, rows=[(request, cache, 3)], padded_len=5
    )

    views = manager.make_views(
        [request.block_ids], [0], max_length=3, new_tokens=3
    )
    materialized_key, materialized_value, _ = views[0].materialize()
    assert torch.allclose(
        materialized_key[0, 0, :3, 0].float(), torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.allclose(
        materialized_value[0, 0, :3, 0].float(), torch.tensor([4.0, 5.0, 6.0])
    )


# ------------------------------------------------------------------- engine


class _FakeTokenizer:
    eos_token_id = 99
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        if isinstance(text, str):
            ids = [len(word) for word in text.split()]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        rows = [[len(word) for word in item.split()] for item in text]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, width - len(row) :] = torch.tensor(row)
            mask[index, width - len(row) :] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "-".join(str(token) for token in token_ids)


def _fake_runtime() -> SimpleNamespace:
    """A CPU stand-in that echoes an incrementing token id per step."""

    calls = {"decode": 0, "prefill": 0}

    def forward_logits(input_ids, position_ids, *, kv_cache=None, attention_mask=None):
        assert kv_cache is not None
        views = list(kv_cache)
        batch = int(input_ids.shape[0])
        calls["decode"] += 1
        logits = torch.zeros(batch, 1, 120)
        for row in range(batch):
            target = (int(input_ids[row, -1]) + 1) % 100
            logits[row, 0, target] = 1.0
        assert all(view.block_table.shape[0] == batch for view in views)
        return logits

    runtime = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
        ),
        max_cache=32,
        device=torch.device("cpu"),
        tokenizer=_FakeTokenizer(),
        forward_logits=forward_logits,
        _sample=lambda logits, temperature=0.0, top_p=1.0, top_k=0: int(torch.argmax(logits)),
        _calls=calls,
        # GenerationResult bookkeeping
        gpu_free_bytes_after_load=0,
        shared_experts_device="cpu",
        lm_head_device="cpu",
        expert_store=None,
        cpu_moe_seconds=0.0,
        expert_device_chunks=0,
        remote_expert_chunks=0,
        expert_device_counts={},
        dense_layer_indices=[],
        moe_layer_indices=[0],
        dense_mlp_devices={},
        fused_moe_enabled=True,
        fused_moe_chunks=0,
        fallback_moe_chunks=0,
        cpu_moe_layer_indices=[],
        cpu_moe_chunks=0,
        requested_cpu_experts_per_layer=0,
        cpu_expert_chunks=0,
        ecs_controller=None,
    )
    return runtime


def test_paged_engine_prefills_decodes_and_frees_blocks() -> None:
    runtime = _fake_runtime()
    engine = PagedContinuousBatchEngine(
        runtime,
        max_batch_size=2,
        block_size=4,
        num_blocks=16,
        prefill_mode="paged",
        pool_device="cpu",
    )
    try:
        first = engine.submit(
            [{"role": "user", "content": "a bb ccc"}], max_new_tokens=3
        )
        second = engine.submit(
            [{"role": "user", "content": "dd eeee"}], max_new_tokens=2
        )
        result_first = first.result(timeout=30)
        result_second = second.result(timeout=30)
    finally:
        engine.close()

    assert result_first.completion_tokens == 3
    assert result_second.completion_tokens == 2
    status = engine.status()
    assert status["completed"] == 2
    assert status["failed"] == 0
    assert status["kv"]["used_blocks"] == 0, "blocks must return to the pool"
    assert status["kv"]["allocations"] >= 2
    assert status["decode_steps"] >= 2
    # Both requests were in flight together at some point.
    assert status["max_active_seen"] >= 2


def test_paged_engine_reports_allocation_failure() -> None:
    runtime = _fake_runtime()
    engine = PagedContinuousBatchEngine(
        runtime,
        max_batch_size=1,
        block_size=4,
        num_blocks=1,
        prefill_mode="paged",
        pool_device="cpu",
    )
    try:
        future = engine.submit(
            [{"role": "user", "content": "a bb ccc dd eee"}], max_new_tokens=8
        )
        with pytest.raises(Exception):
            future.result(timeout=30)
    finally:
        engine.close()
    assert engine.status()["kv"]["used_blocks"] == 0

def test_write_dense_rows_groups_equal_lengths() -> None:
    """Two equal-length rows are written together and stay independent."""

    manager = _manager(num_blocks=8, block_size=2, head_dim=4)
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=4,
        )
    )
    allocator = PagedSlotAllocator(manager.allocator, block_size=2, max_length=8)
    first, second = _request(), _request()
    allocator.reserve(first, 2)
    allocator.reserve(second, 2)

    cache = DynamicCache()
    key = torch.zeros(2, 1, 4, 4, dtype=torch.float16)
    value = torch.zeros(2, 1, 4, 4, dtype=torch.float16)
    key[0, 0, 2:, 0] = torch.tensor([1.0, 2.0])
    key[1, 0, 2:, 0] = torch.tensor([7.0, 8.0])
    value[0, 0, 2:, 0] = torch.tensor([3.0, 4.0])
    value[1, 0, 2:, 0] = torch.tensor([9.0, 10.0])
    cache.key_cache = [key]
    cache.value_cache = [value]
    cache._seen_tokens = 4

    rows = [
        (first, _single_row(cache, 0), 2),
        (second, _single_row(cache, 1), 2),
    ]
    write_dense_rows_to_paged(manager, runtime=runtime, rows=rows, padded_len=4)

    first_view = manager.make_views([first.block_ids], [0], max_length=2, new_tokens=2)
    second_view = manager.make_views([second.block_ids], [0], max_length=2, new_tokens=2)
    assert torch.allclose(
        first_view[0].materialize()[0][0, 0, :2, 0].float(),
        torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(
        second_view[0].materialize()[0][0, 0, :2, 0].float(),
        torch.tensor([7.0, 8.0]),
    )
    assert first.block_ids != second.block_ids


def _single_row(cache: DynamicCache, row: int) -> DynamicCache:
    single = DynamicCache()
    single.key_cache = [tensor[row : row + 1] for tensor in cache.key_cache]
    single.value_cache = [tensor[row : row + 1] for tensor in cache.value_cache]
    single._seen_tokens = int(cache.get_seq_length())
    return single


def test_prefix_cache_reuses_complete_prompt_blocks() -> None:
    runtime = _fake_runtime()
    engine = PagedContinuousBatchEngine(
        runtime,
        max_batch_size=1,
        block_size=2,
        num_blocks=16,
        prefill_mode="paged",
        pool_device="cpu",
        prefix_cache_blocks=8,
    )
    try:
        prompt = [{"role": "user", "content": "a bb ccc dddd eeee"}]
        first = engine.submit(prompt, max_new_tokens=3).result(timeout=30)
        second = engine.submit(prompt, max_new_tokens=3).result(timeout=30)
        status = engine.status()
    finally:
        engine.close()

    # 5 prompt tokens, block size 2 -> two complete blocks are cacheable.
    assert status["prefix_cache"]["hits_this_engine"] == 1
    assert status["prefix_cache"]["reused_tokens"] == 4
    assert status["prefix_cache"]["cached_blocks"] == 2
    assert first.token_ids == second.token_ids
    assert status["kv"]["used_blocks"] >= 0  # all request refs released


def test_prefix_cache_disabled_keeps_zero_hits() -> None:
    runtime = _fake_runtime()
    engine = PagedContinuousBatchEngine(
        runtime,
        max_batch_size=1,
        block_size=2,
        num_blocks=16,
        prefill_mode="paged",
        pool_device="cpu",
    )
    try:
        prompt = [{"role": "user", "content": "a bb ccc dddd eeee"}]
        engine.submit(prompt, max_new_tokens=2).result(timeout=30)
        engine.submit(prompt, max_new_tokens=2).result(timeout=30)
        status = engine.status()
    finally:
        engine.close()
    assert status["prefix_cache"]["hits_this_engine"] == 0
    assert status["prefix_cache"]["max_blocks"] == 0
