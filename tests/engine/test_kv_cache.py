from __future__ import annotations

import pytest
import torch

from deepinfer.attrdict import AttrDict
from deepinfer.engine.kv_cache import (
    BlockAllocator,
    CacheOutOfMemoryError,
    PagedKVCacheManager,
    PrefixCache,
)
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM


def _tiny_cfg() -> AttrDict:
    return AttrDict(
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=True,
        attention_dropout=0.0,
        hidden_act="silu",
        tie_word_embeddings=True,
        use_sliding_window=False,
        sliding_window=None,
        window_size=(-1, -1),
    )


def test_block_allocator_reference_counting_and_oom() -> None:
    allocator = BlockAllocator(3)
    blocks = allocator.allocate(2)
    assert blocks == [0, 1]
    allocator.retain([blocks[0]])
    assert allocator.stats()["shared_blocks"] == 1
    allocator.free([blocks[0]])
    assert allocator.available_blocks == 1
    allocator.free(blocks)
    assert allocator.available_blocks == 3

    with pytest.raises(CacheOutOfMemoryError):
        allocator.allocate(4)
    accepted = allocator.allocate(3)
    assert len(accepted) == 3


def test_prefix_cache_reuses_and_retains_blocks() -> None:
    allocator = BlockAllocator(8)
    prefix = PrefixCache(allocator, max_blocks=2)
    blocks = allocator.allocate(2)
    hashes = PrefixCache.block_hashes(list(range(8)), block_size=4)
    assert len(hashes) == 2
    assert prefix.register(hashes, blocks) == 2
    assert allocator.stats()["shared_blocks"] == 2

    matched = prefix.match(hashes)
    assert matched == blocks
    allocator.free(matched)
    allocator.free(blocks)
    assert allocator.available_blocks == 6
    assert prefix.stats()["reused_blocks"] == 2

    extra_blocks = allocator.allocate(2)
    more_hashes = PrefixCache.block_hashes(list(range(8, 16)), block_size=4)
    prefix.register(more_hashes, extra_blocks)
    assert prefix.stats()["evictions"] == 2
    prefix.clear()
    allocator.free(extra_blocks)
    assert allocator.available_blocks == 8


def test_paged_cache_write_append_and_materialize() -> None:
    manager = PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        block_size=4,
        num_blocks=4,
        dtype=torch.float32,
        device="cpu",
    )
    blocks = manager.allocate_tokens(6)
    assert len(blocks) == 2
    key = torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2)
    value = key + 100
    manager.write_tokens(0, blocks, key, value)

    # Six prompt tokens already occupy two blocks; position six appends into
    # the partial tail of the second block.
    view = manager.make_layer_view(0, [blocks], [6], max_length=7)
    new_key = torch.tensor([[[[200.0, 201.0]]]])
    new_value = torch.tensor([[[[300.0, 301.0]]]])
    view.update(new_key, new_value, torch.tensor([[6]]))
    materialized_key, materialized_value, valid = view.materialize()

    assert tuple(materialized_key.shape) == (1, 1, 7, 2)
    assert valid.all()
    assert torch.equal(materialized_key[:, :, :6], key)
    assert torch.equal(materialized_value[:, :, :6], value)
    assert torch.equal(materialized_key[:, :, 6], new_key[:, :, 0])


def test_batched_paged_decode_matches_dynamic_cache() -> None:
    torch.manual_seed(20261027)
    model = Qwen2ForCausalLM(_tiny_cfg()).eval().requires_grad_(False)
    manager = PagedKVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        num_blocks=24,
        dtype=torch.float32,
        device="cpu",
    )
    prompts = [
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
        torch.tensor([[3, 1, 4, 1, 5]]),
    ]
    next_tokens = [torch.tensor([[8]]), torch.tensor([[9]])]

    references = []
    paged_caches = []
    blocks = []
    lengths = []
    with torch.no_grad():
        for prompt, next_token in zip(prompts, next_tokens, strict=True):
            cache = model.new_kv_cache()
            positions = torch.arange(prompt.shape[1]).unsqueeze(0)
            model.forward_logits(prompt, positions, kv_cache=cache)
            reference = model.forward_logits(
                next_token,
                torch.tensor([[prompt.shape[1]]]),
                kv_cache=cache,
            )[:, -1]
            references.append(reference)
            paged_caches.append(cache)
            blocks.append(manager.allocate_tokens(prompt.shape[1]))
            lengths.append(int(prompt.shape[1]))

        for layer_index in range(2):
            for row, cache in enumerate(paged_caches):
                assert cache[layer_index] is not None
                manager.write_tokens(
                    layer_index,
                    blocks[row],
                    cache[layer_index].key,
                    cache[layer_index].value,
                )

        max_length = max(lengths) + 1
        views = manager.make_views(blocks, lengths, max_length=max_length)
        actual = model.forward_logits(
            torch.cat(next_tokens, dim=0),
            torch.tensor([[length] for length in lengths]),
            kv_cache=views,
        )[:, -1]

    for reference, row in zip(references, actual, strict=True):
        torch.testing.assert_close(row, reference[0], atol=2e-5, rtol=2e-4)


def test_paged_decode_matches_dynamic_cache_with_gemma2_masks() -> None:
    torch.manual_seed(20261027)
    cfg = AttrDict(
        num_hidden_layers=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="gelu_pytorch_tanh",
        tie_word_embeddings=True,
        use_sliding_window=True,
        sliding_window=4,
        window_size=(4, -1),
        sliding_window_pattern=(True, False, True, False),
        rms_norm_weight_offset=1.0,
        embedding_multiplier=32**0.5,
        sandwich_norm=True,
        attn_logit_softcapping=5.0,
        final_logit_softcapping=10.0,
        query_pre_attn_scalar=8.0,
    )
    model = Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33]])
    next_token = torch.tensor([[37]])
    prompt_length = int(prompt.shape[1])

    with torch.no_grad():
        dynamic_cache = model.new_kv_cache()
        model.forward_logits(
            prompt,
            torch.arange(prompt_length).unsqueeze(0),
            kv_cache=dynamic_cache,
        )
        reference = model.forward_logits(
            next_token,
            torch.tensor([[prompt_length]]),
            kv_cache=dynamic_cache,
        )[:, -1]

    manager = PagedKVCacheManager(
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        num_blocks=16,
        dtype=torch.float32,
        device="cpu",
    )
    blocks = manager.allocate_tokens(prompt_length)
    for layer_index, layer_cache in enumerate(dynamic_cache):
        assert layer_cache is not None
        # dynamic cache now includes the reference decode token too; copy only
        # the prompt prefix into the paged pool.
        manager.write_tokens(
            layer_index,
            blocks,
            layer_cache.key[:, :, :prompt_length, :],
            layer_cache.value[:, :, :prompt_length, :],
        )
    view_cache = manager.make_views(
        [blocks],
        [prompt_length],
        max_length=prompt_length + 1,
    )
    with torch.no_grad():
        actual = model.forward_logits(
            next_token,
            torch.tensor([[prompt_length]]),
            kv_cache=view_cache,
        )[:, -1]

    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
