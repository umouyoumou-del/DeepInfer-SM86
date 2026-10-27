from __future__ import annotations

import pytest
import torch

from apps.paged_scheduler import PagedBatchScheduler
from deepinfer.attrdict import AttrDict
from deepinfer.engine.kv_cache import CacheOutOfMemoryError
from deepinfer.model_executor.models import qwen2 as qwen2_module
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM


class _Tokenizer:
    eos_token_id = 63

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return str(messages[0]["content"])

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        length = int(prompt)
        return {"input_ids": torch.arange(1, length + 1).unsqueeze(0)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(token)) for token in token_ids)


class _Runtime:
    device = torch.device("cpu")
    max_cache = 32
    qwen2_module = qwen2_module

    def __init__(self) -> None:
        cfg = AttrDict(
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
        torch.manual_seed(20261027)
        self.model = Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
        self.tokenizer = _Tokenizer()

    def _sample(self, logits, temperature=0.0, top_p=0.95, top_k=50) -> int:
        return int(logits.argmax().item())


def _make_scheduler(
    runtime: _Runtime,
    *,
    max_num_seqs: int = 2,
    blocks: int = 24,
    chunk: int = 64,
    prefix_blocks: int = 8,
    kv_cache_dtype: str = "fp16",
) -> PagedBatchScheduler:
    return PagedBatchScheduler(
        runtime,
        max_num_seqs=max_num_seqs,
        block_size=4,
        prefill_chunk_tokens=chunk,
        kv_cache_dtype=kv_cache_dtype,
        num_gpu_blocks=blocks,
        prefix_cache_blocks=prefix_blocks,
        max_queue_size=8,
    )


def test_paged_scheduler_batches_decode() -> None:
    runtime = _Runtime()
    scheduler = _make_scheduler(runtime)
    try:
        first = scheduler.submit(
            [{"role": "user", "content": "12"}],
            max_new_tokens=4,
            temperature=0.0,
        )
        second = scheduler.submit(
            [{"role": "user", "content": "12"}],
            max_new_tokens=4,
            temperature=0.0,
        )
        first_result = first.future.result(timeout=30)
        second_result = second.future.result(timeout=30)

        assert first_result.completion_tokens == 4
        assert second_result.completion_tokens == 4
        assert len(first_result.inter_token_seconds) == 3
        assert first_result.tpot_seconds >= 0.0
        # The two requests were admitted before either prefill completed, so
        # their same prompt could not hit the prefix cache yet.
        assert first_result.prefix_cached_tokens == 0
        assert second_result.prefix_cached_tokens == 0
        # First tokens come from prefill; the scheduler observed at least one
        # decode forward containing both requests.
        assert scheduler.metrics.get("max_decode_batch_size") >= 2
        assert scheduler.metrics.get("decode_steps_total") < 6
        status = scheduler.status()
        assert status["active"] == 0
        assert status["memory_budget"]["mode"] == "cpu"
        assert "deepinfer_generated_tokens_total 8" in scheduler.metrics.render()
    finally:
        scheduler.stop()


def test_prefix_cache_reuse_and_tenant_isolation() -> None:
    runtime = _Runtime()
    scheduler = _make_scheduler(runtime, max_num_seqs=1)
    messages = [{"role": "user", "content": "12"}]
    try:
        first = scheduler.submit(
            messages, max_new_tokens=2, temperature=0.0, tenant="tenant-a"
        )
        first_result = first.future.result(timeout=30)

        other_tenant = scheduler.submit(
            messages, max_new_tokens=2, temperature=0.0, tenant="tenant-b"
        )
        other_result = other_tenant.future.result(timeout=30)
        assert other_result.prefix_cached_tokens == 0

        same_tenant = scheduler.submit(
            messages, max_new_tokens=2, temperature=0.0, tenant="tenant-a"
        )
        same_result = same_tenant.future.result(timeout=30)
        assert same_result.prefix_cached_tokens == 8
        assert same_result.text == first_result.text
        assert scheduler.manager.prefix_cache.stats()["hits"] >= 1
    finally:
        scheduler.stop()


def test_chunked_prefill_interleaves_with_decode() -> None:
    runtime = _Runtime()
    scheduler = _make_scheduler(
        runtime,
        max_num_seqs=2,
        blocks=32,
        chunk=2,
        prefix_blocks=0,
    )
    try:
        long_request = scheduler.submit(
            [{"role": "user", "content": "20"}],
            max_new_tokens=4,
            temperature=0.0,
        )
        short_request = scheduler.submit(
            [{"role": "user", "content": "3"}],
            max_new_tokens=4,
            temperature=0.0,
        )
        long_result = long_request.future.result(timeout=60)
        short_result = short_request.future.result(timeout=60)

        assert long_result.completion_tokens == 4
        assert short_result.completion_tokens == 4
        # 20 prompt tokens / 2 + 3 prompt tokens / 2 = 12 chunked steps.
        assert scheduler.metrics.get("prefill_steps_total") == 12
        assert scheduler.metrics.get("prefill_decode_overlap_steps") > 0
        assert short_result.ttft_seconds < long_result.ttft_seconds
        assert scheduler.status()["prefill_chunk_tokens"] == 2
    finally:
        scheduler.stop()


def test_decode_oom_is_isolated_to_one_request() -> None:
    runtime = _Runtime()
    scheduler = _make_scheduler(
        runtime,
        max_num_seqs=2,
        blocks=4,
        chunk=2,
        prefix_blocks=0,
    )
    try:
        small = scheduler.submit(
            [{"role": "user", "content": "3"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        too_large = scheduler.submit(
            [{"role": "user", "content": "12"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        small_result = small.future.result(timeout=30)
        assert small_result.completion_tokens == 2
        with pytest.raises(CacheOutOfMemoryError):
            too_large.future.result(timeout=30)
        assert scheduler.metrics.get("kv_cache_oom_total") == 1
        assert scheduler.metrics.get("request_rejected_total") >= 1
        assert scheduler.status()["active"] == 0
    finally:
        scheduler.stop()


def test_stop_sequence_and_completion_prompt() -> None:
    runtime = _Runtime()
    runtime._sample = lambda *args, **kwargs: 5  # type: ignore[method-assign]
    scheduler = _make_scheduler(runtime, max_num_seqs=1)
    try:
        request = scheduler.submit_completion(
            "3",
            max_new_tokens=8,
            temperature=0.0,
            stop="5 5",
        )
        result = request.future.result(timeout=30)
        assert result.finish_reason == "stop"
        assert result.stop_sequence == "5 5"
        # Two sampled tokens form the stop string; the textual result must
        # exclude the stop sequence itself.
        assert result.completion_tokens == 2
        assert result.text == ""
    finally:
        scheduler.stop()


def test_prefill_oom_isolated_to_request() -> None:
    runtime = _Runtime()
    original_forward = runtime.model.forward_logits
    calls = {"count": 0}

    def flaky_forward(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return original_forward(*args, **kwargs)

    runtime.model.forward_logits = flaky_forward  # type: ignore[method-assign]
    scheduler = _make_scheduler(
        runtime,
        max_num_seqs=2,
        blocks=24,
        chunk=2,
        prefix_blocks=0,
    )
    try:
        failing = scheduler.submit(
            [{"role": "user", "content": "8"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        healthy = scheduler.submit(
            [{"role": "user", "content": "3"}],
            max_new_tokens=2,
            temperature=0.0,
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            failing.future.result(timeout=30)
        assert healthy.future.result(timeout=30).completion_tokens == 2
        assert scheduler.metrics.get("runtime_oom_total") == 1
        assert scheduler.metrics.get("requests_failed_total") == 1
        assert scheduler.status()["active"] == 0
    finally:
        scheduler.stop()


def test_scheduler_with_int8_kv_cache() -> None:
    runtime = _Runtime()
    scheduler = _make_scheduler(
        runtime,
        max_num_seqs=1,
        blocks=16,
        chunk=2,
        prefix_blocks=0,
        kv_cache_dtype="int8",
    )
    try:
        request = scheduler.submit(
            [{"role": "user", "content": "9"}],
            max_new_tokens=3,
            temperature=0.0,
        )
        result = request.future.result(timeout=30)
        assert result.completion_tokens == 3
        assert scheduler.status()["kv_cache_dtype"] == "int8"
        assert scheduler.manager.key_pool.dtype == torch.int8
    finally:
        scheduler.stop()
