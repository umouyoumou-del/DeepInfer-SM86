from __future__ import annotations

import torch

from apps.paged_scheduler import PagedBatchScheduler
from deepinfer.attrdict import AttrDict
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


def test_paged_scheduler_batches_decode_and_reuses_prefix() -> None:
    runtime = _Runtime()
    scheduler = PagedBatchScheduler(
        runtime,
        max_num_seqs=2,
        block_size=4,
        num_gpu_blocks=24,
        prefix_cache_blocks=8,
        max_queue_size=8,
    )
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
        assert sorted(
            (first_result.prefix_cached_tokens, second_result.prefix_cached_tokens)
        ) == [0, 8]
        # Both sequences consume the same batched decode steps (the first
        # token of each request is sampled during its serial prefill).
        assert scheduler.metrics.get("decode_steps_total") == 3
        assert scheduler.manager.prefix_cache.stats()["hits"] >= 1
        status = scheduler.status()
        assert status["active"] == 0
        assert status["kv_cache"]["allocator"]["used_blocks"] >= 0
        assert "deepinfer_generated_tokens_total 8" in scheduler.metrics.render()
    finally:
        scheduler.stop()
