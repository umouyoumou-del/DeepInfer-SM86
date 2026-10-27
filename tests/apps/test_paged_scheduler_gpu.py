"""CUDA-only checks for real prefill/decode stream overlap."""

from __future__ import annotations

import pytest
import torch

from apps.paged_scheduler import PagedBatchScheduler
from deepinfer.attrdict import AttrDict
from deepinfer.model_executor.models import qwen2 as qwen2_module
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


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


class _CudaRuntime:
    device = torch.device("cuda")
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
        torch.manual_seed(20261106)
        self.model = (
            Qwen2ForCausalLM(cfg)
            .eval()
            .requires_grad_(False)
            .half()
            .cuda()
        )
        self.tokenizer = _Tokenizer()

    def _sample(self, logits, temperature=0.0, top_p=0.95, top_k=50) -> int:
        return int(logits.argmax().item())


def test_prefill_decode_stream_overlap_runs_both_forwards() -> None:
    runtime = _CudaRuntime()
    scheduler = PagedBatchScheduler(
        runtime,
        max_num_seqs=2,
        block_size=4,
        prefill_chunk_tokens=2,
        kv_cache_dtype="fp16",
        num_gpu_blocks=32,
        prefix_cache_blocks=0,
        max_queue_size=8,
        enable_prefill_decode_overlap=True,
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
        assert scheduler.metrics.get("prefill_steps_total") == 12
        assert scheduler.metrics.get("decode_steps_total") > 0
        assert scheduler.metrics.get("prefill_decode_overlap_steps") > 0
        assert scheduler.metrics.get("prefill_decode_stream_overlap_steps") > 0
        status = scheduler.status()
        assert status["prefill_decode_overlap"] is True
        assert status["prefill_decode_stream_overlap_steps"] > 0
        assert long_result.ttft_seconds >= 0.0
        assert short_result.tpot_seconds >= 0.0
    finally:
        scheduler.stop()
