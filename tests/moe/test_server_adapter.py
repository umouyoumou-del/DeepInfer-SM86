from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from deepinfer.moe.server_adapter import DeepSeekServerRuntime


class _FakeTokenizer:
    pass


class _FakeNative:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=100001)
        )
        self.tokenizer = _FakeTokenizer()
        self.device = torch.device("cpu")
        self.max_cache = 256
        self.fused_moe_enabled = True
        self.supports_prefill_decode_overlap = True
        self.closed = False
        self.calls: list[dict] = []

    def generate(self, messages=None, *, prompt=None, **kwargs):
        self.calls.append(
            {"messages": messages, "prompt": prompt, **kwargs}
        )
        return SimpleNamespace(
            text="hello world",
            prompt_tokens=4,
            completion_tokens=2,
            prefill_seconds=0.1,
            decode_seconds=0.2,
            decode_tokens_per_second=10.0,
            token_ids=[1, 2],
        )

    def close(self) -> None:
        self.closed = True


def test_deepseek_server_adapter_prompt_stats_and_stream(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "fake-gpu")
    native = _FakeNative()
    adapter = DeepSeekServerRuntime(
        native, model_id="deepseek-test", checkpoint=Path("model.gguf")
    )
    prompt = "The capital of France is"
    assert adapter.generate(prompt=prompt, max_new_tokens=2) == "hello world"
    assert native.calls[-1]["prompt"] == prompt
    assert native.calls[-1]["messages"] is None
    assert adapter.last_stats.prompt_tokens == 4
    assert adapter.last_stats.completion_tokens == 2
    assert adapter.last_generated_ids == [1, 2]
    assert adapter.last_timings["decode_tokens_per_second"] == 10.0

    chunks = list(
        adapter.stream_generate(prompt=prompt, max_new_tokens=2)
    )
    assert "".join(chunks) == "hello world"
    health = adapter.health()
    assert health["model_id"] == "deepseek-test"
    assert health["backend"] == "deepseek"
    assert health["paged_overlap_supported"] is True
    adapter.close()
    assert native.closed


def test_deepseek_server_adapter_chat_messages(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "fake-gpu")
    native = _FakeNative()
    adapter = DeepSeekServerRuntime(native, model_id="deepseek-test")
    messages = [{"role": "user", "content": "hi"}]
    assert adapter.generate(messages, max_new_tokens=2) == "hello world"
    assert native.calls[-1]["messages"] == messages
    assert native.calls[-1]["prompt"] is None
