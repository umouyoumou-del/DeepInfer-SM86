"""The hierarchical service indexes streamed text without a full KV cache."""

from __future__ import annotations

from deepinfer.moe.hierarchical_service import HierarchicalChat


class ToyTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def test_add_text_status_and_reset(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    source = chat.add_text(
        "The special magic number for onyx badger is: 7646461.\n",
        "sample.txt",
    )
    assert source.tokens == len(b"The special magic number for onyx badger is: 7646461.\n")
    status = chat.status()
    assert status["mode"] == "hierarchical"
    assert status["context"] == source.tokens
    assert status["index_chunks"] >= 1
    chat.reset()
    assert chat.status()["context"] == 0
    assert chat.status()["index_chunks"] == 0


def test_structured_fast_path_event(tmp_path) -> None:
    chat = HierarchicalChat(None, ToyTokenizer(), index_path=tmp_path / "index.sqlite", capacity=1_000_000)
    chat.add_text("The special magic number for onyx badger is: 7646461.\n", "facts.txt")
    events = chat.ask("What is the special magic number for onyx badger?")
    seen = []
    while True:
        event = events.get(timeout=10)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert done["final"] == "7646461"
    assert done["rounds"] == 1
