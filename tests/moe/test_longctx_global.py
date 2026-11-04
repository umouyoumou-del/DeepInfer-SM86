"""Full-KV service routes global queries before locate."""

from __future__ import annotations

import queue
import threading

from deepinfer.moe.longctx_service import LongContextChat


class ToyTokenizer:
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class FakeSession:
    def __init__(self, text: str) -> None:
        self.slot_tokens = list(text.encode("utf-8"))
        self.length = len(self.slot_tokens)


def test_full_kv_service_handles_global_count() -> None:
    text = "padding " * 60 + "GLOBAL_MARKER_X one.\nGLOBAL_MARKER_X two.\nGLOBAL_MARKER_X three.\n"
    chat = object.__new__(LongContextChat)
    chat.session = FakeSession(text)
    chat.tok = ToyTokenizer()
    chat.header_ids = []
    chat.lock = threading.Lock()
    chat.turns = 0
    chat.history = []
    chat.last = {}
    chat._global_text_cache = None
    chat._global_text_length = -1
    chat.fact_graph = None
    events = queue.Queue()
    handled = chat._try_global("How many times does GLOBAL_MARKER_X occur in the document?", events)
    assert handled is True
    seen = []
    while True:
        event = events.get(timeout=5)
        seen.append(event)
        if event.get("event") in ("done", "error"):
            break
    done = next(event for event in seen if event.get("event") == "done")
    assert done["final"] == "3"
    assert done["global_kind"] == "count"
