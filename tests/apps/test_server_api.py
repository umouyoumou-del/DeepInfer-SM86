from __future__ import annotations

import json
import queue
import threading
import types
import urllib.request
from concurrent.futures import Future
from http.server import ThreadingHTTPServer

from apps import server
from apps.paged_scheduler import GenerationResult


class _Request:
    def __init__(self, *, stream: bool) -> None:
        self.finished = False
        self.finish_reason = "stop"
        self.future: Future[GenerationResult] = Future()
        self.stream_queue: queue.Queue[str | None] | None = (
            queue.Queue() if stream else None
        )
        if stream:
            assert self.stream_queue is not None
            self.stream_queue.put("hello ")
            self.stream_queue.put("world")
            self.stream_queue.put(None)
        else:
            self.future.set_result(
                GenerationResult(
                    text="hello world",
                    prompt_tokens=3,
                    completion_tokens=2,
                    finish_reason="stop",
                )
            )


class _Scheduler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def submit(self, messages, **kwargs):
        self.calls.append(("chat", {"messages": messages, **kwargs}))
        return _Request(stream=bool(kwargs.get("stream")))

    def submit_completion(self, prompt, **kwargs):
        self.calls.append(("completion", {"prompt": prompt, **kwargs}))
        return _Request(stream=bool(kwargs.get("stream")))

    def cancel(self, request) -> None:
        request.finished = True

    def status(self):
        return {"active": 0}

    def stop(self) -> None:
        return None


class _Runtime:
    last_stats = types.SimpleNamespace(prompt_tokens=3, completion_tokens=2)

    def health(self):
        return {"status": "ok", "model_id": "fake"}

    def generate(self, messages=None, *, prompt=None, **kwargs):
        return "hello world"

    def stream_generate(self, messages=None, *, prompt=None, **kwargs):
        yield "hel"
        yield "lo "
        yield "world"

    def _sample(self, *args, **kwargs):
        return 0


def _post(base: str, path: str, payload: dict) -> tuple[int, str, str]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read().decode("utf-8")


def test_openai_chat_and_completions_routes(monkeypatch) -> None:
    scheduler = _Scheduler()
    monkeypatch.setattr(server, "RUNTIME", _Runtime())
    monkeypatch.setattr(server, "SCHEDULER", scheduler)
    monkeypatch.setattr(server, "MODEL_ID", "fake-model")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, _, body = _post(
            base,
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "user": "tenant-a",
                "stop": ["world"],
            },
        )
        assert status == 200
        chat = json.loads(body)
        assert chat["object"] == "chat.completion"
        assert chat["choices"][0]["message"]["content"] == "hello world"
        assert scheduler.calls[-1][1]["stop"] == ["world"]
        assert scheduler.calls[-1][1]["tenant"] == "tenant-a"

        status, _, body = _post(
            base,
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "hi",
                "max_tokens": 2,
                "user": "tenant-b",
            },
        )
        assert status == 200
        completion = json.loads(body)
        assert completion["object"] == "text_completion"
        assert completion["choices"][0]["text"] == "hello world"
        assert scheduler.calls[-1][0] == "completion"

        status, _, body = _post(
            base,
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "hi",
                "max_tokens": 2,
                "stream": True,
                "user": "tenant-b",
            },
        )
        assert status == 200
        assert "text/event-stream" in _
        assert "hello " in body
        assert "data: [DONE]" in body
        assert scheduler.calls[-1][0] == "completion"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_server_stop_stream_filter() -> None:
    stream = iter(["hel", "lo ", "ST", "OP", " tail"])
    filtered = server._filter_stop_stream(stream, ["STOP"])
    output: list[str] = []
    stop_sequence = None
    while True:
        try:
            output.append(next(filtered))
        except StopIteration as done:
            stop_sequence = done.value
            break
    assert "".join(output) == "hello "
    assert stop_sequence == "STOP"


def test_completions_without_scheduler_uses_raw_prompt_and_stop(monkeypatch) -> None:
    monkeypatch.setattr(server, "RUNTIME", _Runtime())
    monkeypatch.setattr(server, "SCHEDULER", None)
    monkeypatch.setattr(server, "MODEL_ID", "fake-model")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, _, body = _post(
            base,
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "raw prompt",
                "max_tokens": 16,
                "stop": ["world"],
            },
        )
        assert status == 200
        payload = json.loads(body)
        assert payload["choices"][0]["text"] == "hello "

        status, content_type, body = _post(
            base,
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "raw prompt",
                "max_tokens": 16,
                "stream": True,
                "stop": ["world"],
            },
        )
        assert status == 200
        assert "text/event-stream" in content_type
        deltas = []
        for line in body.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            deltas.append(chunk["choices"][0]["text"])
        assert "".join(deltas) == "hello "
        assert "data: [DONE]" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
