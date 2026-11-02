"""deepinfer-server's HTTP layer, over a stand-in service."""

from __future__ import annotations

import importlib.util
import json
import queue
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from deepinfer.cli import main as cli
from deepinfer.cli import serve

ROOT = Path(__file__).resolve().parents[2]


class _Service:
    session = types.SimpleNamespace(length=7)

    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []
        self.instructions: list[str | None] = []

    def status(self) -> dict:
        return {"context": 7, "capacity": 100, "queued": 0, "sources": []}

    def add_text(self, text: str, name: str):
        if not text:
            raise ValueError("nothing to add")
        self.added.append((name, text))
        return types.SimpleNamespace(name=name, tokens=len(text))

    def ask(self, question: str, *, deep: bool = False, instruction: str | None = None, variants=(), rounds=None):
        self.instructions.append(instruction)
        self.rounds = rounds
        events: queue.Queue = queue.Queue()
        events.put({"event": "start", "question": question, "context": 7, "deep": deep})
        events.put({"event": "text", "round": 1, "text": "Answer: 42"})
        events.put({"event": "done", "answer": "Answer: 42", "final": "42"})
        return events

    def reset(self) -> None:
        self.session.length = 2


@pytest.fixture
def server():
    serve.SERVICE = _Service()
    serve.KEY = "secret"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read()


def _post(url: str, body: bytes):
    request = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read()


def test_every_route_needs_the_key(server) -> None:
    for path in ("/", "/status", "/status?key=wrong"):
        with pytest.raises(urllib.error.HTTPError) as error:
            _get(server + path)
        assert error.value.code == 403


def test_page_status_and_ingest(server) -> None:
    status, page = _get(server + "/?key=secret")
    assert status == 200 and "上下文".encode() in page
    status, body = _get(server + "/status?key=secret")
    assert json.loads(body)["context"] == 7
    status, body = _post(server + "/ingest?key=secret&name=%E4%B9%A6", "第一章".encode())
    assert json.loads(body) == {"name": "书", "tokens": 3}
    assert serve.SERVICE.added == [("书", "第一章")]


def test_ask_streams_events_until_done(server) -> None:
    status, body = _post(server + "/ask?key=secret", json.dumps({"question": "q?", "deep": True}).encode())
    assert status == 200
    events = [json.loads(line[6:]) for line in body.decode().split("\n\n") if line.startswith("data: ")]
    assert [event["event"] for event in events] == ["start", "text", "done"]
    assert events[0]["deep"] is True and events[-1]["final"] == "42"


def test_demo_questions_carry_their_evaluation_instruction(server) -> None:
    serve.DEMO["instructions"] = {"Which VAR?": "Simulate the VAR lines."}
    try:
        for question in ("Which VAR?", "Something else?"):
            _post(server + "/ask?key=secret", json.dumps({"question": question}).encode())
        assert serve.SERVICE.rounds is None  # a user's question: the service's default cap
    finally:
        serve.DEMO["instructions"] = {}
    assert serve.SERVICE.instructions == ["Simulate the VAR lines.", None]


def test_an_empty_question_is_a_bad_request(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(server + "/ask?key=secret", json.dumps({"question": " "}).encode())
    assert error.value.code == 400


def test_without_a_key_every_request_is_served(server) -> None:
    serve.KEY = ""
    status, body = _get(server + "/status")
    assert status == 200 and json.loads(body)["context"] == 7


def test_only_loopback_addresses_go_without_a_key() -> None:
    assert serve._loopback("127.0.0.1") and serve._loopback("::1") and serve._loopback("localhost")
    assert not serve._loopback("0.0.0.0") and not serve._loopback("192.168.1.5")


def test_the_earlier_script_keeps_its_defaults_and_takes_overrides() -> None:
    spec = importlib.util.spec_from_file_location("old_chat", ROOT / "apps" / "longctx_chat.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = cli.build_parser().parse_args(old.server_argv(["--models-dir", "/m", "--port", "7000", "--key", "k"]))
    assert args.handler is serve.run
    assert (args.host, args.port, args.api_key, args.ctx_size) == ("0.0.0.0", 7000, "k", 3407872)
    assert Path(args.model) == Path("/m/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf")
    assert Path(args.meta) == Path("/root/autodl-tmp/meta_coder")
