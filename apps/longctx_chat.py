"""Chat with a context that grows to millions of tokens: feed text in, ask, watch the answer stream.

Serves a page and a small JSON / server-sent-events API over one
:class:`~deepinfer.moe.longctx_service.LongContextChat`:

    GET  /?key=K                 the page
    GET  /status?key=K           context length, queue, read rate, sources, last answer
    POST /ingest?key=K&name=N    body: UTF-8 text, queued for encoding (the page sends files in pieces)
    POST /ask?key=K              {"question": ..., "deep": false} -> text/event-stream
    POST /demo?key=K             {"tokens": 3145728} -> the complex suite's haystack and its 39 needles
    POST /reset?key=K            back to an empty context

Every request needs the key printed at start (``--key`` or a random one).  On AutoDL, port 6006
is what the console's custom-service link opens.

    python apps/longctx_chat.py --port 6006 --capacity 3407872
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from deepinfer.moe.longctx_service import LongContextChat, Source, build_session

PAGE = Path(__file__).with_name("longctx_chat.html")
SERVICE: LongContextChat | None = None
KEY = ""
DEMO: dict = {"questions": []}
MAX_BODY = 64 << 20


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # one line per request is enough
        if not self.path.startswith("/status"):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {urlparse(self.path).path}\n")

    # ------------------------------------------------------------ helpers

    def _query(self) -> dict[str, str]:
        return {name: values[0] for name, values in parse_qs(urlparse(self.path).query).items()}

    def _authorized(self) -> bool:
        if secrets.compare_digest(self._query().get("key", ""), KEY):
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong key"})
        return False

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"request body over {MAX_BODY >> 20} MB: send it in pieces")
        return self.rfile.read(length) if length else b""

    # ------------------------------------------------------------- routes

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if not self._authorized():
            return
        if path == "/":
            body = PAGE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/status":
            self._json(HTTPStatus.OK, {**SERVICE.status(), "demo_questions": DEMO["questions"]})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "no such page"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not self._authorized():
            return
        try:
            body = self._body()
            if path == "/ingest":
                name = self._query().get("name") or "文本"
                source = SERVICE.add_text(body.decode("utf-8", errors="replace"), name)
                self._json(HTTPStatus.OK, {"name": source.name, "tokens": source.tokens})
            elif path == "/ask":
                request = json.loads(body or b"{}")
                question = str(request.get("question", "")).strip()
                if not question:
                    raise ValueError("empty question")
                self._stream(SERVICE.ask(question, deep=bool(request.get("deep"))))
            elif path == "/demo":
                request = json.loads(body or b"{}")
                self._json(HTTPStatus.OK, load_demo(int(request.get("tokens") or 3145728)))
            elif path == "/reset":
                SERVICE.reset()
                DEMO["questions"] = []
                self._json(HTTPStatus.OK, {"context": SERVICE.session.length})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "no such endpoint"})
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _stream(self, events: queue.Queue) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        while True:
            try:
                event = events.get(timeout=900)
            except queue.Empty:
                event = {"event": "error", "message": "no answer within 15 minutes"}
            try:
                self.wfile.write(("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # the page went away; the answer still finishes and joins the context
            if event["event"] in ("done", "error"):
                return


def load_demo(tokens: int) -> dict:
    """Queue the complex suite's document (stdlib haystack, 39 needles) and remember its questions."""

    import needle_complex as nc
    from needle_longctx import header_ids

    needles, questions = nc.build_suite(11)
    document = nc.build_document(SERVICE.tok, tokens, needles, os.path.dirname(os.__file__))
    body = document[len(header_ids(SERVICE.tok)):]
    source = SERVICE.add_ids(body, f"演示：Python 标准库草堆 + 39 根针（{len(document):,} token）", kind="demo")
    DEMO["questions"] = [q["question"] for q in questions]
    return {"name": source.name, "tokens": source.tokens, "questions": DEMO["questions"]}


def main() -> None:
    global SERVICE, KEY

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--key", default="", help="access key (default: a random one, printed)")
    parser.add_argument("--capacity", type=int, default=3407872)
    parser.add_argument("--encode-window", type=int, default=98304)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--load-cache", default="", help="start from a cache saved by needle_complex --save-cache")
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    args = parser.parse_args()

    from needle_longctx import MODELS, header_ids
    from transformers import AutoTokenizer

    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    gguf, meta, trained = MODELS["coder"]
    tok = AutoTokenizer.from_pretrained(os.path.join(args.meta_root, meta), trust_remote_code=True)
    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf), meta_dir=os.path.join(args.meta_root, meta), device="cuda:0", verbose=False,
    )
    header = header_ids(tok)
    if args.load_cache:
        import torch

        from deepinfer.moe.latent_mla import LatentKVCache

        per_token = LatentKVCache.bytes_per_token(27, "fp8")
        gpu_tokens = int(max(0, torch.cuda.mem_get_info()[0] - 4 * 2**30) // per_token)
        session, _saved = DeepSeekLongContext.load(
            rt, args.load_cache, capacity=args.capacity, gpu_tokens=min(gpu_tokens, args.capacity),
            prefill_chunk=args.chunk, encode_window=args.encode_window,
        )
    else:
        session = build_session(rt, capacity=args.capacity, trained=trained, encode_window=args.encode_window,
                                chunk=args.chunk)
    SERVICE = LongContextChat(rt, tok, session, header_ids=header, chunk=args.chunk)
    if args.load_cache:
        loaded = Source(name=f"已存盘的缓存（{session.length:,} token）", tokens=session.length - len(header),
                        kind="demo", done=session.length - len(header))
        loaded.add(len(header), session.length - len(header))
        SERVICE.sources.append(loaded)
    KEY = args.key or secrets.token_urlsafe(12)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ready: capacity {args.capacity:,} tokens, {session.cache.gpu_tokens:,} of them in VRAM", flush=True)
    print(f"open http://<host>:{args.port}/?key={KEY}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
