"""deepinfer-server: a page and an API over a context that grows to millions of tokens.

Feed text in (paste, files, a demo document), ask at any time - also while it is still being
read - and watch the answer stream, with where in the context it was found.  Served over one
:class:`~deepinfer.moe.longctx_service.LongContextChat`:

    GET  /                      the page
    GET  /status                context length, queue, read rate, sources, last answer
    POST /ingest?name=N         body: UTF-8 text, queued for encoding (the page sends files in pieces)
    POST /ask                   {"question": ..., "deep": true, "instruction": optional} -> text/event-stream
    POST /demo                  {"tokens": 3145728} -> the long-context suite's haystack and its 39 needles
    POST /reset                 back to an empty context

Listening on a loopback address needs no key; on any other address every request needs
``?key=`` (``--api-key``, or a random one printed at start).
"""

from __future__ import annotations

import argparse
import ipaddress
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

from deepinfer.moe.longctx_service import LongContextChat, Source, build_session

PAGE = Path(__file__).with_name("static") / "longctx_chat.html"
SERVICE: LongContextChat | None = None
KEY = ""
DEMO: dict = {"questions": [], "instructions": {}, "variants": {}}
MAX_BODY = 64 << 20
# answer checks (deepinfer.moe.answer_check), on: over 315 answer contexts (scripts/answer_bench.py)
# they turned 12 wrong answers right and 2 right ones wrong; --no-check-copies / --no-self-check
CHECK_COPIES = True
SELF_CHECK = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # one line per request is enough
        if not self.path.startswith("/status"):
            sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {urlparse(self.path).path}\n")

    # ------------------------------------------------------------ helpers

    def _query(self) -> dict[str, str]:
        return {name: values[0] for name, values in parse_qs(urlparse(self.path).query).items()}

    def _authorized(self) -> bool:
        if not KEY or secrets.compare_digest(self._query().get("key", ""), KEY):
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
                # the demo's questions are answered with the instructions they were evaluated with
                instruction = request.get("instruction") or DEMO["instructions"].get(question)
                demo = question in DEMO["instructions"]
                self._stream(SERVICE.ask(
                    question, deep=bool(request.get("deep")), instruction=instruction,
                    variants=DEMO["variants"].get(question, ()), rounds=8 if demo else None,
                ))
            elif path == "/demo":
                request = json.loads(body or b"{}")
                self._json(HTTPStatus.OK, load_demo(int(request.get("tokens") or 3145728)))
            elif path == "/reset":
                SERVICE.reset()
                DEMO["questions"], DEMO["instructions"], DEMO["variants"] = [], {}, {}
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
    """Queue the suite's document (stdlib haystack, 39 needles), as long as the room allows, and its questions."""

    from deepinfer.longctx_suite import build_document, build_suite, header_ids

    header = header_ids(SERVICE.tok)
    tokens = min(int(tokens), SERVICE.room() + len(header))
    if tokens < 65536:
        raise ValueError(f"the demo needs room for 65,536 tokens; {max(0, SERVICE.room()):,} left")
    needles, questions = build_suite(11)
    document = build_document(SERVICE.tok, tokens, needles, os.path.dirname(os.__file__))
    source = SERVICE.add_ids(
        document[len(header):], f"演示：Python 标准库草堆 + 39 根针（{len(document):,} token）", kind="demo"
    )
    DEMO["questions"] = [q["question"] for q in questions]
    DEMO["instructions"] = {q["question"]: q["instruction"] for q in questions}
    DEMO["variants"] = {q["question"]: q.get("variants", []) for q in questions if q.get("variants")}
    return {"name": source.name, "tokens": source.tokens, "questions": DEMO["questions"]}


def add_arguments(parser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default %(default)s)")
    parser.add_argument("--port", type=int, default=8080, help="port (default %(default)s)")
    parser.add_argument("--api-key", "--key", dest="api_key", default="",
                        help="access key; default: none on a loopback address, else a random one (printed)")
    parser.add_argument("-c", "--ctx-size", "--capacity", dest="ctx_size", type=int, default=0,
                        help="context size in tokens; 0 = as much as fits in VRAM. More spills to pinned host "
                             "memory (17.3 KiB a token): 3407872 holds the 3M demo with 41 GiB of RAM")
    parser.add_argument("--encode-window", type=int, default=0,
                        help="tokens each token sees while text is read; 0 = the position chunk (3/4 of trained)")
    parser.add_argument("--chunk", type=int, default=16384, help="tokens encoded between two questions")
    parser.add_argument("--check-copies", action=argparse.BooleanOptionalAction, default=CHECK_COPIES,
                        help="ask again when an answer copies a line of the text wrong")
    parser.add_argument("--self-check", action=argparse.BooleanOptionalAction, default=SELF_CHECK,
                        help="a second turn checks the final answer against its own steps")
    parser.add_argument("--load-cache", default="", help="start from a cache saved by needle_complex --save-cache")
    parser.add_argument("--hierarchical", action="store_true",
                        help="use the persistent FTS5 index + on-demand KV path instead of a full KV cache")
    parser.add_argument("--hierarchical-text", default="", help="text file backing the hierarchical index")
    parser.add_argument("--hierarchical-index", default="", help="SQLite FTS5 index for hierarchical mode")
    parser.add_argument("--hierarchical-top-k", type=int, default=24, help="retrieved chunks per question")


def _loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def run(args) -> None:
    global SERVICE, KEY

    import torch
    from transformers import AutoTokenizer

    from deepinfer.cli.models import resolve
    from deepinfer.longctx_suite import header_ids
    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
    from deepinfer.moe.latent_mla import LatentKVCache

    model = resolve(args.model, args.hf, args.meta, args.meta_hf)
    print(f"loading {model.gguf.name} (config and tokenizer from {model.meta})", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model.meta), trust_remote_code=True)
    rt = DeepSeekV2LiteGGUFRuntime(model.gguf, meta_dir=model.meta, device=args.device, verbose=False)
    if args.hierarchical:
        from deepinfer.moe.hierarchical_service import HierarchicalChat

        SERVICE = HierarchicalChat(
            rt, tok, text_path=args.hierarchical_text or None, index_path=args.hierarchical_index or None,
            capacity=args.ctx_size or 10_000_000, top_k=args.hierarchical_top_k,
        )
        KEY = args.api_key or ("" if _loopback(args.host) else secrets.token_urlsafe(12))
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        status = SERVICE.status()
        print(f"ready: hierarchical index {status['index_chunks']:,} chunks / {status['context']:,} tokens "
              f"({status['index_bytes'] / 1e6:.1f} MB)", flush=True)
        print(f"open http://{shown}:{args.port}/" + (f"?key={KEY}" if KEY else ""), flush=True)
        server.serve_forever()
        return
    window = args.encode_window or model.trained * 3 // 4
    per_token = LatentKVCache.bytes_per_token(len(rt.model.model.layers), "fp8")
    fits = int(max(0, torch.cuda.mem_get_info()[0] - 4 * 2**30) // per_token)
    capacity = args.ctx_size or max(65536, fits)
    header = header_ids(tok)
    if args.load_cache:
        session, _saved = DeepSeekLongContext.load(
            rt, args.load_cache, capacity=capacity, gpu_tokens=min(fits, capacity),
            prefill_chunk=args.chunk, encode_window=window,
        )
    else:
        session = build_session(rt, capacity=capacity, trained=model.trained, encode_window=window, chunk=args.chunk)
    SERVICE = LongContextChat(rt, tok, session, header_ids=header, chunk=args.chunk, near_window=window,
                              check_copies=args.check_copies, self_check=args.self_check)
    if args.load_cache:
        loaded = Source(name=f"已存盘的缓存（{session.length:,} token）", tokens=session.length - len(header),
                        kind="demo", done=session.length - len(header))
        loaded.add(len(header), session.length - len(header))
        SERVICE.sources.append(loaded)
    KEY = args.api_key or ("" if _loopback(args.host) else secrets.token_urlsafe(12))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"ready: context up to {session.cache.capacity:,} tokens, {session.cache.gpu_tokens:,} of them in VRAM",
          flush=True)
    print(f"open http://{shown}:{args.port}/" + (f"?key={KEY}" if KEY else ""), flush=True)
    server.serve_forever()
