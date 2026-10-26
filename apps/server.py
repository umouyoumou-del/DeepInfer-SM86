"""Minimal OpenAI-compatible HTTP server for DeepInfer-SM86."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.runtime import DEFAULT_CHECKPOINT, QwenSm86Runtime

MODEL_ID = "deepinfer-sm86-7b-mixed"
RUNTIME: QwenSm86Runtime | None = None
STARTED = time.time()


def _json_response(
    handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _completion_payload(
    text: str, prompt_tokens: int, completion_tokens: int
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "DeepInferSM86/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        return json.loads(payload.decode("utf-8"))

    def _messages(self, body: dict[str, Any]) -> list[dict[str, str]]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        cleaned = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("each message must be an object")
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
            cleaned.append({"role": role, "content": str(content)})
        return cleaned

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        assert RUNTIME is not None
        if self.path == "/health":
            status = RUNTIME.health()
            status["uptime_seconds"] = round(time.time() - STARTED, 1)
            _json_response(self, HTTPStatus.OK, status)
            return
        if self.path == "/v1/models":
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": MODEL_ID,
                            "object": "model",
                            "created": int(STARTED),
                            "owned_by": "local",
                        }
                    ],
                },
            )
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _sse(self, payload: dict[str, Any]) -> None:
        line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _stream_chat(
        self, body: dict[str, Any], messages: list[dict[str, str]], max_tokens: int
    ) -> None:
        assert RUNTIME is not None
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        stream = RUNTIME.stream_generate(
            messages,
            max_new_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 50)),
        )
        first = True
        try:
            for delta in stream:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": (
                                {"role": "assistant", "content": delta}
                                if first
                                else {"content": delta}
                            ),
                            "finish_reason": None,
                        }
                    ],
                }
                first = False
                self._sse(chunk)
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self._sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        finally:
            stream.close()

    def do_POST(self) -> None:
        assert RUNTIME is not None
        if self.path != "/v1/chat/completions":
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            body = self._read_json()
            messages = self._messages(body)
            max_tokens = int(body.get("max_tokens") or 256)
            stream = bool(body.get("stream", False))
        except Exception as exc:  # noqa: BLE001
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            if stream:
                self._stream_chat(body, messages, max_tokens)
                return
            text = RUNTIME.generate(
                messages,
                max_new_tokens=max_tokens,
                temperature=float(body.get("temperature", 0.0)),
                top_p=float(body.get("top_p", 0.95)),
                top_k=int(body.get("top_k", 50)),
            )
            stats = RUNTIME.last_stats
            _json_response(
                self,
                HTTPStatus.OK,
                _completion_payload(text, stats.prompt_tokens, stats.completion_tokens),
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as response_exc:  # noqa: BLE001
                print(
                    f"[http] failed to send error response: {response_exc}", flush=True
                )


def main() -> int:
    global RUNTIME
    parser = argparse.ArgumentParser(
        description="DeepInfer-SM86 OpenAI-compatible local server"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backend", choices=("mixed", "w4a16"), default="mixed")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument(
        "--no-graph", action="store_true", help="disable CUDA Graph (debug/fallback)"
    )
    args = parser.parse_args()

    RUNTIME = QwenSm86Runtime(
        args.checkpoint,
        max_cache=args.max_cache,
        use_graph=not args.no_graph,
        backend=args.backend,
        model_id=args.model_id,
    )
    MODEL_ID = RUNTIME.model_id
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"DeepInfer-SM86 listening on http://{args.host}:{args.port}", flush=True)
    print(
        f"model={MODEL_ID} checkpoint={args.checkpoint} max_cache={args.max_cache} graph={not args.no_graph}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
