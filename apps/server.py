"""Minimal OpenAI-compatible HTTP server for DeepInfer-SM86."""

from __future__ import annotations

import argparse
import json
import queue
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

from apps.paged_scheduler import PagedBatchScheduler
from apps.runtime import DEFAULT_CHECKPOINT, QwenSm86Runtime
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.server_adapter import DeepSeekServerRuntime
from apps.scheduler import DynamicBatchScheduler
from deepinfer.engine.kv_cache import CacheOutOfMemoryError
from deepinfer.engine.metrics import EngineMetrics

MODEL_ID = "deepinfer-sm86-7b-mixed"
DEFAULT_DEEPSEEK_CHECKPOINT = Path(
    r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
    r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
)
DEFAULT_DEEPSEEK_META = Path(r"E:\DeepInfer-MoE\meta")
RUNTIME: QwenSm86Runtime | DeepSeekServerRuntime | None = None
SCHEDULER: DynamicBatchScheduler | PagedBatchScheduler | None = None
METRICS = EngineMetrics()
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


def _text_response(
    handler: BaseHTTPRequestHandler, status: int, body: str, content_type: str
) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(encoded)


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion_payload(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
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
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _text_completion_payload(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def _parse_stop(body: dict[str, Any]) -> str | list[str] | None:
    stop = body.get("stop")
    if stop is None:
        return None
    if isinstance(stop, str):
        return stop
    if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        return stop
    raise ValueError("stop must be a string or an array of strings")


def _stop_values(body: dict[str, Any]) -> list[str]:
    stop = _parse_stop(body)
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _truncate_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    hits = [(text.find(stop), stop) for stop in stops if stop in text]
    if not hits:
        return text, None
    index, matched = min(hits, key=lambda item: item[0])
    return text[:index], matched


def _filter_stop_stream(stream: Any, stops: list[str]) -> Any:
    """Yield stream deltas while withholding a possible stop sequence tail."""
    buffer = ""
    emitted = 0
    holdback = max((len(stop) for stop in stops), default=0)
    holdback = max(0, holdback - 1)
    for delta in stream:
        buffer += delta
        truncated, matched = _truncate_stop(buffer, stops)
        if matched is not None:
            if len(truncated) > emitted:
                yield truncated[emitted:]
            return matched
        visible_end = max(0, len(buffer) - holdback)
        if visible_end > emitted:
            yield buffer[emitted:visible_end]
            emitted = visible_end
    if len(buffer) > emitted:
        yield buffer[emitted:]
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "DeepInferSM86/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {fmt % args}", flush=True)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length)
        return json.loads(payload.decode("utf-8"))

    def _tenant_id(self, body: dict[str, Any]) -> str:
        header = self.headers.get("X-Tenant-ID", "")
        return str(header or body.get("user") or "")

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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Tenant-ID",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        assert RUNTIME is not None
        if self.path == "/health":
            status = RUNTIME.health()
            status["uptime_seconds"] = round(time.time() - STARTED, 1)
            if SCHEDULER is not None:
                status["scheduler"] = SCHEDULER.status()
            status["metrics"] = getattr(SCHEDULER, "metrics", METRICS).snapshot()
            _json_response(self, HTTPStatus.OK, status)
            return
        if self.path == "/metrics":
            metrics = getattr(SCHEDULER, "metrics", METRICS)
            _text_response(
                self,
                HTTPStatus.OK,
                metrics.render(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
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
        filtered = _filter_stop_stream(stream, _stop_values(body))
        stop_sequence = None
        try:
            while True:
                try:
                    delta = next(filtered)
                except StopIteration as done:
                    stop_sequence = done.value
                    break
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
            finish_reason = "stop"
            if (
                stop_sequence is None
                and RUNTIME.last_stats.completion_tokens >= max_tokens
            ):
                finish_reason = "length"
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            self._sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        finally:
            stream.close()

    def _completions_prompt(self, body: dict[str, Any]) -> str:
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if len(prompt) != 1 or not isinstance(prompt[0], str):
                raise ValueError(
                    "only one string prompt is supported per /v1/completions request"
                )
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        return prompt

    def _stream_completion(
        self,
        body: dict[str, Any],
        prompt: str,
        max_tokens: int,
    ) -> None:
        assert RUNTIME is not None
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        stream = RUNTIME.stream_generate(
            None,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 50)),
        )
        filtered = _filter_stop_stream(stream, _stop_values(body))
        stop_sequence = None
        try:
            while True:
                try:
                    delta = next(filtered)
                except StopIteration as done:
                    stop_sequence = done.value
                    break
                self._sse(
                    {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            finish_reason = "stop"
            if (
                stop_sequence is None
                and RUNTIME.last_stats.completion_tokens >= max_tokens
            ):
                finish_reason = "length"
            self._sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        finally:
            stream.close()

    def _scheduled_chat(
        self,
        body: dict[str, Any],
        messages: list[dict[str, str]],
        max_tokens: int,
        stream: bool,
    ) -> None:
        assert SCHEDULER is not None
        request = SCHEDULER.submit(
            messages,
            max_new_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 50)),
            stream=stream,
            stop=_parse_stop(body),
            tenant=self._tenant_id(body),
        )
        if not stream:
            result = request.future.result(timeout=600)
            _json_response(
                self,
                HTTPStatus.OK,
                _completion_payload(
                    result.text,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.finish_reason,
                ),
            )
            return

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        first = True
        assert request.stream_queue is not None
        completed = False
        try:
            while True:
                delta = request.stream_queue.get(timeout=600)
                if delta is None:
                    break
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
            if request.future.done() and not request.future.cancelled():
                error = request.future.exception()
                if error is not None:
                    self._sse_error(request_id, error)
                    completed = True
                    return
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": (
                            getattr(request, "finish_reason", "") or "stop"
                        ),
                    }
                ],
            }
            self._sse(final)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            completed = True
        except (queue.Empty, BrokenPipeError, ConnectionResetError):
            return
        finally:
            if not completed and not request.finished:
                cancel = getattr(SCHEDULER, "cancel", None)
                if cancel is not None:
                    cancel(request)

    def _scheduled_completion(
        self,
        body: dict[str, Any],
        prompt: str,
        max_tokens: int,
        stream: bool,
    ) -> None:
        assert SCHEDULER is not None
        submit_completion = getattr(SCHEDULER, "submit_completion", None)
        if submit_completion is None:
            raise ValueError(
                "/v1/completions requires the paged scheduler; restart with "
                "--scheduler --scheduler-backend paged"
            )
        request = submit_completion(
            prompt,
            max_new_tokens=max_tokens,
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 0.95)),
            top_k=int(body.get("top_k", 50)),
            stream=stream,
            stop=_parse_stop(body),
            tenant=self._tenant_id(body),
        )
        if not stream:
            result = request.future.result(timeout=600)
            _json_response(
                self,
                HTTPStatus.OK,
                _text_completion_payload(
                    result.text,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.finish_reason,
                ),
            )
            return

        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        assert request.stream_queue is not None
        completed = False
        try:
            while True:
                delta = request.stream_queue.get(timeout=600)
                if delta is None:
                    break
                self._sse(
                    {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            if request.future.done() and not request.future.cancelled():
                error = request.future.exception()
                if error is not None:
                    self._sse_error(request_id, error)
                    completed = True
                    return
            self._sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": (
                                getattr(request, "finish_reason", "") or "stop"
                            ),
                        }
                    ],
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")

            self.wfile.flush()
            self.close_connection = True
            completed = True
        except (queue.Empty, BrokenPipeError, ConnectionResetError):
            return
        finally:
            if not completed and not request.finished:
                cancel = getattr(SCHEDULER, "cancel", None)
                if cancel is not None:
                    cancel(request)

    def do_POST(self) -> None:
        assert RUNTIME is not None
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            body = self._read_json()
            max_tokens = int(
                body.get("max_tokens")
                or body.get("max_completion_tokens")
                or 256
            )
            stream = bool(body.get("stream", False))
            is_chat = self.path == "/v1/chat/completions"
            messages = self._messages(body) if is_chat else None
            prompt = None if is_chat else self._completions_prompt(body)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            if is_chat:
                assert messages is not None
                if SCHEDULER is not None:
                    self._scheduled_chat(body, messages, max_tokens, stream)
                    return
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
                text, matched_stop = _truncate_stop(text, _stop_values(body))
                finish_reason = "stop"
                if matched_stop is None and stats.completion_tokens >= max_tokens:
                    finish_reason = "length"
                _json_response(
                    self,
                    HTTPStatus.OK,
                    _completion_payload(
                        text,
                        stats.prompt_tokens,
                        stats.completion_tokens,
                        finish_reason,
                    ),
                )
                return

            assert prompt is not None
            if SCHEDULER is not None:
                self._scheduled_completion(body, prompt, max_tokens, stream)
                return
            if stream:
                self._stream_completion(body, prompt, max_tokens)
                return
            text = RUNTIME.generate(
                None,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=float(body.get("temperature", 0.0)),
                top_p=float(body.get("top_p", 0.95)),
                top_k=int(body.get("top_k", 50)),
            )
            stats = RUNTIME.last_stats
            text, matched_stop = _truncate_stop(text, _stop_values(body))
            finish_reason = "stop"
            if matched_stop is None and stats.completion_tokens >= max_tokens:
                finish_reason = "length"
            _json_response(
                self,
                HTTPStatus.OK,
                _text_completion_payload(
                    text,
                    stats.prompt_tokens,
                    stats.completion_tokens,
                    finish_reason,
                ),
            )
        except (BrokenPipeError, ConnectionResetError):
            return
        except CacheOutOfMemoryError as exc:
            _json_response(
                self,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": str(exc), "type": "insufficient_kv_memory"},
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            status = (
                HTTPStatus.SERVICE_UNAVAILABLE
                if "queue is full" in str(exc)
                else HTTPStatus.BAD_REQUEST
            )
            try:
                _json_response(
                    self,
                    status,
                    {"error": str(exc), "type": type(exc).__name__},
                )
            except Exception as response_exc:  # noqa: BLE001
                print(
                    f"[http] failed to send error response: {response_exc}", flush=True
                )


def main() -> int:
    global MODEL_ID, RUNTIME, SCHEDULER
    parser = argparse.ArgumentParser(
        description="DeepInfer-SM86 OpenAI-compatible local server"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backend",
        choices=("mixed", "w4a16", "deepseek"),
        default="mixed",
        help="Qwen mixed/W4A16 or full DeepSeek-V2-Lite MoE runtime",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=DEFAULT_DEEPSEEK_META,
        help="DeepSeek tokenizer/config directory",
    )
    parser.add_argument(
        "--expert-chunk-size",
        type=int,
        default=8,
        help="DeepSeek grouped-MoE expert chunk size",
    )
    parser.add_argument(
        "--gpu-expert-cache-mb",
        type=int,
        default=0,
        help="DeepSeek GPU packed-expert cache MiB (0 disables)",
    )
    parser.add_argument(
        "--no-fused-moe",
        action="store_true",
        help="debug: disable DeepSeek fused grouped-MoE kernels",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument(
        "--no-graph", action="store_true", help="disable CUDA Graph (debug/fallback)"
    )
    parser.add_argument(
        "--scheduler", action="store_true", help="enable continuous batching"
    )
    parser.add_argument(
        "--scheduler-backend",
        choices=("paged", "static"),
        default="paged",
        help="paged/elastic KV cache or the legacy fixed-slot scheduler",
    )
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=64)
    parser.add_argument(
        "--no-prefill-decode-overlap",
        action="store_true",
        help="disable the dual-CUDA-stream prefill/decode overlap path",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp16", "int8", "fp8", "fp8_e5m2"),
        default="fp16",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="swap cold paged-KV requests to CPU under GPU block pressure",
    )
    parser.add_argument(
        "--evict-policy",
        choices=("offload", "drop"),
        default="offload",
        help="drop frees cold KV instead of staging it to the host",
    )
    parser.add_argument("--max-cpu-offload-mb", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--reserve-memory-mb", type=int, default=512)
    parser.add_argument("--kv-budget-mb", type=int, default=None)
    parser.add_argument(
        "--kv-blocks",
        type=int,
        default=None,
        help="physical KV block count; default sizes the pool for max-num-seqs",
    )
    parser.add_argument(
        "--prefix-cache-blocks",
        type=int,
        default=None,
        help="blocks reserved for shared prompt-prefix KV (default: up to 16)",
    )
    parser.add_argument("--max-queue-size", type=int, default=64)
    args = parser.parse_args()

    scheduler_enabled = bool(args.scheduler)
    if args.backend == "deepseek":
        if args.scheduler and args.scheduler_backend != "paged":
            raise SystemExit("DeepSeek requires --scheduler-backend paged")
        checkpoint = (
            DEFAULT_DEEPSEEK_CHECKPOINT
            if args.checkpoint == DEFAULT_CHECKPOINT
            else args.checkpoint
        )
        native = DeepSeekV2LiteGGUFRuntime(
            checkpoint,
            meta_dir=args.meta_dir,
            max_cache=args.max_cache,
            expert_chunk_size=args.expert_chunk_size,
            gpu_expert_cache_mb=args.gpu_expert_cache_mb,
            fused_moe=not args.no_fused_moe,
            verbose=False,
        )
        RUNTIME = DeepSeekServerRuntime(
            native,
            model_id=args.model_id or "deepseek-v2-lite-q4km",
            checkpoint=checkpoint,
        )
        args.checkpoint = checkpoint
    else:
        RUNTIME = QwenSm86Runtime(
            args.checkpoint,
            max_cache=args.max_cache,
            use_graph=not args.no_graph and not scheduler_enabled,
            single_request_cache=not scheduler_enabled,
            backend=args.backend,
            model_id=args.model_id,
            allow_extended_context=(
                scheduler_enabled and args.scheduler_backend == "paged"
            ),
        )
    MODEL_ID = RUNTIME.model_id
    if scheduler_enabled and args.scheduler_backend == "paged":
        SCHEDULER = PagedBatchScheduler(
            RUNTIME,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            kv_cache_dtype=args.kv_cache_dtype,
            enable_prefill_decode_overlap=not args.no_prefill_decode_overlap,
            enable_cpu_offload=args.cpu_offload,
        evict_policy=args.evict_policy,
            max_cpu_offload_mb=args.max_cpu_offload_mb,
            num_gpu_blocks=args.kv_blocks,
            prefix_cache_blocks=args.prefix_cache_blocks,
            gpu_memory_utilization=args.gpu_memory_utilization,
            reserve_memory_mb=args.reserve_memory_mb,
            kv_budget_mb=args.kv_budget_mb,
            max_queue_size=args.max_queue_size,
            metrics=METRICS,
        )
    elif scheduler_enabled:
        SCHEDULER = DynamicBatchScheduler(RUNTIME, max_batch_size=args.max_batch_size)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"DeepInfer-SM86 listening on http://{args.host}:{args.port}", flush=True)
    if SCHEDULER is None:
        scheduler_summary = "none"
    elif isinstance(SCHEDULER, PagedBatchScheduler):
        scheduler_summary = (
            f"paged max_seqs={args.max_num_seqs} block={args.block_size} "
            f"prefill_chunk={args.prefill_chunk_tokens} "
            f"prefill_decode_overlap={not args.no_prefill_decode_overlap} "
            f"kv_dtype={args.kv_cache_dtype} "
            f"cpu_offload={args.cpu_offload} "
            f"kv_blocks={SCHEDULER.manager.num_blocks} "
            f"kv_bytes={SCHEDULER.memory_budget.kv_budget_bytes}"
        )
    else:
        scheduler_summary = f"static max_batch={args.max_batch_size}"
    print(
        f"model={MODEL_ID} checkpoint={args.checkpoint} max_cache={args.max_cache} "
        f"graph={not args.no_graph and not scheduler_enabled} "
        f"scheduler={scheduler_summary}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if SCHEDULER is not None:
            SCHEDULER.stop()
        close = getattr(RUNTIME, "close", None)
        if close is not None:
            close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
