"""A bounded concurrent request queue for a single NCCL expert-parallel runtime.

The model itself is still executed by one coordinator runtime, but multiple
callers can submit requests concurrently.  This is the service-boundary queue
needed by an HTTP worker; it does not claim batched transformer execution.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class _QueuedRequest:
    future: Future
    messages: list[dict[str, str]]
    kwargs: dict[str, Any]


class NCCLRequestService:
    def __init__(
        self,
        runtime: Any,
        *,
        device: int = 0,
        max_queue_size: int = 16,
    ) -> None:
        self.runtime = runtime
        self.device = int(device)
        self.queue: queue.Queue[_QueuedRequest | None] = queue.Queue(
            maxsize=max(1, int(max_queue_size))
        )
        self._stop = threading.Event()
        self._completed = 0
        self._failed = 0
        self._active = 0
        self._last_latency = 0.0
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="ecs-request-service", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            import torch

            torch.cuda.set_device(self.device)
            try:
                import cupy

                cupy.cuda.Device(self.device).use()
            except Exception:
                pass
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            started = time.perf_counter()
            self._active = 1
            try:
                result = self.runtime.generate(
                    item.messages, **item.kwargs
                )
                item.future.set_result(result)
                self._completed += 1
            except Exception as exc:
                self._failed += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                item.future.set_exception(exc)
            finally:
                self._last_latency = time.perf_counter() - started
                self._active = 0
                self.queue.task_done()

    def submit(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Future:
        if self._stop.is_set():
            raise RuntimeError("NCCL request service is closed")
        future: Future = Future()
        item = _QueuedRequest(future=future, messages=messages, kwargs=kwargs)
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError("NCCL request service queue is full") from exc
        return future

    def status(self) -> dict[str, Any]:
        remote = getattr(self.runtime, "remote_expert_client", None)
        remote_metrics = remote.metrics() if remote is not None else None
        return {
            "queue_depth": int(self.queue.qsize()),
            "active": int(self._active),
            "completed": int(self._completed),
            "failed": int(self._failed),
            "last_latency_seconds": self._last_latency,
            "last_error": self._last_error,
            "remote_transport": remote_metrics,
        }

    def close(self, timeout: float = 30.0) -> None:
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


__all__ = ["NCCLRequestService"]
