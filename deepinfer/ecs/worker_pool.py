"""ECS-managed multi-device runtime worker pool.

This is a data-parallel ECS worker layer: each GPU owns an independent DeepSeek
runtime and the ECS controller routes requests to the least-queued healthy
worker.  It is the safe stepping stone before expert-sharded all-to-all.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


@dataclass
class ECSWorker:
    device: int
    queue: queue.Queue[Any] = field(default_factory=queue.Queue)
    runtime: Any = None
    future: Future | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    healthy: bool = True
    completed: int = 0
    failed: int = 0
    last_error: str | None = None
    last_latency_seconds: float = 0.0


class ECSRuntimeWorkerPool:
    """Queue-depth scheduling over one runtime per GPU.

    ``runtime_factory(device_index)`` must return an object with
    ``generate(messages, ...)`` and ``close()``.  It is called lazily in each
    worker thread so CUDA contexts/runtime state stay thread-local.
    """

    def __init__(
        self,
        runtime_factory: Callable[[int], Any],
        devices: Sequence[int],
        *,
        max_queue_size: int = 16,
        start: bool = True,
    ) -> None:
        if not devices:
            raise ValueError("at least one ECS device is required")
        self.runtime_factory = runtime_factory
        self.devices = tuple(int(device) for device in devices)
        self.max_queue_size = max(1, int(max_queue_size))
        self.workers: dict[int, ECSWorker] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._factory_lock = threading.Lock()
        for device in self.devices:
            worker = ECSWorker(device=device)
            worker.queue = queue.Queue(maxsize=self.max_queue_size)
            self.workers[device] = worker
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                name=f"ecs-gpu-{device}",
                daemon=True,
            )
            self._threads[device] = thread
            if start:
                thread.start()

    def _worker_loop(self, worker: ECSWorker) -> None:
        try:
            # Transformers' dynamic-module/meta initialization is not safe to
            # run concurrently for two runtimes in the same process.
            with self._factory_lock:
                worker.runtime = self.runtime_factory(worker.device)
        except Exception as exc:
            worker.healthy = False
            worker.last_error = f"{type(exc).__name__}: {exc}"
            worker.ready.set()
            return
        worker.ready.set()
        while not self._stop.is_set():
            try:
                task = worker.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if task is None:
                break
            future, messages, kwargs = task
            started = time.perf_counter()
            try:
                result = worker.runtime.generate(messages, **kwargs)
                future.set_result(result)
                worker.completed += 1
            except Exception as exc:
                worker.failed += 1
                worker.last_error = f"{type(exc).__name__}: {exc}"
                future.set_exception(exc)
            finally:
                worker.last_latency_seconds = time.perf_counter() - started
                worker.queue.task_done()
        if worker.runtime is not None:
            try:
                worker.runtime.close()
            except Exception:
                pass

    def wait_ready(self, timeout: float = 300.0) -> bool:
        deadline = time.perf_counter() + timeout
        for worker in self.workers.values():
            remaining = max(0.0, deadline - time.perf_counter())
            if not worker.ready.wait(remaining):
                return False
        return all(worker.healthy for worker in self.workers.values())

    def _queue_depth(self, worker: ECSWorker) -> int:
        return int(worker.queue.qsize())

    def choose_worker(
        self,
        *,
        queue_depths: dict[int, float] | None = None,
    ) -> ECSWorker:
        with self._lock:
            healthy = [worker for worker in self.workers.values() if worker.healthy]
            if not healthy:
                raise RuntimeError("no healthy ECS GPU workers")
            depths = queue_depths or {}
            return min(
                healthy,
                key=lambda worker: (
                    depths.get(worker.device, self._queue_depth(worker))
                    + (0.25 if worker.future is not None and not worker.future.done() else 0.0),
                    worker.device,
                ),
            )

    def submit(
        self,
        messages: list[dict[str, str]],
        *,
        worker_device: int | None = None,
        **kwargs: Any,
    ) -> tuple[Future, int]:
        with self._lock:
            worker = (
                self.workers[int(worker_device)]
                if worker_device is not None
                else self.choose_worker()
            )
            if not worker.healthy:
                raise RuntimeError(f"ECS worker {worker.device} is unhealthy")
            future: Future = Future()
            worker.future = future
            try:
                worker.queue.put_nowait((future, messages, kwargs))
            except queue.Full as exc:
                raise RuntimeError(
                    f"ECS worker {worker.device} queue is full"
                ) from exc
            return future, worker.device

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "devices": list(self.devices),
                "workers": {
                    str(worker.device): {
                        "healthy": worker.healthy,
                        "queue_depth": self._queue_depth(worker),
                        "completed": worker.completed,
                        "failed": worker.failed,
                        "last_error": worker.last_error,
                        "last_latency_seconds": worker.last_latency_seconds,
                    }
                    for worker in self.workers.values()
                },
            }

    def close(self, timeout: float = 30.0) -> None:
        self._stop.set()
        for worker in self.workers.values():
            try:
                worker.queue.put_nowait(None)
            except queue.Full:
                pass
        for thread in self._threads.values():
            thread.join(timeout=timeout)


__all__ = ["ECSRuntimeWorkerPool", "ECSWorker"]
