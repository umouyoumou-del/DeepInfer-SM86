"""NCCL transport for expert-sharded activation exchange.

The coordinator keeps attention, shared experts and the LM head.  A second rank
owns the remote routed-expert subset and executes the same grouped gate/up/down
kernels on its own GPU.  Unlike the first dual-GPU prototype, the remote rank
does not load the transformer model or duplicate non-routed weights.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist

from deepinfer.moe.deepseek_v2_runtime import GGUFExpertStore
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.grouped_moe import (
    fused_down_gemm_grouped,
    fused_gate_up_gemm_grouped,
)

_HEADER_BYTES = 1024
_DTYPE_NAMES = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}
_NAME_DTYPES = {value: key for key, value in _DTYPE_NAMES.items()}


def _pack_header(payload: dict[str, Any]) -> torch.Tensor:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(encoded) > _HEADER_BYTES:
        raise ValueError(f"expert transport header too large: {len(encoded)} bytes")
    encoded += b" " * (_HEADER_BYTES - len(encoded))
    return torch.tensor(list(encoded), dtype=torch.uint8)


def _unpack_header(tensor: torch.Tensor) -> dict[str, Any]:
    data = bytes(int(value) for value in tensor.cpu().tolist())
    return json.loads(data.rstrip(b" \x00").decode("utf-8"))


@dataclass(slots=True)
class PendingRemoteChunk:
    output: torch.Tensor
    header: torch.Tensor
    tokens: torch.Tensor
    works: tuple[Any, ...]
    started: float
    done: threading.Event | None = None
    transport_error: BaseException | None = None

    def wait(self) -> torch.Tensor:
        if self.done is not None:
            self.done.wait()
            if self.transport_error is not None:
                raise self.transport_error
            return self.output
        for work in self.works:
            work.wait()
        return self.output


class HeartbeatWriter:
    """Write a monotonic-ish heartbeat file for a remote worker process."""

    def __init__(self, path: str | Path, *, interval: float = 0.5) -> None:
        self.path = Path(path)
        self.interval = max(0.1, float(interval))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="ecs-heartbeat-writer", daemon=True
        )

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        self._thread.start()

    def _write(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(str(time.time()), encoding="ascii")
        os.replace(temporary, self.path)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._write()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class HeartbeatMonitor:
    """Read a remote worker heartbeat without sharing GPU state."""

    def __init__(self, path: str | Path, *, timeout: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout = max(0.5, float(timeout))

    def age(self) -> float | None:
        try:
            return max(0.0, time.time() - float(
                self.path.read_text(encoding="ascii").strip()
            ))
        except Exception:
            return None

    def healthy(self) -> bool:
        age = self.age()
        return age is not None and age <= self.timeout

    def status(self) -> dict[str, float | bool | None]:
        age = self.age()
        return {
            "path": str(self.path),
            "age_seconds": age,
            "healthy": age is not None and age <= self.timeout,
            "timeout_seconds": self.timeout,
        }


class _TransportRequest:
    """One queued remote expert chunk waiting for the I/O thread."""

    __slots__ = ("pending", "layer_idx", "expert_ids", "counts")

    def __init__(
        self,
        pending: PendingRemoteChunk,
        layer_idx: int,
        expert_ids: Sequence[int],
        counts: Sequence[int],
    ) -> None:
        self.pending = pending
        self.layer_idx = int(layer_idx)
        self.expert_ids = [int(value) for value in expert_ids]
        self.counts = [int(value) for value in counts]


class NCCLExpertClient:
    """Coordinator-side request/reply client with a single I/O thread.

    The wire protocol is strictly FIFO and has no request id: the worker
    answers chunks in the order they arrive.  Anything that issues sends from
    more than one thread therefore corrupts the stream -- with a prefill thread
    and a decode thread submitting at the same time, the NCCL watchdog aborts
    the process.

    Submissions are now queued and drained by one dedicated I/O thread, so the
    protocol still sees a single ordered producer while callers may submit from
    any thread and wait independently on their own event.
    """

    def __init__(
        self,
        *,
        dst_rank: int = 1,
        device: int = 0,
        heartbeat_path: str | Path | None = None,
        queue_depth: int = 64,
    ) -> None:
        self.dst_rank = int(dst_rank)
        self.device = int(device)
        self.heartbeat = (
            HeartbeatMonitor(heartbeat_path)
            if heartbeat_path is not None
            else None
        )
        self.requests = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.latency_seconds = 0.0
        self._active_sends = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue[_TransportRequest | None] = queue.Queue(
            maxsize=max(1, int(queue_depth))
        )
        self._closed = False
        self._error: BaseException | None = None
        self._max_in_flight = 0
        self._thread = threading.Thread(
            target=self._io_loop, name="nccl-expert-io", daemon=True
        )
        self._thread.start()

    @property
    def in_flight(self) -> int:
        return int(self._active_sends)

    def submit_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: torch.Tensor,
        counts: Sequence[int],
    ) -> PendingRemoteChunk:
        """Queue a chunk and return a handle; never touches the wire itself."""

        tokens = tokens.contiguous()
        if not tokens.is_cuda:
            raise ValueError("NCCL expert transport requires CUDA tokens")
        dtype_name = _DTYPE_NAMES.get(tokens.dtype)
        if dtype_name is None:
            raise TypeError(f"unsupported expert activation dtype {tokens.dtype}")
        shape = tuple(int(value) for value in tokens.shape)
        header = _pack_header(
            {
                "stop": False,
                "layer": int(layer_idx),
                "experts": [int(value) for value in expert_ids],
                "counts": [int(value) for value in counts],
                "shape": list(shape),
                "dtype": dtype_name,
            }
        ).to(tokens.device)
        output = torch.empty(shape, dtype=tokens.dtype, device=tokens.device)
        pending = PendingRemoteChunk(
            output=output,
            header=header,
            tokens=tokens,
            works=(),
            started=time.perf_counter(),
            done=threading.Event(),
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("expert transport is closed")
            self._active_sends += 1
            self._max_in_flight = max(self._max_in_flight, self._active_sends)
        self._queue.put(
            _TransportRequest(pending, layer_idx, expert_ids, counts)
        )
        return pending

    def _io_loop(self) -> None:
        """Drain submissions in FIFO order on one thread."""

        while True:
            item = self._queue.get()
            if item is None:
                return
            pending = item.pending
            try:
                header_work = dist.isend(pending.header, dst=self.dst_rank)
                token_work = dist.isend(pending.tokens, dst=self.dst_rank)
                recv_work = dist.irecv(pending.output, src=self.dst_rank)
                header_work.wait()
                token_work.wait()
                recv_work.wait()
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                pending.transport_error = exc
                with self._lock:
                    self._active_sends = max(0, self._active_sends - 1)
                pending.done.set()
                continue
            pending.done.set()

    def wait_chunk(self, pending: PendingRemoteChunk) -> torch.Tensor:
        output = pending.wait()
        with self._lock:
            self.latency_seconds += time.perf_counter() - pending.started
            self.requests += 1
            self.bytes_sent += int(
                pending.tokens.numel() * pending.tokens.element_size()
            )
            self.bytes_received += int(output.numel() * output.element_size())
            self._active_sends = max(0, self._active_sends - 1)
        return output

    def compute_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: torch.Tensor,
        counts: Sequence[int],
    ) -> torch.Tensor:
        return self.wait_chunk(
            self.submit_chunk(layer_idx, expert_ids, tokens, counts)
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put(None, timeout=30.0)
        except queue.Full:
            pass
        self._thread.join(timeout=30.0)
        header = _pack_header({"stop": True}).to(f"cuda:{self.device}")
        dist.send(header, dst=self.dst_rank)

    def metrics(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "latency_seconds": self.latency_seconds,
            "active_sends": self._active_sends,
            "max_in_flight": self._max_in_flight,
            "queued": self._queue.qsize(),
            "transport_error": str(self._error) if self._error is not None else None,
            "heartbeat": (
                self.heartbeat.status() if self.heartbeat is not None else None
            ),
        }


class RemoteExpertWorker:
    """Second-rank routed-expert executor without a transformer model."""

    def __init__(
        self,
        gguf_path: str | Path,
        *,
        layer_indices: Sequence[int],
        device: int = 1,
        expert_cache_mb: int = 0,
        gpu_expert_cache_mb: int = 0,
        max_gpu_spill_mb: int = 512,
        parallel_read_workers: int = 8,
    ) -> None:
        torch.cuda.set_device(int(device))
        try:
            import cupy

            cupy.cuda.Device(int(device)).use()
        except Exception:
            pass
        self.device = torch.device(f"cuda:{int(device)}")
        self.index = GGUFIndex(Path(gguf_path))
        self.store = GGUFExpertStore(
            self.index,
            layer_indices=tuple(int(layer) for layer in layer_indices),
            device=self.device,
            dtype=torch.float16,
            max_gpu_spill_mb=int(max_gpu_spill_mb),
            parallel_read_workers=int(parallel_read_workers),
            expert_cache_mb=int(expert_cache_mb),
            gpu_expert_cache_mb=int(gpu_expert_cache_mb),
        )
        self.requests = 0
        self.bytes_received = 0
        self.bytes_sent = 0

    def compute_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: torch.Tensor,
        counts: Sequence[int],
    ) -> torch.Tensor:
        tokens = tokens.to(self.device)
        n_experts = len(expert_ids)
        max_count = int(tokens.shape[1])
        count_tensor = torch.tensor(
            [int(value) for value in counts],
            dtype=torch.int32,
            device=self.device,
        )
        batch = self.store.load_packed_experts(
            layer_idx,
            expert_ids,
            device=self.device,
        )
        gate_spec, up_spec, down_spec = batch.specs
        batch.wait_for_gate_up()
        gate_activation = fused_gate_up_gemm_grouped(
            tokens,
            batch.packed[0],
            batch.packed[1],
            count_tensor,
            n_experts=n_experts,
            max_count=max_count,
            out_features=gate_spec.out_features,
            k_size=gate_spec.in_features,
            gate_qtype=gate_spec.quant_type,
            up_qtype=up_spec.quant_type,
            gate_row_bytes=gate_spec.row_bytes,
            up_row_bytes=up_spec.row_bytes,
        )
        batch.wait_for_down()
        return fused_down_gemm_grouped(
            gate_activation,
            batch.packed[2],
            count_tensor,
            n_experts=n_experts,
            max_count=max_count,
            out_features=down_spec.out_features,
            k_size=down_spec.in_features,
            down_qtype=down_spec.quant_type,
            down_row_bytes=down_spec.row_bytes,
        )

    def serve_forever(
        self,
        *,
        src_rank: int = 0,
        heartbeat_path: str | Path | None = None,
    ) -> None:
        writer = (
            HeartbeatWriter(heartbeat_path)
            if heartbeat_path is not None
            else None
        )
        if writer is not None:
            writer.start()
        try:
            self._serve_loop(src_rank=src_rank)
        finally:
            if writer is not None:
                writer.stop()

    def _serve_loop(self, *, src_rank: int = 0) -> None:
        device = self.device
        header = torch.empty(_HEADER_BYTES, dtype=torch.uint8, device=device)
        while True:
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print("[nccl-worker] wait header", flush=True)
            dist.recv(header, src=src_rank)
            meta = _unpack_header(header)
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print(
                    f"[nccl-worker] header stop={meta.get('stop')} "
                    f"layer={meta.get('layer')} experts={len(meta.get('experts', []))}",
                    flush=True,
                )
            if meta.get("stop"):
                break
            dtype = _NAME_DTYPES[str(meta["dtype"])]
            shape = tuple(int(value) for value in meta["shape"])
            tokens = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(tokens, src=src_rank)
            self.bytes_received += int(tokens.numel() * tokens.element_size())
            output = self.compute_chunk(
                int(meta["layer"]),
                [int(value) for value in meta["experts"]],
                tokens,
                [int(value) for value in meta["counts"]],
            )
            if os.environ.get("DEEPINFER_NCCL_DEBUG") == "1":
                print(
                    f"[nccl-worker] send layer={meta['layer']} shape={tuple(output.shape)}",
                    flush=True,
                )
            dist.send(output.contiguous(), dst=src_rank)
            self.bytes_sent += int(output.numel() * output.element_size())
            self.requests += 1
            fault_after = int(
                os.environ.get("DEEPINFER_REMOTE_FAULT_AFTER", "0")
            )
            if fault_after > 0 and self.requests >= fault_after:
                os._exit(42)

    def close(self) -> None:
        self.store.close()

    def metrics(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "bytes_received": self.bytes_received,
            "bytes_sent": self.bytes_sent,
        }


__all__ = [
    "HeartbeatMonitor",
    "HeartbeatWriter",
    "NCCLExpertClient",
    "RemoteExpertWorker",
]
