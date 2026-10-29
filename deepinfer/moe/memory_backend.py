"""Transport back-ends for remote expert execution.

ECS originally assumed one topology: a discrete GPU coordinator that must
stage expert activations through pinned host memory and hand them to another
device over PCIe/NCCL.  Section 8.2 of the paper argues that most of that cost
disappears on a unified-memory SoC, where the coordinator and the expert
executor read the same physical memory.

This module makes that assumption explicit and replaceable:

* :class:`ExpertTransport` is the contract the runtime already relies on
  (``submit_chunk`` / ``wait_chunk`` / ``close`` / ``metrics``).
* :class:`StagedTransport` wraps the existing NCCL client, which moves bytes.
* :class:`UnifiedMemoryTransport` performs no transfer at all: the activation
  tensor the caller produced is the tensor the executor consumes.

The point is not speed in this repo (there is no SoC to run on yet); it is that
the scheduler no longer hard-codes the assumption that a copy must happen.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ExpertTransport(Protocol):
    """What the MoE runtime needs from an expert transport."""

    name: str

    def submit_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: Any,
        counts: Sequence[int],
    ) -> Any: ...

    def wait_chunk(self, pending: Any) -> Any: ...

    def close(self) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class LocalPending:
    """A chunk that is already resident; nothing to wait for."""

    tokens: Any
    layer_idx: int = 0

    def wait(self) -> Any:
        return self.tokens


@dataclass(slots=True)
class UnifiedMemoryTransport:
    """Zero-copy executor for shared-memory topologies.

    ``submit_chunk`` hands back the very tensor it was given.  On a real
    unified-memory device the expert kernel would read it in place, so the
    staging buffers, the H2D copy and the double-buffered slots that the PCIe
    path needs are all unnecessary.
    """

    name: str = "unified_memory"
    supports_concurrent_submit: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    chunks: int = 0
    bytes_staged: int = 0
    peak_inflight: int = 0
    _inflight: int = 0
    _closed: bool = False

    def submit_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: Any,
        counts: Sequence[int],
    ) -> LocalPending:
        if self._closed:
            raise RuntimeError("expert transport is closed")
        with self._lock:
            self.chunks += 1
            self._inflight += 1
            self.peak_inflight = max(self.peak_inflight, self._inflight)
        return LocalPending(tokens=tokens, layer_idx=int(layer_idx))

    def wait_chunk(self, pending: LocalPending) -> Any:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
        return pending.wait()

    def compute_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: Any,
        counts: Sequence[int],
    ) -> Any:
        return self.wait_chunk(
            self.submit_chunk(layer_idx, expert_ids, tokens, counts)
        )

    def close(self) -> None:
        self._closed = True

    @property
    def in_flight(self) -> int:
        return self._inflight

    def metrics(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "chunks": self.chunks,
            "bytes_staged": self.bytes_staged,
            "peak_inflight": self.peak_inflight,
            "staging_required": False,
        }


@dataclass(slots=True)
class StagedTransport:
    """Adapter that marks an existing copy-based client as staging-required."""

    client: Any
    name: str = "staged_pcie"

    @property
    def supports_concurrent_submit(self) -> bool:
        return bool(getattr(self.client, "supports_concurrent_submit", False))

    def submit_chunk(self, layer_idx, expert_ids, tokens, counts):
        return self.client.submit_chunk(layer_idx, expert_ids, tokens, counts)

    def wait_chunk(self, pending):
        return self.client.wait_chunk(pending)

    def compute_chunk(self, layer_idx, expert_ids, tokens, counts):
        if hasattr(self.client, "compute_chunk"):
            return self.client.compute_chunk(
                layer_idx, expert_ids, tokens, counts
            )
        return self.wait_chunk(
            self.submit_chunk(layer_idx, expert_ids, tokens, counts)
        )

    def close(self) -> None:
        self.client.close()

    @property
    def in_flight(self) -> int:
        return int(getattr(self.client, "in_flight", 0))

    def metrics(self) -> dict[str, Any]:
        payload = dict(self.client.metrics()) if hasattr(
            self.client, "metrics"
        ) else {}
        payload.setdefault("backend", self.name)
        payload["staging_required"] = True
        return payload


def describe(transport: Any) -> dict[str, Any]:
    """Small helper for the scheduler's status payload."""

    metrics = transport.metrics() if hasattr(transport, "metrics") else {}
    return {
        "backend": getattr(transport, "name", type(transport).__name__),
        "staging_required": bool(metrics.get("staging_required", True)),
        "concurrent_submit": bool(
            getattr(transport, "supports_concurrent_submit", False)
        ),
    }


__all__ = [
    "ExpertTransport",
    "LocalPending",
    "StagedTransport",
    "UnifiedMemoryTransport",
    "describe",
]
