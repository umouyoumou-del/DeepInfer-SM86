"""Fan-out client for several remote expert workers.

The single-worker NCCL transport serialises every remote chunk: one worker, one
queue, one internal communication stream.  Replicating the worker is the only
way past that ceiling, and replication needs a dispatcher that

* sends each chunk to the least busy worker,
* keeps the submit/wait contract of :class:`NCCLExpertClient`, and
* aggregates per-worker counters so the imbalance is visible.

Routing is by in-flight count with a round-robin tie-break, which behaves like
queue-depth assignment: a worker that is still busy with a previous chunk stops
receiving new ones until it catches up.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

from deepinfer.moe.nccl_expert_transport import PendingRemoteChunk


class PooledExpertClient:
    """Dispatch remote expert chunks across several transport clients."""

    supports_concurrent_submit = True

    def __init__(
        self,
        clients: Sequence[Any],
        *,
        strategy: str = "least_in_flight",
    ) -> None:
        clients = list(clients)
        if not clients:
            raise ValueError("at least one worker client is required")
        if strategy not in {"least_in_flight", "round_robin"}:
            raise ValueError("strategy must be 'least_in_flight' or 'round_robin'")
        self.clients = clients
        self.strategy = strategy
        self._lock = threading.Lock()
        self._next = 0
        self._assigned = [0] * len(clients)
        self._submitted = 0
        self._closed = False

    # ------------------------------------------------------------- internal

    def _in_flight(self, client: Any) -> int:
        value = getattr(client, "in_flight", None)
        if value is None:
            value = getattr(client, "_active_sends", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _pick(self) -> tuple[int, Any]:
        with self._lock:
            if self.strategy == "round_robin":
                index = self._next % len(self.clients)
                self._next += 1
            else:
                counts = [self._in_flight(client) for client in self.clients]
                lowest = min(counts)
                candidates = [i for i, value in enumerate(counts) if value == lowest]
                index = candidates[self._next % len(candidates)]
                self._next += 1
            self._assigned[index] += 1
            self._submitted += 1
            return index, self.clients[index]

    # ------------------------------------------------------------ public API

    def submit_chunk(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        tokens: Any,
        counts: Sequence[int],
    ) -> PendingRemoteChunk:
        if self._closed:
            raise RuntimeError("expert worker pool is closed")
        _, client = self._pick()
        pending = client.submit_chunk(layer_idx, expert_ids, tokens, counts)
        try:
            pending.owner = client
        except AttributeError:
            pass
        return pending

    def wait_chunk(self, pending: PendingRemoteChunk) -> Any:
        # The pending handle already knows which client produced it, so the
        # reply is waited on where it was submitted.
        owner = getattr(pending, "owner", None)
        if owner is None:
            raise RuntimeError(
                "pending chunk has no owning worker; submit it through the pool"
            )
        return owner.wait_chunk(pending)

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
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for client in self.clients:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    def metrics(self) -> dict[str, Any]:
        workers = []
        for index, client in enumerate(self.clients):
            entry: dict[str, Any] = {
                "index": index,
                "type": type(client).__name__,
                "assigned": self._assigned[index],
                "in_flight": self._in_flight(client),
            }
            if hasattr(client, "metrics"):
                try:
                    entry["metrics"] = client.metrics()
                except Exception as exc:  # noqa: BLE001
                    entry["metrics_error"] = str(exc)
            workers.append(entry)
        totals = {
            key: sum(
                int(entry.get("metrics", {}).get(key, 0) or 0)
                for entry in workers
            )
            for key in ("requests", "bytes_sent", "bytes_received")
        }
        assigned = [entry["assigned"] for entry in workers]
        return {
            "strategy": self.strategy,
            "workers": len(self.clients),
            "submitted": self._submitted,
            "assigned": assigned,
            "imbalance": (
                max(assigned) - min(assigned) if assigned else 0
            ),
            **totals,
            "per_worker": workers,
        }


__all__ = ["PooledExpertClient"]
