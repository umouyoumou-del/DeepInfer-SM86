"""The transport abstraction must not change results, only cost."""

from __future__ import annotations

import numpy as np
import pytest

from deepinfer.moe.memory_backend import (
    ExpertTransport,
    StagedTransport,
    UnifiedMemoryTransport,
    describe,
)


class _FakeClient:
    """Copy-based client stand-in."""

    supports_concurrent_submit = True

    def __init__(self) -> None:
        self.calls = 0
        self.in_flight = 0
        self.closed = False

    def submit_chunk(self, layer_idx, expert_ids, tokens, counts):
        self.calls += 1
        self.in_flight += 1
        return tokens

    def wait_chunk(self, pending):
        self.in_flight -= 1
        return pending

    def metrics(self):
        return {"chunks": self.calls, "transport_error": None}

    def close(self):
        self.closed = True


def test_unified_transport_satisfies_the_protocol() -> None:
    assert isinstance(UnifiedMemoryTransport(), ExpertTransport)


def test_unified_transport_is_zero_copy() -> None:
    transport = UnifiedMemoryTransport()
    tokens = np.arange(12, dtype=np.float32).reshape(3, 4)
    pending = transport.submit_chunk(2, [1, 5], tokens, [2, 1])
    result = transport.wait_chunk(pending)
    assert result is tokens, "unified memory must not copy activations"
    metrics = transport.metrics()
    assert metrics["bytes_staged"] == 0
    assert metrics["staging_required"] is False
    assert metrics["chunks"] == 1


def test_unified_transport_tracks_inflight_and_close() -> None:
    transport = UnifiedMemoryTransport()
    tokens = np.zeros((1, 4), dtype=np.float32)
    first = transport.submit_chunk(0, [0], tokens, [1])
    second = transport.submit_chunk(1, [1], tokens, [1])
    assert transport.in_flight == 2
    assert transport.metrics()["peak_inflight"] == 2
    transport.wait_chunk(first)
    transport.wait_chunk(second)
    assert transport.in_flight == 0
    transport.close()
    with pytest.raises(RuntimeError):
        transport.submit_chunk(0, [0], tokens, [1])


def test_staged_transport_marks_staging_and_delegates() -> None:
    client = _FakeClient()
    transport = StagedTransport(client=client)
    pending = transport.submit_chunk(0, [3], "tokens", [1])
    assert transport.wait_chunk(pending) == "tokens"
    metrics = transport.metrics()
    assert metrics["staging_required"] is True
    assert metrics["chunks"] == 1
    transport.close()
    assert client.closed


def test_describe_reports_the_topology() -> None:
    assert describe(UnifiedMemoryTransport()) == {
        "backend": "unified_memory",
        "staging_required": False,
        "concurrent_submit": True,
    }
    staged = describe(StagedTransport(client=_FakeClient()))
    assert staged["staging_required"] is True
    assert staged["concurrent_submit"] is True
