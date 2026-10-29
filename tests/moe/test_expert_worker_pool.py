from __future__ import annotations

import pytest

from deepinfer.moe.expert_worker_pool import PooledExpertClient


class _Pending:
    owner = None

    def __init__(self, client, layer_idx, tokens) -> None:
        self.owner = client
        self.layer = layer_idx
        self.tokens = tokens


class _FakeClient:
    supports_concurrent_submit = True

    def __init__(self, name: str) -> None:
        self.name = name
        self.in_flight = 0
        self.submitted = 0
        self.requests = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        self.closed = False

    def submit_chunk(self, layer_idx, expert_ids, tokens, counts):
        self.submitted += 1
        self.in_flight += 1
        pending = _Pending(self, layer_idx, tokens)
        return pending

    def wait_chunk(self, pending):
        self.in_flight = max(0, self.in_flight - 1)
        self.requests += 1
        self.bytes_sent += 10
        self.bytes_received += 20
        return pending

    def close(self):
        self.closed = True

    def metrics(self):
        return {
            "requests": self.requests,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


def test_pool_requires_a_worker() -> None:
    with pytest.raises(ValueError):
        PooledExpertClient([])


def test_pool_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        PooledExpertClient([_FakeClient("a")], strategy="random")


def test_least_in_flight_avoids_a_busy_worker() -> None:
    first, second = _FakeClient("a"), _FakeClient("b")
    pool = PooledExpertClient([first, second])

    pending_one = pool.submit_chunk(0, [1], "t", [1])   # both idle -> a
    pending_two = pool.submit_chunk(0, [2], "t", [1])   # a busy -> b
    assert pending_one.owner is first
    assert pending_two.owner is second

    pool.wait_chunk(pending_one)
    third = pool.submit_chunk(0, [3], "t", [1])         # a idle again -> a
    assert third.owner is first
    assert pool.metrics()["assigned"] == [2, 1]


def test_round_robin_alternates() -> None:
    first, second = _FakeClient("a"), _FakeClient("b")
    pool = PooledExpertClient([first, second], strategy="round_robin")
    owners = [
        pool.submit_chunk(0, [index], "t", [1]).owner
        for index in range(4)
    ]
    assert owners == [first, second, first, second]
    assert pool.metrics()["imbalance"] == 0


def test_metrics_aggregate_and_close_propagates() -> None:
    first, second = _FakeClient("a"), _FakeClient("b")
    pool = PooledExpertClient([first, second])
    for index in range(4):
        pool.wait_chunk(pool.submit_chunk(0, [index], "t", [1]))

    metrics = pool.metrics()
    assert metrics["workers"] == 2
    assert metrics["submitted"] == 4
    assert metrics["requests"] == 4
    assert metrics["bytes_sent"] == 40
    assert metrics["imbalance"] <= 1
    assert [worker["assigned"] for worker in metrics["per_worker"]] == [2, 2]

    pool.close()
    assert first.closed and second.closed
