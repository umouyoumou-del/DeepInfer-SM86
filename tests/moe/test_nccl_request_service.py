from __future__ import annotations

import time
from types import SimpleNamespace

from deepinfer.moe.nccl_expert_transport import HeartbeatMonitor, HeartbeatWriter
from deepinfer.moe.nccl_request_service import NCCLRequestService


def test_heartbeat_writer_and_monitor(tmp_path) -> None:
    path = tmp_path / "heartbeat"
    writer = HeartbeatWriter(path, interval=0.1)
    writer.start()
    try:
        time.sleep(0.2)
        monitor = HeartbeatMonitor(path, timeout=2.0)
        assert monitor.healthy()
        assert monitor.status()["healthy"] is True
    finally:
        writer.stop()


def test_request_service_queues_concurrent_submissions() -> None:
    class FakeRuntime:
        def generate(self, messages, **kwargs):
            time.sleep(0.02)
            return SimpleNamespace(
                text=messages[0]["content"],
                token_ids=[int(kwargs.get("index", 0))],
            )

    service = NCCLRequestService(FakeRuntime(), device=0, max_queue_size=4)
    try:
        futures = [
            service.submit([{"role": "user", "content": str(index)}], index=index)
            for index in range(3)
        ]
        results = [future.result(timeout=5) for future in futures]
        assert [result.token_ids[0] for result in results] == [0, 1, 2]
        assert service.status()["completed"] == 3
        assert service.status()["queue_depth"] == 0
    finally:
        service.close()
