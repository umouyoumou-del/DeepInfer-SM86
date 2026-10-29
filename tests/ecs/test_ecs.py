from __future__ import annotations

import pytest

from deepinfer.ecs import (
    ECSController,
    ECSRouter,
    ExpertKey,
    ResilientAllToAll,
    SimulatedAllToAll,
    Tier,
    decode_packet,
    encode_packet,
)
from deepinfer.ecs.core import available as cpp_available


@pytest.mark.parametrize("use_cpp", [False, True])
def test_ecs_heat_cold_selection_and_chunks(use_cpp: bool) -> None:
    if use_cpp and not cpp_available():
        pytest.skip("ECS C++ compiler is unavailable")
    controller = ECSController(use_cpp=use_cpp)
    layer = 3
    controller.observe_routes(layer, [1, 2, 3], [10, 1, 5])
    cold = controller.choose_cold_experts(layer, [1, 2, 3], count=1)
    assert cold == {2}
    chunks = controller.plan_chunks(
        [1, 2, 3], [4, 4, 4], max_experts=2, max_tokens=8
    )
    assert chunks == [(0, 2), (2, 1)]
    assert controller.snapshot()["entries"] == 3


@pytest.mark.parametrize("use_cpp", [False, True])
def test_ecs_tier_eviction_uses_heat(use_cpp: bool) -> None:
    if use_cpp and not cpp_available():
        pytest.skip("ECS C++ compiler is unavailable")
    controller = ECSController(use_cpp=use_cpp, l1_capacity_bytes=100)
    hot = ExpertKey(1, "gate", 1)
    cold = ExpertKey(1, "gate", 2)
    assert controller.place(hot, tier=Tier.L1, bytes=60, heat_delta=10)
    assert controller.place(cold, tier=Tier.L1, bytes=60, heat_delta=1) is True
    assert controller.lookup(hot) is None
    assert controller.lookup(cold) is not None
    assert controller.snapshot()["metrics"]["evictions"] == 1


def test_ecs_router_transport_and_packet_roundtrip() -> None:
    if not cpp_available():
        pytest.skip("ECS C++ compiler is unavailable")
    controller = ECSController(devices=(0, 1), use_cpp=True)
    # Pin expert 7 to device 1 via a placement touch.
    controller.touch(
        ExpertKey(2, "route", 7), tier=Tier.L3, device=1, bytes=32
    )
    router = ECSRouter(controller, devices=(0, 1), replicate_hot=False)
    plan = router.plan(
        2,
        {7: [10, 11], 8: [12]},
        expert_bytes=32,
        src_device=0,
    )
    assert plan.assigned_experts[7] == 1
    assert len(plan.packets) == 2
    assert plan.cross_device_packets >= 1
    mesh = SimulatedAllToAll((0, 1))
    outbound = {0: plan.packets, 1: []}
    inbox = mesh.exchange(outbound)
    assert sum(len(packets) for packets in inbox.values()) == 2
    packet = plan.packets[0]
    assert decode_packet(encode_packet(packet)).token_ids == packet.token_ids


def test_ecs_evictor_callback_and_queue_balancing() -> None:
    if not cpp_available():
        pytest.skip("ECS C++ compiler is unavailable")
    controller = ECSController(
        devices=(0, 1), l1_capacity_bytes=100, use_cpp=True
    )
    evicted: list[int] = []
    controller.register_evictor(Tier.L1, evicted.append)
    first = ExpertKey(1, "expert", 1)
    second = ExpertKey(1, "expert", 2)
    assert controller.place(first, tier=Tier.L1, bytes=60, heat_delta=10)
    assert controller.place(second, tier=Tier.L1, bytes=60, heat_delta=1)
    assert evicted == [first.stable_id]
    assert controller.lookup(first) is None

    hot = ExpertKey(2, "route", 7)
    controller.touch(hot, tier=Tier.L3, device=0, heat_delta=20)
    router = ECSRouter(
        controller,
        devices=(0, 1),
        replicate_hot=True,
        hot_threshold=8,
    )
    plan = router.plan(
        2,
        {7: [1, 2, 3, 4], 8: [5]},
        expert_bytes=100,
        queue_depths={0: 10.0, 1: 0.0},
    )
    assert plan.hot_replications >= 1
    assert plan.assigned_experts[7] in (0, 1)
    assert len(plan.packets) >= 3
    mesh = SimulatedAllToAll((0, 1))
    inbox = mesh.exchange({0: plan.packets, 1: []})
    assert sum(len(packets) for packets in inbox.values()) == len(plan.packets)


def test_ecs_profile_roundtrip_and_prometheus(tmp_path) -> None:
    controller = ECSController(use_cpp=cpp_available())
    controller.observe_routes(4, [1, 2], [10, 2])
    profile = tmp_path / "ecs_profile.json"
    controller.save_profile(profile)
    restored = ECSController(use_cpp=cpp_available())
    assert restored.load_profile(profile)
    assert restored.heat(4, 1) == 10.0
    assert restored.heat(4, 2) == 2.0
    text = restored.render_prometheus()
    assert "deepinfer_ecs_entries" in text
    assert 'tier="L3"' in text


def test_resilient_all_to_all_reroutes_failed_worker() -> None:
    if not cpp_available():
        pytest.skip("ECS C++ compiler is unavailable")
    controller = ECSController(devices=(0, 1), use_cpp=True)
    controller.touch(
        ExpertKey(1, "route", 3), tier=Tier.L3, device=1, bytes=16
    )
    router = ECSRouter(controller, devices=(0, 1), replicate_hot=False)
    plan = router.plan(1, {3: [1, 2]}, expert_bytes=16, src_device=0)
    backend = SimulatedAllToAll((0, 1), failed_devices=(1,))
    resilient = ResilientAllToAll(backend, fallback_device=0)
    inbox = resilient.exchange({0: plan.packets, 1: []})
    assert resilient.metrics["failovers"] == 1
    assert resilient.metrics["retries"] >= 1
    assert all(packet.dst_device == 0 for packet in inbox[0])


def test_ecs_worker_pool_queue_scheduling() -> None:
    from types import SimpleNamespace

    from deepinfer.ecs import ECSRuntimeWorkerPool

    class FakeRuntime:
        def __init__(self, device):
            self.device = device
            self.closed = False

        def generate(self, messages, **kwargs):
            return SimpleNamespace(
                text=f"gpu{self.device}:{messages[0]['content']}",
                token_ids=[self.device],
            )

        def close(self):
            self.closed = True

    pool = ECSRuntimeWorkerPool(
        lambda device: FakeRuntime(device), devices=(0, 1), max_queue_size=2
    )
    try:
        assert pool.wait_ready(timeout=5)
        futures = [
            pool.submit([{"role": "user", "content": str(index)}])
            for index in range(2)
        ]
        assert [future.result(timeout=5).token_ids[0] for future, _ in futures] == [0, 1]
        status = pool.status()
        assert status["workers"]["0"]["completed"] == 1
        assert status["workers"]["1"]["completed"] == 1
    finally:
        pool.close()


def test_runtime_expert_owner_uses_ecs_placement() -> None:
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    runtime = object.__new__(DeepSeekV2LiteGGUFRuntime)
    runtime.device = __import__("torch").device("cpu")
    runtime.expert_devices = (0, 1)
    runtime._multi_device_experts = True
    controller = ECSController(
        devices=(0, 1), use_cpp=cpp_available()
    )
    controller.place(
        ExpertKey(4, "route", 7),
        tier=Tier.L3,
        device=1,
        bytes=0,
    )
    runtime.ecs_controller = controller
    assert runtime._expert_device_index(4, 7) == 1
    runtime.ecs_controller = None
    assert runtime._expert_device_index(4, 7) in (0, 1)
