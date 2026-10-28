from __future__ import annotations

import pytest

from deepinfer.ecs import (
    ECSController,
    ECSRouter,
    ExpertKey,
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
