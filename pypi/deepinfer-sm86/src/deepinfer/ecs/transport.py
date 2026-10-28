"""Cross-device ECS transport primitives.

The tensor math remains on the GPU.  This module only packs expert/token
routing metadata into device-addressed packets and provides a simulated
all-to-all for multi-GPU tests plus an optional torch.distributed backend.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any, Iterable

from deepinfer.ecs.controller import ECSController, ExpertKey


@dataclass(slots=True, frozen=True)
class ExpertPacket:
    src_device: int
    dst_device: int
    layer: int
    expert_id: int
    token_ids: tuple[int, ...]
    expert_bytes: int = 0
    payload: bytes = b""

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def transfer_bytes(self) -> int:
        return int(self.expert_bytes) + len(self.payload)


@dataclass(slots=True)
class RoutingPlan:
    packets: list[ExpertPacket]
    by_device: dict[int, list[ExpertPacket]] = field(default_factory=dict)
    cross_device_packets: int = 0
    cross_device_bytes: int = 0
    assigned_experts: dict[int, int] = field(default_factory=dict)

    def device_batches(self) -> dict[int, list[ExpertPacket]]:
        if self.by_device:
            return self.by_device
        grouped: dict[int, list[ExpertPacket]] = {}
        for packet in self.packets:
            grouped.setdefault(packet.dst_device, []).append(packet)
        self.by_device = grouped
        return grouped


class ECSRouter:
    """Turn expert→token assignments into device-addressed packets."""

    def __init__(
        self,
        controller: ECSController,
        *,
        devices: Iterable[int] | None = None,
        replicate_hot: bool = True,
        hot_threshold: float = 8.0,
    ) -> None:
        self.controller = controller
        self.devices = tuple(devices or controller.devices)
        self.replicate_hot = bool(replicate_hot)
        self.hot_threshold = float(hot_threshold)

    def plan(
        self,
        layer: int,
        assignments: dict[int, list[int]],
        *,
        expert_bytes: int = 0,
        src_device: int = 0,
    ) -> RoutingPlan:
        loads = {device: 0 for device in self.devices}
        packets: list[ExpertPacket] = []
        assigned: dict[int, int] = {}
        for expert_id in sorted(assignments):
            token_ids = tuple(int(token) for token in assignments[expert_id])
            if not token_ids:
                continue
            key = ExpertKey(layer, "route", int(expert_id))
            dst = self.controller.choose_device(key, bytes=expert_bytes)
            heat = self.controller.heat(layer, expert_id)
            if (
                self.replicate_hot
                and heat >= self.hot_threshold
                and len(self.devices) > 1
            ):
                dst = min(self.devices, key=lambda device: loads[device])
                self.controller.metrics["hot_replications"] += 1.0
            loads[dst] += max(0, expert_bytes)
            assigned[int(expert_id)] = dst
            packets.append(
                ExpertPacket(
                    src_device=int(src_device),
                    dst_device=int(dst),
                    layer=int(layer),
                    expert_id=int(expert_id),
                    token_ids=token_ids,
                    expert_bytes=int(expert_bytes),
                )
            )
        cross = [packet for packet in packets if packet.src_device != packet.dst_device]
        plan = RoutingPlan(
            packets=packets,
            cross_device_packets=len(cross),
            cross_device_bytes=sum(packet.transfer_bytes for packet in cross),
            assigned_experts=assigned,
        )
        plan.by_device = plan.device_batches()
        self.controller.metrics["cross_device_packets"] += float(len(cross))
        self.controller.metrics["cross_device_bytes"] += float(plan.cross_device_bytes)
        return plan


class SimulatedAllToAll:
    """Deterministic in-process all-to-all used by tests and CPU simulation."""

    def __init__(self, devices: Iterable[int]) -> None:
        self.devices = tuple(int(device) for device in devices)
        self.received: dict[int, list[ExpertPacket]] = {
            device: [] for device in self.devices
        }

    def exchange(
        self, outbound: dict[int, list[ExpertPacket]]
    ) -> dict[int, list[ExpertPacket]]:
        self.received = {device: [] for device in self.devices}
        for source in self.devices:
            for packet in outbound.get(source, []):
                if packet.dst_device not in self.received:
                    raise ValueError(f"unknown destination device {packet.dst_device}")
                if packet.dst_device != source:
                    self.received[packet.dst_device].append(packet)
                else:
                    self.received[source].append(packet)
        return self.received


class TorchCollectiveAllToAll:
    """Optional torch.distributed packet exchange.

    ``all_to_all_object`` is available in recent PyTorch builds.  When it is
    absent we fall back to ``all_gather_object`` and filter by destination;
    that path is correct but less bandwidth-efficient.
    """

    def __init__(self, process_group: Any = None) -> None:
        import torch.distributed as dist

        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first")
        self.dist = dist
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

    def exchange(
        self, outbound: list[ExpertPacket] | dict[int, list[ExpertPacket]]
    ) -> list[ExpertPacket]:
        if isinstance(outbound, list):
            by_dst: dict[int, list[ExpertPacket]] = {}
            for packet in outbound:
                by_dst.setdefault(packet.dst_device, []).append(packet)
        else:
            by_dst = outbound
        send = [by_dst.get(rank, []) for rank in range(self.world_size)]
        if hasattr(self.dist, "all_to_all_object"):
            received: list[list[ExpertPacket]] = [None] * self.world_size  # type: ignore[list-item]
            self.dist.all_to_all_object(
                received, send, group=self.process_group
            )
            return [packet for packets in received for packet in packets]
        gathered: list[list[ExpertPacket]] = [None] * self.world_size  # type: ignore[list-item]
        self.dist.all_gather_object(gathered, send, group=self.process_group)
        rank = self.rank
        return [
            packet
            for per_rank in gathered
            for packets in per_rank
            for packet in packets
            if packet.dst_device == rank
        ]


def encode_packet(packet: ExpertPacket) -> bytes:
    header = struct.pack(
        "<iiiqq",
        int(packet.src_device),
        int(packet.dst_device),
        int(packet.layer),
        int(packet.expert_id),
        int(len(packet.token_ids)),
    )
    token_blob = struct.pack(f"<{len(packet.token_ids)}q", *packet.token_ids)
    return header + token_blob + packet.payload


def decode_packet(data: bytes, *, expert_bytes: int = 0) -> ExpertPacket:
    if len(data) < 28:
        raise ValueError("packet is too short")
    src, dst, layer, expert_id, count = struct.unpack_from("<iiiqq", data, 0)
    offset = 28
    token_ids = struct.unpack_from(f"<{count}q", data, offset)
    offset += 8 * count
    return ExpertPacket(
        src_device=int(src),
        dst_device=int(dst),
        layer=int(layer),
        expert_id=int(expert_id),
        token_ids=tuple(int(token) for token in token_ids),
        expert_bytes=int(expert_bytes),
        payload=data[offset:],
    )


__all__ = [
    "ECSAllToAll",
    "ECSRouter",
    "ExpertPacket",
    "RoutingPlan",
    "SimulatedAllToAll",
    "TorchCollectiveAllToAll",
    "decode_packet",
    "encode_packet",
]

# Backwards-friendly alias for callers that prefer the full name.
ECSAllToAll = SimulatedAllToAll
