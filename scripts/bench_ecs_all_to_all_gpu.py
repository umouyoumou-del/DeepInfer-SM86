"""Validate ECS packet/All-to-All transport on two physical GPUs.

Run with:
    torchrun --standalone --nproc_per_node=2 scripts/bench_ecs_all_to_all_gpu.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from deepinfer.ecs import (
    ECSController,
    ECSRouter,
    ExpertKey,
    Tier,
    TorchCollectiveAllToAll,
)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "2"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    transport = TorchCollectiveAllToAll()
    controller = ECSController(devices=tuple(range(world)), use_cpp=True)
    # Pin one expert to the remote device and make it hot enough to shard.
    controller.touch(
        ExpertKey(2, "route", 7),
        tier=Tier.L3,
        device=1 if world > 1 else 0,
        bytes=1024,
        heat_delta=20.0,
    )
    controller.touch(
        ExpertKey(2, "route", 8),
        tier=Tier.L3,
        device=0,
        bytes=1024,
        heat_delta=2.0,
    )
    router = ECSRouter(
        controller,
        devices=tuple(range(world)),
        replicate_hot=True,
        hot_threshold=8.0,
    )
    plan = router.plan(
        2,
        {7: [101, 102, 103, 104], 8: [201, 202]},
        expert_bytes=1024,
        src_device=local_rank,
        queue_depths={0: 3.0, 1: 0.0},
    )
    outbound = plan.packets if rank == 0 else []
    started = time.perf_counter()
    received = transport.exchange(outbound)
    latency = time.perf_counter() - started
    dist.barrier()
    payload = {
        "rank": rank,
        "world_size": world,
        "gpu": torch.cuda.get_device_name(local_rank),
        "transport": type(transport).__name__,
        "sent_packets": len(outbound),
        "received_packets": len(received),
        "received": [
            {
                "src": packet.src_device,
                "dst": packet.dst_device,
                "expert": packet.expert_id,
                "tokens": list(packet.token_ids),
            }
            for packet in received
        ],
        "hot_replications": plan.hot_replications,
        "latency_ms": latency * 1000.0,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    output = Path(os.environ.get("ECS_ALL_TO_ALL_OUTPUT", "/tmp/ecs_all_to_all.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    rank_file = output.with_suffix(f".rank{rank}.json")
    rank_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
