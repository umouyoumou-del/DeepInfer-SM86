"""Paged-KV continuous-batch benchmark for the ECS expert runtime.

Runs the same staggered-arrival workload as
``bench_ecs_continuous_batch.py`` but keeps KV in a paged pool instead of one
dense ``DynamicCache``.  Use ``--single-gpu`` to validate correctness on one
card; the default path launches the two-rank NCCL expert worker.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
from transformers import AutoConfig

from deepinfer.moe.layout import split_layer_indices
from deepinfer.moe.paged_continuous_batching import PagedContinuousBatchEngine


def _moe_layers(meta_dir: Path) -> tuple[int, ...]:
    config = AutoConfig.from_pretrained(
        str(meta_dir), local_files_only=True, trust_remote_code=True
    )
    _, moe_layers = split_layer_indices(config)
    return tuple(int(layer) for layer in moe_layers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF"
            "/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ),
    )
    parser.add_argument("--meta-dir", type=Path, default=Path("/root/autodl-tmp/meta"))
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "只回答数字：5-2=?",
            "只回答数字：2+3=?",
            "只回答数字：9-4=?",
            "只回答数字：1+1=?",
        ],
    )
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--arrival-delay", type=float, default=0.15)
    parser.add_argument("--remote-fraction", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-blocks", type=int, default=None)
    parser.add_argument("--kv-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--prefill-mode",
        choices=("batched", "paged"),
        default="batched",
    )
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=Path("/root/autodl-tmp/results/nccl_paged_heartbeat"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _record(prompt: str, result) -> dict[str, object]:
    return {
        "prompt": prompt,
        "text": result.text,
        "token_ids": result.token_ids,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "prefill_seconds": result.prefill_seconds,
        "decode_seconds": result.decode_seconds,
        "decode_tokens_per_second": result.decode_tokens_per_second,
    }


def _run_engine(engine, args, *, load_seconds: float, transport) -> dict[str, object]:
    submit_started = time.perf_counter()
    futures = []
    for index, prompt in enumerate(args.prompts):
        if index:
            time.sleep(max(0.0, float(args.arrival_delay)))
        futures.append(
            engine.submit(
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
        )
    results = [future.result(timeout=1800) for future in futures]
    wall_seconds = time.perf_counter() - submit_started
    status = engine.status()
    records = [
        _record(prompt, result)
        for prompt, result in zip(args.prompts, results, strict=True)
    ]
    total_decode = sum(max(0, record["completion_tokens"] - 1) for record in records)
    payload = {
        "mode": "paged_continuous_batch",
        "prefill_mode": engine.prefill_mode,
        "block_size": args.block_size,
        "batch_size_capacity": args.max_batch_size,
        "arrival_delay_seconds": args.arrival_delay,
        "load_seconds": load_seconds,
        "wall_seconds": wall_seconds,
        "completed": status["completed"],
        "failed": status["failed"],
        "aggregate_decode_tokens_per_second": (
            total_decode / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "engine_status": status,
        "records": records,
    }
    if transport is not None:
        payload["remote_transport"] = dict(transport)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    if args.single_gpu:
        return _run_single_gpu(args)

    from deepinfer.moe.nccl_expert_transport import NCCLExpertClient, RemoteExpertWorker
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "2"))
    if world != 2:
        raise RuntimeError("this benchmark requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=300))
    moe_layers = _moe_layers(args.meta_dir)
    worker: RemoteExpertWorker | None = None
    client: NCCLExpertClient | None = None
    runtime = None
    engine = None
    try:
        if rank == 1:
            worker = RemoteExpertWorker(
                args.model,
                layer_indices=moe_layers,
                device=local_rank,
                expert_cache_mb=0,
                gpu_expert_cache_mb=0,
                parallel_read_workers=8,
            )
        else:
            client = NCCLExpertClient(
                dst_rank=1,
                device=local_rank,
                heartbeat_path=args.heartbeat_file,
            )
        dist.barrier()
        if rank == 1:
            assert worker is not None
            worker.serve_forever(src_rank=0, heartbeat_path=args.heartbeat_file)
            dist.barrier()
            print(
                json.dumps(
                    {
                        "rank": 1,
                        "role": "remote_expert_worker",
                        **worker.metrics(),
                        "peak_gpu_allocated_bytes": int(
                            torch.cuda.max_memory_allocated(local_rank)
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return 0

        assert client is not None
        load_started = time.perf_counter()
        runtime = DeepSeekV2LiteGGUFRuntime(
            args.model,
            meta_dir=args.meta_dir,
            device="cuda:0",
            expert_devices=(0, 1),
            remote_expert_client=client,
            remote_expert_device=1,
            remote_expert_fraction=args.remote_fraction,
            max_cache=args.max_cache,
            expert_chunk_size=8,
            verbose=False,
            ecs_enabled=True,
        )
        load_seconds = time.perf_counter() - load_started
        engine = PagedContinuousBatchEngine(
            runtime,
            max_batch_size=args.max_batch_size,
            block_size=args.block_size,
            num_blocks=args.kv_blocks,
            kv_memory_utilization=args.kv_memory_utilization,
            max_cache=args.max_cache,
            device=local_rank,
            prefill_mode=args.prefill_mode,
        )
        payload = _run_engine(
            engine,
            args,
            load_seconds=load_seconds,
            transport=json.loads(json.dumps(client.metrics())),
        )
        engine.close()
        engine = None
        client.close()
        dist.barrier()
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0
    finally:
        if engine is not None:
            engine.close()
        if runtime is not None:
            runtime.close()
        if worker is not None:
            worker.close()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def _run_single_gpu(args) -> int:
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

    torch.cuda.set_device(args.device)
    load_started = time.perf_counter()
    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        device=args.device,
        expert_devices=(0,),
        max_cache=args.max_cache,
        expert_chunk_size=8,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_started
    engine = None
    try:
        engine = PagedContinuousBatchEngine(
            runtime,
            max_batch_size=args.max_batch_size,
            block_size=args.block_size,
            num_blocks=args.kv_blocks,
            kv_memory_utilization=args.kv_memory_utilization,
            max_cache=args.max_cache,
            device=int(str(args.device).split(":")[-1] or 0),
            prefill_mode=args.prefill_mode,
        )
        payload = _run_engine(engine, args, load_seconds=load_seconds, transport=None)
        payload["device"] = str(args.device)
        payload["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return 0
    finally:
        if engine is not None:
            engine.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
