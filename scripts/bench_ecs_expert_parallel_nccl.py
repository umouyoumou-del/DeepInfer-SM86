"""Run routed experts on a second GPU through real NCCL activation exchange.

Launch with:
    torchrun --standalone --nproc_per_node=2 scripts/bench_ecs_expert_parallel_nccl.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig

from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.layout import split_layer_indices
from deepinfer.moe.nccl_expert_transport import NCCLExpertClient, RemoteExpertWorker


def _moe_layers(meta_dir: Path) -> tuple[int, ...]:
    config = AutoConfig.from_pretrained(
        str(meta_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    _, moe_layers = split_layer_indices(config)
    return tuple(int(layer) for layer in moe_layers)


def main() -> int:
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
        "--remote-fraction",
        type=float,
        default=0.5,
        help="fraction of routed experts owned by the remote rank",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="number of sequential prompts to serve over the same NCCL worker",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="NCCL process-group timeout used to detect a dead remote rank",
    )
    parser.add_argument(
        "--prompt", default="只回答数字：5-2=?"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "2"))
    if world != 2:
        raise RuntimeError("this benchmark requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=max(1.0, float(args.timeout_seconds))),
    )
    moe_layers = _moe_layers(args.meta_dir)
    worker: RemoteExpertWorker | None = None
    client: NCCLExpertClient | None = None
    runtime: DeepSeekV2LiteGGUFRuntime | None = None
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
            client = NCCLExpertClient(dst_rank=1, device=local_rank)

        print(f"[rank{rank}] initialized; entering barrier", flush=True)
        dist.barrier()
        print(f"[rank{rank}] barrier passed", flush=True)
        if rank == 1:
            assert worker is not None
            worker.serve_forever(src_rank=0)
            dist.barrier()
            payload = {
                "rank": rank,
                "gpu": torch.cuda.get_device_name(local_rank),
                "role": "remote_expert_worker",
                **worker.metrics(),
                "peak_gpu_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(local_rank)
                ),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
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
        # Rank 1 is already blocked in serve_forever waiting for the first
        # expert request.  A second barrier here would deadlock rank 0.
        print(f"[rank0] runtime ready; starting generate", flush=True)
        prompts = [args.prompt]
        extra = ["只回答数字：2+3=?", "只回答数字：9-4=?"]
        prompts.extend(extra[: max(0, int(args.requests) - 1)])
        records = []
        for prompt in prompts:
            result = runtime.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            records.append(
                {
                    "prompt": prompt,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "decode_tokens_per_second": result.decode_tokens_per_second,
                    "peak_gpu_allocated_bytes": result.peak_gpu_allocated_bytes,
                    "timings": result.timings,
                    "ecs": result.ecs,
                }
            )
        print("[rank0] generate finished; stopping worker", flush=True)
        client.close()
        dist.barrier()
        result = records[0]
        payload = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(local_rank),
            "role": "coordinator",
            "load_seconds": load_seconds,
            "remote_fraction": args.remote_fraction,
            "requests": len(records),
            "text": result["text"],
            "token_ids": result["token_ids"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "prefill_seconds": result["prefill_seconds"],
            "decode_seconds": result["decode_seconds"],
            "decode_tokens_per_second": result["decode_tokens_per_second"],
            "peak_gpu_allocated_bytes": result["peak_gpu_allocated_bytes"],
            "remote_transport": client.metrics(),
            "timings": result["timings"],
            "ecs": result.get("ecs"),
            "records": records,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0
    finally:
        if client is not None:
            # If generation failed before close, still make a best effort to
            # unblock rank 1.
            try:
                if dist.is_initialized():
                    pass
            except Exception:
                pass
        if runtime is not None:
            runtime.close()
        if worker is not None:
            worker.close()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
