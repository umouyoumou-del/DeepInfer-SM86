"""Smoke-test ECS queue-depth scheduling over two independent GPU runtimes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from deepinfer.ecs import ECSRuntimeWorkerPool
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime


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
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    def factory(device: int):
        torch.cuda.set_device(device)
        try:
            import cupy

            cupy.cuda.Device(device).use()
        except Exception:
            pass
        return DeepSeekV2LiteGGUFRuntime(
            args.model,
            meta_dir=args.meta_dir,
            device=f"cuda:{device}",
            max_cache=args.max_cache,
            expert_chunk_size=8,
            verbose=False,
        )

    pool = ECSRuntimeWorkerPool(
        factory,
        devices=args.devices,
        max_queue_size=4,
    )
    try:
        ready = pool.wait_ready(timeout=900)
        if not ready:
            raise RuntimeError(f"ECS workers did not become ready: {pool.status()}")
        prompts = [
            [{"role": "user", "content": "只回答数字：5-2=?"}],
            [{"role": "user", "content": "只回答数字：2+3=?"}],
        ]
        started = time.perf_counter()
        futures = [
            pool.submit(prompt, max_new_tokens=args.max_new_tokens, temperature=0.0)
            for prompt in prompts
        ]
        results = [future.result(timeout=900) for future, _ in futures]
        elapsed = time.perf_counter() - started
        report = {
            "devices": args.devices,
            "elapsed_seconds": elapsed,
            "assigned_devices": [device for _, device in futures],
            "results": [
                {
                    "device": device,
                    "text": result.text,
                    "token_ids": result.token_ids,
                    "decode_tokens_per_second": result.decode_tokens_per_second,
                }
                for (_, device), result in zip(futures, results, strict=True)
            ],
            "status": pool.status(),
            "gpu_names": [
                torch.cuda.get_device_name(device) for device in args.devices
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return 0
    finally:
        pool.close()


if __name__ == "__main__":
    raise SystemExit(main())
