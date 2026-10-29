"""Run the native DeepSeek-V2-Lite MoE executor on a Q4_K_M GGUF.

Example:
    F:\\QwenQLoRA\\venv\\Scripts\\python.exe ^
      scripts\\smoke_deepseek_v2_lite_moe.py ^
      --model E:\\DeepInfer-MoE\\models\\DeepSeek-V2-Lite-Chat-GGUF\\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf ^
      --meta-dir E:\\DeepInfer-MoE\\meta ^
      --prompt "用一句话解释什么是混合专家模型（MoE）。" ^
      --max-new-tokens 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native DeepSeek-V2-Lite MoE smoke/benchmark on SM86"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
            r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
        ),
        help="Q4_K_M GGUF checkpoint",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(r"E:\DeepInfer-MoE\meta"),
        help="directory containing config.json, modeling_deepseek.py and tokenizer files",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="user prompt; repeat this flag to benchmark multiple prompts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=4,
        help="run a short warmup generation before measured prompts",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-cache", type=int, default=256)
    parser.add_argument("--expert-chunk-size", type=int, default=8)
    parser.add_argument(
        "--cpu-moe-layers",
        type=int,
        default=0,
        help="offload the last N routed-expert layers to the CPU FP32 path",
    )
    parser.add_argument(
        "--cpu-experts-per-layer",
        type=int,
        default=0,
        help="offload the coldest N routed experts per layer to CPU AVX2",
    )
    parser.add_argument(
        "--cpu-expert-layers",
        type=int,
        default=0,
        help="apply expert-level CPU split only to the last N MoE layers; 0=all",
    )
    parser.add_argument(
        "--cpu-expert-cache-mb",
        type=int,
        default=512,
        help="LRU cache for dequantized CPU expert weights",
    )
    parser.add_argument(
        "--expert-cache-mb",
        type=int,
        default=0,
        help="CPU packed-expert LRU cache size in MiB; 0 disables it",
    )
    parser.add_argument(
        "--gpu-expert-cache-mb",
        type=int,
        default=0,
        help="GPU packed-expert LRU cache size in MiB; 0 disables it",
    )
    parser.add_argument(
        "--ecs",
        action="store_true",
        help="enable ECS heat/LRU/tier control plane",
    )
    parser.add_argument("--ecs-l1-mb", type=int, default=0)
    parser.add_argument("--ecs-l2-mb", type=int, default=0)
    parser.add_argument(
        "--ecs-chunk-max-tokens",
        type=int,
        default=0,
        help="ECS token budget per expert chunk; 0 keeps expert-count chunks",
    )
    parser.add_argument(
        "--ecs-profile",
        type=Path,
        default=None,
        help="persist/load ECS heat profile JSON",
    )
    parser.add_argument(
        "--prefill-prefetch",
        action="store_true",
        default=True,
        help="background-read the next prefill expert chunk (default on)",
    )
    parser.add_argument(
        "--no-prefill-prefetch",
        action="store_true",
        help="disable background prefill expert prefetch",
    )
    parser.add_argument(
        "--prefill-pipeline",
        action="store_true",
        help="experimental copy-stream pipeline for multi-token prefill chunks",
    )
    parser.add_argument(
        "--no-decode-prefetch",
        action="store_true",
        help="disable early background reads for the single decode expert chunk",
    )
    parser.add_argument(
        "--predictive-prefetch",
        action="store_true",
        help="experimental: warm next-layer experts from previous-token routing",
    )
    parser.add_argument(
        "--no-predictive-prefetch",
        action="store_true",
        help="kept for compatibility; predictive warmup is off by default",
    )
    parser.add_argument(
        "--predictive-cache-mb",
        type=int,
        default=256,
        help="CPU byte cache for predicted next-layer experts",
    )
    parser.add_argument(
        "--no-prefill-pipeline",
        action="store_true",
        help="kept for compatibility; pipeline is off by default",
    )
    parser.add_argument(
        "--prefill-all-experts",
        action="store_true",
        help="experimental: coalesce all routed experts into one prefill chunk",
    )
    parser.add_argument(
        "--no-prefill-all-experts",
        action="store_true",
        help="kept for compatibility; all-expert coalescing is off by default",
    )
    parser.add_argument(
        "--no-fused-moe",
        action="store_true",
        help="disable fused packed-weight MoE GEMV/GEMM and use dequant+bmm",
    )
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expert-devices",
        type=int,
        nargs="+",
        default=None,
        help="GPU indices that own routed experts (e.g. 0 1)",
    )
    parser.add_argument(
        "--shared-experts-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--dense-mlp-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--lm-head-device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--disable-down-copy-overlap",
        action="store_true",
        help="disable the pinned-memory copy stream (useful for debugging)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main() -> int:
    args = _parse_args()
    prompts = args.prompt or [
        "用一句话解释什么是混合专家模型（MoE）。",
        "中国的首都是哪里？请只回答城市名。",
    ]
    load_started = time.perf_counter()
    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        device=args.device,
        expert_devices=args.expert_devices,
        max_cache=args.max_cache,
        cpu_threads=args.cpu_threads,
        expert_chunk_size=args.expert_chunk_size,
        cpu_moe_layers=args.cpu_moe_layers,
        cpu_experts_per_layer=args.cpu_experts_per_layer,
        cpu_expert_layers=args.cpu_expert_layers,
        cpu_expert_cache_mb=args.cpu_expert_cache_mb,
        prefill_pipeline=(
            args.prefill_pipeline and not args.no_prefill_pipeline
        ),
        prefill_prefetch=(
            args.prefill_prefetch and not args.no_prefill_prefetch
        ),
        decode_prefetch=not args.no_decode_prefetch,
        predictive_prefetch=(
            args.predictive_prefetch and not args.no_predictive_prefetch
        ),
        predictive_cache_mb=args.predictive_cache_mb,
        ecs_enabled=args.ecs,
        ecs_l1_mb=args.ecs_l1_mb,
        ecs_l2_mb=args.ecs_l2_mb,
        ecs_chunk_max_tokens=args.ecs_chunk_max_tokens,
        ecs_profile=args.ecs_profile,
        prefill_all_experts=(
            args.prefill_all_experts and not args.no_prefill_all_experts
        ),
        expert_cache_mb=args.expert_cache_mb,
        gpu_expert_cache_mb=args.gpu_expert_cache_mb,
        fused_moe=not args.no_fused_moe,
        shared_experts_device=args.shared_experts_device,
        dense_mlp_device=args.dense_mlp_device,
        lm_head_device=args.lm_head_device,
        verbose=not args.quiet,
    )
    if args.disable_down_copy_overlap:
        runtime._overlap_down_copy = False
    load_seconds = time.perf_counter() - load_started

    records: list[dict[str, Any]] = []
    warmup_seconds = 0.0
    try:
        if args.warmup_tokens > 0:
            warmup_started = time.perf_counter()
            runtime.generate(
                [{"role": "user", "content": "你好。"}],
                max_new_tokens=args.warmup_tokens,
                temperature=0.0,
            )
            warmup_seconds = time.perf_counter() - warmup_started
        for prompt in prompts:
            result = runtime.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            record = {"prompt": prompt, **result.as_dict()}
            records.append(record)
            if not args.quiet:
                print("\n=== prompt ===")
                print(prompt)
                print("=== response ===")
                print(result.text)
                print(
                    "decode %.3f tok/s | prefill %.3fs | decode %.3fs | "
                    "peak GPU %.1f MiB"
                    % (
                        result.decode_tokens_per_second,
                        result.prefill_seconds,
                        result.decode_seconds,
                        result.peak_gpu_allocated_bytes / 1024**2,
                    )
                )
    finally:
        runtime.close()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(args.model),
        "meta_dir": str(args.meta_dir),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "device": torch.cuda.get_device_name(
            int(torch.device(args.device).index or 0)
        ),
        "expert_devices": args.expert_devices,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_bytes_after_close": int(free_bytes),
        "max_cache": args.max_cache,
        "expert_chunk_size": args.expert_chunk_size,
        "cpu_moe_layers": args.cpu_moe_layers,
        "cpu_experts_per_layer": args.cpu_experts_per_layer,
        "cpu_expert_layers": args.cpu_expert_layers,
        "cpu_expert_cache_mb": args.cpu_expert_cache_mb,
        "prefill_pipeline": (
            args.prefill_pipeline and not args.no_prefill_pipeline
        ),
        "prefill_prefetch": (
            args.prefill_prefetch and not args.no_prefill_prefetch
        ),
        "decode_prefetch": not args.no_decode_prefetch,
        "predictive_prefetch": (
            args.predictive_prefetch and not args.no_predictive_prefetch
        ),
        "predictive_cache_mb": args.predictive_cache_mb,
        "ecs": args.ecs,
        "ecs_l1_mb": args.ecs_l1_mb,
        "ecs_l2_mb": args.ecs_l2_mb,
        "ecs_chunk_max_tokens": args.ecs_chunk_max_tokens,
        "ecs_profile": str(args.ecs_profile) if args.ecs_profile else None,
        "prefill_all_experts": (
            args.prefill_all_experts and not args.no_prefill_all_experts
        ),
        "expert_cache_mb": args.expert_cache_mb,
        "gpu_expert_cache_mb": args.gpu_expert_cache_mb,
        "dense_mlp_device": args.dense_mlp_device,
        "fused_moe_requested": not args.no_fused_moe,
        "results": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.quiet:
        print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
