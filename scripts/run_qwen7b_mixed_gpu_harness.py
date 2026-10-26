"""Run the 7B mixed W4/W3 DeepInfer model on RTX 3050 with native GQA + Graph."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qwen2_w4a16_gpu_harness import (
    _install_deepinfer_import_stubs,
    _load_module,
)

DEFAULT_CHECKPOINT = Path(r"D:\Qwen7B-Mixed-SM86")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepInfer Qwen2.5-7B mixed W4/W3 SM86 harness"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup-replays", type=int, default=3)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "doc" / "results" / "qwen7b_mixed_gpu.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _install_deepinfer_import_stubs()
    _load_module("deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py")
    qwen2_module = _load_module(
        "deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py"
    )
    _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )
    mixed_module = _load_module(
        "deepinfer.model_executor.models.qwen2_mixed_sm86",
        "model_executor/models/qwen2_mixed_sm86.py",
    )

    checkpoint_manifest = json.loads(
        (args.checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    report: dict[str, Any] = {
        "status": "running",
        "route": "DeepInfer Qwen2.5-7B SM86 model_executor",
        "full_engine": False,
        "native_w4a16_kernel": True,
        "native_w3a16_kernel": checkpoint_manifest["mlp_bits"] == 3
        or checkpoint_manifest["lm_head_bits"] == 3,
        "native_attention_kernel": True,
        "checkpoint": str(args.checkpoint),
        "quantization": {
            "attention": checkpoint_manifest["attention_bits"],
            "mlp": checkpoint_manifest["mlp_bits"],
            "embed": checkpoint_manifest["embed_bits"],
            "lm_head": checkpoint_manifest["lm_head_bits"],
            "group_size": args.group_size,
        },
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.checkpoint),
            local_files_only=True,
            trust_remote_code=True,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = mixed_module.Qwen2MixedForCausalLM.from_checkpoint(
            args.checkpoint,
            device="cuda",
            group_size=args.group_size,
        )
        torch.cuda.synchronize()
        report["load_seconds"] = round(time.perf_counter() - load_started, 3)
        report["gpu_peak_alloc_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        report["gpu_peak_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        )
        report["gpu_alloc_after_load_gib"] = round(
            torch.cuda.memory_allocated() / 1024**3, 3
        )
        report["gpu_reserved_after_load_gib"] = round(
            torch.cuda.memory_reserved() / 1024**3, 3
        )
        print(f"loaded in {report['load_seconds']}s", flush=True)

        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].cuda()
        positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        dynamic_cache = model.new_kv_cache()

        with torch.inference_mode():
            started = time.perf_counter()
            logits = model.forward_logits(input_ids, positions, kv_cache=dynamic_cache)
            generated_ids = [int(logits[:, -1].argmax(dim=-1).item())]
            torch.cuda.synchronize()
            report["prefill_ttft_seconds"] = round(time.perf_counter() - started, 6)
            report["gpu_alloc_after_prefill_gib"] = round(
                torch.cuda.memory_allocated() / 1024**3, 3
            )
            report["gpu_reserved_after_prefill_gib"] = round(
                torch.cuda.memory_reserved() / 1024**3, 3
            )
        torch.cuda.empty_cache()

        max_seq = args.max_cache
        prompt_length = int(input_ids.shape[1])
        required = prompt_length + args.max_new_tokens + 2
        if required > max_seq:
            raise ValueError(f"max_cache={max_seq} too small for {required} positions")
        static_cache = []
        for layer_cache in dynamic_cache:
            assert (
                layer_cache is not None
                and layer_cache.key is not None
                and layer_cache.value is not None
            )
            static = qwen2_module.Qwen2StaticLayerKVCache(
                key=torch.zeros(
                    (1, layer_cache.key.shape[1], max_seq, layer_cache.key.shape[3]),
                    dtype=layer_cache.key.dtype,
                    device="cuda",
                ),
                value=torch.zeros(
                    (
                        1,
                        layer_cache.value.shape[1],
                        max_seq,
                        layer_cache.value.shape[3],
                    ),
                    dtype=layer_cache.value.dtype,
                    device="cuda",
                ),
            )
            static.key[:, :, :prompt_length, :].copy_(layer_cache.key)
            static.value[:, :, :prompt_length, :].copy_(layer_cache.value)
            static_cache.append(static)

        token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device="cuda")
        position = torch.tensor([[prompt_length]], dtype=torch.long, device="cuda")
        mask = torch.zeros((1, max_seq), dtype=torch.bool, device="cuda")
        mask[:, : prompt_length + 1] = True
        warm_started = time.perf_counter()
        with torch.no_grad():
            warm_logits = model.forward_logits(
                token,
                position,
                kv_cache=static_cache,
                attention_mask=mask,
            )
            torch.cuda.synchronize()
        report["warm_eager_decode_seconds"] = round(
            time.perf_counter() - warm_started, 6
        )
        report["gpu_alloc_before_capture_gib"] = round(
            torch.cuda.memory_allocated() / 1024**3, 3
        )
        report["gpu_reserved_before_capture_gib"] = round(
            torch.cuda.memory_reserved() / 1024**3, 3
        )
        generated_ids.append(int(warm_logits[:, -1].argmax(dim=-1).item()))

        token.fill_(generated_ids[-1])
        position.fill_(prompt_length + 1)
        mask.zero_()
        mask[:, : prompt_length + 2] = True
        torch.cuda.empty_cache()
        capture_started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            graph_logits = model.forward_logits(
                token,
                position,
                kv_cache=static_cache,
                attention_mask=mask,
            )
        torch.cuda.synchronize()
        report["graph_capture_seconds"] = round(
            time.perf_counter() - capture_started, 6
        )
        report["gpu_alloc_after_capture_gib"] = round(
            torch.cuda.memory_allocated() / 1024**3, 3
        )
        report["gpu_reserved_after_capture_gib"] = round(
            torch.cuda.memory_reserved() / 1024**3, 3
        )
        first_started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        report["graph_first_replay_seconds"] = round(
            time.perf_counter() - first_started, 6
        )
        generated_ids.append(int(graph_logits[:, -1].argmax(dim=-1).item()))

        eos_token_id = tokenizer.eos_token_id
        # Warm up the graph after capture and advance the cache.  These tokens
        # are real generated tokens but are excluded from the timed average.
        with torch.no_grad():
            for warmup_index in range(args.warmup_replays):
                if eos_token_id is not None and generated_ids[-1] == eos_token_id:
                    break
                token.fill_(generated_ids[-1])
                next_position = prompt_length + 2 + warmup_index
                position.fill_(next_position)
                mask.zero_()
                mask[:, : next_position + 1] = True
                graph.replay()
                torch.cuda.synchronize()
                generated_ids.append(int(graph_logits[:, -1].argmax(dim=-1).item()))

        latencies: list[float] = []
        with torch.no_grad():
            for replay_index in range(args.max_new_tokens):
                if eos_token_id is not None and generated_ids[-1] == eos_token_id:
                    break
                token.fill_(generated_ids[-1])
                next_position = prompt_length + 2 + args.warmup_replays + replay_index
                position.fill_(next_position)
                mask.zero_()
                mask[:, : next_position + 1] = True
                started = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)
                generated_ids.append(int(graph_logits[:, -1].argmax(dim=-1).item()))

        replay_seconds = sum(latencies)
        replay_median = statistics.median(latencies) if latencies else 0.0
        report.update(
            {
                "status": "passed",
                "prompt": args.prompt,
                "prompt_tokens": prompt_length,
                "generated_tokens": len(generated_ids),
                "generated_ids": generated_ids,
                "generated_text": tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ),
                "graph_replay_tokens": len(latencies),
                "graph_replay_seconds": round(replay_seconds, 6),
                "graph_tokens_per_second": round(
                    len(latencies) / max(replay_seconds, 1e-9), 3
                ),
                "graph_median_latency_ms": round(replay_median * 1000, 3),
                "graph_median_tokens_per_second": round(1.0 / replay_median, 3)
                if replay_median
                else 0.0,
                "decode_mode": "synchronized_greedy",
                "warmup_replays": args.warmup_replays,
                "gpu_memory_after_generation_gib": round(
                    torch.cuda.memory_allocated() / 1024**3, 3
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 1024**3, 3),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
