"""Run the SM86 W4A16 DeepInfer Qwen2 model with real weights on the RTX 3050.

The dense BF16 model is loaded on CPU first, merged with the PEFT adapter by the
existing Qwen2 loader, then converted layer-by-layer to the DeepInfer W4A16
model_executor backend.  This is still a model_executor path, not the complete
DeepSeek native engine.
"""

from __future__ import annotations

import argparse
import gc
import json
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

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM
from deepinfer.model_executor.models.qwen2_w4a16 import Qwen2W4A16ForCausalLM

DEFAULT_MODEL = Path(r"F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards")
DEFAULT_ADAPTER = Path(r"F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepInfer Qwen2 SM86 W4A16 GPU smoke")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "doc" / "results" / "qwen2_w4a16_gpu_smoke.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.cuda.init()
    report: dict[str, Any] = {
        "status": "running",
        "route": "deepinfer.model_executor.models.qwen2_w4a16.Qwen2W4A16ForCausalLM",
        "full_engine": False,
        "native_w4a16_kernel": True,
        "native_attention_kernel": False,
        "model": str(args.model),
        "adapter": str(args.adapter),
        "group_size": args.group_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model),
            local_files_only=True,
            trust_remote_code=True,
        )
        model_cfg = load_hf_model_config(args.model / "config.json")

        load_started = time.perf_counter()
        dense = Qwen2ForCausalLM.load_model(
            str(args.model),
            model_cfg,
            adapter_path=args.adapter,
            device="cpu",
        )
        report["dense_cpu_load_seconds"] = round(time.perf_counter() - load_started, 3)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        convert_started = time.perf_counter()
        model = Qwen2W4A16ForCausalLM.from_dense_model(
            dense,
            device="cuda",
            group_size=args.group_size,
        )
        torch.cuda.synchronize()
        report["convert_seconds"] = round(time.perf_counter() - convert_started, 3)
        report["gpu_peak_alloc_gib"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
        report["gpu_peak_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        )
        del dense
        gc.collect()
        torch.cuda.empty_cache()

        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].cuda()
        positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        cache = model.new_kv_cache()

        generated_ids: list[int] = []
        with torch.inference_mode():
            prefill_started = time.perf_counter()
            logits = model.forward_logits(input_ids, positions, kv_cache=cache)
            next_token = int(logits[:, -1].argmax(dim=-1).item())
            torch.cuda.synchronize()
            prefill_seconds = time.perf_counter() - prefill_started
            generated_ids.append(next_token)

            token_latencies: list[float] = []
            for _ in range(max(0, args.max_new_tokens - 1)):
                token = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device="cuda"
                )
                position = torch.tensor(
                    [[input_ids.shape[1] + len(generated_ids) - 1]],
                    dtype=torch.long,
                    device="cuda",
                )
                started = time.perf_counter()
                logits = model.forward_logits(token, position, kv_cache=cache)
                generated_ids.append(int(logits[:, -1].argmax(dim=-1).item()))
                torch.cuda.synchronize()
                token_latencies.append(time.perf_counter() - started)

        decode_seconds = sum(token_latencies)
        report.update(
            {
                "status": "passed",
                "prompt": args.prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": len(generated_ids),
                "generated_ids": generated_ids,
                "generated_text": tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ),
                "prefill_ttft_seconds": round(prefill_seconds, 6),
                "decode_seconds": round(decode_seconds, 6),
                "decode_tokens_per_second": round(
                    max(0, len(token_latencies)) / max(decode_seconds, 1e-9),
                    3,
                ),
                "token_latencies_ms": [
                    round(value * 1000, 3) for value in token_latencies
                ],
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
