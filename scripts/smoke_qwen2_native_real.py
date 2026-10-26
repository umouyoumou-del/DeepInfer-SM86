"""Load the real Qwen2.5-3B checkpoint through DeepInfer's Qwen2 model code.

This is a source-level model_executor smoke test, not a full DeepInfer engine or
SM100-kernel benchmark.  It is intentionally usable on CPU because the current
host has no SM100/Linux native build and only 4 GB of VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.hugging_face import load_hf_model_config
from deepinfer.model_executor.models.qwen2 import Qwen2ForCausalLM

DEFAULT_MODEL = Path(r"F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards")
DEFAULT_ADAPTER = Path(r"F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-weight DeepInfer Qwen2 model smoke test"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "doc"
        / "results"
        / "qwen2_native_real_smoke.json",
    )
    return parser.parse_args()


def rss_gib() -> float:
    return round(psutil.Process().memory_info().rss / 1024**3, 3)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    report: dict[str, Any] = {
        "status": "running",
        "route": "deepinfer.model_executor.models.qwen2.Qwen2ForCausalLM",
        "full_engine": False,
        "native_kernel": False,
        "model": str(args.model),
        "adapter": str(args.adapter),
        "device": args.device,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model),
            local_files_only=True,
            trust_remote_code=True,
        )
        model_cfg = load_hf_model_config(args.model / "config.json")
        load_started = time.perf_counter()
        model = Qwen2ForCausalLM.load_model(
            str(args.model),
            model_cfg,
            adapter_path=args.adapter,
            device=args.device,
        )
        load_seconds = time.perf_counter() - load_started
        report["load_seconds"] = round(load_seconds, 3)
        report["rss_after_load_gib"] = rss_gib()
        report["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        report["parameter_dtype"] = str(next(model.parameters()).dtype)

        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].to(args.device)
        positions = torch.arange(input_ids.shape[1], device=args.device).unsqueeze(0)
        cache = model.new_kv_cache()

        generated_ids: list[int] = []
        with torch.inference_mode():
            first_token_started = time.perf_counter()
            logits = model.forward_logits(input_ids, positions, kv_cache=cache)
            next_token = int(logits[:, -1].argmax(dim=-1).item())
            first_token_seconds = time.perf_counter() - first_token_started
            generated_ids.append(next_token)

            decode_started = time.perf_counter()
            for _ in range(max(0, args.max_new_tokens - 1)):
                token = torch.tensor(
                    [[generated_ids[-1]]], dtype=torch.long, device=args.device
                )
                position = torch.tensor(
                    [[input_ids.shape[1] + len(generated_ids) - 1]],
                    dtype=torch.long,
                    device=args.device,
                )
                logits = model.forward_logits(token, position, kv_cache=cache)
                generated_ids.append(int(logits[:, -1].argmax(dim=-1).item()))
            decode_seconds = time.perf_counter() - decode_started

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        report.update(
            {
                "status": "passed",
                "prompt": args.prompt,
                "prompt_tokens": int(input_ids.shape[1]),
                "generated_tokens": len(generated_ids),
                "generated_ids": generated_ids,
                "generated_text": generated_text,
                "ttft_seconds": round(first_token_seconds, 6),
                "decode_seconds": round(decode_seconds, 6),
                "decode_tokens_per_second": round(
                    max(0, len(generated_ids) - 1) / max(decode_seconds, 1e-9),
                    3,
                ),
                "rss_final_gib": rss_gib(),
            }
        )
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "rss_gib": rss_gib(),
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
