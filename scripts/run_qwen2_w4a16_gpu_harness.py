"""Run DeepInfer's Qwen2 W4A16 model code in the existing Python 3.10 CUDA env.

The native DeepInfer package itself needs Python 3.12, but the Qwen2 model
source and W4A16 backend are self-contained enough to load with lightweight
package stubs on Python 3.10.  This harness is only for the SM86 model/kernel
speed probe; it is not the final engine integration test.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import sys
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(r"F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards")
DEFAULT_ADAPTER = Path(r"F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120")


def _install_deepinfer_import_stubs() -> None:
    """Load the two DeepInfer source modules without importing the 3.12 package."""

    if not hasattr(sys, "version_info") or sys.version_info >= (3, 12):
        # The real package should be used when Python 3.12 is available.
        return

    import typing

    if not hasattr(typing, "Self"):
        typing.Self = typing.TypeVar("Self")  # type: ignore[attr-defined]

    def _module(name: str, path: Path) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = name.rpartition(".")[0]
        sys.modules[name] = module
        return module

    deepinfer = _module("deepinfer", ROOT / "deepinfer" / "__init__.py")
    deepinfer.__path__ = [str(ROOT / "deepinfer")]

    # attrdict.py is the only package module needed by Qwen config/model code.
    spec = importlib.util.spec_from_file_location(
        "deepinfer.attrdict", ROOT / "deepinfer" / "attrdict.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load deepinfer.attrdict")
    attrdict_module = importlib.util.module_from_spec(spec)
    sys.modules["deepinfer.attrdict"] = attrdict_module
    spec.loader.exec_module(attrdict_module)
    deepinfer.AttrDict = attrdict_module.AttrDict  # type: ignore[attr-defined]

    logger_module = _module("deepinfer.logger", ROOT / "deepinfer" / "logger.py")

    def init_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    logger_module.init_logger = init_logger  # type: ignore[attr-defined]

    model_executor = _module(
        "deepinfer.model_executor",
        ROOT / "deepinfer" / "model_executor" / "__init__.py",
    )
    model_executor.__path__ = [str(ROOT / "deepinfer" / "model_executor")]

    models_pkg = _module(
        "deepinfer.model_executor.models",
        ROOT / "deepinfer" / "model_executor" / "models" / "__init__.py",
    )
    models_pkg.__path__ = [str(ROOT / "deepinfer" / "model_executor" / "models")]

    output_module = _module(
        "deepinfer.model_executor.model_output",
        ROOT / "deepinfer" / "model_executor" / "model_output.py",
    )

    @dataclass(frozen=True, slots=True)
    class CausalLMForwardOutput:
        hidden_states: torch.Tensor

    output_module.CausalLMForwardOutput = CausalLMForwardOutput  # type: ignore[attr-defined]

    hugging_face_pkg = _module(
        "deepinfer.hugging_face",
        ROOT / "deepinfer" / "hugging_face" / "__init__.py",
    )
    hugging_face_pkg.__path__ = [str(ROOT / "deepinfer" / "hugging_face")]


def _load_module(name: str, relative_path: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "deepinfer" / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepInfer Qwen2 W4A16 SM86 GPU harness"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "doc" / "results" / "qwen2_w4a16_gpu_harness.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _install_deepinfer_import_stubs()
    config_module = _load_module(
        "deepinfer.hugging_face.qwen2_config", "hugging_face/qwen2_config.py"
    )
    qwen2_module = _load_module(
        "deepinfer.model_executor.models.qwen2", "model_executor/models/qwen2.py"
    )
    w4a16_module = _load_module(
        "deepinfer.model_executor.models.qwen2_w4a16",
        "model_executor/models/qwen2_w4a16.py",
    )

    report: dict[str, Any] = {
        "status": "running",
        "route": "DeepInfer Qwen2/W4A16 source + standalone Python3.10 harness",
        "full_engine": False,
        "native_w4a16_kernel": True,
        "native_attention_kernel": True,
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
        hf = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
        model_cfg = config_module.build_qwen2_model_config(
            hf, args.model / "config.json"
        )
        dense_cls = qwen2_module.Qwen2ForCausalLM
        quant_cls = w4a16_module.Qwen2W4A16ForCausalLM

        dense_started = time.perf_counter()
        dense = dense_cls.load_model(
            str(args.model),
            model_cfg,
            adapter_path=args.adapter,
            device="cpu",
        )
        report["dense_cpu_load_seconds"] = round(time.perf_counter() - dense_started, 3)
        print(f"dense loaded in {report['dense_cpu_load_seconds']}s", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        convert_started = time.perf_counter()
        model = quant_cls.from_dense_model(
            dense, device="cuda", group_size=args.group_size
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
        print(f"converted in {report['convert_seconds']}s", flush=True)

        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt_text, return_tensors="pt")
        input_ids = encoded["input_ids"].cuda()
        positions = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        cache = model.new_kv_cache()
        generated_ids: list[int] = []

        with torch.inference_mode():
            started = time.perf_counter()
            logits = model.forward_logits(input_ids, positions, kv_cache=cache)
            generated_ids.append(int(logits[:, -1].argmax(dim=-1).item()))
            torch.cuda.synchronize()
            report["prefill_ttft_seconds"] = round(time.perf_counter() - started, 6)

        # Build a fixed-size static GQA cache from the prompt cache.  All
        # decode steps and CUDA Graph replays use this cache.
        max_seq = args.max_cache
        required_seq = input_ids.shape[1] + args.max_new_tokens + 2
        if required_seq > max_seq:
            raise ValueError(
                f"max_cache={max_seq} is too small for {required_seq} positions"
            )
        static_cache = []
        for layer_cache in cache:
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
            prompt_length = int(layer_cache.key.shape[2])
            static.key[:, :, :prompt_length, :].copy_(layer_cache.key)
            static.value[:, :, :prompt_length, :].copy_(layer_cache.value)
            static_cache.append(static)

        prompt_length = int(input_ids.shape[1])
        token = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device="cuda")
        position = torch.tensor([[prompt_length]], dtype=torch.long, device="cuda")
        mask = torch.zeros((1, max_seq), dtype=torch.bool, device="cuda")
        mask[:, : prompt_length + 1] = True

        with torch.no_grad():
            warm_logits = model.forward_logits(
                token,
                position,
                kv_cache=static_cache,
                attention_mask=mask,
            )
            torch.cuda.synchronize()
        generated_ids.append(int(warm_logits[:, -1].argmax(dim=-1).item()))

        # Capture the single-token decode step.  Cache writes use a fixed
        # position/index tensor, so replay can advance one slot at a time.
        token.fill_(generated_ids[-1])
        position.fill_(prompt_length + 1)
        mask.zero_()
        mask[:, : prompt_length + 2] = True
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
        # CUDA Graph capture records custom CUDA kernels but does not execute
        # them; the first replay produces the captured step's real output.
        first_replay_started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        report["graph_first_replay_seconds"] = round(
            time.perf_counter() - first_replay_started, 6
        )
        generated_ids.append(int(graph_logits[:, -1].argmax(dim=-1).item()))

        replay_latencies: list[float] = []
        with torch.no_grad():
            for replay_index in range(args.max_new_tokens):
                token.fill_(generated_ids[-1])
                next_position = prompt_length + 2 + replay_index
                position.fill_(next_position)
                mask.zero_()
                mask[:, : next_position + 1] = True
                started = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                replay_latencies.append(time.perf_counter() - started)
                generated_ids.append(int(graph_logits[:, -1].argmax(dim=-1).item()))

        replay_seconds = sum(replay_latencies)
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
                "graph_replay_tokens": len(replay_latencies),
                "graph_replay_seconds": round(replay_seconds, 6),
                "graph_tokens_per_second": round(
                    max(0, len(replay_latencies)) / max(replay_seconds, 1e-9),
                    3,
                ),
                "graph_token_latencies_ms": [
                    round(value * 1000, 3) for value in replay_latencies
                ],
                "max_cache": max_seq,
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
