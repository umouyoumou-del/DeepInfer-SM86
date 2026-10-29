"""Two-GPU layer-pipeline smoke using two DeepSeek runtimes.

This is a stepping stone between independent replicas and fully expert-sharded
All-to-All: rank 0 runs the first layers, rank 1 runs the remaining layers and
the LM head.  Hidden state crosses the two GPUs once per layer-stage per token.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime


def _set_device(device: int) -> None:
    torch.cuda.set_device(device)
    try:
        import cupy

        cupy.cuda.Device(device).use()
    except Exception:
        pass


def _run_layers(
    runtime: DeepSeekV2LiteGGUFRuntime,
    layer_indices: range,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache: DynamicCache,
) -> torch.Tensor:
    device = int(runtime.device.index or 0)
    _set_device(device)
    hidden = hidden.to(runtime.device)
    attention_mask = attention_mask.to(runtime.device)
    position_ids = position_ids.to(runtime.device)
    layers = runtime.model.model.layers
    for index in layer_indices:
        output = layers[index](
            hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            output_attentions=False,
        )
        hidden = output[0] if isinstance(output, tuple) else output
    return hidden


def _logits(runtime: DeepSeekV2LiteGGUFRuntime, hidden: torch.Tensor) -> torch.Tensor:
    _set_device(int(runtime.device.index or 0))
    normalized = runtime.model.model.norm(hidden.to(runtime.device))
    return runtime._logits(normalized)


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
    parser.add_argument("--split", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument(
        "--prompt", default="只回答数字：5-2?"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    load_started = time.perf_counter()
    _set_device(0)
    stage0 = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        device="cuda:0",
        max_cache=args.max_cache,
        expert_chunk_size=8,
        verbose=False,
    )
    _set_device(1)
    stage1 = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        device="cuda:1",
        max_cache=args.max_cache,
        expert_chunk_size=8,
        verbose=False,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        prompt_text = stage0.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = stage0.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        prompt_tokens = int(input_ids.shape[1])
        cache0 = DynamicCache()
        cache1 = DynamicCache()
        _set_device(0)
        hidden = stage0._embed_tokens(input_ids)
        positions = torch.arange(prompt_tokens, device="cuda:0").unsqueeze(0)
        mask = torch.ones((1, prompt_tokens), dtype=torch.long, device="cuda:0")
        mask4d = _prepare_4d_causal_attention_mask(
            mask, (1, prompt_tokens), hidden, 0
        )
        prefill_started = time.perf_counter()
        hidden = _run_layers(
            stage0, range(0, args.split), hidden, mask4d, positions, cache0
        )
        hidden = _run_layers(
            stage1,
            range(args.split, len(stage1.model.model.layers)),
            hidden,
            mask4d,
            positions,
            cache1,
        )
        logits = _logits(stage1, hidden)
        next_token = stage1._sample(
            logits[0], temperature=0.0, top_p=1.0, top_k=0
        )
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [next_token]
        total_tokens = prompt_tokens + 1
        decode_started = time.perf_counter()
        for _ in range(args.max_new_tokens - 1):
            if next_token == int(stage0.tokenizer.eos_token_id):
                break
            _set_device(0)
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            hidden = stage0._embed_tokens(token_tensor)
            token_mask = torch.ones(
                (1, total_tokens), dtype=torch.long, device="cuda:0"
            )
            token_mask4d = _prepare_4d_causal_attention_mask(
                token_mask,
                (1, 1),
                hidden,
                total_tokens - 1,
            )
            token_positions = torch.tensor(
                [[total_tokens - 1]], dtype=torch.long, device="cuda:0"
            )
            hidden = _run_layers(
                stage0,
                range(0, args.split),
                hidden,
                token_mask4d,
                token_positions,
                cache0,
            )
            hidden = _run_layers(
                stage1,
                range(args.split, len(stage1.model.model.layers)),
                hidden,
                token_mask4d,
                token_positions,
                cache1,
            )
            logits = _logits(stage1, hidden)
            next_token = stage1._sample(
                logits[0], temperature=0.0, top_p=1.0, top_k=0
            )
            generated.append(next_token)
            total_tokens += 1
        decode_seconds = time.perf_counter() - decode_started
        text = stage0.tokenizer.decode(generated, skip_special_tokens=True)
        report = {
            "prompt": args.prompt,
            "split": args.split,
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                max(0, len(generated) - 1) / decode_seconds
                if decode_seconds
                else 0.0
            ),
            "text": text,
            "token_ids": generated,
            "gpus": [torch.cuda.get_device_name(0), torch.cuda.get_device_name(1)],
            "peak_allocated": [
                int(torch.cuda.max_memory_allocated(0)),
                int(torch.cuda.max_memory_allocated(1)),
            ],
            "stage_devices": [
                str(stage0.device),
                str(stage1.device),
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
        stage1.close()
        stage0.close()


if __name__ == "__main__":
    raise SystemExit(main())
