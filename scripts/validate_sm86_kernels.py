"""Validate the optional SM86 paged-attention and batched W4 GEMV kernels.

The first two checks use synthetic tensors and compare against PyTorch
references.  Passing ``--checkpoint`` additionally compares native paged
decode logits against the SDPA fallback on a real W4A16 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.engine.kv_cache import PagedKVCacheManager
from deepinfer.engine.paged_attention import native_paged_gqa_decode
from deepinfer.model_executor.models.qwen2_w4a16 import W4A16Linear


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            left.flatten().float(), right.flatten().float(), dim=0
        ).item()
    )


def validate_paged_attention(kv_cache_dtype: str = "fp16") -> dict:
    torch.manual_seed(20261030)
    batch, heads, kv_heads, head_dim = 2, 4, 2, 16
    block_size = 4
    manager = PagedKVCacheManager(
        num_layers=1,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=16,
        dtype=torch.float16,
        kv_cache_dtype=kv_cache_dtype,
        device="cuda",
    )
    lengths = [6, 3]
    block_ids = [manager.allocate_tokens(length + 1) for length in lengths]
    for ids, length in zip(block_ids, lengths):
        key = torch.randn(
            1, kv_heads, length, head_dim, device="cuda", dtype=torch.float16
        )
        value = torch.randn_like(key)
        manager.write_tokens(0, ids, key, value)
    views = manager.make_views(
        block_ids, lengths, max_length=max(lengths) + 1, new_tokens=1
    )
    new_key = torch.randn(
        batch, kv_heads, 1, head_dim, device="cuda", dtype=torch.float16
    )
    new_value = torch.randn_like(new_key)
    views[0].update(
        new_key,
        new_value,
        torch.tensor(lengths, device="cuda").unsqueeze(1),
    )
    query = torch.randn(batch, heads, 1, head_dim, device="cuda", dtype=torch.float16)
    scale = 1.0 / math.sqrt(head_dim)
    window = 3
    softcap = 5.0
    output = native_paged_gqa_decode(
        query,
        views[0],
        scale=scale,
        sliding_window=window,
        softcap=softcap,
    )
    if output is None:
        raise RuntimeError("native paged attention is unavailable")
    torch.cuda.synchronize()
    key, value, valid = views[0].materialize()
    repeat = heads // kv_heads
    key = (
        key[:, :, None, :, :]
        .expand(batch, kv_heads, repeat, key.shape[2], head_dim)
        .reshape(batch, heads, key.shape[2], head_dim)
    )
    value = (
        value[:, :, None, :, :]
        .expand(batch, kv_heads, repeat, value.shape[2], head_dim)
        .reshape(batch, heads, value.shape[2], head_dim)
    )
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    scores = torch.tanh(scores / softcap) * softcap
    key_pos = torch.arange(key.shape[2], device="cuda")[None, None, None, :]
    query_pos = torch.tensor(lengths, device="cuda")[:, None, None, None]
    mask = valid[:, None, None, :] & (key_pos <= query_pos) & (
        (query_pos - key_pos) < window
    )
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    reference = torch.matmul(torch.softmax(scores, dim=-1), value)
    return {
        "kv_cache_dtype": kv_cache_dtype,
        "cosine": _cosine(output, reference),
        "max_abs_error": float((output.float() - reference.float()).abs().max()),
    }


def validate_batched_w4() -> dict:
    torch.manual_seed(20261031)
    in_features, out_features, group_size, batch = 512, 256, 32, 4
    dense = torch.nn.Linear(in_features, out_features, bias=True).cuda().half()
    quantized = W4A16Linear(
        in_features,
        out_features,
        True,
        group_size=group_size,
        dtype=torch.float16,
    ).cuda()
    quantized.load_from_linear(dense)
    inputs = torch.randn(
        batch, in_features, device="cuda", dtype=torch.float16
    )
    output = quantized(inputs)
    weight = quantized._dequantize_rows(0, out_features).to(torch.float16)
    reference = torch.nn.functional.linear(inputs, weight, dense.bias)
    return {
        "cosine": _cosine(output, reference),
        "max_abs_error": float((output.float() - reference.float()).abs().max()),
        "batch": batch,
    }


def validate_checkpoint(checkpoint: Path, max_cache: int) -> dict:
    from apps.runtime import QwenSm86Runtime

    runtime = QwenSm86Runtime(
        checkpoint,
        max_cache=max_cache,
        use_graph=False,
        single_request_cache=False,
        backend="w4a16",
        allow_extended_context=True,
    )
    model = runtime.model
    tokenizer = runtime.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "用一句话解释分页注意力。"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    prompt_length = int(input_ids.shape[1])
    cfg = model.cfg
    compute_dtype = next(model.parameters()).dtype

    def prefill(manager: PagedKVCacheManager) -> list[int]:
        blocks = manager.allocate_tokens(prompt_length + 2)
        for start in range(0, prompt_length, 16):
            end = min(prompt_length, start + 16)
            views = manager.make_views(
                [blocks], [start], max_length=end, new_tokens=end - start
            )
            with torch.inference_mode():
                model.forward_logits(
                    input_ids[:, start:end],
                    torch.arange(start, end, device="cuda").unsqueeze(0),
                    kv_cache=views,
                )
        return blocks

    base_manager = PagedKVCacheManager(
        num_layers=int(cfg.num_hidden_layers),
        num_kv_heads=int(cfg.num_key_value_heads),
        head_dim=int(cfg.head_dim),
        block_size=16,
        num_blocks=32,
        dtype=compute_dtype,
        device="cuda",
    )
    base_blocks = prefill(base_manager)
    last_views = base_manager.make_views(
        [base_blocks], [prompt_length - 1], max_length=prompt_length, new_tokens=1
    )
    with torch.inference_mode():
        logits = model.forward_logits(
            input_ids[:, -1:],
            torch.tensor([[prompt_length - 1]], device="cuda"),
            kv_cache=last_views,
        )
    next_token = int(logits[:, -1].argmax(-1).item())

    def decode(native: bool) -> tuple[torch.Tensor, float]:
        manager = PagedKVCacheManager(
            num_layers=int(cfg.num_hidden_layers),
            num_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.head_dim),
            block_size=16,
            num_blocks=32,
            dtype=compute_dtype,
            device="cuda",
        )
        blocks = prefill(manager)
        for layer in model.model.layers:
            layer.self_attn.native_paged_attention = native
        views = manager.make_views(
            [blocks], [prompt_length], max_length=prompt_length + 1, new_tokens=1
        )
        token = torch.tensor([[next_token]], device="cuda")
        position = torch.tensor([[prompt_length]], device="cuda")
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.forward_logits(token, position, kv_cache=views)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        return result[:, -1].clone(), elapsed

    native_logits, native_seconds = decode(True)
    fallback_logits, fallback_seconds = decode(False)
    return {
        "prompt_tokens": prompt_length,
        "cosine": _cosine(native_logits, fallback_logits),
        "max_abs_error": float(
            (native_logits.float() - fallback_logits.float()).abs().max()
        ),
        "native_argmax": int(native_logits.argmax(-1).item()),
        "fallback_argmax": int(fallback_logits.argmax(-1).item()),
        "native_decode_seconds": native_seconds,
        "fallback_decode_seconds": fallback_seconds,
        "speedup": fallback_seconds / native_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-cache", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "paged_attention_fp16": validate_paged_attention("fp16"),
        "paged_attention_int8": validate_paged_attention("int8"),
        "paged_attention_fp8": validate_paged_attention("fp8"),
        "paged_attention_fp8_e5m2": validate_paged_attention("fp8_e5m2"),
        "batched_w4": validate_batched_w4(),
    }
    if args.checkpoint is not None:
        report["checkpoint"] = str(args.checkpoint)
        report["end_to_end"] = validate_checkpoint(args.checkpoint, args.max_cache)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
