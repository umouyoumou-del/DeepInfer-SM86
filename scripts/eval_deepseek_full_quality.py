"""Run the DeepSeek full-category quality regression suite.

This harness is dataset-driven and deliberately records its sampling budget.
It evaluates every MMLU subject represented in the suite, a fixed GSM8K
sample, deterministic long-chain arithmetic, and a fixed WikiText-2
perplexity slice. llama.cpp comparison is optional because CPU generation is
much slower than the native GPU runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from eval_deepseek_moe_quality import _run_llama_cli, _run_llama_ppl

DEFAULT_MODEL = Path(
    r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
    r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
)
DEFAULT_META = Path(r"E:\DeepInfer-MoE\meta")
DEFAULT_SUITE = ROOT / "docs" / "quality" / "deepseek_v2_lite_full_quality_suite.json"
DEFAULT_OUTPUT = ROOT / "docs" / "results" / "deepseek_v2_lite_q4km_moe_full_quality.json"
DEFAULT_LLAMA = Path(r"F:\QwenQLoRA\llama.cpp\cpu\llama-cli.exe")
DEFAULT_LLAMA_PPL = Path(r"F:\QwenQLoRA\llama.cpp\cpu\llama-perplexity.exe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--meta-dir", type=Path, default=DEFAULT_META)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--max-cache", type=int, default=1024)
    p.add_argument("--limit-per-category", type=int, default=0)
    p.add_argument("--mmlu-max-tokens", type=int, default=8)
    p.add_argument("--gsm8k-max-tokens", type=int, default=32)
    p.add_argument("--long-chain-max-tokens", type=int, default=32)
    p.add_argument("--skip-perplexity", action="store_true")
    p.add_argument("--skip-llama", action="store_true")
    p.add_argument("--llama-limit-per-category", type=int, default=5)
    p.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA)
    p.add_argument("--llama-ppl", type=Path, default=DEFAULT_LLAMA_PPL)
    p.add_argument("--llama-model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--llama-threads", type=int, default=6)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def safe_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def choice_prompt(item: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip("ABCD", item["choices"], strict=True)
    )
    return (
        "Answer the following multiple-choice question. Reply with only one "
        "letter: A, B, C, or D.\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )


def numeric_prompt(item: dict[str, Any]) -> str:
    return (
        "Solve the following reasoning problem. Reply with only the final "
        "numeric answer, no explanation.\n"
        f"Problem: {item['question']}\nAnswer:"
    )


def extract_choice(text: str) -> str | None:
    upper = text.strip().upper()
    matches = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", upper)
    if matches:
        return matches[0]
    return upper[:1] if upper[:1] in "ABCD" else None


def extract_number(text: str) -> str | None:
    cleaned = text.replace(",", "").replace("，", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    value = matches[-1]
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def normalize(value: str) -> str:
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else str(number)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * total)) / total
    ) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def run_generation(
    runtime: DeepSeekV2LiteGGUFRuntime,
    category: str,
    items: list[dict[str, Any]],
    prompt_fn,
    max_new_tokens: int,
    quiet: bool,
) -> tuple[list[dict[str, Any]], float]:
    tasks: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(items, 1):
        prompt = prompt_fn(item)
        result = runtime.generate(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        parsed = (
            extract_choice(result.text)
            if category == "mmlu"
            else extract_number(result.text)
        )
        expected = normalize(str(item["answer"]))
        actual = "" if parsed is None else normalize(str(parsed))
        task = {
            "id": item["id"],
            "category": category,
            "subject": item.get("subject"),
            "prompt": prompt,
            "expected": expected,
            "response": result.text,
            "parsed": parsed,
            "correct": actual == expected,
            "prefill_seconds": result.prefill_seconds,
            "decode_seconds": result.decode_seconds,
            "token_ids": result.token_ids,
        }
        tasks.append(task)
        if not quiet:
            print(
                f"{category:9s} {index:3d}/{len(items):3d} "
                f"{item['id']:44s} {'PASS' if task['correct'] else 'FAIL'} "
                f"expected={expected!r} got={actual!r}",
                flush=True,
            )
    return tasks, time.perf_counter() - started


def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(tasks)
    correct = sum(int(t["correct"]) for t in tasks)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
        "wilson_95": wilson_interval(correct, total),
    }


def summarize_mmlu(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summarize(tasks)
    by_subject: dict[str, dict[str, Any]] = {}
    for task in tasks:
        by_subject.setdefault(str(task.get("subject", "unknown")), []).append(task)
    overall["by_subject"] = {
        subject: summarize(subset) for subject, subset in sorted(by_subject.items())
    }
    overall["subjects"] = len(by_subject)
    return overall


def run_native(args, suite, runtime) -> dict[str, Any]:
    limit = max(0, int(args.limit_per_category))
    mmlu = suite["mmlu"][:limit] if limit else suite["mmlu"]
    gsm8k = suite["gsm8k"][:limit] if limit else suite["gsm8k"]
    long_chain = suite["long_chain"][:limit] if limit else suite["long_chain"]

    started = time.perf_counter()
    mmlu_tasks, mmlu_seconds = run_generation(
        runtime, "mmlu", mmlu, choice_prompt, args.mmlu_max_tokens, args.quiet
    )
    gsm_tasks, gsm_seconds = run_generation(
        runtime,
        "gsm8k",
        gsm8k,
        numeric_prompt,
        args.gsm8k_max_tokens,
        args.quiet,
    )
    chain_tasks, chain_seconds = run_generation(
        runtime,
        "long_chain",
        long_chain,
        numeric_prompt,
        args.long_chain_max_tokens,
        args.quiet,
    )
    generation_seconds = time.perf_counter() - started

    perplexity: dict[str, Any] = {"skipped": True}
    if not args.skip_perplexity:
        ppl_started = time.perf_counter()
        ppl_text = suite["perplexity"]["text"]
        scored = runtime.score_text(ppl_text)
        perplexity = {
            **scored,
            "seconds": time.perf_counter() - ppl_started,
            "source": suite["perplexity"].get("source"),
            "declared_tokens": suite["perplexity"].get("tokens"),
        }

    return {
        "mmlu": summarize_mmlu(mmlu_tasks),
        "gsm8k": summarize(gsm_tasks),
        "long_chain": summarize(chain_tasks),
        "perplexity": perplexity,
        "timings": {
            "mmlu_seconds": mmlu_seconds,
            "gsm8k_seconds": gsm_seconds,
            "long_chain_seconds": chain_seconds,
            "generation_seconds": generation_seconds,
        },
        "tasks": {
            "mmlu": mmlu_tasks,
            "gsm8k": gsm_tasks,
            "long_chain": chain_tasks,
        },
    }


def run_llama(args, suite) -> dict[str, Any]:
    limit = max(0, int(args.llama_limit_per_category))
    if limit <= 0:
        return {"skipped": True, "reason": "llama generation comparison disabled"}
    items = {
        "mmlu": suite["mmlu"][:limit],
        "gsm8k": suite["gsm8k"][:limit],
        "long_chain": suite["long_chain"][:limit],
    }
    prompts: list[str] = []
    meta: list[dict[str, Any]] = []
    for category, subset in items.items():
        for item in subset:
            prompts.append(
                choice_prompt(item)
                if category == "mmlu"
                else numeric_prompt(item)
            )
            meta.append(
                {
                    "category": category,
                    "id": item["id"],
                    "expected": normalize(str(item["answer"])),
                }
            )
    started = time.perf_counter()
    responses = _run_llama_cli(
        args.llama_cli,
        args.llama_model,
        prompts,
        threads=args.llama_threads,
        max_new_tokens=max(
            args.mmlu_max_tokens,
            args.gsm8k_max_tokens,
            args.long_chain_max_tokens,
        ),
    )
    tasks: list[dict[str, Any]] = []
    for item_meta, response in zip(meta, responses, strict=True):
        parsed = (
            extract_choice(response)
            if item_meta["category"] == "mmlu"
            else extract_number(response)
        )
        actual = "" if parsed is None else normalize(str(parsed))
        tasks.append(
            {
                **item_meta,
                "response": response,
                "parsed": parsed,
                "correct": actual == item_meta["expected"],
            }
        )
    scores = {
        category: summarize(
            [task for task in tasks if task["category"] == category]
        )
        for category in ("mmlu", "gsm8k", "long_chain")
    }
    result: dict[str, Any] = {
        "scores": scores,
        "tasks": tasks,
        "generation_seconds": time.perf_counter() - started,
    }
    if not args.skip_perplexity:
        try:
            ppl = _run_llama_ppl(
                args.llama_ppl,
                args.llama_model,
                suite["perplexity"]["text"],
                threads=args.llama_threads,
            )
            result["perplexity"] = ppl
        except Exception as exc:
            result["perplexity"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


def main() -> int:
    args = parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": str(args.suite),
        "suite_name": suite.get("name"),
        "suite_counts": suite.get("counts"),
        "suite_sources": suite.get("sources"),
        "suite_sampling": suite.get("sampling"),
        "model": str(args.model),
        "max_cache": args.max_cache,
        "seed": suite.get("seed"),
        "native": {},
        "llama_cpp": {},
    }
    load_started = time.perf_counter()
    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        max_cache=args.max_cache,
        cpu_threads=6,
        expert_chunk_size=8,
        verbose=False,
    )
    try:
        report["device"] = torch.cuda.get_device_name(0)
        report["load_seconds"] = time.perf_counter() - load_started
        report["native"] = run_native(args, suite, runtime)
        if not args.skip_llama:
            try:
                report["llama_cpp"] = run_llama(args, suite)
            except Exception as exc:
                report["llama_cpp"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
    finally:
        runtime.close()

    if not args.quiet:
        native = report["native"]
        print(
            "native MMLU %.3f (%d/%d) GSM8K %.3f (%d/%d) "
            "long-chain %.3f (%d/%d) PPL %s"
            % (
                native["mmlu"]["accuracy"] or 0.0,
                native["mmlu"]["correct"],
                native["mmlu"]["total"],
                native["gsm8k"]["accuracy"] or 0.0,
                native["gsm8k"]["correct"],
                native["gsm8k"]["total"],
                native["long_chain"]["accuracy"] or 0.0,
                native["long_chain"]["correct"],
                native["long_chain"]["total"],
                native["perplexity"].get("perplexity"),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe_json(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
