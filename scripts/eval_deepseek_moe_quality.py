"""Fast quality regression gate for the native DeepSeek-V2-Lite MoE runtime.

This is deliberately a *small* auditable suite rather than full MMLU/GSM8K:
it uses the same Q4_K_M GGUF as llama.cpp, greedy sampling, raw-text
perplexity, MMLU-style multiple choice, GSM8K-style word problems and
multi-step reasoning.  The llama.cpp CLI is optional; native-only runs still
produce a regression JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

DEFAULT_MODEL = Path(
    r"E:\DeepInfer-MoE\models\DeepSeek-V2-Lite-Chat-GGUF"
    r"\DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
)
DEFAULT_META = Path(r"E:\DeepInfer-MoE\meta")
DEFAULT_LLAMA = Path(r"F:\QwenQLoRA\llama.cpp\cpu\llama-cli.exe")
DEFAULT_LLAMA_PPL = Path(r"F:\QwenQLoRA\llama.cpp\cpu\llama-perplexity.exe")
DEFAULT_LLAMA_MODEL = DEFAULT_MODEL
DEFAULT_SUITE = ROOT / "docs" / "quality" / "deepseek_v2_lite_quality_suite.json"
DEFAULT_OUTPUT = ROOT / "docs" / "results" / "deepseek_v2_lite_q4km_moe_quality_eval.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--meta-dir", type=Path, default=DEFAULT_META)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cache", type=int, default=512)
    parser.add_argument("--expert-chunk-size", type=int, default=8)
    parser.add_argument("--gpu-expert-cache-mb", type=int, default=0)
    parser.add_argument("--no-fused-moe", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="limit each task category")
    parser.add_argument("--skip-llama", action="store_true")
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--llama-ppl", type=Path, default=DEFAULT_LLAMA_PPL)
    parser.add_argument("--llama-model", type=Path, default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--llama-threads", type=int, default=6)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _choice_prompt(item: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip("ABCD", item["choices"], strict=True)
    )
    return (
        "Answer the following multiple-choice question. Reply with only one "
        "letter: A, B, C, or D.\n"
        f"Question: {item['question']}\n{choices}\nAnswer:"
    )


def _numeric_prompt(item: dict[str, Any], kind: str) -> str:
    if kind == "gsm8k":
        prefix = (
            "Solve the following grade-school math problem. Reply with only "
            "the final numeric answer, no explanation.\n"
        )
    else:
        prefix = (
            "Solve the following multi-step reasoning problem. Reply with only "
            "the final answer, no explanation.\n"
        )
    return f"{prefix}Problem: {item['question']}\nAnswer:"


def _extract_choice(text: str) -> str | None:
    upper = text.strip().upper()
    matches = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", upper)
    if matches:
        return matches[0]
    first = upper[:1]
    return first if first in "ABCD" else None


def _extract_number(text: str) -> str | None:
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


def _normalize_answer(value: str) -> str:
    value = value.strip()
    try:
        number = float(value)
    except ValueError:
        return value.lower()
    return str(int(number)) if number.is_integer() else str(number)


def _run_native(
    runtime: DeepSeekV2LiteGGUFRuntime,
    suite: dict[str, Any],
    limit: int,
    quiet: bool,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    categories = (
        ("mmlu_style", 64, "choice"),
        ("gsm8k_style", 32, "number"),
        ("long_chain", 32, "number"),
    )
    for category, default_tokens, answer_kind in categories:
        items = suite[category]
        if limit > 0:
            items = items[:limit]
        for item in items:
            prompt = (
                _choice_prompt(item)
                if category == "mmlu_style"
                else _numeric_prompt(item, category)
            )
            result = runtime.generate(
                [{"role": "user", "content": prompt}],
                max_new_tokens=default_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            parsed = (
                _extract_choice(result.text)
                if answer_kind == "choice"
                else _extract_number(result.text)
            )
            expected = _normalize_answer(str(item["answer"]))
            actual = "" if parsed is None else _normalize_answer(str(parsed))
            correct = actual == expected
            tasks.append(
                {
                    "category": category,
                    "id": item["id"],
                    "prompt": prompt,
                    "expected": expected,
                    "response": result.text,
                    "parsed": parsed,
                    "correct": correct,
                    "token_ids": result.token_ids,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                }
            )
            if not quiet:
                print(
                    f"{category:10s} {item['id']:24s} "
                    f"{'PASS' if correct else 'FAIL'} "
                    f"expected={expected!r} got={actual!r}"
                )

    scores: dict[str, dict[str, Any]] = {}
    for category in ("mmlu_style", "gsm8k_style", "long_chain"):
        subset = [task for task in tasks if task["category"] == category]
        scores[category] = {
            "correct": sum(int(task["correct"]) for task in subset),
            "total": len(subset),
            "accuracy": (
                sum(int(task["correct"]) for task in subset) / len(subset)
                if subset
                else None
            ),
        }

    ppl_started = time.perf_counter()
    per_text = []
    total_nll = 0.0
    total_scored = 0
    for item in suite["perplexity_texts"]:
        scored = runtime.score_text(item["text"])
        per_text.append({"id": item["id"], **scored})
        total_nll += float(scored["nll"])
        total_scored += int(scored["scored_tokens"])
    native_ppl = math.exp(total_nll / total_scored) if total_scored else None
    return {
        "scores": scores,
        "perplexity": {
            "perplexity": native_ppl,
            "nll": total_nll,
            "scored_tokens": total_scored,
            "seconds": time.perf_counter() - ppl_started,
            "per_text": per_text,
        },
        "tasks": tasks,
    }


def _run_llama_cli(
    exe: Path,
    model: Path,
    prompts: list[str],
    *,
    threads: int,
    max_new_tokens: int = 64,
) -> list[str]:
    if not exe.exists():
        raise FileNotFoundError(exe)
    # llama-cli is line-oriented in interactive mode.  Keep each prompt on one
    # line and clear chat history between prompts so every item is independent.
    clean = [" ".join(prompt.splitlines()).strip() for prompt in prompts]
    input_parts: list[str] = []
    for prompt in clean:
        input_parts.extend((prompt, "/clear"))
    input_parts.append("/exit")
    command = [
        str(exe),
        "-m",
        str(model),
        "-n",
        str(max_new_tokens),
        "--temp",
        "0",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--seed",
        "42",
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "512",
        "-t",
        str(threads),
        "--no-display-prompt",
        "--simple-io",
        "--no-warmup",
    ]
    completed = subprocess.run(
        command,
        input=("\n".join(input_parts) + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    output = completed.stdout.decode("utf-8", "replace")
    blocks = output.split("[ Prompt:")
    if len(blocks) < len(prompts) + 1:
        raise RuntimeError(
            f"llama-cli returned {len(blocks) - 1} response blocks for "
            f"{len(prompts)} prompts; tail={output[-1000:]!r}"
        )
    responses: list[str] = []
    for block in blocks[: len(prompts)]:
        marker = block.rfind("> ")
        if marker < 0:
            raise RuntimeError(
                f"llama-cli response block has no '> ' marker: {block[-500:]!r}"
            )
        response = block[marker + 2 :].strip()
        responses.append(response)
    return responses


def _run_llama_ppl(
    exe: Path,
    model: Path,
    corpus: str,
    *,
    threads: int,
) -> dict[str, Any]:
    if not exe.exists():
        raise FileNotFoundError(exe)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
        dir=str(Path(tempfile.gettempdir())),
    ) as handle:
        handle.write(corpus)
        corpus_path = Path(handle.name)
    try:
        command = [
            str(exe),
            "-m",
            str(model),
            "-f",
            str(corpus_path),
            "-c",
            "128",
            "--chunks",
            "-1",
            "-t",
            str(threads),
            "--no-warmup",
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
        )
        text = (
            completed.stdout.decode("utf-8", "replace")
            + "\n"
            + completed.stderr.decode("utf-8", "replace")
        )
        match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", text)
        if match is None:
            raise RuntimeError(f"could not parse llama.cpp PPL; tail={text[-1200:]!r}")
        return {
            "perplexity": float(match.group(1)),
            "returncode": completed.returncode,
            "raw_tail": text[-1200:],
        }
    finally:
        corpus_path.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "suite": str(args.suite),
        "model": str(args.model),
        "fused_moe": not args.no_fused_moe,
        "device": None,
        "native": {},
        "llama_cpp": {},
    }
    load_started = time.perf_counter()
    runtime = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta_dir,
        max_cache=args.max_cache,
        cpu_threads=6,
        expert_chunk_size=args.expert_chunk_size,
        gpu_expert_cache_mb=args.gpu_expert_cache_mb,
        fused_moe=not args.no_fused_moe,
        verbose=False,
    )
    try:
        report["device"] = torch.cuda.get_device_name(0)
        report["load_seconds"] = time.perf_counter() - load_started
        native = _run_native(runtime, suite, args.limit, args.quiet)
        report["native"] = native

        prompts: list[str] = []
        prompt_meta: list[dict[str, Any]] = []
        for category, max_tokens, answer_kind in (
            ("mmlu_style", 64, "choice"),
            ("gsm8k_style", 32, "number"),
            ("long_chain", 32, "number"),
        ):
            items = suite[category]
            if args.limit > 0:
                items = items[:args.limit]
            for item in items:
                prompts.append(
                    _choice_prompt(item)
                    if category == "mmlu_style"
                    else _numeric_prompt(item, category)
                )
                prompt_meta.append(
                    {
                        "category": category,
                        "id": item["id"],
                        "expected": _normalize_answer(str(item["answer"])),
                        "answer_kind": answer_kind,
                    }
                )

        if not args.skip_llama:
            try:
                llama_responses = _run_llama_cli(
                    args.llama_cli,
                    args.llama_model,
                    prompts,
                    threads=args.llama_threads,
                )
                llama_tasks = []
                for meta, response in zip(prompt_meta, llama_responses, strict=True):
                    parsed = (
                        _extract_choice(response)
                        if meta["answer_kind"] == "choice"
                        else _extract_number(response)
                    )
                    actual = "" if parsed is None else _normalize_answer(str(parsed))
                    llama_tasks.append(
                        {
                            **meta,
                            "response": response,
                            "parsed": parsed,
                            "correct": actual == meta["expected"],
                        }
                    )
                llama_scores = {}
                for category in ("mmlu_style", "gsm8k_style", "long_chain"):
                    subset = [
                        task for task in llama_tasks if task["category"] == category
                    ]
                    llama_scores[category] = {
                        "correct": sum(int(task["correct"]) for task in subset),
                        "total": len(subset),
                        "accuracy": (
                            sum(int(task["correct"]) for task in subset) / len(subset)
                            if subset
                            else None
                        ),
                    }
                corpus = "\n\n".join(
                    item["text"] for item in suite["perplexity_texts"]
                )
                llama_ppl = _run_llama_ppl(
                    args.llama_ppl,
                    args.llama_model,
                    corpus,
                    threads=args.llama_threads,
                )
                report["llama_cpp"] = {
                    "scores": llama_scores,
                    "perplexity": llama_ppl,
                    "tasks": llama_tasks,
                }
            except Exception as exc:  # keep native regression usable
                report["llama_cpp"] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        runtime.close()

    if not args.quiet:
        native_scores = report["native"]["scores"]
        print(
            "native  MMLU-style %.3f  GSM8K-style %.3f  long-chain %.3f  PPL %.3f"
            % (
                native_scores["mmlu_style"]["accuracy"] or 0.0,
                native_scores["gsm8k_style"]["accuracy"] or 0.0,
                native_scores["long_chain"]["accuracy"] or 0.0,
                report["native"]["perplexity"]["perplexity"] or 0.0,
            )
        )
        if "scores" in report["llama_cpp"]:
            llama_scores = report["llama_cpp"]["scores"]
            print(
                "llama.cpp MMLU-style %.3f  GSM8K-style %.3f  long-chain %.3f  PPL %.3f"
                % (
                    llama_scores["mmlu_style"]["accuracy"] or 0.0,
                    llama_scores["gsm8k_style"]["accuracy"] or 0.0,
                    llama_scores["long_chain"]["accuracy"] or 0.0,
                    report["llama_cpp"]["perplexity"]["perplexity"] or 0.0,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
