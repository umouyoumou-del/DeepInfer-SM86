"""Benchmark 3B native, 7B native and Ollama sequentially on the same prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_3B = Path(r"D:\Qwen3B-W4A16-SM86")
DEFAULT_7B = Path(r"D:\Qwen7B-Mixed-SM86")
DEFAULT_OLLAMA = "maibot-catgirl-v3:latest"
DEFAULT_OUTPUT = ROOT / "docs/results/productivity_backends_benchmark.json"

SYSTEM = "你是一个简洁、可靠的中文个人助理。只输出最终答案，不要复述任务。"
PROMPTS = [
    "用一句话解释 CUDA Graph。",
    "把下面这段话压缩成一句待办：明天下午三点前把实验报告发给导师，并附上结果表格。",
]
MAX_TOKENS = 64
REPEATS = 2
SEED = 20261026


def http_json(
    url: str, payload: dict[str, Any] | None = None, timeout: int = 300
) -> dict[str, Any]:
    data = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(url: str, timeout: float = 180.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return http_json(url + "/health", timeout=5)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last_error}")


def stop_process(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def start_native(
    checkpoint: Path,
    backend: str,
    model_id: str,
    port: int,
    log_path: Path,
) -> tuple[subprocess.Popen[Any], dict[str, Any], float]:
    log = log_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "apps/server.py",
            "--checkpoint",
            str(checkpoint),
            "--backend",
            backend,
            "--model-id",
            model_id,
            "--max-cache",
            "256",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**dict(__import__("os").environ), "PYTHONIOENCODING": "utf-8"},
    )
    try:
        health = wait_health(f"http://127.0.0.1:{port}", timeout=180)
    except Exception:
        stop_process(proc)
        log.close()
        raise
    return proc, health, time.perf_counter() - started


def native_request(
    port: int, prompt: str, max_tokens: int, temperature: float = 0.0
) -> tuple[dict[str, Any], dict[str, Any], float]:
    body = {
        "model": "benchmark",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "stream": False,
    }
    started = time.perf_counter()
    response = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    wall = time.perf_counter() - started
    health = http_json(f"http://127.0.0.1:{port}/health")
    return response, health, wall


def run_native(
    label: str,
    checkpoint: Path,
    backend: str,
    model_id: str,
    port: int,
    log_path: Path,
) -> dict[str, Any]:
    proc, health, load_seconds = start_native(
        checkpoint, backend, model_id, port, log_path
    )
    runs: list[dict[str, Any]] = []
    try:
        native_request(port, PROMPTS[0], 16)
        for repeat in range(REPEATS):
            for prompt_index, prompt in enumerate(PROMPTS):
                response, timing, wall = native_request(port, prompt, MAX_TOKENS)
                choice = response["choices"][0]
                usage = response.get("usage", {})
                last = timing.get("last_timings", {})
                runs.append(
                    {
                        "repeat": repeat,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "wall_seconds": round(wall, 6),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "prefill_seconds": last.get("prefill_seconds"),
                        "decode_seconds": last.get("decode_seconds"),
                        "decode_tokens_per_second": last.get(
                            "decode_tokens_per_second"
                        ),
                        "text": choice["message"]["content"],
                    }
                )
    finally:
        stop_process(proc)
        if proc.stdout is not None:
            proc.stdout.close()
    measured = [r for r in runs if r["decode_tokens_per_second"] is not None]
    return {
        "label": label,
        "model_id": health.get("model_id", model_id),
        "backend": backend,
        "checkpoint": str(checkpoint),
        "load_seconds": round(load_seconds, 3),
        "gpu": health.get("gpu"),
        "max_cache": health.get("max_cache"),
        "runs": runs,
        "summary": {
            "median_wall_seconds": round(
                statistics.median(r["wall_seconds"] for r in measured), 4
            ),
            "median_prefill_seconds": round(
                statistics.median(r["prefill_seconds"] for r in measured), 4
            ),
            "median_decode_tokens_per_second": round(
                statistics.median(r["decode_tokens_per_second"] for r in measured), 3
            ),
            "median_completion_tokens": round(
                statistics.median(r["completion_tokens"] for r in measured), 3
            ),
        },
    }


def stop_ollama(model: str) -> None:
    try:
        subprocess.run(
            ["ollama", "stop", model], check=False, capture_output=True, timeout=30
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ollama stop failed: {exc}", flush=True)


def ollama_request(
    model: str, prompt: str, max_tokens: int
) -> tuple[dict[str, Any], float]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 50,
            "seed": SEED,
        },
    }
    started = time.perf_counter()
    response = http_json("http://127.0.0.1:11434/api/chat", body, timeout=600)
    return response, time.perf_counter() - started


def run_ollama(model: str) -> dict[str, Any]:
    stop_ollama(model)
    warm, _ = ollama_request(model, PROMPTS[0], 16)
    runs: list[dict[str, Any]] = []
    for repeat in range(REPEATS):
        for prompt_index, prompt in enumerate(PROMPTS):
            response, wall = ollama_request(model, prompt, MAX_TOKENS)
            load_seconds = response.get("load_duration", 0) / 1e9
            prefill_seconds = response.get("prompt_eval_duration", 0) / 1e9
            decode_seconds = response.get("eval_duration", 0) / 1e9
            completion = int(response.get("eval_count", 0))
            runs.append(
                {
                    "repeat": repeat,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "wall_seconds": round(wall, 6),
                    "prompt_tokens": response.get("prompt_eval_count"),
                    "completion_tokens": completion,
                    "load_seconds": round(load_seconds, 6),
                    "prefill_seconds": round(prefill_seconds, 6),
                    "decode_seconds": round(decode_seconds, 6),
                    "decode_tokens_per_second": round(completion / decode_seconds, 3)
                    if decode_seconds > 0
                    else None,
                    "ttft_proxy_seconds": round(load_seconds + prefill_seconds, 6),
                    "text": response.get("message", {}).get("content", ""),
                }
            )
    stop_ollama(model)
    measured = [r for r in runs if r["decode_tokens_per_second"] is not None]
    return {
        "label": "ollama_3b_q4km",
        "model_id": model,
        "backend": "ollama",
        "model_detail": warm.get("model"),
        "runs": runs,
        "summary": {
            "median_wall_seconds": round(
                statistics.median(r["wall_seconds"] for r in measured), 4
            ),
            "median_prefill_seconds": round(
                statistics.median(r["prefill_seconds"] for r in measured), 4
            ),
            "median_decode_tokens_per_second": round(
                statistics.median(r["decode_tokens_per_second"] for r in measured), 3
            ),
            "median_completion_tokens": round(
                statistics.median(r["completion_tokens"] for r in measured), 3
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-b-checkpoint", type=Path, default=DEFAULT_3B)
    parser.add_argument("--seven-b-checkpoint", type=Path, default=DEFAULT_7B)
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-3b", action="store_true")
    parser.add_argument("--skip-7b", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
            "vram_gib": 4.0,
            "os": "Windows",
        },
        "protocol": {
            "system": SYSTEM,
            "prompts": PROMPTS,
            "max_tokens": MAX_TOKENS,
            "repeats": REPEATS,
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 50,
            "seed": SEED,
            "native_context": 256,
            "note": "3B native includes maibot-catgirl-v3 LoRA; 7B native is the base model; Ollama is maibot-catgirl-v3 Q4_K_M.",
        },
        "backends": {},
    }
    if not args.skip_3b:
        report["backends"]["qwen3b_native"] = run_native(
            "qwen3b_w4a16_native",
            args.three_b_checkpoint,
            "w4a16",
            "deepinfer-sm86-3b-w4a16",
            18001,
            ROOT / "bench_3b.log",
        )
    if not args.skip_7b:
        report["backends"]["qwen7b_native"] = run_native(
            "qwen7b_mixed_native",
            args.seven_b_checkpoint,
            "mixed",
            "deepinfer-sm86-7b-mixed",
            18002,
            ROOT / "bench_7b.log",
        )
    if not args.skip_ollama:
        report["backends"]["ollama_3b"] = run_ollama(args.ollama_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {name: data["summary"] for name, data in report["backends"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
