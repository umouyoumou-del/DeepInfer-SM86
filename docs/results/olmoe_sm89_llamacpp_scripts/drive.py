#!/usr/bin/env python3
"""llama.cpp vs DeepInfer OLMoE decode on one host, interleaved.

Every run is bracketed by the pure-Python CPU probe that
scripts/bench_olmoe_decode.py records and by the cgroup's cpu.stat, so runs
taken in the host's slow state, or under CPU-quota throttling, can be found
afterwards.

usage: drive.py explore | round N [llama-first|deepinfer-first] THREADS
"""
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

WORK = "/root/autodl-tmp/llamacpp-vs-deepinfer"
BIN = glob.glob(WORK + "/llama.cpp-*/build/bin")[0]
MODEL = "/root/autodl-tmp/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
REPO = "/root/autodl-tmp/DeepInfer-SM86"
PY = "/root/miniconda3/bin/python"
PROMPT = "The capital of France is"
OUT = WORK + "/results"


def cpu_probe_ms():
    # identical to scripts/bench_olmoe_decode.py:cpu_probe_ms
    began = time.perf_counter()
    total = 0
    for value in range(300_000):
        total += value * value % 7
    return (time.perf_counter() - began) * 1e3


def cpu_stat():
    with open("/sys/fs/cgroup/cpu.stat", encoding="utf-8") as handle:
        return {k: int(v) for k, v in (line.split() for line in handle)}


def run(label, cmd, env=None, timeout=3600):
    probe0, stat0, began = cpu_probe_ms(), cpu_stat(), time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False,
                          stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    wall, stat1, probe1 = time.time() - began, cpu_stat(), cpu_probe_ms()
    record = {
        "label": label, "cmd": cmd, "env": env or {}, "returncode": proc.returncode,
        "wall_s": wall, "cpu_probe_ms": [probe0, probe1],
        "throttled_periods": stat1["nr_throttled"] - stat0["nr_throttled"],
        "throttled_ms": (stat1["throttled_usec"] - stat0["throttled_usec"]) / 1e3,
        "periods": stat1["nr_periods"] - stat0["nr_periods"],
        "cpu_s": (stat1["usage_usec"] - stat0["usage_usec"]) / 1e6,
        "stdout": proc.stdout, "stderr": proc.stderr,
    }
    print(f"[{label}] rc={proc.returncode} wall={wall:.1f}s cpu={record['cpu_s']:.1f}s"
          f" probe={probe0:.1f}/{probe1:.1f} ms throttled={record['throttled_periods']}"
          f"/{record['periods']} periods ({record['throttled_ms']:.0f} ms)", flush=True)
    if proc.returncode:
        print(proc.stderr[-3000:], flush=True)
    return record


def llama_bench(label, args, env=None):
    record = run(label, [BIN + "/llama-bench", "-m", MODEL, "-ngl", "99", "-o", "json", *args],
                 env)
    try:
        record["results"] = json.loads(record["stdout"])
    except json.JSONDecodeError:
        record["results"] = []
    for res in record["results"]:
        per_rep = [res["n_gen"] * 1e9 / ns if res["n_gen"] else res["n_prompt"] * 1e9 / ns
                   for ns in res["samples_ns"]]
        res["median_ts"] = statistics.median(per_rep)
        kind = f"tg{res['n_gen']}" if res["n_gen"] else f"pp{res['n_prompt']}"
        print(f"   t={res['n_threads']:>2} fa={res['flash_attn']!s:5} {kind:6s} d={res['n_depth']}"
              f"  avg {res['avg_ts']:8.2f} ± {res['stddev_ts']:6.2f} t/s"
              f"  median {res['median_ts']:8.2f}  ({1e3 / res['median_ts']:.3f} ms/token)"
              f"  reps {len(per_rep)}", flush=True)
    return record


def llama_greedy(label, threads, fa):
    record = run(label, [BIN + "/llama-completion", "-m", MODEL, "-ngl", "99", "-p", PROMPT,
                         "-n", "33", "--temp", "0", "--top-k", "1", "--ignore-eos", "-no-cnv",
                         "--verbose-prompt", "--no-display-prompt", "--simple-io",
                         "-t", str(threads), "-fa", fa, "--seed", "1"])
    match = re.search(r"eval time =\s*([\d.]+) ms /\s*(\d+) runs", record["stderr"])
    prompt_ids = re.findall(r"^\s*(\d+) -> '", record["stderr"], re.MULTILINE)
    record["eval_ms"], record["eval_runs"] = (float(match[1]), int(match[2])) if match else (None, None)
    record["prompt_ids"] = [int(i) for i in prompt_ids]
    record["text"] = record["stdout"]
    if match:
        per_token = record["eval_ms"] / record["eval_runs"]
        print(f"   greedy eval {per_token:.3f} ms/token ({1e3 / per_token:.2f} t/s) over"
              f" {record['eval_runs']} runs; prompt ids {record['prompt_ids']}", flush=True)
    print(f"   text: {record['text']!r}", flush=True)
    return record


def deepinfer(label, extra):
    out = f"{OUT}/{label}.json"
    record = run(label, [PY, REPO + "/scripts/bench_olmoe_decode.py", "--gguf", MODEL,
                         "--repeats", "5", "--output", out, *extra], {"PYTHONPATH": REPO})
    for line in record["stdout"].splitlines():
        if line.startswith(("L=", "  text:", "loaded")):
            print("  ", line, flush=True)
    return record


def save(name, records):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "explore":
        records = [
            llama_bench("explore_threads", ["-p", "0", "-n", "128", "-t", "1,2,4,8,16,64",
                                            "-fa", "off,on", "-r", "3"]),
            llama_bench("explore_nograph", ["-p", "0", "-n", "128", "-t", "4,16,64",
                                            "-fa", "on", "-r", "3"],
                        {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
            llama_greedy("explore_greedy", 16, "on"),
        ]
        save("explore", records)
        return
    number, order, threads = int(sys.argv[2]), sys.argv[3], sys.argv[4]

    def llama_side():
        return [
            llama_bench(f"r{number}_llama_bench", ["-p", "0", "-d", "5", "-n", "32,128",
                                                   "-t", f"{threads},64", "-fa", "on,off",
                                                   "-r", "10"]),
            llama_bench(f"r{number}_llama_nograph", ["-p", "0", "-d", "5", "-n", "32",
                                                     "-t", threads, "-fa", "on", "-r", "10"],
                        {"GGML_CUDA_DISABLE_GRAPHS": "1"}),
            llama_bench(f"r{number}_llama_pp", ["-p", "512", "-n", "0", "-t", threads,
                                                "-fa", "on", "-r", "5"]),
            llama_greedy(f"r{number}_llama_greedy", threads, "on"),
        ]

    def deepinfer_side():
        return [deepinfer(f"r{number}_deepinfer", ["--mode", "graph", "eager",
                                                   "--expert-kernel", "gemv", "grouped"])]

    sides = [llama_side, deepinfer_side]
    if order == "deepinfer-first":
        sides.reverse()
    records = [record for side in sides for record in side()]
    save(f"round{number}", records)


if __name__ == "__main__":
    main()
