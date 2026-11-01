#!/usr/bin/env python3
"""Interleaved rounds: llama-bench against bench_olmoe_decode.py (graph + fused).

usage: drive_fused.py TAG ROUND llama-first|deepinfer-first

Each run is bracketed by the cgroup's cpu.stat; the DeepInfer bench records its
own in-process CPU probe per repeat.
"""
import json
import os
import statistics
import subprocess
import sys
import time

WORK = "/root/autodl-tmp/fused-bench"
LLAMA = ("/root/autodl-tmp/llamacpp-vs-deepinfer/llama.cpp-"
         "ddf48f85c933c1f438212a02eb21e7b001d2d37d/build/bin/llama-bench")
MODEL = "/root/autodl-tmp/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
REPO = "/root/autodl-tmp/DeepInfer-SM86"
PY = "/root/miniconda3/bin/python"


def cpu_stat():
    with open("/sys/fs/cgroup/cpu.stat", encoding="utf-8") as handle:
        return {k: int(v) for k, v in (line.split() for line in handle)}


def run(label, cmd, env=None):
    stat0, began = cpu_stat(), time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False,
                          stdin=subprocess.DEVNULL, env={**os.environ, **(env or {})})
    stat1 = cpu_stat()
    record = {"label": label, "cmd": cmd, "returncode": proc.returncode,
              "wall_s": time.time() - began,
              "throttled_periods": stat1["nr_throttled"] - stat0["nr_throttled"],
              "stdout": proc.stdout, "stderr": proc.stderr[-4000:]}
    print(f"[{label}] rc={proc.returncode} wall={record['wall_s']:.1f}s "
          f"throttled={record['throttled_periods']}", flush=True)
    return record


def llama(label):
    record = run(label, [LLAMA, "-m", MODEL, "-ngl", "99", "-p", "0", "-d", "5",
                         "-n", "32,128", "-fa", "on", "-t", "16", "-r", "10", "-o", "json"])
    record["results"] = json.loads(record["stdout"])
    for res in record["results"]:
        per_rep = [res["n_gen"] * 1e9 / ns for ns in res["samples_ns"]]
        res["median_ts"] = statistics.median(per_rep)
        print(f"   llama.cpp tg{res['n_gen']}@d5  median {res['median_ts']:7.2f} tok/s"
              f"  ({1e3 / res['median_ts']:.3f} ms)  avg {res['avg_ts']:.2f}", flush=True)
    return record


def deepinfer(label):
    out = f"{WORK}/{label}.json"
    record = run(label, [PY, REPO + "/scripts/bench_olmoe_decode.py", "--gguf", MODEL,
                         "--mode", "graph", "fused", "--expert-kernel", "gemv",
                         "--repeats", "5", "--output", out], {"PYTHONPATH": REPO})
    for line in record["stdout"].splitlines():
        if line.startswith(("L=", "  text:")):
            print("  ", line, flush=True)
    with open(out, encoding="utf-8") as handle:
        record["bench"] = json.load(handle)
    return record


def main():
    tag, number, order = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    sides = [lambda: llama(f"{tag}_r{number}_llama"),
             lambda: deepinfer(f"{tag}_r{number}_deepinfer")]
    if order == "deepinfer-first":
        sides.reverse()
    records = [side() for side in sides]
    with open(f"{WORK}/{tag}_round{number}.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)


if __name__ == "__main__":
    main()
