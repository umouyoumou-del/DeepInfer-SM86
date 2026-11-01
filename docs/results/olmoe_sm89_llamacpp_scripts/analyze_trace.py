"""Per-decode-step GPU time by kernel, from kernel_trace.cpp CSV output.

A decode step is one CUDA-graph replay: its kernels share the correlation id
of the cudaGraphLaunch.  Steps are the replays whose kernel count is the most
common one; the first ``--skip`` of them are dropped as warm-up.
"""
import argparse
import collections
import json
import statistics
import subprocess


def load(path):
    kernels = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("K,"):
                continue
            _, start, end, _stream, graph, corr, grid, block, name = line.rstrip("\n").split(",", 8)
            kernels.append((int(start), int(end), int(graph), int(corr), grid, block, name))
    return kernels


def demangle(names):
    proc = subprocess.run(["/usr/local/cuda/bin/cu++filt"], input="\n".join(names),
                          capture_output=True, text=True, check=True)
    return dict(zip(names, proc.stdout.splitlines()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--replays-per-step", type=int, default=1,
                        help="consecutive graph replays that make one decode step")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json")
    args = parser.parse_args()

    kernels = load(args.trace)
    replays = collections.defaultdict(list)
    for kernel in kernels:
        if kernel[2]:
            replays[kernel[3]].append(kernel)
    sizes = collections.Counter(len(v) for v in replays.values())
    print("graph replays by kernel count:", dict(sizes.most_common(6)))
    shapes = {size for size, _ in sizes.most_common(args.replays_per_step)}
    ordered = [v for _, v in sorted(replays.items(), key=lambda kv: min(k[0] for k in kv[1]))
               if len(v) in shapes]
    group = args.replays_per_step
    steps = [[k for replay in ordered[i:i + group] for k in replay]
             for i in range(0, len(ordered) - group + 1, group)]
    typical = collections.Counter(len(s) for s in steps).most_common(1)[0][0]
    steps = [s for s in steps if len(s) == typical][args.skip:]
    busy = [sum(k[1] - k[0] for k in s) / 1e3 for s in steps]
    span = [(max(k[1] for k in s) - min(k[0] for k in s)) / 1e3 for s in steps]
    print(f"steps {len(steps)}, kernels/step {typical}, busy {statistics.median(busy):.1f} us,"
          f" span {statistics.median(span):.1f} us (median)")

    per_name = collections.defaultdict(list)
    for step in steps:
        acc = collections.defaultdict(lambda: [0, 0])
        for k in step:
            acc[(k[6], k[4], k[5])][0] += k[1] - k[0]
            acc[(k[6], k[4], k[5])][1] += 1
        for name, (dur, count) in acc.items():
            per_name[name].append((dur, count))
    names = demangle(sorted({key[0] for key in per_name}))
    rows = []
    for (name, grid, block), values in per_name.items():
        us = statistics.median(v[0] for v in values) / 1e3
        count = statistics.median(v[1] for v in values)
        rows.append({"name": names[name], "mangled": name, "grid": grid, "block": block,
                     "us_per_step": us, "calls_per_step": count})
    rows.sort(key=lambda r: -r["us_per_step"])
    total = sum(r["us_per_step"] for r in rows)
    for r in rows[:args.top]:
        print(f"{r['us_per_step']:8.1f} us {100 * r['us_per_step'] / total:5.1f}%"
              f"  x{r['calls_per_step']:<5g} g{r['grid']:<12} b{r['block']:<9} {r['name'][:110]}")
    print(f"{total:8.1f} us total (sum of per-name medians)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"trace": args.trace, "steps": len(steps), "kernels_per_step": typical,
                       "busy_us": busy, "span_us": span, "kernels": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
