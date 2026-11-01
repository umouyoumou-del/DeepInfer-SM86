"""Group the per-kernel trace summaries into categories; print and save the table."""
import json
import re
import statistics
import sys

RULES = {
    "llama.cpp": [
        ("expert GEMV gate+up", lambda n, g: "mul_mat_vec_q" in n and g.startswith("1024x8")),
        ("expert GEMV down", lambda n, g: "mul_mat_vec_q" in n and g.startswith("512x8")),
        ("dense GEMV attention", lambda n, g: "mul_mat_vec_q" in n and g.startswith("2048x1")),
        ("dense GEMV lm_head", lambda n, g: "mul_mat_vec_q" in n and g.startswith("50304")),
        ("activation quantize (q8_1)", lambda n, g: "quantize_q8_1" in n),
        ("attention", lambda n, g: "flash_attn" in n),
        ("RMSNorm", lambda n, g: "rms_norm" in n),
        ("RoPE", lambda n, g: "rope_" in n),
        ("router + top-k + combine", lambda n, g: re.search(r"mul_mat_vec_f|topk_moe|moe_weighted", n)),
        ("KV write", lambda n, g: "set_rows" in n),
        ("residual add / other", lambda n, g: True),
    ],
    "DeepInfer": [
        ("expert GEMV gate+up", lambda n, g: "pg_gate_up_gemv" in n),
        ("expert GEMV down", lambda n, g: "pg_down_gemv_combine" in n),
        ("dense GEMV attention", lambda n, g: "gemvx" in n and g.startswith("512x1")),
        ("dense GEMV lm_head", lambda n, g: "gemvx" in n and g.startswith("12576")),
        ("dtype casts", lambda n, g: "direct_copy_kernel" in n),
        ("attention", lambda n, g: "fmha_cutlass" in n),
        ("attention mask", lambda n, g: re.search(r"FillFunctor<c10::Half>|where_kernel|logical_not", n)),
        ("RMSNorm", lambda n, g: re.search(r"MeanOps<float|pow_tensor_scalar|rsqrt|CUDAFunctorOnSelf_add<float>|BinaryFunctor<float, float, float", n)),
        ("RoPE", lambda n, g: re.search(r"BinaryFunctor<c10::Half|CatArrayBatchedCopy|indexSelectSmallIndex", n) or ("CUDAFunctor_add<c10::Half>" in n and g.startswith("2x1"))),
        ("router + top-k + combine", lambda n, g: re.search(r"dot_kernel|reduce_1Block|gatherTopK|bitonicSort|softmax_warp", n)),
        ("KV write", lambda n, g: "index_copy" in n and g.startswith("4x1")),
        ("residual add / other", lambda n, g: True),
    ],
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def bandwidth(entry):
    return f"{entry['gb_s']:6.0f}" if "gb_s" in entry else "      "


def categorize(path, rules):
    data = read_json(path)
    table = {}
    for row in data["kernels"]:
        for category, rule in rules:
            if rule(row["name"], row["grid"]):
                entry = table.setdefault(category, {"us": 0.0, "kernels": 0})
                entry["us"] += row["us_per_step"]
                entry["kernels"] += row["calls_per_step"]
                break
    return data, table


bytes_mb = read_json(sys.argv[3])
moved = {
    "llama.cpp": {"expert GEMV gate+up": bytes_mb["quantized_mb"]["gate_up"],
                  "expert GEMV down": bytes_mb["quantized_mb"]["down"],
                  "dense GEMV attention": bytes_mb["quantized_mb"]["attn"],
                  "dense GEMV lm_head": bytes_mb["quantized_mb"]["lm_head"]},
    "DeepInfer": {"expert GEMV gate+up": bytes_mb["quantized_mb"]["gate_up"],
                  "expert GEMV down": bytes_mb["quantized_mb"]["down"],
                  "dense GEMV attention": bytes_mb["fp16_mb"]["attn"],
                  "dense GEMV lm_head": bytes_mb["fp16_mb"]["lm_head"]},
}
result = {}
for engine, path in (("DeepInfer", sys.argv[1]), ("llama.cpp", sys.argv[2])):
    data, table = categorize(path, RULES[engine])
    for category, entry in table.items():
        if category in moved[engine]:
            entry["mb"] = moved[engine][category]
            entry["gb_s"] = entry["mb"] / entry["us"] * 1e3  # MB/us -> GB/s
    result[engine] = {"kernels_per_step": data["kernels_per_step"], "steps": data["steps"],
                      "busy_us": statistics.median(data["busy_us"]),
                      "span_us": statistics.median(data["span_us"]), "categories": table}
order = [c for c, _ in RULES["DeepInfer"]] + [c for c, _ in RULES["llama.cpp"]]
order = list(dict.fromkeys(order))
print(f"{'category':28s} {'DeepInfer us':>12s} {'k':>5s} {'GB/s':>6s} | {'llama.cpp us':>12s} {'k':>5s} {'GB/s':>6s} | {'diff us':>8s}")
for c in order:
    d = result["DeepInfer"]["categories"].get(c, {"us": 0, "kernels": 0})
    l = result["llama.cpp"]["categories"].get(c, {"us": 0, "kernels": 0})
    print(f"{c:28s} {d['us']:12.1f} {d['kernels']:5g} {bandwidth(d)} | {l['us']:12.1f}"
          f" {l['kernels']:5g} {bandwidth(l)} | {d['us'] - l['us']:8.1f}")
for e in ("DeepInfer", "llama.cpp"):
    r = result[e]
    print(f"{e}: {r['kernels_per_step']} kernels/step, busy {r['busy_us']:.1f} us, span {r['span_us']:.1f} us, steps {r['steps']}")
with open(sys.argv[4], "w", encoding="utf-8") as out:
    json.dump(result, out, indent=1)
