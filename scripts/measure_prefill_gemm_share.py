import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, r"D:\DeepInfer-SM86")
from deepinfer.model_executor.models import qwen2 as Q, qwen2_w4a16 as W
from deepinfer.hugging_face import qwen2_config as C

MODEL = Path(r"D:\Qwen2.5-3B-DeepInfer-SM86\model")
h = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
cfg = C.build_qwen2_model_config(h, MODEL / "config.json")
print("loading dense on CPU...", flush=True)
t0 = time.perf_counter()
dense = Q.Qwen2ForCausalLM.load_model(str(MODEL), cfg, device="cpu")
print("dense loaded %.1fs" % (time.perf_counter() - t0), flush=True)
model = W.Qwen2W4A16ForCausalLM.from_dense_model(dense, device="cuda", group_size=128)
del dense
print("quantized", flush=True)

orig = W._native_w4a16_tiled_gemm
acc = {"ms": 0.0, "calls": 0}
def timed(*a, **kw):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); out = orig(*a, **kw); e.record(); e.synchronize()
    acc["ms"] += s.elapsed_time(e); acc["calls"] += 1
    return out
W._native_w4a16_tiled_gemm = timed

res = []
for L in (128, 256):
    ids = torch.randint(0, 1000, (1, L), device="cuda")
    with torch.inference_mode():
        model(ids, use_cache=True)
        acc["ms"] = 0.0; acc["calls"] = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model(ids, use_cache=True)
        torch.cuda.synchronize(); total = (time.perf_counter() - t0) * 1e3
    res.append({"tokens": L, "total_ms": round(total,1), "gemm_ms": round(acc["ms"],1),
                "calls": acc["calls"], "gemm_share_pct": round(100*acc["ms"]/total,1)})
print(json.dumps({"results": res, "peak_gib": round(torch.cuda.max_memory_allocated()/2**30,2)},
                 ensure_ascii=False, indent=2))
