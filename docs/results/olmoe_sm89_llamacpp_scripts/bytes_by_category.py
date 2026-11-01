"""Bytes one decode token reads, by category, as stored (llama.cpp) and as DeepInfer holds them."""
import json
import sys

sys.path.insert(0, "/root/autodl-tmp/DeepInfer-SM86")
from deepinfer.moe.gguf_index import GGUFIndex

index = GGUFIndex(sys.argv[1])
top_k, experts = 8, 64
quant = {"attn": 0, "lm_head": 0, "gate_up": 0, "down": 0, "small": 0}
fp16 = {"attn": 0, "lm_head": 0}
for name, t in index.tensors.items():
    if name.endswith(("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")):
        quant["attn"] += t.nbytes
        fp16["attn"] += t.element_count * 2
    elif name == "output.weight":
        quant["lm_head"] += t.nbytes
        fp16["lm_head"] += t.element_count * 2
    elif "ffn_gate_exps" in name or "ffn_up_exps" in name:
        quant["gate_up"] += t.nbytes * top_k // experts
    elif "ffn_down_exps" in name:
        quant["down"] += t.nbytes * top_k // experts
    elif name == "token_embd.weight":
        quant["small"] += t.nbytes // t.dims[1]
    else:
        quant["small"] += t.nbytes  # norms, router: F32
print(json.dumps({"quantized_mb": {k: v / 1e6 for k, v in quant.items()},
                  "fp16_mb": {k: v / 1e6 for k, v in fp16.items()},
                  "total_llamacpp_gb": sum(quant.values()) / 1e9}, indent=1))
