export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/DeepInfer-SM86 || exit 1
cat > /tmp/genc.py <<'PY'
import sys, time, torch, numpy as np
sys.path.insert(0,"/root/autodl-tmp/DeepInfer-SM86")
from transformers import AutoTokenizer
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.olmoe_spec import spec_from_gguf
from deepinfer.moe.olmoe_weights import layer_tensors, load_layer_tensor
from deepinfer.moe.olmoe_packed import load_packed_experts
from deepinfer.moe.olmoe_layer import OLMoEKVKache
from deepinfer.moe.olmoe_model import forward
P="/root/autodl-tmp/models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
idx=GGUFIndex(P); spec=spec_from_gguf(idx)
tok=AutoTokenizer.from_pretrained("/root/autodl-tmp/meta_olmoe")
dev="cuda"; dt=torch.float16
print("loading...", flush=True)
layers=[]
for L in range(spec.block_count):
    t=layer_tensors(spec,L); d={}
    for role in list(t.attention)+list(t.norms)+["router"]:
        d[role]=load_layer_tensor(idx,spec,L,role,dtype=torch.float32).to(dev,dt)
    d["packed"]={}
    for role,sfx,out in (("gate","ffn_gate_exps",1024),("up","ffn_up_exps",1024),("down","ffn_down_exps",2048)):
        d["packed"][role]=load_packed_experts(idx,f"blk.{L}.{sfx}.weight",experts=spec.expert_count,out_features=out)
    d["packed_gate_out"]=1024
    for role,sfx in (("expert_gate","ffn_gate_exps"),("expert_up","ffn_up_exps"),("expert_down","ffn_down_exps")):
        parts=[idx.expert_tensor(f"blk.{L}.{sfx}.weight",e,expert_count=spec.expert_count) for e in range(spec.expert_count)]
        d[role]=torch.from_numpy(np.stack(parts)).to(dev,dt); del parts
    layers.append(d)
emb=torch.from_numpy(idx.read_text("token_embd.weight") if False else idx.read_tensor("token_embd.weight")).to(dev,dt)
outw=torch.from_numpy(idx.read_tensor("output.weight")).to(dev,dt)
fin=torch.from_numpy(idx.read_tensor("output_norm.weight")).to(dev,dt)
W={"embed_tokens":emb,"lm_head":outw,"final_norm":fin}
ids=tok("The capital of France is",return_tensors="pt")["input_ids"].to(dev)
cache=OLMoEKVKache()
with torch.no_grad():
    lg=forward(ids,W,layers,num_heads=spec.head_count,eps=spec.rms_eps,rope_base=spec.rope_base,
               top_k=spec.expert_used_count,cache=cache)
# free the fp16 experts; decode uses the packed pointers only
for L in layers:
    for k in ("expert_gate","expert_up","expert_down"): L.pop(k,None)
torch.cuda.empty_cache()
print("after prefill, VRAM %.1f GiB" % (torch.cuda.memory_allocated()/2**30), flush=True)
gen=[int(lg[0,-1].argmax(-1))]
with torch.no_grad():
    t0=time.perf_counter()
    for _ in range(12):
        nxt=torch.tensor([[gen[-1]]],device=dev)
        lg=forward(nxt,W,layers,num_heads=spec.head_count,eps=spec.rms_eps,rope_base=spec.rope_base,
                   top_k=spec.expert_used_count,cache=cache)
        gen.append(int(lg[0,-1].argmax(-1)))
el=time.perf_counter()-t0
print("generated ids:", gen)
print("text:", repr(tok.decode(gen)))
print(f"decode {el/12*1000:.1f} ms/token ({12/el:.2f} tok/s)")
PY
PYTHONPATH=$PWD python /tmp/genc.py 2>&1 | tail -8
echo GENC_DONE
