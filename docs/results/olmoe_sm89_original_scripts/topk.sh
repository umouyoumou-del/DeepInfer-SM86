export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/DeepInfer-SM86 || exit 1
cat > /tmp/topk.py <<'PY'
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
layers=[]
for L in range(spec.block_count):
    t=layer_tensors(spec,L); d={}
    for role in list(t.attention)+list(t.norms)+["router"]:
        d[role]=load_layer_tensor(idx,spec,L,role,dtype=torch.float32).to(dev,dt)
    d["packed"]={}
    for role,sfx,out in (("gate","ffn_gate_exps",1024),("up","ffn_up_exps",1024),("down","ffn_down_exps",2048)):
        d["packed"][role]=load_packed_experts(idx,f"blk.{L}.{sfx}.weight",experts=spec.expert_count,out_features=out)
    d["packed_gate_out"]=1024
    layers.append(d)
emb=torch.from_numpy(idx.read_tensor("token_embd.weight")).to(dev,dt)
outw=torch.from_numpy(idx.read_tensor("output.weight")).to(dev,dt)
fin=torch.from_numpy(idx.read_tensor("output_norm.weight")).to(dev,dt)
W={"embed_tokens":emb,"lm_head":outw,"final_norm":fin}
ids=tok("The capital of France is",return_tensors="pt")["input_ids"].to(dev)[:, :1]
# bytes per token scales with top_k: 16 layers x k experts x 12.58 MB (fp16-equivalent packed)
PER_EXPERT = (1024*2048 + 1024*2048 + 2048*1024) * 4.5/8 / 1e9   # packed Q4_K/Q6_K mixed ~4.5bit
print("top_k  ms/token   tok/s   GB/token   GB/s")
for k in (8,6,4,2,1):
    cache=OLMoEKVKache()
    with torch.no_grad():
        lg=forward(ids,W,layers,num_heads=spec.head_count,eps=spec.rms_eps,
                   rope_base=spec.rope_base,top_k=k,cache=cache)
    gen=int(lg[0,-1].argmax(-1))
    n=12
    with torch.no_grad():
        t0=time.perf_counter()
        for _ in range(n):
            nxt=torch.tensor([[gen]],device=dev)
            lg=forward(nxt,W,layers,num_heads=spec.head_count,eps=spec.rms_eps,
                       rope_base=spec.rope_base,top_k=k,cache=cache)
            gen=int(lg[0,-1].argmax(-1))
        el=time.perf_counter()-t0
    ms=el/n*1000; tps=1000/ms
    gb=spec.block_count*k*PER_EXPERT
    print(f"{k:5d}  {ms:7.1f}  {tps:6.2f}   {gb:7.3f}  {gb/(ms/1000):6.1f}")
PY
PYTHONPATH=$PWD python /tmp/topk.py 2>&1 | tail -8
echo TOPK_DONE
