"""Greedy decode with DeepInfer's eager OLMoE path, printing top-5 logits per step."""
import sys

import gguf
import torch

REPO = "/root/autodl-tmp/DeepInfer-SM86"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/scripts")
import bench_olmoe_decode as bench

from deepinfer.moe import olmoe_packed
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.olmoe_layer import OLMoEKVKache
from deepinfer.moe.olmoe_model import forward
from deepinfer.moe.olmoe_spec import spec_from_gguf

model, n_gen, kernel = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = [int(t) for t in sys.argv[4:]]
index = GGUFIndex(model)
spec = spec_from_gguf(index)
weights, stack = bench.load_model(index, spec, spec.block_count, torch.device("cuda"))
olmoe_packed.USE_WARP_GEMV = kernel == "gemv"

reader = gguf.GGUFReader(model)
field = reader.fields["tokenizer.ggml.tokens"]
vocab = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]


def text(i):
    return vocab[i].replace("Ġ", " ").replace("Ċ", "\\n")


cache = OLMoEKVKache()
with torch.no_grad():
    for t in prompt:
        logits = forward(torch.tensor([[t]], device="cuda"), weights, stack,
                         num_heads=spec.head_count, eps=spec.rms_eps, rope_base=spec.rope_base,
                         top_k=spec.expert_used_count, cache=cache)
    for step in range(n_gen):
        values, ids = torch.topk(logits[0, -1].float(), 5)
        print(step, *(f"{i}|{v:.4f}|{text(i)}" for v, i in zip(values.tolist(), ids.tolist())),
              sep="\t", flush=True)
        logits = forward(ids[:1].view(1, 1), weights, stack, num_heads=spec.head_count,
                         eps=spec.rms_eps, rope_base=spec.rope_base,
                         top_k=spec.expert_used_count, cache=cache)
