"""Greedy decode with the fused decoder (olmoe_fused), printing top-5 logits per step.

usage: margins_fused.py MODEL N_GEN PROMPT_ID...
"""
import sys

import gguf
import torch

REPO = "/root/autodl-tmp/DeepInfer-SM86"
sys.path.insert(0, REPO)
from deepinfer.moe.gguf_index import GGUFIndex
from deepinfer.moe.olmoe_fused import OLMoEFusedDecoder, load_fused_model
from deepinfer.moe.olmoe_spec import spec_from_gguf

model, n_gen = sys.argv[1], int(sys.argv[2])
prompt = [int(t) for t in sys.argv[3:]]
index = GGUFIndex(model)
spec = spec_from_gguf(index)
decoder = OLMoEFusedDecoder(load_fused_model(index, spec), num_heads=spec.head_count,
                            eps=spec.rms_eps, rope_base=spec.rope_base,
                            top_k=spec.expert_used_count, max_len=64)
decoder.capture()
reader = gguf.GGUFReader(model)
field = reader.fields["tokenizer.ggml.tokens"]
vocab = [bytes(field.parts[i]).decode("utf-8", "replace") for i in field.data]


def text(i: int) -> str:
    return vocab[i].replace("Ġ", " ").replace("Ċ", "\\n")


for token in prompt:
    decoder.step(token)
for step in range(n_gen):
    values, ids = torch.topk(decoder.logits[0, -1], 5)
    print(step, *(f"{i}|{v:.4f}|{text(i)}" for v, i in zip(values.tolist(), ids.tolist())),
          sep="\t", flush=True)
    decoder.step()  # feeds its own argmax back
