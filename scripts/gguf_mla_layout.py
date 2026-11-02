"""Rewrite an older DeepSeek-V2 GGUF in llama.cpp's MLA layout; the weights stay the same.

GGUFs converted before llama.cpp supported MLA (mid-2024 - second-state's DeepSeek-V2-Lite files
among them) store each layer's ``kv_b`` projection whole, and current llama.cpp then keeps a full
per-head K/V cache: 270 KiB a token for DeepSeek-V2-Lite instead of the 576-wide latent (31 KiB at
f16).  llama.cpp's own converter (``conversion/deepseek.py``) writes ``attn_k_b`` (transposed) and
``attn_v_b`` instead and describes the cache as a single 576-wide head (key_length 576, value_length
512, head_count_kv 1, key/value_length_mla 192/128).  This does the same starting from the GGUF:
``attn_kv_b`` is dequantized and split into F16 ``attn_k_b`` / ``attn_v_b``; every other tensor and
all metadata are copied unchanged.  Used to give llama.cpp its best case in the long-context
comparison (docs/LONG_CONTEXT_1M.md).

    python scripts/gguf_mla_layout.py old.gguf new.gguf --gguf-py LLAMA_CPP/gguf-py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--gguf-py", default="", help="llama.cpp's gguf-py folder, to write the format it reads")
    args = parser.parse_args()
    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)
    import gguf
    from gguf.quants import dequantize

    reader = gguf.GGUFReader(args.source)

    def value(name: str):
        return reader.fields[name].contents()

    arch = value(gguf.Keys.General.ARCHITECTURE)
    if arch != "deepseek2":
        raise SystemExit(f"{args.source} is a {arch} model, not deepseek2")
    if f"{arch}.attention.key_length_mla" in reader.fields:
        raise SystemExit(f"{args.source} is already in the MLA layout")
    heads = value(f"{arch}.attention.head_count")
    rank = value(f"{arch}.attention.kv_lora_rank")
    key_length = value(f"{arch}.attention.key_length")  # nope + rope, per head
    value_length = value(f"{arch}.attention.value_length")
    rope = value(f"{arch}.rope.dimension_count")
    nope = key_length - rope
    changed = {
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.attention.key_length": rank + rope,
        f"{arch}.attention.value_length": rank,
    }

    writer = gguf.GGUFWriter(args.target, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue  # written by GGUFWriter itself
        kind = field.types[0]
        sub_type = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, changed.get(field.name, field.contents()), kind, sub_type=sub_type)
    writer.add_key_value(f"{arch}.attention.key_length_mla", key_length, gguf.GGUFValueType.UINT32)
    writer.add_key_value(f"{arch}.attention.value_length_mla", value_length, gguf.GGUFValueType.UINT32)

    tensors = []  # (name, array, type)
    for tensor in reader.tensors:
        if not tensor.name.endswith(".attn_kv_b.weight"):
            tensors.append((tensor.name, tensor.data, tensor.tensor_type))
            continue
        weight = dequantize(tensor.data, tensor.tensor_type).reshape(heads, nope + value_length, rank)
        k_b = np.ascontiguousarray(weight[:, :nope, :].transpose(0, 2, 1)).astype(np.float16)  # [heads, rank, nope]
        v_b = np.ascontiguousarray(weight[:, nope:, :]).astype(np.float16)  # [heads, value_length, rank]
        tensors.append((tensor.name.replace("attn_kv_b", "attn_k_b"), k_b, gguf.GGMLQuantizationType.F16))
        tensors.append((tensor.name.replace("attn_kv_b", "attn_v_b"), v_b, gguf.GGMLQuantizationType.F16))
    for name, data, kind in tensors:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    split = sum(name.endswith(".attn_k_b.weight") for name, _, _ in tensors)
    print(f"{args.target}: {len(tensors)} tensors, attn_kv_b split in {split} layers; "
          f"cache head {rank + rope} (k) / {rank} (v), per-head {key_length} / {value_length}")


if __name__ == "__main__":
    main()
