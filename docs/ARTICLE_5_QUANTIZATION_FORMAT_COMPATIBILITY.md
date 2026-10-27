# 给 DeepInfer-SM86 接入 GPTQ、AWQ 和 GGUF：量化格式兼容改造复盘

> 日期：2026-10-27
> 目标硬件：RTX 3050 Laptop 4 GB / SM86 / Windows
> 定位：把社区量化 checkpoint 转换成 DeepInfer-SM86 原生 W4A16 分层格式
> 本文：实现说明 + 走过的弯路 + 当前结果与边界。不是 GPU 性能榜单。

## 1. 先说结论

这次让 DeepInfer-SM86 能够导入三种社区量化格式：

```text
GPTQ
AWQ
GGUF
```

入口是一个统一转换器：

```text
社区量化 checkpoint
        │
        ▼
  解包 / 反量化 / 张量映射
        │
        ▼
  按层重打包为 SM86 symmetric int4
        │
        ▼
  DeepInfer-SM86 layered W4A16 checkpoint
        │
        ▼
  现有 W4A16 runtime / paged scheduler 直接加载
```

最终支持的能力：

| 格式 | 能力 |
|---|---|
| GPTQ | 2/4/8-bit、group quant、`qzeros`、`g_idx` / desc_act |
| AWQ | AutoAWQ GEMM 4-bit packed 布局 |
| GGUF | 单文件，F32/F16/BF16 和 `gguf` 包支持反量化的 ggml 类型 |
| 输出 | layered W4A16，group size 可配置，默认 128 |

需要明确一句：

> 这是**离线转换兼容**，不是原格式零拷贝直接执行。

社区格式不能直接喂给当前的 SM86 GEMV kernel。强行“直接兼容”只会得到两种结果：要么逐 token 在线反量化，速度很差；要么把错误布局当成正确权重，生成看起来能跑、实际 logits 错误的模型。最终选择了更稳的路线：先转换成 native W4A16。

## 2. 为什么不做原格式直跑

当前 SM86 kernel 的输入约定是：

```text
packed: [out_features, in_features / 2] uint8
scales: [out_features, in_features / group_size] fp16
```

GPTQ、AWQ、GGUF 的存储布局都不是这个形状：

- GPTQ 的 `qweight` 沿输入维打包；
- AWQ 的 `qweight` 沿输出维打包，而且还有一个 8-value reorder；
- GGUF 是容器格式，内部 tensor 可能是 Q4_0、Q4_K、Q6_K 等不同 block；
- GGUF 的 tensor 命名是 llama.cpp 风格，不是 HF 命名；
- GGUF 常见 fused QKV / gate-up，需要拆开；
- GGUF 文件通常不包含 HF tokenizer 目录。

如果做在线直跑，意味着：

1. 每个 linear 都要保留原格式的 packed tensor；
2. 每个 token 都要按原格式解包；
3. 每个线性层都要做一次 scale/zero 反量化；
4. attention、cache、scheduler 仍按现在的 W4A16 形状工作；
5. GGUF block 反量化尤其不适合逐 token 做。

在 4 GB 显存和当前 kernel 结构下，这会让“兼容”变成“能跑但极慢”，不如一次性离线转换。

## 3. 新增的代码结构

```text
deepinfer/quantization/
  formats.py       格式检测、bits/group_size/布局校验
  dequant.py       GPTQ/AWQ int32 解包与反量化
  pack.py          W4A16 symmetric int4 重打包
  sources.py       HF safetensors 源读取
  gguf_source.py   GGUF metadata/tensor 映射与反量化
  convert.py       统一转换器

scripts/
  import_quantized_checkpoint.py

tests/quantization/
  test_dequant.py
  test_convert_hf.py
  test_convert_gguf.py
```

统一接口大致是：

```python
convert_checkpoint(
    source_path,
    output_path,
    group_size=128,
    tokenizer=None,
    overwrite=False,
    source_format=None,
)
```

输出目录结构：

```text
config.json
generation_config.json
manifest.json
embed.safetensors
norm.safetensors
lm_head.safetensors       # tied embedding 时不存在
layer_00.safetensors
layer_01.safetensors
...
```

每个 layer 文件包含：

```text
self_attn.q_proj.packed / scales / bias?
self_attn.k_proj.packed / scales / bias?
self_attn.v_proj.packed / scales / bias?
self_attn.o_proj.packed / scales
mlp.gate_proj.packed / scales
mlp.up_proj.packed / scales
mlp.down_proj.packed / scales
input_layernorm.weight
post_attention_layernorm.weight
pre_feedforward_layernorm.weight?   # Gemma 2
post_feedforward_layernorm.weight?  # Gemma 2
```

## 4. GPTQ 是怎么处理的

GPTQ 常见布局：

```text
qweight: [in_features / (32 / bits), out_features]
qzeros:  [num_groups, out_features / (32 / bits)]
scales:  [num_groups, out_features]
```

每一步 int32 中打包多个低位整数。反量化的关键是两件事。

### 4.1 解包方向

GPTQ 的 qweight 沿**输入维**打包，所以需要把每个 int32 的多个 shift 结果还原到输入特征维，再 reshape 成：

```text
[in_features, out_features]
```

最后转置成目标 W4A16 需要的：

```text
[out_features, in_features]
```

### 4.2 qzeros 的 +1

GPTQ 的 qzeros 里存的是 `zero - 1`，反量化时要：

```text
real_zero = unpack(qzeros) + 1
```

这就是第一次合成 GPTQ 测试时余弦只有约 0.96、误差明显偏大的原因。最初以为是 packing 方向错了，后来发现是零点少加了一。

### 4.3 desc_act / g_idx

GPTQ 的 `desc_act=true` 时，输入特征到 scale group 的映射不一定是：

```text
g_idx[i] = i // group_size
```

而是 checkpoint 里保存的 `g_idx`。实现中：

- 有 `g_idx` 就按它取 `scales[g_idx]` 和 `zeros[g_idx]`；
- 没有 `g_idx` 才回退到默认 group 映射；
- 如果配置声称 `desc_act=true` 但 checkpoint 没有 `g_idx`，直接报错，而不是猜。

这条严格检查是有必要的。Activation-order checkpoint 如果被当成普通 group checkpoint，模型通常还是能输出文本，但精度会悄悄变差，很难排查。

## 5. AWQ 是怎么处理的

AWQ GEMM 的布局：

```text
qweight: [in_features, out_features / 8]
qzeros:  [num_groups, out_features / 8]
scales:  [num_groups, out_features]
```

解包后还需要做 AutoAWQ 的重新排序：

```python
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

正确的处理顺序是：

1. 把 qweight 的每个 int32 按 4-bit shift 解成 8 个值；
2. reshape 到 `[in_features, out_features]`；
3. 对每 8 个输出值按 reverse order 重排；
4. 对 qzeros 做同样的重排；
5. 按 group 做 `(value - zero) * scale`；
6. 转置到 `[out_features, in_features]`。

AWQ 最容易错的地方不是 shift，而是这个输出维 reorder。直接按 GPTQ 的方向解包，结果不会报 shape 错误，但权重会错。

另外，AWQ 有 GEMM / GEMV / ExLlama 等不同 packing 约定。本次只实现 AutoAWQ GEMM。格式检测会检查 `version`，遇到 `gemv` / `exllama` 直接拒绝，避免生成错误 checkpoint。

## 6. GGUF 是怎么处理的

GGUF 和 GPTQ/AWQ 不是一个层级的问题：

- GPTQ/AWQ 是 packed int32 布局；
- GGUF 是文件容器，内部可以是 F16、Q4_0、Q4_K、Q6_K 等很多类型；
- GGUF 还带 architecture、hparams、tokenizer metadata 和 llama.cpp tensor 命名。

本次依赖纯 Python 的 `gguf` 包负责：

- 解析 GGUF 文件；
- 读取 metadata 和 tensor info；
- 按 ggml block 类型反量化。

导入器负责：

- architecture 映射；
- hparams -> HF config；
- llama.cpp tensor 名 -> 目标模型权重名；
- fused QKV / gate-up 拆分；
- 最终按 W4A16 重新打包。

### 6.1 GGUF tensor 名映射

典型映射：

```text
token_embd.weight      -> model.embed_tokens.weight
output_norm.weight     -> model.norm.weight
output.weight          -> lm_head.weight

blk.N.attn_norm        -> model.layers.N.input_layernorm
blk.N.attn_q/k/v       -> self_attn.q/k/v_proj
blk.N.attn_output      -> self_attn.o_proj
blk.N.ffn_norm         -> model.layers.N.post_attention_layernorm
blk.N.ffn_gate/up/down -> mlp.gate/up/down_proj
```

Gemma 2 则额外映射：

```text
blk.N.post_attention_norm -> post_attention_layernorm
blk.N.ffn_norm            -> pre_feedforward_layernorm
blk.N.post_ffw_norm       -> post_feedforward_layernorm
```

### 6.2 GGUF architecture 和 hparams

从 metadata 生成目标 `config.json`：

- `general.architecture`；
- block_count；
- embedding_length；
- feed_forward_length；
- attention.head_count / head_count_kv；
- key_length / value_length；
- layer_norm_rms_epsilon；
- rope.freq_base；
- context_length；
- vocab_size；
- tokenizer 的 bos/eos/pad id。

Mistral 有一个容易踩的坑：很多 GGUF 使用 `general.architecture=llama`，只通过 `attention.sliding_window` 表达 SWA。导入器看到这个组合会把模型类型提升为 Mistral，保留 sliding-window 语义。

### 6.3 Fused tensor

GGUF 可能存成：

```text
blk.N.attn_qkv.weight
blk.N.ffn_gate_up.weight
```

导入器会按配置里的 head 数和中间维度拆分，分别写入目标 q/k/v 和 gate/up。

### 6.4 Tokenizer

GGUF 不像 HF 目录那样天然带 `tokenizer_config.json` 和 `tokenizer.json`。因此：

- 只做权重转换时，`--tokenizer` 可省略；
- 要接 OpenAI gateway 时，建议显式提供：

```powershell
--tokenizer F:\models\Qwen2.5-3B-Instruct
```

导入器会复制 tokenizer 文件，并在存在 GGUF chat template 时补齐 `tokenizer_config.json`。

## 7. 输出格式和运行时修正

### 7.1 Symmetric int4 重打包

转换器把源 dense 权重按目标 kernel 要求重新量化：

```text
scale = max(abs(group)) / 7
q = clamp(round(weight / scale), -8, 7)
```

两个 signed int4 装进一个 uint8，scales 保存为 FP16。

### 7.2 runtime 不再硬编码 group_size

以前 `QwenSm86Runtime` 固定用 128 加载 W4A16。现在会读取：

```text
manifest.json -> group_size
```

这样转换器使用 `--group-size 64/32` 时，runtime 也能正确加载并分配 scale 张量。

### 7.3 CPU checkpoint 加载修正

测试转换后 checkpoint 时发现：

```text
Qwen2W4A16ForCausalLM.from_checkpoint(..., device="cpu")
```

仍然会无条件调用 `_bind_cupy()`，在 CPU 上直接报错。

修复为只有 packed tensor 在 CUDA 上时才绑定 CuPy。这个修正让转换器的 CPU 端到端测试成为可能，也避免 CPU 环境执行无关的 CUDA bridge 初始化。

### 7.4 防止输出覆盖源

`--overwrite` 是危险参数。转换器会检查：

```text
output == source 或者 output 是 source 的祖先目录
```

如果是，直接拒绝。否则一次错误的重定向可能把源 checkpoint 删掉。

## 8. 结果

### 8.1 自动化测试

完整 CPU 回归：

```text
26 passed, 1 skipped
ruff: All checks passed
```

其中量化格式模块：

```text
8 passed
```

覆盖：

- GPTQ desc_act / g_idx 反量化；
- AWQ GEMM reverse-order 反量化；
- W4A16 pack roundtrip；
- GPTQ -> SM86 W4A16 端到端；
- AWQ -> SM86 W4A16 端到端；
- GGUF Q4_0 -> SM86 W4A16 端到端；
- GGUF Q4_K block 反量化；
- 转换后 checkpoint 重新加载、logits 有限，并与 FP reference 做 cosine 对齐。

### 8.2 Tiny Qwen2 端到端观测

用一层的 tiny Qwen2 做合成量化 checkpoint，再转换、加载、比较 logits：

| 源格式 | logits cosine（tiny 模型，非生产指标） |
|---|---:|
| GPTQ 合成 | 约 0.9966 |
| AWQ 合成 | 约 0.9959 |
| GGUF Q4_0 合成 | 约 0.9863 |

这些数字只说明解包、重排和转换链路方向正确，不代表真实 3B/7B 模型的质量。

### 8.3 没有做的结果

必须明确：

- 没有在真实 Qwen2.5-3B/7B 社区 checkpoint 上跑完整转换；
- 没有在 RTX 3050 上测 GPTQ/AWQ/GGUF 转换后的吞吐；
- 没有测真实模型量化前后的 perplexity / 任务质量；
- 没有 GPU 显存峰值数据。

当前环境只有 CPU 版 PyTorch，所以这次只给正确性结果，不给性能结论。

## 9. 走过的弯路

下面这些是这次真正踩过的坑，不是事后总结出来的漂亮经验。

### 9.1 一开始想过“直接支持社区格式运行”

最初的直觉是：既然已经有 W4A16 GEMV，能不能把 GPTQ/AWQ 的 qweight 直接塞给 kernel？

答案是不行。qweight 的 packing 方向、group 顺序、qzeros 语义和 GGUF block
布局都不一样。在线逐 token 解包会带来额外开销，且很容易把布局错误藏进
“模型好像还能说话”的结果里。

最后走了“离线转换到 native 格式”的保守路线。它的代价是转换时间和临时磁盘，
但换来两个好处：

- 执行路径只有一套 W4A16 kernel；
- 转换结果可以被 safetensors/manifest 检查，不依赖运行时 luck。

### 9.2 GPTQ 的 qzeros 少加了 1

第一次写 GPTQ 合成测试时，反量化权重的平均误差明显偏大，cosine 只有约
0.96。检查 packing 方向和 scale 后才发现问题在零点：

```text
qzeros 里存的是 zero - 1
反量化时要用 unpack + 1
```

这个问题最阴险的地方是：如果直接拿真实 checkpoint 跑，模型可能仍然输出一些
token，但数值已经偏了。合成 roundtrip 测试在这里救了一次。

### 9.3 GPTQ 的 desc_act 不能靠猜

一开始只按 `group_size` 做：

```text
g_idx[i] = i // group_size
```

这对普通 group quant 没问题，但 `desc_act=true` 时 checkpoint 会提供 `g_idx`。
如果忽略它，scale/zero 会被应用到错误的输入行。

最终规则是：

- 有 `g_idx` 就严格使用；
- 配置声称 desc_act 但 checkpoint 没有 g_idx 就报错；
- 不自动假设“可能是顺序的”。

宁可拒绝，也不悄悄生成精度错误的权重。

### 9.4 AWQ 的 reverse order 比 shift 更容易错

AWQ 的 int32 unpack 本身不复杂，但输出维有一个顺序重排：

```text
[0, 4, 1, 5, 2, 6, 3, 7]
```

最初按直觉直接 reshape，shape 完全正确，数值却不对。只看 shape 是发现不了的，
必须用合成权重的 roundtrip cosine 才能暴露。

### 9.5 AWQ 还有 GEMV / ExLlama 等不同布局

即使都叫 AWQ，不同 backend 的 qweight packing 也可能不同。本次只支持
AutoAWQ GEMM。检测到 `version=gemv` 或 `version=exllama` 时直接拒绝。

这是故意做得“不宽容”的：错误的反量化公式比明确拒绝更危险。

### 9.6 GGUF 的维度顺序与普通张量相反

GGUF/ggml 的 tensor info 里保存的维度顺序和 PyTorch 逻辑 shape 相反。第一次
构造测试 GGUF 时，把原始 shape 当成 raw shape 传给 writer，直接触发了：

```text
Quantized tensor bytes per row is not a multiple of ... type size
```

后来确认 writer/reader 会做 byte-shape <-> logical-shape 转换，反量化后的线性
权重已经是目标需要的 `[out, in]`，不应再额外转置。

### 9.7 想用 gguf 包 quantize Q4_K 来造测试，结果它只支持 dequantize

为了覆盖常见的 Q4_K，第一版测试想用 `gguf.quants.quantize` 合成一个完整
Q4_K checkpoint。结果直接报：

```text
NotImplementedError
```

这个包可以反量化 Q4_K，但没有实现 Q4_K 的 quantize。

最后的测试策略改成：

- Q4_0 做完整 GGUF -> W4A16 端到端；
- Q4_K 手工构造一个 144-byte block，验证我们的读取和反量化路径。

这个弯路提醒：测试夹具本身也可能“不具备生产能力”。

### 9.8 GGUF 没有天然 HF tokenizer

考虑过从 GGUF 的 tokenizer metadata 自动生成 `tokenizer.json`，但 GGUF 可能使用
BPE、SentencePiece 等不同 tokenizer 后端，自动还原 HF tokenizer 的 special
tokens、chat template 和 added tokens 很容易出错。

这次没有伪造一个不完整的 tokenizer。权重转换可以独立完成；如果需要接网关，
提供 `--tokenizer` 指向一个 HF tokenizer 目录。

### 9.9 lm_head 很想一起量化，但 runtime 还不支持

从显存角度，7B 的 dense FP16 lm_head 很贵。最初考虑过在转换时直接把 lm_head
也打成 W4A16。

但当前 runtime 的 lm_head 是 dense `nn.Linear`，如果只在转换器里写一个 packed
head，加载器根本不会使用它。那会得到一个“文件里有 head packed 张量，但推理
没用上”的假兼容。

因此本次明确：

- tied embedding：不写 head；
- 非 tied：写 FP16 dense head；
- 大 head 打印提示；
- 把量化 head 留到 runtime 支持 W4/W3 head 之后再实现。

### 9.10 runtime 的 group_size 硬编码

转换测试中用 `group_size=32` 加载时发现了问题：runtime 一直固定传
`group_size=128`，不读输出 manifest。

修复后 runtime 从 manifest 读取 group_size。这个问题说明“转换器支持”不仅是转换
器自己的事，加载端也必须消费同样的元数据。

### 9.11 CPU 加载仍无条件绑定 CuPy

转换后 checkpoint 用 CPU 加载时又遇到一个隐藏假设：

```python
layer.self_attn.fused_qkv._bind_cupy()
```

这里默认模型一定在 CUDA 上。测试环境没有 CUDA，于是直接报错。

最后改成只在 packed tensor 实际位于 CUDA 时绑定。这个改动也让转换器的 CPU
端到端回归可以真正执行。

### 9.12 `--overwrite` 可能误删源目录

输出目录如果被错误地设成源目录或源目录的祖先，`--overwrite` 的清理逻辑可能先
删掉源 checkpoint，再去读它。

虽然这是命令行使用错误，但工具应该保护用户。现在转换器会在任何清理前检查：

```text
source.is_relative_to(output) -> 拒绝
```

这个检查看起来很小，但比事后恢复模型目录便宜得多。

## 10. 仍未完成

| 能力 | 状态 |
|---|---|
| GPTQ 2/4/8-bit -> W4A16 | ✅ |
| GPTQ desc_act / g_idx | ✅ |
| AWQ GEMM 4-bit -> W4A16 | ✅ |
| GGUF Q4_0 完整转换 | ✅ |
| GGUF Q4_K block 反量化 | ✅ |
| 原格式零拷贝直跑 | ❌ |
| AWQ GEMV / ExLlama layout | ❌ |
| per-channel `group_size=-1` | ❌ |
| GGUF split shards | ❌ |
| 量化 embedding / 量化 norm | ❌ |
| 量化 lm_head 原生执行 | ❌ |
| MoE / MLA | ❌ |
| 真实 3B/7B 质量与吞吐数据 | 未测 |

尤其不要把 tiny-model cosine 当成生产精度结论。它只能证明“这个格式的解包和
重排方向是对的”。

## 11. 如何复现

安装依赖：

```powershell
pip install -r requirements-sm86.txt
```

运行全部 CPU 回归：

```powershell
python -m pytest -q tests
```

只跑量化模块：

```powershell
python -m pytest -q tests\quantization
```

GPTQ：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-GPTQ-Int4 `
  --output D:\Qwen3B-GPTQ-W4A16-SM86 `
  --format auto --group-size 128
```

AWQ：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-AWQ `
  --output D:\Qwen3B-AWQ-W4A16-SM86 `
  --format awq --group-size 128
```

GGUF：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\qwen2.5-3b-q4_k_m.gguf `
  --tokenizer F:\models\Qwen2.5-3B-Instruct `
  --output D:\Qwen3B-GGUF-W4A16-SM86 `
  --format gguf --group-size 128
```

转换完成后可以继续使用现有 W4A16 gateway：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-GGUF-W4A16-SM86 `
  --backend w4a16 `
  --scheduler --scheduler-backend paged `
  --port 18001
```

## 12. 最后的判断

这次改造的价值不是“支持了多少个缩写”，而是把量化格式问题从主执行路径里隔离
出来了：

```text
格式识别 / 解包 / 反量化
        ↓
统一 W4A16 转换
        ↓
稳定执行路径
```

GPTQ 的 qzeros、AWQ 的 reorder、GGUF 的 block 布局和 tensor 命名，都是很容易
被“能跑通”表象掩盖的问题。最后选择以合成 roundtrip + CPU 端到端 + 明确拒绝
未知布局作为底线。

一句话总结：

> 现在 DeepInfer-SM86 可以吃下 GPTQ、AWQ 和多种 GGUF，但它是通过一个严格的
> 离线转换器吃下，而不是假装这些格式和 native W4A16 kernel 是同一种东西。

---

- 实现提交：`183e92b`
- 兼容说明：`docs/QUANTIZATION_FORMAT_COMPATIBILITY.md`
- 测试：`tests/quantization/`
- CLI：`scripts/import_quantized_checkpoint.py`
