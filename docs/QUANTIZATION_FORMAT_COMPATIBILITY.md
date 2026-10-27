# 量化格式兼容：GPTQ、AWQ、GGUF -> SM86 W4A16

> 日期：2026-10-27  
> 目标：让社区现成量化 checkpoint 能进入 DeepInfer-SM86 的原生 W4A16 执行路径。

## 1. 结论

新增统一导入器：

```text
GPTQ / AWQ / GGUF
        │
        ▼
  dequantize to dense layer weights
        │
        ▼
  symmetric int4, group 128 (可配置)
        │
        ▼
  DeepInfer-SM86 layered W4A16 checkpoint
```

支持的是**离线转换兼容**，不是“原格式零拷贝直接运行”。

转换后的目录可以被现有 W4A16 runtime 使用：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --scheduler --scheduler-backend paged `
  --port 18001
```

## 2. 支持的格式

| 格式 | 支持状态 | 说明 |
|---|---|---|
| GPTQ | ✅ | 2/4/8-bit packed int32、group_size、qzeros、`g_idx`/desc_act |
| AWQ | ✅ | AutoAWQ GEMM 4-bit packed 布局 |
| GGUF | ✅ | 单文件；F32/F16/BF16 以及 `gguf` 包能反量化的 ggml 类型 |
| 本仓库 W4A16 layered | ✅ | 原生格式，无需转换 |
| BitsAndBytes NF4/INT8 | ❌ | 未实现 |
| AWQ GEMV/ExLlama 布局 | ❌ | 当前只支持 GEMM packing |
| GGUF split shards | ❌ | 需要先合并为单个 `.gguf` |
| MoE / MLA 模型 | ❌ | 当前执行器只支持 dense decoder |

GGUF 常见可用类型包括：

```text
F32, F16, BF16
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
IQ1/IQ2/IQ3/IQ4 系列
```

具体能否转换取决于已安装的 `gguf` 包是否实现了该类型的 dequantizer。遇到不支持的 ggml 类型时，导入器会明确报错，不会静默写出错误权重。

## 3. 安装依赖

```powershell
pip install -r requirements-sm86.txt
```

新增依赖：

```text
numpy>=1.24
gguf>=0.19,<1
```

## 4. 使用

### 4.1 GPTQ

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-Instruct-GPTQ-Int4 `
  --output D:\Qwen3B-W4A16-SM86 `
  --format auto `
  --group-size 128 `
  --summary D:\Qwen3B-W4A16-SM86\import-summary.json
```

自动检测 `config.json` 中的：

```json
"quantization_config": {
  "quant_method": "gptq",
  "bits": 4,
  "group_size": 128,
  "desc_act": false
}
```

### 4.2 AWQ

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\Qwen2.5-3B-Instruct-AWQ `
  --output D:\Qwen3B-AWQ-to-W4A16-SM86 `
  --format awq `
  --group-size 128
```

AWQ 只支持 AutoAWQ GEMM 布局。配置里出现 `version: gemv` 或 `exllama` 会直接拒绝，避免用错误的反量化公式生成看似成功、实际权重错误的 checkpoint。

### 4.3 GGUF

GGUF 文件通常不自带 HF tokenizer 目录，所以服务端使用前需要额外提供 tokenizer：

```powershell
python scripts\import_quantized_checkpoint.py `
  --source F:\models\qwen2.5-3b-instruct-q4_k_m.gguf `
  --tokenizer F:\models\Qwen2.5-3B-Instruct `
  --output D:\Qwen3B-GGUF-to-W4A16-SM86 `
  --format gguf `
  --group-size 128
```

如果只做权重转换、不打算接网关，`--tokenizer` 可以省略；导入器会输出 warning，目标目录仍然可以用于检查 manifest 和 layer tensors。

## 5. 转换后的目录

```text
output/
  config.json
  generation_config.json
  manifest.json
  embed.safetensors
  norm.safetensors
  lm_head.safetensors        # tied embedding 时不存在
  layer_00.safetensors
  layer_01.safetensors
  ...
  tokenizer*.json            # 如果提供了 tokenizer
```

每个 layer 包含：

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

## 6. 反量化实现细节

### 6.1 GPTQ

GPTQ 的 `qweight` 沿输入维打包：

```text
qweight: [in_features / (32 / bits), out_features]
qzeros:  [num_groups, out_features / (32 / bits)]
scales:  [num_groups, out_features]
```

反量化：

```text
weight[in, out] =
    scales[g_idx[in]] *
    (unpack(qweight)[in, out] - (unpack(qzeros)[g_idx[in], out] + 1))
```

`g_idx` 缺失时按：

```text
g_idx[i] = i // group_size
```

处理，因此 `desc_act=true` 的 checkpoint 也能转换。

### 6.2 AWQ

AWQ GEMM 的 `qweight` 沿输出维打包：

```text
qweight: [in_features, out_features / 8]
qzeros:  [num_groups, out_features / 8]
scales:  [num_groups, out_features]
```

解包后还要按 AutoAWQ 的顺序恢复每个 8-value 分组：

```text
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

### 6.3 GGUF

GGUF reader 负责解析 metadata、tensor info 和 ggml block layout；导入器负责：

- `general.architecture` -> ModelSpec 架构；
- `token_embd`、`blk.N.attn_q` 等 llama.cpp 命名 -> HF/DeepInfer 权重命名；
- `Q4_K` 等类型 -> dense tensor；
- 重新打包成 SM86 symmetric int4 W4A16。

GGUF 的 fused `attn_qkv` / `ffn_gate_up` 也能在导入时拆开。

## 7. 转换器中特意设的限制

### 7.1 lm_head 仍然写 FP16

当前原生 W4A16 runtime 的 `lm_head` 仍是 dense `nn.Linear`。导入器不会假装它支持 3-bit 或 4-bit head：

- `tie_word_embeddings=true`：不写 `lm_head.safetensors`；
- 未 tied：写 FP16 dense `lm_head.safetensors`；
- 如果 head 很大，会打印显存/磁盘提示。

7B 全 4-bit + dense lm_head 可能超过 4 GB；这类模型应继续使用已有的 mixed W4/W3 导出路径，而不是期待这次导入器直接解决 4 GB 极限。

### 7.2 不做的兼容

- 量化 embedding/norm 的特殊格式；
- 非 GEMM AWQ 布局；
- GPTQ/AWQ 的 per-channel `group_size=-1`；
- GGUF 多分片；
- MoE 权重；
- 直接 mmap 社区格式并零拷贝执行。

目标 W4A16 的 `group_size` 必须整除每个 linear 的输入维度；真实 Qwen/Llama
模型通常使用 128，因此 CLI 默认值就是 128。

这些都会在检测或读取阶段报错，而不是生成一个“看起来能跑、实际 logits 错”的 checkpoint。

## 8. 验证

量化模块测试：

```text
tests/quantization/test_dequant.py
tests/quantization/test_convert_hf.py
tests/quantization/test_convert_gguf.py
```

覆盖：

- GPTQ desc_act/g_idx 反量化；
- AWQ GEMM reverse-order 反量化；
- W4A16 pack roundtrip；
- GPTQ -> SM86 W4A16 端到端；
- AWQ -> SM86 W4A16 端到端；
- GGUF Q4_0 -> SM86 W4A16 端到端；
- Q4_K block dequant；
- 转换后 checkpoint 重新加载并生成有限 logits，与 FP reference cosine 对齐。

当前 CPU 测试目标：

```text
8 passed
```

完整的 3B 真实模型转换和 GPU 吞吐需要在目标机器上实测，不应把 tiny-model cosine 当作生产精度或性能数据。

## 9. 下一步

- 直接 mmap / streaming 加载社区量化格式，避免完整 dense 中转；
- 量化 lm_head 的原生 W4/W3 路径；
- AWQ GEMV 和 ExLlama 布局；
- GGUF 分片；
- GPTQ Marlin 等变体；
- 3B/7B 真实 checkpoint 的端到端质量与显存基准。
