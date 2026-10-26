> **归档说明**：本文已拆分为两篇独立文章：
> `ARTICLE_1_DEEPINFER_SM86_KERNEL.md`（内核与量化）和
> `ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`（个人生产力与基准）。
> 本文件保留为合并稿。

# 从 0.4 到 42.7 tokens/s：RTX 3050 4 GB 上的 DeepInfer SM86 Qwen2.5 3B/7B 原生推理工程记录

> 时间：2026-10-25 ～ 2026-10-26  
> 硬件：NVIDIA RTX 3050 Laptop GPU，4 GB，SM86  
> 系统：Windows  
> 目标：在不换硬件的前提下，把 Qwen2.5 接进 DeepInfer 的原生模型执行路径，并尽可能接近 Ollama 的速度。

> 术语约定：本文中的 **native** 一律指 **DeepInfer SM86 native**
> （DeepInfer 的 model_executor / W4A16 / W3A16 / GQA / CUDA Graph 路径），
> 不是 Ollama、llama.cpp、Transformers 兼容层或任何第三方 native engine。

---

## 0. 结论先行

这两天最终得到三条可复现路径：

| 路径 | 量化 | 显存 | decode 速度 | 性质 |
|---|---|---:|---:|---|
| 3B NF4 兼容层 | NF4 | 2.28 GiB reserved | 5.859 tokens/s | 兼容性基线，不是原生 |
| 3B DeepInfer SM86 原生 | W4A16 + GQA | 2.49 GiB allocated | **42.7 tokens/s（稳态）** | model_executor 级原生 |
| 7B 混合精度原生 | 4-bit attention/embed + 3-bit MLP/head | 3.20 GiB allocated | **25.96 / 26.19 tokens/s** | 4 GB 容量折中 |
| 7B 整体 Q4 | 全 4-bit | 3.92 GiB allocated | 7.94 tokens/s | 能跑，但持续 paging |

最重要的结论有三个：

1. **兼容层速度和原生速度不能混为一谈。**  
   NF4 兼容层约 5.9 tokens/s，Ollama 约 53 tokens/s，而 DeepInfer 的 3B 原生路径最终达到约 42.7 tokens/s。

2. **4 GB 卡跑 7B，纯 4-bit 全量 GPU 放不下。**  
   7B 全 4-bit 权重约 3.657 GiB，而这张卡实测只有 3.237 GiB CUDA 可用显存。

3. **7B 想在这张卡上达到二十多 tokens/s，必须使用混合精度。**  
   最终采用 4-bit attention/embed + 3-bit MLP/lm_head，checkpoint 2.930 GiB，decode 约 26 tokens/s。

本文只记录 DeepInfer / Qwen 推理线，不包含 B 站插件。

---

## 1. 背景：为什么要做这件事

DeepInfer 原本的原生路径面向 DeepSeek 系列，依赖：

- SM100 / SM103；
- Linux；
- CUDA 13；
- FlashMLA、DeepGEMM、DeepEP、TileLang；
- 压缩 KV / MLA；
- DeepSeek 专用 CacheEngine 和 Sampler。

本地只有：

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Compute Capability: 8.6
VRAM: 4096 MiB
OS: Windows
```

这意味着：

- DeepInfer 的 SM100 kernel 全部不能直接用；
- 完整 engine 的 Linux/CUDA 13 构建链也不具备；
- 只能先在 **model_executor 级** 做可验证的原生实现。

目标不是“让 Ollama 跑起来”，而是回答：

> 在 4 GB 的 SM86 上，DeepInfer 自己的模型代码、量化格式、CUDA kernel、GQA 和 CUDA Graph，究竟能跑到什么程度？

---

## 2. 环境与模型

### 2.1 软件环境

最终 GPU 测速使用：

```text
Python 3.10.3
torch 2.5.1+cu121
CUDA 12.1
CuPy 14.2.0
transformers 4.50.0
peft 0.14.0
safetensors 0.5.3
```

DeepInfer 主包本身需要 Python 3.12，但本机 Python 3.12 CUDA wheel 在解压 `torch_cuda.dll` 时卡住超过 20 分钟。因此 GPU 测速采用了“Python 3.10 + 轻量 package stub + 加载同一份 DeepInfer source”的 harness。  
这不是最终的 Python 3.12 package 集成，文章中会明确标注。

### 2.2 模型

- 3B：Qwen2.5-3B-Instruct + `checkpoint-120` PEFT LoRA；
- 7B：Qwen2.5-7B-Instruct（无 7B LoRA，使用基座模型）；
- 7B 原始 BF16 权重约 15.2 GB；
- 7B 配置关键值：

```text
hidden_size          = 3584
intermediate_size    = 18944
num_hidden_layers    = 28
num_attention_heads  = 28
num_key_value_heads  = 4
head_dim             = 128
vocab_size           = 152064
tie_word_embeddings  = false
```

`tie_word_embeddings=false` 是后面显存预算的一个关键点：embedding 和 lm_head 不能共享。

配套的 `maibot-catgirl-v3` LoRA 训练已完成 120/120 steps，`checkpoint-120`
被合并进 3B 兼容层和早期 DeepInfer SM86 native 验证；Ollama 侧对应 Q4_K_M 模型约 2.4 GB，
运行时报告 100% GPU、context 4096。

---

## 3. 第一阶段：兼容层基线

在写任何 DeepInfer SM86 native kernel 之前，先测了 Transformers + PEFT + bitsandbytes NF4 和 Ollama 两条路径。

### 3.1 Transformers + NF4

正式结果：

| 指标 | 结果 |
|---|---:|
| 模型加载 | 267.613 秒 |
| TTFT 中位 | 0.316656 秒 |
| decode 中位 | 5.859 tokens/s |
| end-to-end 中位 | 5.554 tokens/s |
| 峰值 reserved | 2332 MiB |
| 稳定性 | 10/10 |

### 3.2 Ollama Q4_K_M

| 指标 | 结果 |
|---|---:|
| 独立请求 decode 中位 | 53.178 tokens/s |
| 热态稳定性 decode 中位 | 55.7495 tokens/s |
| Ollama 报告 VRAM | 2,390,300,672 bytes |
| 稳定性 | 10/10 |

### 3.3 输出差异

三个固定问题，NF4 和 Ollama 的输出：

- 0/3 完全一致；
- 字符相似度分别为 0.25、0.6452、0.6364。

这说明两条路径不仅速度不同，输出分布也不同。  
因此后面所有 DeepInfer SM86 native 结果都必须单独标注，不能写成“兼容层结果”。

### 3.4 这一阶段的结论

- NF4 约 5.9 tokens/s；
- Ollama 约 53 tokens/s；
- 差距主要来自量化 kernel、推理引擎和 graph/调度，而不是模型本身。

---

## 4. 第二阶段：DeepInfer 原生 source 接入

### 4.1 做了什么

新增/修改了：

```text
deepinfer/hugging_face/qwen2_config.py
deepinfer/hugging_face/config.py
deepinfer/model_config_loader.py
deepinfer/model_executor/models/qwen2.py
deepinfer/model_executor/models/__init__.py
```

主要能力：

- Qwen2 HF config → DeepInfer `AttrDict`；
- HF safetensors 分片加载；
- tied embedding；
- Qwen2 GQA / RoPE / RMSNorm / SwiGLU；
- PEFT LoRA 合并；
- KV cache 续算；
- 与 HF logits 对照。

测试结果：

```text
6 passed
```

包括：

- HF logits 对齐；
- KV cache 对齐；
- PEFT LoRA 合并；
- 真实 Qwen2.5-3B config/key 覆盖；
- W4A16 反量化 prefill 路径；
- 真实 3B 权重 key 覆盖。

真实 3B 权重的统计：

```text
parameter_count = 3,085,938,688
HF tensors     = 434
missing        = {"lm_head.weight"}（tied）
```

### 4.2 CPU 真实权重 smoke

用真实 3B + LoRA 在 CPU 上跑：

| 指标 | 结果 |
|---|---:|
| 加载 | 345.469 秒 |
| TTFT | 66.131 秒 |
| decode | 4.801 秒 / 2 个后续 token |
| decode 速度 | 0.417 tokens/s |
| 文本 | `你好呀！` |

这个结果证明权重加载、adapter 合并和参考前向是通的，但**不能当作性能结果**。

### 4.3 这一阶段的弯路

1. **Python 3.10 无法导入 DeepInfer**  
   DeepInfer 主包使用了 PEP 695 语法，Python 3.10 直接语法错误。后来单独建立 Python 3.12 CPU-only venv。

2. **Windows 缺少 `os.uname()` 和 `ForkProcess`**  
   增加了可导入回退，否则 source 测试无法在 Windows 启动。

3. **PEFT LoRA key 前缀不匹配**  
   最初把模块路径剥掉前缀后去对比未剥前缀的 key，导致 0 个矩阵被合并。修复为先用完整 key 查找 `lora_B`，再做路径映射。

4. **`safe_open` 不可直接迭代**  
   恢复为显式 `shard.keys()`。

---

## 5. 第三阶段：SM86 原生 W4A16 decode

### 5.1 为什么不能复用现有 kernel

DeepInfer 的 attention/kernel 路径面向 DeepSeek MLA/MQA 和 SM100。  
Qwen2.5 是标准 GQA：

```text
16 Q heads / 2 KV heads（3B）
28 Q heads / 4 KV heads（7B）
```

因此不能直接使用现有 MLA kernel，也不能假设 `deepinfer._C` 在 SM86 上可构建。

### 5.2 4-bit 格式

第一版使用：

```text
symmetric int4
group size = 128
scale = max(abs(group)) / 7
两个 int4 打包进一个 uint8
```

即量化权重约：

```text
0.5 byte/weight + 2 bytes / 128 weights
≈ 4.125 bpw
```

### 5.3 自定义 GEMV

实现位置：

```text
deepinfer/model_executor/models/qwen2_w4a16.py
```

核心设计：

- CuPy `RawKernel` + NVRTC 编译 CUDA C；
- 一个 warp 处理一个输出 row；
- 每个 lane 一次处理 8 个 int4 值；
- x 先拷到 shared memory；
- warp 内归约；
- 单 token decode 走自定义 GEMV；
- prefill 走分块 dequant + cuBLAS。

### 5.4 3B 结果

最初没有 CUDA Graph 时，第一 token 有 2.46 秒 NVRTC 编译，后续约 75–85 ms/token，稳态约 12–13 tokens/s。

加入静态 KV cache 和 CUDA Graph 后：

| 指标 | 结果 |
|---|---:|
| decode 平均（含首轮） | 37.157 tokens/s |
| decode 中位 | 42.666 tokens/s |
| 稳态 | 约 42.7 tokens/s |
| TTFT | 4.555 秒 |
| 峰值 allocated | 2.493 GiB |
| 峰值 reserved | 2.627 GiB |
| 文本（EOS 前） | `你好！有什么问题请尽管告诉我，我会尽力回答的。` |

### 5.5 3B 数字的演进

```text
CPU 参考路径          0.417 tokens/s
W4A16 无 Graph 稳态   约 12–13 tokens/s
W4A16 + Graph         37.157 平均 / 42.666 中位
```

这个提升主要来自两件事：

1. 自定义 4-bit kernel 替代通用 dequant + cuBLAS 的 decode 路径；
2. 静态 KV cache + CUDA Graph 去掉逐 token 的 Python/launch 开销。

---

## 6. 第四阶段：原生 GQA、静态 cache 和 CUDA Graph

### 6.1 原生 GQA kernel

新增 `qwen2_gqa_decode`：

- 一个 block 处理一个 query head；
- 128 个线程对应最多 128 个 cache slot；
- 计算 q·k、mask、softmax、加权 V；
- 支持 16/2 和 28/4 两种 GQA 配置；
- 先与 PyTorch SDPA 对照，再进入 Graph。

tiny 模型验证：

```text
DeepInfer SM86 native GQA vs SDPA:
cosine = 1.0
max abs diff = 0.015625

CUDA Graph vs 普通原生前向:
max abs diff = 0.0
```

### 6.2 静态 KV cache

新增 `Qwen2StaticLayerKVCache`：

```text
key/value: [batch, kv_heads, max_seq, head_dim]
```

decode 时：

- 用 position tensor 写入固定 slot；
- attention 读取整个固定长度 cache；
- 外部 mask 控制有效 token；
- CUDA Graph 可以复用固定 shape。

### 6.3 CUDA Graph 的三个坑

#### 坑 1：DLPack 在 capture 中报 stream 错误

报错类似：

```text
operation would make the legacy stream depend on a capturing blocking stream
```

原因是 `cp.from_dlpack(torch_tensor)` 会触发 PyTorch 的 stream wait event，捕获期间非法。

修复：

```python
cp.cuda.UnownedMemory(
    tensor.data_ptr(),
    tensor.numel() * tensor.element_size(),
    owner=tensor,
    device_id=...,
)
```

用零拷贝 `UnownedMemory + MemoryPointer + cp.ndarray` 替代 DLPack。

#### 坑 2：capture 只记录，不执行自定义 CUDA kernel

第一次 capture 后直接读 `graph_logits`，得到的是无效结果。  
必须：

```python
graph.replay()
torch.cuda.synchronize()
```

之后输出才是真实结果。

#### 坑 3：Python float 传给 CUDA kernel 变成 0

GQA kernel 的 `scale` 参数最初直接传 Python `0.5`，结果 kernel 读到 0，softmax 变成均匀分布。

修复：

```python
cp.float32(1.0 / math.sqrt(head_dim))
```

这看起来是小问题，但会造成“能跑、结果全错”。

---

## 7. 第五阶段：7B 的容量墙

### 7.1 纯 4-bit 放不下

7B 参数约 7.615B。  
按 int4 + group128 scales 计算：

| 项目 | 大小 |
|---|---:|
| 28 层线性层 | 3.134 GiB |
| embedding | 0.262 GiB |
| lm_head | 0.262 GiB |
| **合计** | **3.657 GiB** |

实测 CUDA 可用显存：

```text
3.237 GiB
```

权重尚未包含 KV cache、激活、CUDA context 和 Graph 私有内存，已经超出 0.420 GiB。

### 7.2 paging 测试

用 uint8 全量扫描测有效读取带宽：

| 大小 | 有效带宽 |
|---:|---:|
| 3.2 GiB | 170.6 GiB/s |
| 3.5 GiB | 56.4 GiB/s |
| 3.8 GiB | 32.8 GiB/s |

一旦超过约 3.2 GiB，WDDM 开始 page-in/page-out。

### 7.3 CPU offload 测试

把 7B 单层预先反量化成 FP16：

```text
单层 CPU 矩阵乘：约 194.15 ms
8 层 CPU 理论上限：0.644 tokens/s
```

因此 CPU offload 无法达到几十 tokens/s。

### 7.4 结论

4 GB 上纯 4-bit 7B：

- 全量 GPU：放不下；
- CPU offload：低于 1 token/s；
- paging：上限约 5–9 tokens/s。

要在这张卡上达到二十多 tokens/s，只能降低部分权重的位宽。

---

## 8. 第六阶段：7B 混合 W4/W3

### 8.1 量化方案

最终：

```text
attention q/k/v/o : 4-bit
embedding        : 4-bit
MLP gate/up/down : 3-bit
lm_head          : 3-bit
group size       : 128
```

3-bit 格式：

- 8 个 signed 3-bit 值压进 3 bytes；
- 量化范围 `[-4, 3]`；
- scale = max(abs(group)) / 3；
- 自定义 `qwen2_w3a16_gemv` kernel。

checkpoint 实际大小：

```text
2.930 GiB
有效位宽约 3.30 bpw
```

### 8.2 为什么不是 Q4

这一点必须说清楚：

- Q4 单层权重再加 scales 约 4.125 bpw；
- 全 4-bit 7B 约 3.657 GiB；
- 当前混合精度约 3.30 bpw。

所以当前 25–26 tokens/s 的版本是 **Q4/Q3 混合**，不是整体 Q4。  
它的质量低于全 Q4，但能放进 4 GB。

### 8.3 流式量化

7B 的 BF16 权重太大，不能把整模型加载到 CPU 再量化。  
因此写了流式 quantizer：

```text
scripts/quantize_qwen7b_mixed.py
scripts/finish_qwen7b_mixed_quantize.py
scripts/quantize_qwen7b_head3.py
```

流程：

1. 按 shard 逐 tensor 读取；
2. attention/MLP 逐层量化；
3. 逐层写 safetensors；
4. embedding/lm_head 分块量化，避免 CPU OOM；
5. 写 manifest。

产物：

```text
D:\Qwen7B-Mixed-SM86
```

### 8.4 模型执行器

新增：

```text
deepinfer/model_executor/models/qwen2_mixed_sm86.py
```

包含：

- `W3A16Linear`；
- 混合 `Qwen2MixedForCausalLM`；
- int4 `QuantizedEmbedding`；
- 构造期 fused qkv；
- 构造期 fused gate-up；
- DeepInfer SM86 native GQA；
- 静态 KV cache；
- CUDA Graph。

### 8.5 最终结果

| 指标 | 结果 |
|---|---:|
| checkpoint | 2.930 GiB |
| 权重加载 | 4.445 秒 |
| prefill TTFT | 3.061 秒 |
| Graph 捕获 | 0.121 秒 |
| 首次 replay | 0.0331 秒 |
| decode 平均 | **25.962 tokens/s** |
| decode 中位 | **26.188 tokens/s** |
| 中位单 token | 38.186 ms |
| 峰值 allocated | 3.196 GiB |
| 峰值 reserved | 3.357 GiB |
| 生成后 allocated | 2.970 GiB |

生成结果：

```text
你好！您好吗？我可以帮你回答问题或完成任务，请告诉我需要做的，请告诉我你接下来的目的或任务。
```

### 8.6 混合精度速度演进

```text
20.956 tokens/s   3-bit MLP + 4-bit head（早期）
24.545 tokens/s   4-bit head + 构造期融合 + 调 block
25.693 tokens/s   lm_head 降到 3-bit
25.962 tokens/s   通用 bits 重构后的最终回归
```

---

## 9. 第七阶段：整体 Q4

### 9.1 真正全 4-bit

重新生成了纯 Q4 checkpoint：

```text
D:\Qwen7B-Q4-SM86
```

配置：

```text
attention = 4-bit
MLP       = 4-bit
embedding = 4-bit
lm_head   = 4-bit
```

大小：

```text
3.658 GiB
约 4.125 bpw
```

### 9.2 结果

| 指标 | 结果 |
|---|---:|
| 权重加载 | 7.793 秒 |
| prefill TTFT | 86.316 秒 |
| decode 平均 | 7.942 tokens/s |
| decode 中位 | 7.939 tokens/s |
| 中位单 token | 125.955 ms |
| 峰值 allocated | 3.918 GiB |
| 峰值 reserved | 4.076 GiB |
| 生成后 allocated | 3.693 GiB |

生成：

```text
你好！很高兴见到你。你可以叫我
```

### 9.3 结论

整体 Q4 **能加载、能生成**，但 4 GB 显存不足导致持续 paging：

- prefill 约 86 秒；
- decode 只有约 8 tokens/s；
- 峰值 reserved 4.076 GiB，超过物理显存。

所以：

| 配置 | 有效位宽 | 速度 | 适用性 |
|---|---:|---:|---|
| 混合 Q4/Q3 | 3.30 bpw | 25.96 tokens/s | 4 GB 可用 |
| 整体 Q4 | 4.125 bpw | 7.94 tokens/s | 仅质量/兼容验证 |

要同时做到整体 Q4 和几十 tokens/s，需要 8 GB 以上显存。

---

## 10. 两天里走过的弯路

这一节专门记录失败和返工，因为这些比最终数字更有信息量。

### 10.1 把兼容层当成 DeepInfer SM86 native

早期 NF4 结果只有 5.9 tokens/s。  
但这是 Transformers + bitsandbytes 的路径，不是 DeepInfer kernel。  
后来严格分开：

- 兼容层结果；
- Ollama 结果；
- DeepInfer source 结果；
- DeepInfer SM86 native kernel 结果。

### 10.2 试图在 4 GB 上直接跑纯 4-bit 7B

权重 3.657 GiB > 可用 3.237 GiB，必然 paging。  
最终全 Q4 只有 7.9 tokens/s。  
这个路线不是实现失败，而是容量不满足。

### 10.3 加载后融合 qkv/gate-up 导致 allocator 碎片

一开始先加载独立的 q/k/v/gate/up，再创建 fused 模块。  
结果：

```text
reserved 峰值 4.811 GiB
decode 掉到 3.9 tokens/s
```

原因是旧模块和新 fused 模块同时存在，造成 PyTorch allocator 无法回收的碎片。

修复：

- 在模型构造期直接创建 fused qkv / fused gate-up；
- checkpoint 直接往 fused 切片加载；
- 不再构造旧模块。

修复后：

```text
reserved 3.357 GiB
decode 25.962 tokens/s
```

### 10.4 CUDA Graph capture 不等于执行

自定义 CUDA kernel 在 capture 阶段只被记录，不执行。  
最初直接读 capture 的输出，得到乱码 token。  
修复为首个 replay 后才采样。

### 10.5 DLPack 在 capture 中不可用

使用 `cp.from_dlpack()` 会导致 stream wait event，捕获失败。  
改成 `UnownedMemory` 零拷贝视图。

### 10.6 GQA scale 变成 0

Python float 传给 RawKernel 的 `float` 参数时被错误解释为 0，导致 softmax 均匀。  
改成 `cp.float32(...)` 后 DeepInfer SM86 native GQA 与 SDPA cosine 到 1.0。

### 10.7 3-bit 打包的 padding 错误

第一版每个 row 多一个 padding byte，但 kernel 按紧密 row stride 读取，导致第 2 行开始全部错位。

修复：

- 只在整块 tensor 末尾放一个 padding byte；
- row stride 保持 `K * 3 / 8`。

### 10.8 CPU 量化 head/embedding OOM

一次性把 545M 参数转 float32 超出 RAM。  
改成按 8192 行分块量化。

### 10.9 异步 GPU greedy 没有更快

尝试把 argmax、token 写回、position/mask 推进全部放到 GPU 上流水。  
结果约 24.15 tokens/s，低于同步路径的 25.9。  
最终保留同步 greedy，避免为了“看起来异步”牺牲实际速度。

### 10.10 Python 3.12 CUDA 环境失败

`torch_cuda.dll` 解压长时间卡住，最终使用 Python 3.10 harness 加载同一份 source。  
因此当前 GPU 结果不是 Python 3.12 package 集成结果，这一点在结论中必须保留。

---

## 11. 当前实现边界

必须明确：当前是 **model_executor 级** 实现，不是完整 DeepInfer engine。

已完成：

- Qwen2 source 模型；
- HF 分片加载；
- PEFT LoRA 合并；
- W4A16 GEMV；
- W3A16 GEMV；
- 原生 GQA decode；
- 静态 KV cache；
- CUDA Graph；
- 混合精度 checkpoint；
- 7B 流式量化。

未完成：

- `worker/cache_engine.py` full-attention block table / slot mapping；
- DeepInfer 原生 Sampler；
- Qwen InputMetadata producer；
- engine/scheduler/worker 全链路；
- 生产级 continuous batching；
- tiled W4/W3 prefill GEMM；
- Python 3.12 CUDA package 验证；
- BF16 质量对照与正式 perplexity/评测。

当前状态字段仍然是：

```text
full_engine = false
native_runner_ready = false
```

所以不能写成“DeepInfer 完整 engine 已支持 Qwen2.5”。

---

## 12. 复现

### 12.1 3B 原生 W4A16

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen2_w4a16_gpu_harness.py `
  --max-new-tokens 24 `
  --output D:\DeepInfer\doc\results\qwen2_w4a16_native_gqa_gpu.json
```

### 12.2 7B 混合 Q4/Q3

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --max-new-tokens 24 --warmup-replays 3 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_mixed_head3_final.json
```

### 12.3 7B 整体 Q4

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Q4-SM86 `
  --max-new-tokens 4 --warmup-replays 1 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_q4_gpu.json
```

### 12.4 源码测试

```powershell
cd D:\DeepInfer
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest -q `
  tests\models\test_qwen2_source_integration.py
```

结果：

```text
6 passed
```

---

## 13. 产物索引

### 3B

- `doc/QWEN_SM86_W4A16_NATIVE.md`
- `doc/results/qwen2_w4a16_native_gqa_gpu.json`
- `doc/results/qwen2_w4a16_sm86_summary.json`
- `doc/results/qwen2_native_real_smoke.json`

### 7B 混合 Q4/Q3

- `doc/QWEN_7B_MIXED_SM86.md`
- `doc/results/qwen7b_mixed_head3_final.json`
- `doc/results/qwen7b_mixed_sm86_summary.json`
- checkpoint：`D:\Qwen7B-Mixed-SM86`

### 7B 整体 Q4

- `doc/QWEN_7B_Q4_SM86.md`
- `doc/results/qwen7b_q4_gpu.json`
- `doc/results/qwen7b_q4_sm86_summary.json`
- checkpoint：`D:\Qwen7B-Q4-SM86`

### 兼容性

- `doc/QWEN_COMPAT_FORMAL_RESULTS.md`
- `doc/results/qwen_compat_formal_20261025.json`

### 代码

- `deepinfer/hugging_face/qwen2_config.py`
- `deepinfer/model_executor/models/qwen2.py`
- `deepinfer/model_executor/models/qwen2_w4a16.py`
- `deepinfer/model_executor/models/qwen2_mixed_sm86.py`
- `scripts/run_qwen2_w4a16_gpu_harness.py`
- `scripts/run_qwen7b_mixed_gpu_harness.py`
- `scripts/quantize_qwen7b_mixed.py`
- `scripts/quantize_qwen7b_head3.py`
- `scripts/quantize_qwen7b_q4.py`

---

## 14. 仓库状态

当前 DeepInfer 工作区：

```text
HEAD = 8b16149
main = origin/main
ahead/behind = 0/0
```

所有新增源码、报告、结果 JSON 和文章都在工作区，尚未提交、尚未 push。
这不是遗漏，而是按“暂时先不提交”的要求保留的本地工作状态。

## 15. 最后总结

这两天真正完成的不是“把 7B 塞进 4 GB”这么简单，而是建立了一条可解释的工程链路：

1. 先用兼容层得到基线；
2. 再做 source-level DeepInfer SM86 native 模型接入；
3. 再写 SM86 W4A16 / W3A16 / GQA kernel；
4. 再用静态 cache + CUDA Graph 降低 launch 开销；
5. 再根据 4 GB 容量做混合精度；
6. 最后用整体 Q4 验证真实上限。

最终结果：

```text
3B 原生 W4A16 + GQA + Graph：约 42.7 tokens/s
7B 混合 Q4/Q3：约 26.0 tokens/s
7B 整体 Q4：约 7.9 tokens/s
```

最重要的边界：

> 当前是 DeepInfer model_executor 级原生实现，不是完整 engine；  
> 4 GB 上无法同时满足“整体 Q4”和“几十 tokens/s”；  
> 要两者兼得，至少需要 8 GB 以上显存。

这也是这两天最值得保留的结论。

---

## 16. 补充：把成果推到个人生产力

跑分解决的是“能不能快”，个人生产力解决的是“每天会不会用”。

最终增加了一层本地网关：

```text
Chatbox / Obsidian / VS Code / 浏览器插件
                │
                ▼
      http://127.0.0.1:18000/v1
                │
                ▼
        apps/server.py
        /v1/models
        /v1/chat/completions
                │
                ▼
    Qwen2.5-7B mixed Q4/Q3
    常驻模型 + 256-slot static KV + CUDA Graph
```

### 16.1 已实现

- OpenAI 兼容的 `/v1/models`；
- `/v1/chat/completions`；
- SSE 流式输出；
- EOS 停止；
- 256 token context；
- 单请求串行与显存保护；
- `apps/ask.py` CLI；
- `desktop/quick_ask.ps1` 剪贴板总结/改写/翻译/待办；
- 建议用 PowerToys 绑定 `Win+Alt+Q` 等快捷键。

### 16.2 实测

在 4 GB 3050、后台还有 MaiBot 占用显存的情况下：

| 指标 | 结果 |
|---|---:|
| prefill | 约 5.8–6.3 秒 |
| decode | 约 25.7 tokens/s |
| 64-token 请求总延迟 | 约 8.4 秒 |
| 24-token 快捷回答 | 约 7.3 秒 |

时间主要花在 prefill。对于“选中一段文字，总结/改写/翻译”这种短上下文任务，
这是可用水平；对于整篇长文，需要分段 map-reduce。

### 16.3 这一层新增的弯路

#### 512 context 退化

直接测试 512-slot GQA kernel 时 cosine 为 1.0；但接进 7B runtime 后，
256/512 context 的输出出现重复和 `!`。最终定位到 512-thread 路径在集成
环境下不稳定，而 256-slot / 256-thread kernel 输出正常。

处理方式不是把错误路径包装成“能用”，而是**收口到 256 context**，并在
README 和结果中明确标注。512-slot 只保留为后续研究项。

#### UTF-8 流式重复

Qwen 的 byte-level BPE 会在一个 UTF-8 汉字中间切断 token。直接对每个
token prefix 做 `decode` 会得到 `�`，下个 token 补齐后又会把整段文本重新
输出，造成重复。

修复方式：

- 流式时截断第一个 `U+FFFD` 之后的内容；
- 只发送已经稳定的文本前缀；
- 等下一个 token 补齐后再发送增量；
- 非流式则收集完整 token IDs，最后一次性 `decode`。

#### SSE 不关闭

最初发送 `[DONE]` 后仍保持 keep-alive，`curl` 会一直等待。改为
`Connection: close` 并在 `[DONE]` 后显式关闭连接。

#### 端口 8000 被占用

本机 8000 已被 MaiBot 使用。网关默认改到 `18000`，避免“服务启动成功但
请求打到别的进程”。

#### PowerShell 中文乱码

Windows PowerShell 5.1 会把无 BOM 的 UTF-8 脚本按 ANSI 读取，中文
`ValidateSet` 直接语法错误。`desktop/quick_ask.ps1` 必须保存为
UTF-8 with BOM。

### 16.4 当前边界

- 网关仍然只是 model_executor 级原生模型的 HTTP 包装；
- `full_engine=false`；
- 256 context；
- 单请求串行；
- 没有 RAG、工具调用、邮件/日历写操作；
- 3B 已通过 `--backend w4a16` 接入常驻网关，详见第17节；
- 3-bit 量化仍是有损的。

但至少现在，这个成果不再只是 `doc/results/*.json` 里的数字，而是可以：

1. 作为 OpenAI 兼容服务接入现有客户端；
2. 用剪贴板快捷键处理每天遇到的文字；
3. 在断网、本地、4 GB 显卡上完成总结、改写、翻译和待办提取。

---

## 17. 再进一步：3B 生产力适配与三后端基准

### 17.1 3B 不能每次启动都重新量化

原来的 3B 原生 harness 每次启动都要：

```text
加载 3B dense 权重 + merge LoRA（CPU，约 5 分钟）
→ 搬到 GPU
→ 逐层转 W4A16
```

这只能做测速，不能做常驻服务。

新增：

```text
scripts/export_qwen3b_w4_checkpoint.py
deepinfer/model_executor/models/qwen2_w4a16.py::from_checkpoint
```

导出一个分层 checkpoint：

```text
D:\Qwen3B-W4A16-SM86
  embed.safetensors
  norm.safetensors
  layer_00.safetensors
  ...
  layer_35.safetensors
  manifest.json
```

大小 **1.912 GiB**。  
加载时间从约 5 分钟降到 **11.75 秒**。

3B 是 tied embedding，checkpoint 只保存一份 embedding；lm_head 共享它。

### 17.2 3B 网关结果

启动：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

同一协议（system + 两个中文 prompt、max_tokens=64、temperature=0、两轮）：

| 指标 | 3B DeepInfer SM86 native |
|---|---:|
| prefill 中位 | 0.4559 秒 |
| decode 中位 | **44.773 tokens/s** |
| 一次请求 wall 中位 | **0.8948 秒** |
| completion 中位 | 19 tokens |

### 17.3 和 Ollama 3B 对比

Ollama 使用 `maibot-catgirl-v3:latest`，Q4_K_M，4096 context：

| 指标 | 3B DeepInfer SM86 native | Ollama 3B |
|---|---:|---:|
| prefill | 0.4559 s | **0.0603 s** |
| decode | 44.773 t/s | **57.285 t/s** |
| wall | 0.8948 s | **0.4637 s** |
| context | 256 | **4096** |

结论：

- 3B DeepInfer SM86 native 已经进入个人快捷操作可用区间；
- Ollama 在 3B 上仍然更快，主要优势在 prefill 和完整 engine；
- DeepInfer 的 W4A16 decode 本身并不慢，差距在服务层。

### 17.4 7B 当前环境 paging

桌面后台约 500 MiB GPU 占用时，7B 网关 prefill 超过 120 秒仍无法完成。  
用 DeepInfer SM86 native harness 在当前环境测得：

```text
prefill: 53.089 秒
decode:  25.141 tokens/s
峰值 allocated: 3.196 GiB
峰值 reserved: 3.357 GiB
```

同一 7B checkpoint 在更干净环境下是：

```text
prefill: 3.061 秒
decode:  25.962 平均 / 26.188 中位 tokens/s
```

所以 7B 不是 kernel 退化，而是 **4 GB + 后台 GPU 占用 + WDDM paging**。  
当前环境下的 7B 生成还出现退化，不能把该文本当作质量结果。

### 17.5 新的弯路

- 3B dense 加载本身就要约 5 分钟，不能直接做常驻；
- 3B tied embedding 如果重复保存 lm_head 会浪费0.6 GB；
- 7B 网关在前台 HTTP + static cache + CUDA Graph + 后台 GPU 占用下，
  prefill 从3秒变成53秒以上；
- 3B/7B 不能同时常驻4 GB；
- 512 context 仍然没有交付，当前上限是256。

### 17.6 最终建议

| 场景 | 选择 |
|---|---|
| 剪贴板总结/改写/翻译/待办 | 3B DeepInfer SM86 native，约0.9秒 |
| 想用最快3B | Ollama Q4_K_M，约0.46秒 |
| 更高质的短任务 | 7B DeepInfer SM86 native，但必须隔离显存 |
| 长上下文 | 当前 DeepInfer SM86 256 context 不足，Ollama 4096 更合适 |
| 生产级 | 仍需 CacheEngine、Sampler、continuous batching、tiled prefill |

三后端原始数据：

```text
docs/results/productivity_3b_only.json
docs/results/productivity_ollama_only.json
docs/results/qwen7b_mixed_head3_final.json
docs/results/qwen7b_current_paging_harness.json
docs/results/productivity_backends_combined.json
```

完整报告：

```text
docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md
```

---

## 18. 三后端基准的完整协议与原始数据

### 18.1 为什么要先定协议

将 3B DeepInfer SM86 native、7B DeepInfer SM86 native、Ollama 放在一起比较时，最容易犯的错误是：

- 一个用 24 token，另一个用 64 token；
- 一个带 system prompt，另一个不带；
- 一个把模型加载算进 decode，另一个不算；
- 一个看 `eval_duration`，另一个看 HTTP wall；
- 模型温度、top_p、seed 不一致；
- 3B 和 7B 直接拿质量做排名。

因此这次先固定协议，再跑数据。

### 18.2 统一协议

```text
system:
你是一个简洁、可靠的中文个人助理。只输出最终答案，不要复述任务。

prompt 1:
用一句话解释 CUDA Graph。

prompt 2:
把下面这段话压缩成一句待办：
明天下午三点前把实验报告发给导师，并附上结果表格。

max_tokens  = 64
temperature = 0
top_p       = 0.95
top_k       = 50
seed        = 20261026
warmup      = 1
repeats     = 2 × 2 prompts
```

3B DeepInfer SM86 native 与 Ollama 的 tokenizer 都报告：

```text
prompt 1 = 38 tokens
prompt 2 = 58 tokens
```

这说明两条路径确实用了同一组输入。

### 18.3 3B DeepInfer SM86 native 的逐轮数据

| 轮次 | prompt | wall | prefill | decode | completion | decode tps |
|---|---|---:|---:|---:|---:|---:|
| 0 | CUDA Graph | 0.863216 s | 0.456009 s | 0.400732 s | 18 | 44.918 |
| 0 | 待办压缩 | 1.004158 s | 0.549768 s | 0.448161 s | 20 | 44.627 |
| 1 | CUDA Graph | 0.861360 s | 0.454701 s | 0.399181 s | 18 | 45.092 |
| 1 | 待办压缩 | 0.926476 s | 0.455816 s | 0.450063 s | 20 | 44.438 |

中位：

```text
prefill = 0.4559 s
decode  = 44.773 tokens/s
wall    = 0.8948 s
```

### 18.4 Ollama 3B 的逐轮数据

| 轮次 | prompt | wall | prefill | decode | completion | decode tps |
|---|---|---:|---:|---:|---:|---:|
| 0 | CUDA Graph | 0.439561 s | 0.017359 s | 0.257574 s | 15 | 58.236 |
| 0 | 待办压缩 | 0.499255 s | 0.070977 s | 0.276193 s | 16 | 57.930 |
| 1 | CUDA Graph | 0.485539 s | 0.114966 s | 0.264832 s | 15 | 56.640 |
| 1 | 待办压缩 | 0.441871 s | 0.049554 s | 0.283564 s | 16 | 56.425 |

中位：

```text
prefill = 0.0603 s
decode  = 57.285 tokens/s
wall    = 0.4637 s
```

### 18.5 7B 的两组数据

隔离环境：

```text
prefill = 3.061 s
decode  = 25.962 平均 / 26.188 中位 tokens/s
峰值 allocated = 3.196 GiB
峰值 reserved  = 3.357 GiB
```

当前后台约 500 MiB GPU 占用时：

```text
prefill = 53.089 s
decode  = 25.141 tokens/s
峰值 allocated = 3.196 GiB
峰值 reserved  = 3.357 GiB
```

当前 paging 文本：

```text
CUDA Graph是一种编译时<decltypeoriouslylopenngen<decltypeoriouslylopenngen>>>>>>>>来优化计算图以
```

这不是一个可接受的质量输出，因此不能拿它做质量结论。

---

## 19. 3B 生产力适配的实现细节

### 19.1 以前的 3B 启动路径

```text
加载 3B BF16 分片
→ merge PEFT LoRA
→ CPU dense 模型
→ GPU float16
→ 逐层量化 W4A16
→ 创建静态 cache
→ 创建 CUDA Graph
```

这个流程有两个问题：

1. CPU 加载和 LoRA merge 约 5 分钟；
2. 每次启动都重复做一次转换。

作为测速 harness 可以，作为个人网关不行。

### 19.2 分层 checkpoint

新增导出脚本：

```text
scripts/export_qwen3b_w4_checkpoint.py
```

它做的事情：

```text
Qwen2ForCausalLM.load_model(..., device="cpu", adapter_path=...)
→ Qwen2W4A16ForCausalLM.from_dense_model(..., device="cuda")
→ 每层保存 packed/scales/bias + norm
→ 单独保存 embed + norm
→ 写 manifest
```

产物：

```text
D:\Qwen3B-W4A16-SM86
  config.json
  tokenizer.json
  tokenizer_config.json
  manifest.json
  embed.safetensors
  norm.safetensors
  layer_00.safetensors
  ...
  layer_35.safetensors
```

总大小：

```text
1.912 GiB
```

### 19.3 tied embedding

3B 配置是：

```json
"tie_word_embeddings": true
```

因此：

- `embed_tokens.weight` 和 `lm_head.weight` 是同一个张量；
- checkpoint 只保存一份 `embed.safetensors`；
- loader 不重复分配 lm_head；
- 如果重复保存，会多浪费约 0.6 GiB。

### 19.4 loader

新增：

```text
Qwen2W4A16ForCausalLM.from_checkpoint(...)
```

流程：

1. 读取 Qwen2 config；
2. 在 CUDA 上构造空白 W4A16 模型；
3. 加载 dense embedding；
4. 加载 norm；
5. 逐层加载 q/k/v/o + gate/up/down 的 packed/scales/bias；
6. 绑定 CuPy 零拷贝视图；
7. 挂上 `native_gqa_decode`；
8. 进入 eval 模式。

加载时间：

```text
约 11.75 秒（第一次服务器启动）
```

### 19.5 网关

3B：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

7B：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

4 GB 上两者不能同时常驻：

```text
3B 约 2.0–2.5 GiB
7B 约 3.0–3.4 GiB
```

---

## 20. 我走过的弯路（完整版）

这一节不只列“最后怎么解决”，还记录**当时为什么会判断错**。

### 20.1 把兼容层速度当成 DeepInfer 原生速度

**错误判断：**

> NF4 只有 5.9 tokens/s，所以 DeepInfer 的 Qwen 路径也很慢。

**实际情况：**

NF4 是 Transformers + bitsandbytes 的兼容层，不是 DeepInfer 的 W4A16 DeepInfer SM86 native
路径。3B 原生后来到 42.7 tokens/s 稳态。

**教训：**

比较之前先写清楚是“兼容层、Ollama、source 模型还是 DeepInfer SM86 native kernel”。

### 20.2 以为纯 4-bit 7B 可以塞进 4 GB

**错误判断：**

> 4-bit 就是 0.5 byte/weight，7B 应该只有 3.5 GB 左右，也许刚好能跑。

**实际情况：**

- 7.615B 参数；
- 加 group128 scales 后约 3.657 GiB；
- 可用显存只有 3.237 GiB；
- 还没算 KV、激活和 Graph。

最终全 Q4 能跑，但 prefill 86 秒、decode 7.9 tokens/s。

**教训：**

先算权重 + scales + KV + 激活，再看“能不能跑”。

### 20.3 加载后再融合 qkv/gate-up

**错误判断：**

> 先按普通层加载，再替换成 fused 模块，代码更简单。

**实际情况：**

旧模块和 fused 模块同时存在，PyTorch allocator 出现碎片：

```text
reserved 峰值 4.811 GiB
decode 掉到 3.9 tokens/s
```

**修复：**

在模型构造期直接创建 fused qkv / gate-up，checkpoint 直接往 fused 切片加载。

**教训：**

4 GB 边缘上，allocator 碎片和算法一样致命。

### 20.4 CUDA Graph capture 不等于执行

**错误判断：**

> capture 完之后直接读输出就行。

**实际情况：**

capture 只记录自定义 CUDA kernel，不执行；必须 replay 一次。

**修复：**

```python
graph.replay()
torch.cuda.synchronize()
```

**教训：**

Graph 的“第一次正确性”必须单独验证。

### 20.5 DLPack 在 capture 中报 stream 错误

**报错：**

```text
operation would make the legacy stream depend on a capturing blocking stream
```

**原因：**

`cp.from_dlpack(torch_tensor)` 会触发 stream wait event。

**修复：**

使用 `cp.cuda.UnownedMemory` 零拷贝视图。

### 20.6 GQA scale 变成 0

**现象：**

DeepInfer SM86 native GQA 能跑，但输出全错，softmax 近似均匀。

**原因：**

Python float 传给 RawKernel 的 `float` 参数被解释成 0。

**修复：**

```python
cp.float32(1.0 / math.sqrt(head_dim))
```

### 20.7 3-bit 打包 row padding 错位

**现象：**

3-bit kernel 余弦约 `-0.068`。

**原因：**

每行多放了一个 padding byte，但 kernel 按紧密 row stride 读取。

**修复：**

只在整块 tensor 末尾放一个 padding byte。

### 20.8 量化 embedding/head 时 CPU OOM

**现象：**

一次性把 545M 参数转 float32，RAM 不够。

**修复：**

按 8192 行分块量化，写 `finish_qwen7b_mixed_quantize.py`。

### 20.9 Python 3.12 CUDA wheel 解压卡死

**现象：**

`torch_cuda.dll` 解压超过 20 分钟。

**实际选择：**

使用 Python 3.10 CUDA harness 加载同一份 DeepInfer source。

**代价：**

这不是最终 Python 3.12 package 验证；文章和 README 都必须明确这一点。

### 20.10 512 context 的“直接 kernel 正确”不等于“runtime 正确”

**错误判断：**

> 512-slot GQA kernel 与 SDPA 的 cosine 是 1.0，所以 512 context 可以交付。

**实际情况：**

- 直接 kernel 测试 cosine 1.0；
- 但接进 7B runtime 后，256/512 context 输出重复和 `!`；
- 512-thread 路径在集成环境不稳定；
- 256-slot / 256-thread 路径输出正常。

**修复：**

收口到 256 context，不把错误路径包装成“能用”。

**教训：**

单 kernel 正确性 ≠ 模型级正确性。

### 20.11 7B 网关 paging 被低估

**错误判断：**

> 7B 混合模型在 4 GB 上能跑到 26 tokens/s，网关也应该一样。

**实际情况：**

后台约 500 MiB GPU 占用时：

- 7B 网关 prefill 超过 120 秒；
- 7B harness 当前环境 prefill 53.089 秒；
- 同一 checkpoint 在干净环境 prefill 3.061 秒；
- paging 下输出还会退化。

**修复：**

- 3B 作为默认；
- 7B 只在高价值短任务、显存隔离时使用；
- 文档同时保留“隔离最佳”和“当前 paging”两组结果，不混在一起。

### 20.12 3B 每次启动重新转换

**错误判断：**

> 先把 3B 原生 harness 跑通，再考虑服务化。

**实际情况：**

dense 加载 + LoRA merge + W4A16 转换约 5 分钟，根本不能做常驻。

**修复：**

导出 1.912 GiB 的分层 checkpoint，加载降到 11.75 秒。

### 20.13 清理 Ollama 时把基准环境弄坏

**现象：**

基准脚本报：

```text
WinError 10061 目标计算机积极拒绝
```

**原因：**

为了释放显存，把 Ollama 服务进程停掉了，但没有重新启动。

**修复：**

```powershell
ollama serve
```

再跑 `/api/chat`。

**教训：**

“释放显存”和“保持服务可用”要分开操作。

### 20.14 超过 4 GB 后仍试图让 7B 网关完成

**现象：**

7B server 的 warmup 请求：

```text
prefill_seconds = 147.510645
```

第二请求继续 paging。

**错误做法：**

继续等待，希望第一次之后会变快。

**正确做法：**

记录 paging 事实，改用 DeepInfer SM86 native harness 获取可完成的当前数据，同时保留干净环境的
历史 25.962/26.188 结果，不伪造“同协议”数字。

### 20.15 仓库 `.gitignore` 把源码目录也忽略了

**现象：**

GitHub 上只有文档、脚本和 `.cu` 文件，没有：

```text
deepinfer/model_executor/models/qwen2.py
deepinfer/model_executor/models/qwen2_w4a16.py
deepinfer/model_executor/models/qwen2_mixed_sm86.py
```

**原因：**

`.gitignore` 写了 `models/`，把 `deepinfer/model_executor/models/` 也匹配掉了。

**修复：**

```text
/models/
/checkpoints/
```

只忽略仓库根目录的权重目录。

**教训：**

推送后不能只看 README 的 200 响应，还要检查 tree 是否包含真正的源码。

**当前状态：**

这个修复和源码仍然在本地未提交状态；远端还没有新的内核提交。

### 20.16 PowerShell 中文脚本编码

**现象：**

```text
Missing closing ')' in expression
```

**原因：**

Windows PowerShell 5.1 把无 BOM 的 UTF-8 脚本按 ANSI 读取，中文
`ValidateSet` 直接语法错误。

**修复：**

`desktop/quick_ask.ps1` 保存为 UTF-8 with BOM。

### 20.17 SSE 不关闭导致 curl 一直等待

**现象：**

流式内容已经收完，但 `curl` 不退出。

**原因：**

发送 `[DONE]` 后仍然保持 keep-alive。

**修复：**

```text
Connection: close
self.close_connection = True
```

### 20.18 UTF-8 流式重复

**现象：**

流式文本出现：

```text
... CUDA �CUDA Graph 是一种...
```

**原因：**

byte-level BPE 在一个汉字中间切断 token，直接 decode 前缀得到 `�`，下一 token
补齐后又重复输出整段。

**修复：**

- 只发送 `U+FFFD` 之前的稳定前缀；
- 等下一个 token 补齐；
- 非流式收集完整 token IDs 后一次 decode。

---

## 21. 哪些结论可以说，哪些不能说

### 可以说

- 3B DeepInfer SM86 W4A16 在 RTX 3050 4 GB 上达到约 44.8 tokens/s；
- 3B DeepInfer SM86 短任务一次请求约 0.9 秒；
- 7B DeepInfer SM86 mixed 在隔离环境约 26 tokens/s；
- Ollama 3B 在相同输入下约 57.3 tokens/s；
- 当前 DeepInfer SM86 网关 context 是 256，Ollama 是 4096；
- 4 GB 上 3B/7B 不能同时常驻；
- 全 Q4 7B 能跑，但只有约 7.9 tokens/s。

### 不能说

- “DeepInfer 已经全面超过 Ollama”；
- “7B 在 4 GB 上可以稳定生产”；
- “3-bit MLP/head 与 BF16 等价”；
- “当前是完整 DeepInfer engine”；
- “512 context 已经交付”；
- “7B paging 下的退化文本是模型质量问题”；
- “兼容层 5.9 tokens/s 是 DeepInfer SM86 native 性能”。

---

## 22. 复现清单

### 22.1 3B DeepInfer SM86 checkpoint

```powershell
python scripts\export_qwen3b_w4_checkpoint.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --output D:\Qwen3B-W4A16-SM86
```

产物：

```text
D:\Qwen3B-W4A16-SM86
约 1.912 GiB
```

### 22.2 3B DeepInfer SM86 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

### 22.3 7B DeepInfer SM86 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

### 22.4 三后端基准

```powershell
python scripts\bench_productivity_backends.py
```

或：

```powershell
python scripts\bench_productivity_backends.py --skip-7b
python scripts\bench_productivity_backends.py --skip-3b --skip-7b
```

### 22.5 源码测试

```powershell
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest `
  tests\models\test_qwen2_source_integration.py
```

结果：

```text
6 passed
```

### 22.6 Ollama 服务

如果基准报：

```text
WinError 10061 目标计算机积极拒绝
```

先启动：

```powershell
ollama serve
```

---

## 23. 最终选择

| 场景 | 选择 | 理由 |
|---|---|---|
| 剪贴板总结/改写/翻译/待办 | 3B DeepInfer SM86 | 约 0.9 秒，本地、离线 |
| 最快 3B | Ollama Q4_K_M | 约 0.46 秒 |
| 更好的短文本质量 | 7B DeepInfer SM86 mixed | 但必须隔离显存 |
| 长上下文 | Ollama 4096 | DeepInfer SM86 网关当前只有 256 |
| 生产级 | 继续做 engine | 单请求、短上下文、早期 runtime |

如果只保留一句话：

> **3B DeepInfer SM86 已经可以每天用；7B DeepInfer SM86 还是“边缘可用”；Ollama 仍然是 3B 小模型最成熟的基线。**

---

## 24. 当前仓库状态与诚实说明

本地工作区：

```text
D:\DeepInfer-SM86
```

远端：

```text
https://github.com/umouyoumou-del/DeepInfer-SM86
HEAD = 29ddbea
```

当前本地状态：

- 3B W4A16 checkpoint、3B backend、256 GQA、基准脚本、报告全部在本地；
- `.gitignore` 已修复，`deepinfer/model_executor/models/` 源码已在本地；
- 3B 分层 checkpoint 在 `D:\Qwen3B-W4A16-SM86`；
- 7B 混合 checkpoint 在 `D:\Qwen7B-Mixed-SM86`；
- **这些最新改动尚未提交/推送**；
- 远端还缺少 3B/基准这一版，也没有完整的 `deepinfer/model_executor/models/` 源码；
- 按“先不要提交”的要求，没有继续执行 git。

这不是“已经全部上云”的状态。  
这篇文章的价值之一，就是把“做出来了”“能跑”“已经推送”三件事分开写清楚。
