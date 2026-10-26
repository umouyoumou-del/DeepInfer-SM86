# 3B / 7B / Ollama 生产力基准记录

> 系列下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`

> 术语约定：本文中的 **native** 一律指 **DeepInfer SM86 native**
> （DeepInfer model_executor / W4A16 / W3A16 / GQA / CUDA Graph），
> 不是 Ollama、llama.cpp、Transformers 兼容层或第三方 native engine。

> 日期：2026-10-26  
> 硬件：RTX 3050 Laptop 4 GB（SM86）、Windows  
> 目标：验证 DeepInfer-SM86 的 3B 与 7B 常驻模型是否达到个人生产力可用，并与 Ollama 3B 对比。

---

## 1. 结论先行

| 后端 | 模型 | 量化 | context | decode | 一次请求 wall |
|---|---|---|---:|---:|---:|
| **3B DeepInfer SM86 native** | Qwen2.5-3B + catgirl LoRA | W4A16 | 256 | **44.773 tokens/s** | **0.895 s** |
| **Ollama 3B** | maibot-catgirl-v3 | Q4_K_M | 4096 | **57.285 tokens/s** | **0.464 s** |
| **7B DeepInfer SM86 native（隔离最佳）** | Qwen2.5-7B base | 4-bit attn/embed + 3-bit MLP/head | 64 测速配置 | **25.962 / 26.188 tokens/s** | 见历史结果 |
| **7B DeepInfer SM86 native（当前 paging）** | Qwen2.5-7B base | 同上 | 64 | 25.141 tokens/s | prefill 53.089 s |

结论：

1. **3B DeepInfer SM86 native 已经达到个人快捷操作可用水平**：一次 64-token 上限请求约 0.9 秒。
2. **Ollama 3B 仍然更快**：decode 约快 28%，wall 约快 1.9 倍；Ollama 的 prefill 只有约 0.06 秒，native 是 0.46 秒。
3. **7B DeepInfer SM86 native 不是不能跑，而是 4 GB 边界很窄**：隔离环境可以 25.96 tokens/s；后台多占约 500 MiB 时 prefill 会进入严重 paging。
4. **三者的 context 不同**：native 256，Ollama 4096；这是当前模型服务层的差距，不是单纯 kernel 速度差距。

---

## 2. 三个后端

### 2.1 3B DeepInfer SM86 native

- 模型：Qwen2.5-3B-Instruct + `maibot-catgirl-v3` LoRA；
- 格式：W4A16，int4 group 128；
- 权重：dense embedding + W4A16 linear；
- 有效位宽约 4.125 bpw（linear 部分），tied embedding；
- checkpoint：`D:\Qwen3B-W4A16-SM86`；
- checkpoint 大小：**1.912 GiB**；
- 加载：分层 safetensors，**11.75 秒**；
- context：256；
- 后端：`--backend w4a16`。

### 2.2 7B DeepInfer SM86 native

- 模型：Qwen2.5-7B-Instruct（base，无 LoRA）；
- 4-bit attention/embed + 3-bit MLP/head；
- checkpoint 2.930 GiB；
- context 256（网关配置）；
- 隔离环境历史结果：25.962 平均 / 26.188 中位 tokens/s，prefill 3.061 秒；
- 当前环境 paging 结果：decode 25.141 tokens/s，prefill 53.089 秒。

### 2.3 Ollama 3B

- 模型：`maibot-catgirl-v3:latest`；
- 格式：Q4_K_M；
- 大小：1.96 GB；
- Ollama 报告 context：4096；
- 100% GPU；
- 通过 `/api/chat` 测试。

不是完全同模型：

- 3B DeepInfer SM86 native 和 Ollama 都是 3B catgirl 路线；
- 7B DeepInfer SM86 native 是 base 模型，质量定位不同；
- 因此 7B 数字只能做硬件/后端对比，不能直接拿来做质量排名。

---

## 3. 基准协议

3B 与 Ollama 使用完全相同的文本协议：

```text
system: 你是一个简洁、可靠的中文个人助理。只输出最终答案，不要复述任务。

prompt 1: 用一句话解释 CUDA Graph。
prompt 2: 把下面这段话压缩成一句待办：明天下午三点前把实验报告发给导师，并附上结果表格。
```

参数：

```text
max_tokens   = 64
temperature  = 0
top_p        = 0.95
top_k        = 50
seed         = 20261026
warmup       = 1
repeats      = 2（两个 prompt 各跑两轮）
```

测量：

- native：HTTP wall + `/health.last_timings` 的 prefill/decode；
- Ollama：`/api/chat` 的 `load_duration`、`prompt_eval_duration`、`eval_duration`、`eval_count`；
- 不把模型加载时间混入 decode。
- 3B 与 Ollama 顺序运行，避免 4 GB 显存重叠。

7B 例外：

- 当前桌面后台约 500 MiB GPU 占用，7B 网关 prefill 超过 120 秒仍无法完成；
- 因此当前 7B 使用 DeepInfer SM86 native harness 记录 paging 数据；
- 另保留隔离环境的历史 25.962/26.188 结果；
- 这是记录真实环境限制，不把 paging 结果包装成正常吞吐。

---

## 4. 原始结果

### 4.1 3B DeepInfer SM86 native（同协议）

| 指标 | 结果 |
|---|---:|
| load | 11.75 秒 |
| prefill 中位 | 0.4559 秒 |
| decode 中位 | **44.773 tokens/s** |
| wall 中位 | **0.8948 秒** |
| completion 中位 | 19 tokens |

示例输出：

```text
CUDA Graph允许开发者高效管理GPU执行流程，减少不必要的资源准备和清理。
```

第二个 prompt：

```text
明天下午三点前将实验报告发给导师，并附上结果表格。事项已完成。
```

### 4.2 Ollama 3B（同协议）

| 指标 | 结果 |
|---|---:|
| prefill 中位 | 0.0603 秒 |
| decode 中位 | **57.285 tokens/s** |
| wall 中位 | **0.4637 秒** |
| completion 中位 | 15.5 tokens |

示例输出：

```text
CUDA Graph 是一种用于优化 CUDA 应用程序性能的技术。
```

```text
明天下午三点前把实验报告发给导师，并附上结果表格
```

### 4.3 7B DeepInfer SM86 native

隔离环境历史结果：

| 指标 | 结果 |
|---|---:|
| prefill | 3.061 秒 |
| decode 平均 | 25.962 tokens/s |
| decode 中位 | 26.188 tokens/s |
| 峰值 allocated | 3.196 GiB |
| 峰值 reserved | 3.357 GiB |

当前环境 paging 结果：

| 指标 | 结果 |
|---|---:|
| prefill | **53.089 秒** |
| decode 平均 | 25.141 tokens/s |
| decode 中位 | 25.128 tokens/s |
| 峰值 allocated | 3.196 GiB |
| 峰值 reserved | 3.357 GiB |

当前 paging 输出出现退化：

```text
CUDA Graph是一种编译时<decltypeoriouslylopenngen<decltypeoriouslylopenngen>>>>>>>>来优化计算图以
```

这个输出不能作为质量结果，只说明**显存边界下 3-bit + paging 会破坏生成稳定性**。

---

## 5. 对比分析

### 5.1 3B DeepInfer SM86 native vs Ollama

| 指标 | 3B DeepInfer SM86 native | Ollama 3B | 差距 |
|---|---:|---:|---:|
| prefill | 0.4559 s | 0.0603 s | native 慢 7.6 倍 |
| decode | 44.773 t/s | 57.285 t/s | native 慢约 22% |
| wall | 0.8948 s | 0.4637 s | native 慢约 1.9 倍 |
| context | 256 | 4096 | Ollama 大 16 倍 |

这说明：

- DeepInfer SM86 的 **W4A16 decode 已经可用**；
- 当前主要差距在 **prefill、调度、上下文长度和完整 engine**；
- 不是“模型不会跑”，而是“服务层还不够成熟”。

### 5.2 3B vs 7B

- 3B DeepInfer SM86 native：44.8 tokens/s，0.9 秒内完成短任务；
- 7B DeepInfer SM86 native：约 26 tokens/s，但 4 GB 边界更危险；
- 7B 的理论质量更高，但当前 3-bit MLP/head 是有损的；
- 对“剪贴板总结/改写/翻译/待办提取”，3B 的速度和稳定性更适合作为默认；
- 对需要更强推理的短文本，可以在 GPU 隔离时切 7B。

---

## 6. 3B 的生产力适配做了什么

### 6.1 分层 checkpoint

之前 3B 每次启动都要：

```text
CPU dense 加载 + LoRA merge
→ GPU float16
→ 逐层 W4A16 量化
```

耗时约 5 分钟，不能做常驻服务。

新增：

```text
scripts/export_qwen3b_w4_checkpoint.py
deepinfer/model_executor/models/qwen2_w4a16.py::from_checkpoint
```

导出：

```text
D:\Qwen3B-W4A16-SM86
  config.json
  tokenizer.json
  manifest.json
  embed.safetensors
  norm.safetensors
  layer_00.safetensors ... layer_35.safetensors
```

加载时间从约 5 分钟降到 **11.75 秒**。

### 6.2 3B 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

模型名：

```text
deepinfer-sm86-3b-w4a16
```

### 6.3 7B 网关

```powershell
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18002
```

4 GB 上不要同时启动两个模型。

---

## 7. 基准脚本

```powershell
cd D:\DeepInfer-SM86

# 3B + 7B + Ollama
python scripts\bench_productivity_backends.py

# 只跑 3B
python scripts\bench_productivity_backends.py --skip-7b --skip-ollama

# 只跑 Ollama
python scripts\bench_productivity_backends.py --skip-3b --skip-7b
```

脚本会按顺序启动/停止 native server，避免 4 GB 显存重叠。

> 注意：本机 8000 被 MaiBot 占用，native 网关默认使用 18000/18001/18002。

---

## 8. 这次又遇到的坑

### 8.1 3B dense 转换不能每次启动都做

第一次直接加载 dense 3B + LoRA 再量化，约 5 分钟。  
所以必须先导出分层 checkpoint；否则“3B 生产力”只是换了个启动方式。

### 8.2 tied embedding

3B `tie_word_embeddings=true`，`lm_head.weight` 与 embedding 共享。  
checkpoint 只保存 `embed.safetensors`，loader 不再重复保存 lm_head。

### 8.3 7B 网关 paging

后台约 500 MiB GPU 占用时：

- 7B 网关 prefill 超过 120 秒；
- DeepInfer SM86 native harness 可以跑完，但 prefill 53 秒；
- 同一模型在更干净环境下 prefill 3.06 秒。

这不是 kernel 退化，而是 **4 GB 总显存 + WDDM paging**。

### 8.4 512 context 仍未交付

512-thread GQA 路径在集成 runtime 中出现输出退化，  
当前稳定版本上限是 **256 context**，长文必须先分段。

### 8.5 3B 结果的质量边界

3B 输出可读、速度快，但它：

- 是 W4A16 有损量化；
- 带有 catgirl LoRA，和 base 7B 不是同一质量定位；
- context 256；
- 不能把 3B 的 44.8 tokens/s 外推成 7B 速度。

---

## 9. 原始文件

- `docs/results/productivity_3b_only.json`
- `docs/results/productivity_ollama_only.json`
- `docs/results/qwen7b_mixed_head3_final.json`
- `docs/results/qwen7b_current_paging_harness.json`（当前 paging 7B）
- `docs/results/productivity_backends_combined.json`
- 3B checkpoint：`D:\Qwen3B-W4A16-SM86`
- 7B checkpoint：`D:\Qwen7B-Mixed-SM86`

## 10. 最终建议

- **日常快捷操作**：优先 3B DeepInfer SM86 native 或 Ollama 3B；
- **要更快**：目前 Ollama 3B 仍是本机最快的 3B 路径；
- **要 DeepInfer 原生**：3B W4A16 已可用，44.8 tokens/s，0.9 秒级；
- **要 7B**：尽量关闭其它 GPU 程序；否则会 paging，prefill 从 3 秒变成 53 秒以上；
- **要长上下文**：当前 native 网关 256 context 不够，需要分段或继续做 512/长 KV；
- **要生产级**：仍需 CacheEngine、Sampler、continuous batching 和 tiled prefill。

这不是“DeepInfer 已经全面超过 Ollama”，而是：

> 3B 的 SM86 原生路径已经进入个人生产力可用区间；  
> 7B 在 4 GB 上仍然脆弱；  
> Ollama 在 3B 小模型上仍是更成熟的基线。
