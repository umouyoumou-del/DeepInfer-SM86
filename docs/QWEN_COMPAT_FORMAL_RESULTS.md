# DeepInfer-Qwen 兼容层正式复测报告

> 测试日期：2026-10-25
> 测试对象：DeepInfer Qwen compatibility backend
> 结构化原始结果：[`results/qwen_compat_formal_20261025.json`](results/qwen_compat_formal_20261025.json)
> 适用边界：兼容性验证，不代表 DeepInfer SM100 原生内核性能

## 1. 结论摘要

- CPU 微型 Qwen2 + LoRA 冒烟测试通过。
- Qwen2.5-3B + `maibot-catgirl-v3/checkpoint-120` 能在 4 GB RTX 3050 上以 NF4 4-bit 加载并生成。
- 直接 Transformers/NF4 路径在正式复测中的中位端到端吞吐为 **5.554 tokens/s**，中位 decode 吞吐为 **5.859 tokens/s**。
- 直接 NF4 路径的 10 轮连续对话 **10/10 成功**；reserved VRAM 在 2332 MiB 附近进入平台期，未观察到持续线性增长。
- Ollama `maibot-catgirl-v3:latest` 使用 Q4_K_M + GGUF LoRA，模型以 **100% GPU** 运行；正式复测中位 decode 吞吐为 **53.178 tokens/s**，10 轮稳定性 **10/10 成功**，VRAM 分配保持恒定。
- OpenAI 兼容端点 `/v1/chat/completions` 调用成功并返回标准 `chat.completion` 结构。
- 两条路径在 3 个固定 prompt 上输出不完全相同，说明量化格式和运行时差异会影响具体文本；字符相似度只能作为描述性指标。

**最终表述：Ollama 路径在系统层面达到 Ollama 级别的 decode 吞吐，但这是后端委托，不是 Transformers/NF4 内核提速，也不能用于证明 DeepInfer 原生内核性能。**

## 2. 环境与协议

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU，4 GB，SM86 |
| OS | Windows 10.0.26200 |
| Python | 3.10.3 |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.49.0 |
| PEFT | 0.14.0 |
| bitsandbytes | 0.48.2 |
| Accelerate | 1.3.0 |
| Ollama | 0.24.0 |
| 基座 | Qwen2.5-3B-Instruct，本地 small-shards 权重 |
| Adapter | `F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120` |
| Ollama 模型 | `maibot-catgirl-v3:latest`，Qwen2 Q4_K_M + GGUF LoRA |

固定评测参数：

- system prompt：`F:\QwenQLoRA\system_prompt.txt`，318 字符；
- 吞吐测试：3 个固定 prompt，`max_new_tokens=64`，采样温度 0.7、top_p 0.9、top_k 50、seed 20261025；
- 稳定性测试：10 轮连续对话，`max_new_tokens=16`，贪心解码；
- 一致性测试：3 个固定 prompt，贪心解码，`max_new_tokens=64`；
- Transformers：NF4 double quant + FP16 compute + SDPA；
- Ollama 吞吐测试：每个请求使用 `keep_alive=0`，将模型加载时间与 prompt/decode 时间分开记录；
- Ollama 稳定性测试：模型加载后使用 `keep_alive=10m`，逐轮查询 `/api/ps` 和 `nvidia-smi`。

## 3. CPU 冒烟测试

使用真实 Qwen tokenizer、合成 1 层 Qwen2 模型和随机 PEFT LoRA adapter，在 CPU、float32、eager attention 下执行。

| 检查项 | 结果 |
|---|---|
| 模型和 adapter 加载 | 通过 |
| ChatML 模板渲染 | 通过 |
| `chat_result` 生成 | 通过 |
| `chat_completion` OpenAI 形状 | 通过 |
| prompt tokens | 30 |
| completion tokens | 3 |
| OpenAI usage total | 32 |
| TTFT / decode 计时字段 | 已产生且非空 |
| 测试退出码 | 0 |

正式 CPU 冒烟脚本：`scripts/smoke_qwen_cpu.py`
原始结果：`doc/results/qwen_compat_cpu_smoke_20261025.json`

该测试只验证控制流、模板、adapter 加载和接口形状，不代表 3B 模型的性能。

## 4. 直接 Transformers/NF4 路径

### 4.1 加载与显存

| 指标 | 结果 |
|---|---:|
| 加载耗时 | 267.613 秒 |
| 加载峰值 allocated | 2074.2 MiB |
| 加载峰值 reserved | 2224.0 MiB |
| 加载后空闲显存 | 655.7 MiB |
| 运行设备 | `cuda:0` |
| 吞吐测试峰值 reserved | 2288.0 MiB |
| 10 轮稳定性峰值 reserved | 2332.0 MiB |

加载耗时受 F: 盘文件缓存和系统负载影响，不能作为稳定的磁盘 I/O 基准。

### 4.2 吞吐与首 token 延迟

每次运行都输出到 EOS 自然停止；三次均生成 15 个 completion tokens。

| 轮次 | TTFT | decode 时间 | 完成 tokens | 端到端 tokens/s | decode tokens/s |
|---|---:|---:|---:|---:|---:|
| 1 | 1.869 秒 | 2.396 秒 | 15 | 3.514 | 5.842 |
| 2 | 0.317 秒 | 2.382 秒 | 15 | 5.554 | 5.878 |
| 3 | 0.278 秒 | 2.389 秒 | 15 | 5.619 | 5.859 |
| **中位数** | **0.317 秒** | — | 15 | **5.554** | **5.859** |

TTFT 使用 Transformers `StoppingCriteria` 在首个生成 token 被采样时计时；它不是服务端网络延迟。

此前固定 32-token 控制实验的结果为 **6.089 tokens/s**，与本次短序列正式复测的 decode 结果一致。

### 4.3 10 轮稳定性

10 轮对话全部成功，共生成 80 个 completion tokens。

| 轮次 | 完成 tokens | 峰值 reserved | 空闲显存 |
|---|---:|---:|---:|
| 1 | 5 | 2288.0 MiB | 575.7 MiB |
| 2 | 7 | 2288.0 MiB | 575.7 MiB |
| 3 | 10 | 2288.0 MiB | 575.7 MiB |
| 4 | 6 | 2332.0 MiB | 531.7 MiB |
| 5 | 10 | 2332.0 MiB | 531.7 MiB |
| 6 | 16 | 2332.0 MiB | 531.7 MiB |
| 7 | 10 | 2332.0 MiB | 531.7 MiB |
| 8 | 4 | 2332.0 MiB | 531.7 MiB |
| 9 | 6 | 2332.0 MiB | 531.7 MiB |
| 10 | 6 | 2332.0 MiB | 527.5 MiB |

reserved 显存在第 1～4 轮增加 44 MiB，随后 7 轮保持不变；没有观察到持续增长。该结果支持短会话稳定性，但不等于长上下文或长时间服务稳定性。

## 5. Ollama Q4_K_M + GGUF LoRA 路径

### 5.1 模型与服务状态

- 模型：`maibot-catgirl-v3:latest`
- 格式：GGUF，Q4_K_M
- 参数规模：约 3.1B
- 能力：`completion`、`tools`
- Modelfile 中检测到 `ADAPTER`
- `/api/ps`：加载 2,390,300,672 bytes，100% GPU，context 4096
- 直接服务端 OpenAI 兼容端点测试成功

### 5.2 独立请求吞吐

三个请求均使用 `keep_alive=0`，Load 时间单独记录；Ollama decode 吞吐使用原生 `eval_count / eval_duration`。

| 轮次 | 模型 load | prompt eval | 完成 tokens | decode 时间 | decode tokens/s | 请求 wall |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2.108 秒 | 0.223 秒 | 8 | 0.150 秒 | 53.178 | 2.499 秒 |
| 2 | 2.637 秒 | 0.230 秒 | 7 | 0.135 秒 | 51.860 | 3.018 秒 |
| 3 | 0.088 秒 | 0.044 秒 | 8 | 0.146 秒 | 54.766 | 0.290 秒 |
| **中位数** | — | **0.223 秒** | 8 | — | **53.178** | — |

第 3 次请求的 load 和 prompt eval 明显较低，可能受益于模型 relic/cache 状态，因此正式论文不应把此列解释为完全独立的冷启动基准。更早的同 prompt 统一基准在热态下测到 58.8～59.8 decode tokens/s，可作为更高吞吐参考。

### 5.3 10 轮稳定性与显存

10 轮连续对话全部成功：

| 指标 | 结果 |
|---|---:|
| 成功轮次 | 10 / 10 |
| 中位 prompt eval（热态） | 0.0436 秒 |
| 中位 decode 吞吐 | 55.750 tokens/s |
| `/api/ps` `size_vram` | 每轮均为 2,390,300,672 bytes |
| nvidia-smi used | 第 1 轮与第 10 轮均为 3001 MiB |
| context length | 4096 |

Ollama 在 10 轮中保持固定 VRAM 分配，未观察到逐轮增长。

### 5.4 OpenAI 兼容接口

请求 `/v1/chat/completions` 成功：

- `object=chat.completion`
- `finish_reason=stop`
- prompt tokens：229
- completion tokens：8
- total tokens：237
- wall time：0.273 秒
- 输出：`我是黑龙江的高一学生w`

## 6. 输出一致性对照

使用 3 个固定 prompt、贪心解码：

| Prompt | Transformers/NF4 | Ollama Q4_K_M + LoRA | 字符相似度 |
|---|---|---|---:|
| 你是谁？ | 我是猫猫w | 我是 catgirl，黑龙江高一学生w | 0.2500 |
| 用群里的黑话吐槽一下数学考试。 | “这题咋这么难啊，我不会了” | “这题咋这么绕啊，我还没想明白呢” | 0.6452 |
| 群里有人发了很奇怪的东西，你会怎么回？ | “这东西是啥玩意啊”w | “这东西”是什么意思w | 0.6364 |

0/3 输出完全一致。字符相似度只能说明表面文本接近程度，不能证明语义等价；NF4 与 Q4_K_M 的量化方式和推理实现都不同。

## 7. 解释与限制

1. **后端定位必须写清楚**：53～56 tokens/s 来自 Ollama 的优化 CUDA 运行时；Transformers/NF4 直载路径仍为约 6 tokens/s。
2. **不能证明 DeepInfer 原生性能**：兼容层绕过 SM100 专用 FlashMLA、DeepGEMM、DeepEP 和原生调度器。
3. **Ollama 冷/热态存在缓存影响**：正式吞吐拆分了 load、prompt eval 和 eval duration；第 3 次请求显示明显的服务端缓存效应。
4. **自然停止长度不同**：正式吞吐 prompt 在 7～15 tokens 时停止；此前 32-token 控制实验仍应作为主要长度固定参考。
5. **TTFT 定义不同**：Transformers 记录首 token 采样时间；Ollama 记录原生 `prompt_eval_duration`。两者可比性有限。
6. **显存稳定性范围有限**：10 轮短对话不能替代长上下文、并发或小时级压力测试。
7. **输出质量未做人工语义评分**：本报告只做控制流、性能和短期稳定性验证。
8. **Windows 与 Linux 差异**：实验在 Windows 上运行，不是 DeepInfer 原生目标平台。

## 8. 复现命令

CPU 冒烟：

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe scripts\smoke_qwen_cpu.py `
  --output D:\DeepInfer\doc\results\qwen_compat_cpu_smoke_20261025.json
```

正式双后端复测：

```powershell
cd D:\DeepInfer
F:\Ollama\app\ollama.exe stop maibot-catgirl-v3:latest
F:\QwenQLoRA\venv\Scripts\python.exe scripts\bench_qwen_compat.py `
  --output D:\DeepInfer\doc\results\qwen_compat_formal_20261025.json
```

只复测 Ollama 阶段：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe scripts\bench_qwen_compat.py `
  --skip-transformers `
  --output D:\DeepInfer\doc\results\qwen_compat_formal_ollama_rerun.json
```

验证正式结果：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe scripts\validate_qwen_results.py `
  --results D:\DeepInfer\doc\results\qwen_compat_formal_20261025.json
```

## 9. 结果文件

| 文件 | 内容 |
|---|---|
| `doc/results/qwen_compat_formal_20261025.json` | 正式复测结构化原始结果 |
| `doc/results/qwen_compat_formal_ollama_rerun.json` | 补充逐轮显存采样后的 Ollama 原始结果 |
| `doc/results/qwen_compat_cpu_smoke_20261025.json` | CPU 冒烟结构化结果 |
| `doc/results/SHA256SUMS` | 结果 JSON、正式脚本和核心代码的 SHA256 |
| `doc/QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json` | 早期统一 benchmark |
| `doc/QWEN_COMPAT_OLLAMA_BENCHMARK.json` | 早期 Ollama 统一 benchmark |
| `doc/QWEN_COMPAT_OPTIMIZATION_RESULTS.json` | 静态 cache / Flash-SDPA / compile 消融 |

正式结果 JSON SHA256：

```text
e3e45513f834bbf2245ad56420a20d5c9577814d09279d298b9e1af402957720
```
