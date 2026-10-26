# DeepInfer-Qwen 兼容层：设计、实现与实验方法

> 文档版本：0.4
> 状态：CPU 冒烟、真实 GPU 生成、Ollama 对接、TTFT、10 轮稳定性、
> OpenAI 兼容接口与正式双后端复测均已完成；
> 正式结果见 [`QWEN_COMPAT_FORMAL_RESULTS.md`](QWEN_COMPAT_FORMAL_RESULTS.md)；
> 高性能数字来自 Ollama 后端委托，不能表述为 Transformers/NF4 原生内核优化
> 适用对象：DeepInfer + Qwen2.5-3B QLoRA 开发与论文方法部分

## 摘要

DeepInfer 是面向 DeepSeek V4.1 系列模型的高性能推理框架，其原生执行路径依赖 SM100/SM103 GPU、Linux、CUDA 13 和专用 FlashMLA/DeepGEMM/DeepEP 内核。本文设计并实现一个可选的 Qwen2/Qwen2.5 兼容层，使 Qwen2.5-3B 基座模型及其 PEFT LoRA 适配器能够在普通 CUDA 显卡（实验平台为 4 GB 显存的 RTX 3050 Laptop GPU）上完成加载、对话生成和接口验证。兼容层提供两种共享同一 API 的后端：一是 Transformers + PEFT + bitsandbytes 的 NF4/PyTorch 直载后端；二是调用本机 Ollama 服务的高性能后端。实验完成了 CPU 微型 Qwen2 + LoRA 冒烟测试、真实 Qwen2.5-3B + checkpoint-120 的 4-bit GPU 生成、Ollama 模型 `maibot-catgirl-v3:latest` 的 100% GPU 加载，以及同一 system/user prompt 下的双后端对照。结果显示：NF4 直载后端在 32-token 生成时约为 6.1–6.3 tokens/s；静态 KV Cache、Flash-SDPA 和 `torch.compile` 均未带来实质提升；Ollama 高性能后端的原生 decode 指标达到约 59.8 tokens/s（58.8–59.8），从而在系统层面达到 Ollama 性能。正式复测进一步加入重复吞吐、首 token 延迟、10 轮连续对话和 OpenAI 兼容接口验证：NF4 直载路径中位端到端吞吐为 5.554 tokens/s、decode 为 5.859 tokens/s，首 token 中位延迟为 0.317 秒，10 轮全部成功；Ollama 独立请求中位 decode 为 53.178 tokens/s，10 轮热态稳定对话中位 decode 为 55.750 tokens/s，且 10 轮全部成功。两条路径的输出在固定 prompt 上并不完全一致。该高性能结果必须明确表述为**后端委托**，而不是 Transformers/NF4 内核本身达到 Ollama 吞吐。

**关键词：** DeepInfer；Qwen2.5；QLoRA；NF4；兼容层；推理框架；低显存部署

## Abstract

DeepInfer is a high-performance inference framework specialized for DeepSeek V4.1 and SM100-class NVIDIA GPUs. Its DeepInfer SM86 native kernel path is not directly applicable to ordinary CUDA GPUs or Qwen2/Qwen2.5 architectures. This work presents an optional Qwen compatibility layer with two interchangeable backends: a direct Transformers + PEFT + bitsandbytes NF4 backend, and a high-performance backend that delegates to a local Ollama service. Experiments on a 4 GB RTX 3050 Laptop GPU completed a synthetic Qwen2+LoRA smoke test, real Qwen2.5-3B checkpoint-120 generation, and a same-prompt comparison against the exported Ollama model `maibot-catgirl-v3:latest`. The direct NF4 backend reached about 6.1–6.3 tokens/s for 32-token generation; static KV cache, Flash-SDPA and `torch.compile` produced no material improvement. The Ollama backend reported about 59 native decode tokens/s (58.8–59.8) and therefore reaches Ollama-level throughput at the system level. A formal replication additionally measured median end-to-end 5.554 tokens/s and decode 5.859 tokens/s for the direct NF4 backend, with a median first-token latency of 0.317 s and 10/10 successful consecutive rounds. The Ollama path reached 53.178 decode tokens/s for independent requests and 55.750 tokens/s in warm 10-round stability runs, also with 10/10 success; its OpenAI-compatible endpoint was verified successfully. This high-performance result must be reported as backend delegation, not as a native speedup of the Transformers/NF4 execution path.

**Keywords:** DeepInfer; Qwen2.5; QLoRA; NF4; compatibility layer; low-VRAM inference

## 1. 研究背景与问题定义

### 1.1 原始系统约束

原始 DeepInfer 的部署条件与目标模型具有强耦合：

- DeepInfer 当前注册的模型架构仅为 `DeepseekV41ForCausalLM`。
- 原生 FlashMLA 的 CUDA 编译目标为 `sm_100a` / `sm_103a`。
- 原生 DeepGEMM 面向 SM90/SM100，而不是实验平台使用的 SM86。
- 原生运行时依赖 Linux、Python 3.12、CUDA 13 和 NCCL/DeepEP。
- 原生代码没有 CPU/磁盘权重分层加载路径，模型参数需要位于 CUDA 设备上。

实验平台为 Lenovo 笔记本，配置如下：

| 组件 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop，4 GB VRAM |
| Compute Capability | 8.6 (SM86) |
| CPU | AMD Ryzen 5 5600H，6C/12T |
| RAM | 16 GB |
| OS | Windows 11 |
| NVIDIA Driver | 546.30，CUDA 12.3 |
| Python（训练/兼容层） | 3.10 |
| 原有 Qwen 环境 | PyTorch 2.5.1+cu121，Transformers 4.49.0，PEFT 0.14.0，bitsandbytes 0.48.2 |

### 1.2 目标模型

当前正在训练的模型为：

- 基座：`Qwen/Qwen2.5-3B-Instruct`
- 训练方式：4-bit NF4 QLoRA
- 参数规模：约 3B
- 架构：`Qwen2ForCausalLM`
- hidden size：2048
- Transformer layers：36
- attention heads：16
- KV heads：2（GQA）
- intermediate size：11008
- vocab size：151936
- 原始精度：BF16

### 1.3 问题定义

给定 Qwen2.5-3B 基座权重和一个 PEFT LoRA checkpoint，在 4 GB 显存的消费级 GPU 上实现：

1. 不依赖 DeepInfer SM100 专用内核即可加载模型；
2. 支持 NF4 4-bit 量化以控制显存；
3. 支持加载训练产生的 LoRA adapter；
4. 支持 Qwen ChatML 对话模板和多轮消息；
5. 输出可被后续接口层使用的 OpenAI Chat Completions 形状结果；
6. 与 Ollama GGUF + LoRA 路径进行功能对照；
7. 明确区分“兼容性验证”和“DeepInfer 原始高性能推理”。

## 2. 设计目标与非目标

### 2.1 设计目标

- **硬件可运行性**：在 SM86、4 GB VRAM 的 RTX 3050 上完成 3B 模型的 NF4 推理。
- **模型可扩展性**：通过 Transformers 的 `AutoModelForCausalLM` 支持 Qwen2/Qwen2.5 系列。
- **训练产物可对接**：直接加载 PEFT LoRA checkpoint，无需先进行物理权重合并。
- **低侵入性**：作为独立兼容包存在，不改变 DeepInfer 原生 DeepSeek V4.1 执行路径。
- **可复现性**：提供固定模型路径、环境版本、采样参数和冒烟测试方法。
- **可迁移性**：同一套命令可在 CPU 微型模型测试和 GPU 真实模型测试之间切换。

### 2.2 非目标

该兼容层不以以下目标为设计目标：

- 不复现 DeepInfer 的 FlashMLA、DeepGEMM、DeepEP 性能；
- 不实现 DeepInfer 的 paged KV cache、专家并行、调度器和 CUDA Graph；
- 不替代生产级推理服务，如 vLLM、TensorRT-LLM 或 Ollama；
- 不提供工具调用的结构化解析；
- 不声称在 RTX 3050 上达到 DeepSeek V4.1 或 Qwen 的最优吞吐。

因此，论文中建议将该工作定位为：

> 一种面向低显存消费级 GPU 的 DeepInfer-Qwen 兼容性后端与实验对照方法，而不是 DeepInfer 原生内核的移植或性能优化。

## 3. 总体架构

### 3.1 模块划分

兼容层位于 DeepInfer 仓库的独立包中：

```text
D:\DeepInfer\
├── deepinfer_compat\
│   ├── __init__.py       # 导出两种 engine 和 GenerationOptions
│   ├── qwen.py           # 模型加载、NF4、PEFT、生成逻辑
│   ├── ollama.py         # Ollama 高性能后端与原生计时字段
│   ├── benchmark.py      # 双后端可复现基准
│   ├── __main__.py       # 命令行入口与交互式 REPL
│   └── README.md         # 使用说明
├── requirements-qwen-compat.txt
└── doc\
    ├── QWEN_COMPAT_METHODOLOGY_ZH.md
    ├── QWEN_COMPAT_BENCHMARK_RESULTS.json
    ├── QWEN_COMPAT_OPTIMIZATION_RESULTS.json
    ├── QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json
    └── QWEN_COMPAT_OLLAMA_BENCHMARK.json
```

### 3.2 与原始 DeepInfer 的关系

原始 DeepInfer 原生路径：

```text
DeepInfer engine
  -> DeepseekV41ForCausalLM
  -> FlashMLA / DeepGEMM / DeepEP
  -> SM100 CUDA kernels
```

兼容层有两条共享 API 的后端路径：

```text
路径 A：deepinfer_compat.QwenCompatEngine
  -> Transformers AutoModelForCausalLM
  -> bitsandbytes NF4 (optional)
  -> PEFT PeftModel (optional)
  -> PyTorch SDPA / eager attention
  -> generic CUDA or CPU inference

路径 B：deepinfer_compat.OllamaCompatEngine
  -> HTTP /api/chat
  -> local Ollama service
  -> Ollama CUDA runtime / GGUF Q4_K_M + LoRA
  -> native eval_count / eval_duration timings
```

三条执行路径（DeepInfer 原生、Transformers 兼容、Ollama 兼容）是**并行、可选、互不替换**的。兼容层不会导入 DeepInfer 的 `_C` 扩展，也不会初始化 DeepInfer 原生 platform/worker；Ollama 后端也不会返回 Transformers 的内部指标。

### 3.3 数据流

```text
messages/system/tools
        |
        v
Qwen tokenizer.apply_chat_template
        |
        v
input_ids (+ attention_mask)
        |
        v
model.generate
        |
        v
new token ids -> tokenizer.decode
        |
        v
GenerationResult(text, prompt_tokens, completion_tokens, finish_reason)
        |
        v
plain text / OpenAI-shaped chat.completion dictionary
```

Ollama 后端使用相同输入和输出结构，但将 tokenize/generate/decode 交给
`POST /api/chat`，并额外读取 `prompt_eval_count`、`eval_count`、
`prompt_eval_duration`、`eval_duration` 返回原生计时指标。

## 4. 实现方法

### 4.1 模型与分词器加载

兼容层使用 `AutoModelForCausalLM.from_pretrained` 和 `AutoTokenizer.from_pretrained`：

- 默认 `local_files_only=True`，保证实验只使用本地模型，不依赖网络；
- 通过 `device_map` 支持 `"auto"`、`"cpu"` 和单 GPU 设备映射；
- 默认设置 `padding_side="right"`，并在缺少 pad token 时使用 eos token；
- 通过 `trust_remote_code` 开关兼容需要自定义代码的模型；
- 使用 Qwen tokenizer 的 ChatML 模板渲染对话，避免手工拼接特殊 token。

### 4.2 NF4 4-bit 量化

当启用 `load_in_4bit=True` 时，构造：

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

设计理由：

1. NF4 相比普通 INT4 更适合近似正态分布的权重；
2. double quantization 进一步压缩量化常数；
3. 默认使用 FP16 计算：在 RTX 3050 的实测中，FP16 明显快于 BF16；兼容层保留 BF16 作为数值稳定性优先时的可选模式；
4. 该路径只要求普通 CUDA，不要求 SM100 或原生 FP8/FP4 张量核心。

### 4.3 LoRA adapter 加载

当提供 `adapter_path` 时：

```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False,
)
```

该方法直接读取训练脚本保存的 PEFT checkpoint，包括：

- `adapter_config.json`
- `adapter_model.safetensors`

不需要先将 adapter 合并进 BF16 基座，因此避免了 3B 模型完整合并所需的额外内存和磁盘空间。

### 4.4 对话模板与生成

对话消息首先通过 Qwen tokenizer 的 `apply_chat_template` 转换为模型训练时一致的文本格式：

- 支持 `system` 角色；
- 支持多轮 `user` / `assistant` 消息；
- 可选地将 `tools` 定义传入模板；
- 设置 `add_generation_prompt=True`，保证在 assistant 起始位置之后生成。

生成参数由 `GenerationOptions` 统一管理：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `max_new_tokens` | 256 | 最大生成 token 数 |
| `temperature` | 0.7 | 采样温度 |
| `top_p` | 0.9 | nucleus sampling |
| `top_k` | 50 | top-k sampling |
| `repetition_penalty` | 1.05 | 重复惩罚 |
| `do_sample` | True | 是否采样 |
| `seed` | None | 可选随机种子 |

当 `do_sample=False` 时使用贪心解码，用于可重复的冒烟测试。

### 4.5 OpenAI 形状输出

`chat_completion()` 返回与 OpenAI Chat Completions 结构相近的字典，包含：

- `id`：唯一响应标识；
- `object`：`chat.completion`；
- `created`：Unix 时间戳；
- `model`：模型或 adapter 名称；
- `choices[0].message.content`：生成文本；
- `choices[0].finish_reason`：`stop` 或 `length`；
- `usage.prompt_tokens` / `completion_tokens` / `total_tokens`。

该结构便于实验脚本、API 客户端或论文中的接口对照，但当前版本不解析 `tool_calls`。

### 4.6 资源与并发控制

- 生成调用由 `threading.Lock` 串行化，避免同一模型实例上的并发调用破坏生成状态；
- 使用 `torch.inference_mode()` 关闭梯度计算；
- `close()` 释放模型引用并调用 `torch.cuda.empty_cache()`；
- CLI 在支持该分配器的系统上于首次 CUDA 分配前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；Windows 上不请求该选项；
- 所有大型依赖采用延迟导入，便于在没有 Transformers/PEFT 的环境中仅做静态检查。

### 4.7 Ollama 高性能后端

`OllamaCompatEngine` 与 `QwenCompatEngine` 共享 `chat`、`chat_result` 和
`chat_completion` 接口，但内部通过本机 Ollama 的 `POST /api/chat` 执行推理：

- 模型以 Ollama tag 指定，例如 `maibot-catgirl-v3:latest`；
- system/messages 直接传入 Ollama 原生 chat 接口；
- `max_new_tokens` 映射为 `num_predict`；
- `temperature`、`top_p`、`top_k`、`repeat_penalty`、`seed` 映射为 Ollama options；
- 返回 `prompt_eval_count`、`eval_count`、`prompt_eval_duration`、`eval_duration`；
- OpenAI 形状响应额外包含 `x_deepinfer` 字段，报告后端名、prompt 评估时间、decode 时间和原生 decode tokens/s；
- 工具定义和 Ollama 原生 `tool_calls` 透传，不丢失工具调用结构。

该后端的定位是**性能后端委托**。它复用 Ollama 的 CUDA/GGUF 执行路径，因此
系统端吞吐可达到 Ollama 水平，但不能用来证明 Transformers/NF4 直载路径本身
达到了相同性能。

## 5. 实验设置

### 5.1 硬件与软件

| 类别 | 配置 |
|---|---|
| GPU | RTX 3050 Laptop，4 GB，SM86 |
| CPU | AMD Ryzen 5 5600H，6C/12T |
| RAM | 16 GB |
| OS | Windows 11 家庭版 26200 |
| Python | 3.10.3（训练/兼容层 venv） |
| PyTorch | 2.5.1+cu121 |
| Transformers | 4.49.0 |
| PEFT | 0.14.0 |
| bitsandbytes | 0.48.2 |
| Accelerate | 1.3.0 |
| Safetensors | 0.5.2 |

### 5.2 模型配置

- 基座：Qwen2.5-3B-Instruct
- 架构：`Qwen2ForCausalLM`
- hidden size：2048
- 层数：36
- attention heads：16
- KV heads：2
- intermediate size：11008
- vocab size：151936
- 原始 dtype：BF16

### 5.3 QLoRA 训练配置

当前训练配置如下：

| 参数 | 值 |
|---|---|
| 量化方式 | 4-bit NF4 + double quant |
| 计算 dtype | BF16 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 最大序列长度 | 256 |
| batch size | 1 |
| gradient accumulation | 8 |
| 学习率 | 5e-5 |
| 训练目标 | 120 optimizer steps |
| 实际训练样本 | 1146 条训练、65 条验证 |
| 实际训练步数 | 120 |
| 训练耗时 | 10034.14 秒（约 2 小时 47 分） |
| 最终验证 loss | 3.1553 |
| 训练峰值 allocated | 3.8028 GiB |
| 训练峰值 reserved | 4.1523 GiB |
| 当前运行输出 | `F:\QwenQLoRA\train\maibot-catgirl-v3` |
| 最终 adapter | `train/maibot-catgirl-v3/checkpoint-120` |

### 5.4 实验路径

论文中建议报告三条互相对照的执行路径：

**路径 A：DeepInfer-Qwen 兼容层，Transformers/NF4 直载后端**

- 基座 + PEFT adapter；
- `load_in_4bit=True`；
- Transformers SDPA；
- 用于验证算法、接口和低显存推理可行性；
- 实测 32-token 生成约 6.1–6.3 tokens/s。

**路径 B：DeepInfer-Qwen 兼容层，Ollama 高性能后端**

- 以 `qwen2.5:3b-instruct-q4_K_M` 为基座；
- 将 PEFT LoRA 转换为 GGUF adapter；
- 使用 Modelfile `ADAPTER` 创建 Ollama 模型；
- 兼容层通过 Ollama 原生 `/api/chat` 调用并转发原生计时指标；
- 实测原生 decode 约 59.8 tokens/s（58.8–59.8），系统层面达到 Ollama 性能。

**路径 C：Ollama 直接服务**

- 与路径 B 相同的模型和服务；
- 直接用 `/api/chat` 或 OpenAI 兼容 `/v1/chat/completions`；
- 作为路径 B 的对照基线。

路径 B 是**后端委托**，不是路径 A 的内核优化结果。

本机实际导出结果：

- 模型标签：`maibot-catgirl-v3:latest`
- GGUF adapter：`F:\QwenQLoRA\export\maibot-catgirl-v3.adapter.gguf`（29.9 MB）
- Modelfile：`F:\QwenQLoRA\export\Modelfile.maibot-catgirl-v3`
- 模型格式：Qwen2 / Q4_K_M
- 服务：`http://127.0.0.1:11434`，OpenAI 兼容接口 `/v1/chat/completions`

## 6. 实验方法与评价指标

### 6.1 CPU 微型模型冒烟测试

目的：在不占用训练 GPU 的情况下验证代码路径的正确性。

构造一个合成微型 Qwen2：

- hidden size = 32
- 1 个 Transformer layer
- 4 个 attention heads
- 2 个 KV heads
- 使用真实 Qwen tokenizer 的 vocab 和 ChatML 模板
- 使用随机初始化权重
- 生成并保存随机 PEFT LoRA adapter
- 在 CPU、float32、eager attention 下运行

检查项：

1. adapter 配置和权重文件存在；
2. 兼容层能同时加载基座和 PEFT adapter；
3. ChatML 模板能正常渲染中文消息；
4. `generate` 能返回新 token；
5. `chat_completion` 返回合法 OpenAI 形状字典；
6. token 用量字段大于零且可解析。

### 6.2 真实 GPU 功能测试

训练完成后，在 RTX 3050 上执行：

```powershell
D:
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --load-in-4bit `
  --compute-dtype float16 `
  --prompt "你是谁？"
```

记录：

- 是否成功加载；
- 加载耗时；
- 峰值显存；
- 首 token 延迟（如可测）；
- 总生成耗时；
- 生成 token 数；
- tokens/s；
- 输出文本样例；
- 是否出现 OOM、NaN 或模板错误。

### 6.3 Ollama 对照测试

训练完成后执行 LoRA 到 GGUF 的转换和模型创建。本机实际使用的模型标签为
`maibot-catgirl-v3:latest`，转换产物为
`export/maibot-catgirl-v3.adapter.gguf`，Modelfile 继承
`qwen2.5:3b-instruct-q4_K_M` 的工具模板：

```powershell
F:\QwenQLoRA\build_ollama_adapter.ps1 `
  -Adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  -ModelTag maibot-catgirl-v3 `
  -ExportStem maibot-catgirl-v3
```

然后通过：

```text
GET  http://127.0.0.1:11434/api/version
GET  http://127.0.0.1:11434/v1/models
POST http://127.0.0.1:11434/v1/chat/completions
```

记录：

- `ollama show` 的模型格式、量化和 adapter 信息；
- `ollama ps` 的 GPU/CPU 分层情况；
- 峰值显存；
- 相同 prompt 下的输出；
- tokens/s；
- 与兼容层输出的一致性和差异。

### 6.4 优化消融与双后端同条件测试

为区分“兼容层代码开销”和“量化执行后端能力”，在相同模型、system
prompt、用户 prompt、FP16/seed 条件下执行以下消融：

1. Transformers 动态 KV cache；
2. Transformers 静态 KV cache；
3. 静态 KV cache + 显式 Flash-SDPA；
4. `torch.compile(mode="reduce-overhead")`（将 bitsandbytes Linear4bit 标记为 Dynamo opaque）；
5. Ollama 高性能后端（原生 `/api/chat` 计时）。

Transformers 基准命令：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat.benchmark `
  --backend transformers `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --load-in-4bit --compute-dtype float16 `
  --prompt "你是谁？" `
  --system-file F:\QwenQLoRA\system_prompt.txt `
  --warmup-tokens 8 --benchmark-tokens 32 `
  --output D:\DeepInfer\doc\QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json
```

Ollama 基准命令：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat.benchmark `
  --backend ollama `
  --model maibot-catgirl-v3:latest `
  --prompt "你是谁？" `
  --system-file F:\QwenQLoRA\system_prompt.txt `
  --warmup-tokens 8 --benchmark-tokens 32 `
  --output D:\DeepInfer\doc\QWEN_COMPAT_OLLAMA_BENCHMARK.json
```

Ollama 后端使用原生 `eval_count / eval_duration` 计算 decode tokens/s；
Transformers 后端使用外层 wall-clock 统计。两者同时报告 completion token
数和统计口径，避免把 prompt prefill、模型加载或 prompt cache 混入 decode 吞吐。

### 6.5 建议报告的指标

| 指标 | 定义 | 状态 |
|---|---|---|
| 加载成功率 | 模型和 adapter 是否成功加载 | 兼容层与 Ollama 均成功 |
| 峰值显存 | `torch.cuda.max_memory_allocated()` 或 `nvidia-smi` | 兼容层与 Ollama 均已测 |
| 首 token 延迟 | 请求到首个输出 token 的时间 | Transformers 首 token 0.317 秒；Ollama 热态 prompt eval 0.044 秒 |
| 总延迟 | 请求到生成结束的时间 | 已记录 |
| 端到端吞吐 | completion tokens / wall seconds | 双后端已测 |
| 原生 decode 吞吐 | completion tokens / decode duration | 双后端均已拆分计时 |
| 输出一致性 | 与 Ollama 在相同 prompt 下的定性对比 | 已完成 3 个固定 prompt 对照 |
| 量化误差 | 与 BF16 参考输出的困惑度或人工评分 | 可选，尚未进行 |
| 稳定性 | 连续多轮请求是否 OOM/崩溃 | 双后端各 10 轮，均 10/10 成功 |
| OpenAI 兼容 | `/v1/chat/completions` 返回结构 | 已验证成功 |

## 7. 当前已验证结果

正式复测报告与结构化结果：

- [`doc/QWEN_COMPAT_FORMAL_RESULTS.md`](QWEN_COMPAT_FORMAL_RESULTS.md)
- `doc/results/qwen_compat_formal_20261025.json`
- `doc/results/qwen_compat_formal_ollama_rerun.json`
- `doc/results/qwen_compat_cpu_smoke_20261025.json`

早期与消融结果同时保存在：

- `doc/QWEN_COMPAT_BENCHMARK_RESULTS.json`
- `doc/QWEN_COMPAT_OPTIMIZATION_RESULTS.json`
- `doc/QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json`
- `doc/QWEN_COMPAT_OLLAMA_BENCHMARK.json`

这些文件可用于论文绘图、复核和补充材料。

### 7.1 结果总览

| 路径 | 是否成功 | 关键结果 |
|---|---|---|
| CPU 微型 Qwen2 + LoRA | 是 | 模型、adapter、模板、生成和 OpenAI 形状输出全部连通 |
| RTX 3050 + Qwen2.5-3B + checkpoint-120（Transformers/NF4） | 是 | BF16 与 FP16 均成功；FP16 32-token 约 6.1–6.3 tokens/s；正式复测中位 5.554 端到端 / 5.859 decode tokens/s |
| RTX 3050 + Ollama 高性能后端 | 是 | 原生 decode 约 58.8–59.8 tokens/s；正式复测 53.178，热态稳定轮次中位 55.750 |
| 静态 cache / Flash-SDPA / compile 消融 | 完成 | 均未在 NF4 直载路径产生实质提升 |
| Ollama v3（Q4_K_M + GGUF LoRA） | 是 | 模型 100% GPU，OpenAI 兼容接口已验证 |
| 10 轮连续对话 | 完成 | Transformers 10/10；Ollama 10/10；VRAM 进入平台期或保持恒定 |
| 固定 prompt 双后端输出对照 | 完成 | 0/3 完全一致；字符相似度 0.25–0.6452 |

### 7.2 CPU 冒烟测试结果

已完成的 CPU 冒烟测试结果：

- 加载合成 Qwen2 微型模型：成功；
- 加载 PEFT LoRA adapter：成功；
- 中文 ChatML 对话模板渲染：成功；
- `chat_result` 生成：成功；
- `chat_completion` OpenAI 形状输出：成功；
- 输入 token 数：30；
- `chat_result` 输出 token 数：3；
- `chat_completion` 输出 token 数：2，总 token 数：32；
- 首 token 与 decode 计时字段：已产生且非空；
- 运行设备：CPU；
- 测试退出码：0。

该结果证明了兼容层的模型加载、adapter 加载、模板渲染、生成调用和返回结构是连通的，但**不能**作为 3B 模型在 4 GB GPU 上的性能结论。正式脚本为 `scripts/smoke_qwen_cpu.py`。

### 7.3 真实 Qwen2.5-3B NF4 GPU 结果

测试模型为 `Qwen2.5-3B-Instruct-smallshards`，adapter 为
`train/maibot-catgirl-v3/checkpoint-120`，使用 RTX 3050 Laptop 4 GB GPU。

#### 7.3.1 BF16 compute dtype

| 指标 | 测量值 |
|---|---:|
| 模型加载耗时 | 96.27 秒 |
| 加载峰值 allocated | 2072.7 MiB（约 2.02 GiB） |
| 加载峰值 reserved | 2288.0 MiB（约 2.23 GiB） |
| 加载后空闲显存 | 0.0 MiB |
| 冷启动 adapter 8 tokens | 9.941 秒，0.805 tokens/s |
| 无 adapter 8 tokens | 2.970 秒，2.693 tokens/s |
| 热身后 adapter 8 tokens | 3.437 秒，2.328 tokens/s |

#### 7.3.2 FP16 compute dtype

| 指标 | 测量值 |
|---|---:|
| 模型加载耗时 | 271.38 秒 |
| 加载峰值 allocated | 2072.7 MiB（约 2.02 GiB） |
| 加载峰值 reserved | 2288.0 MiB（约 2.23 GiB） |
| 加载后空闲显存 | 1080.5 MiB |
| 冷启动 adapter 8 tokens | 8.164 秒，0.980 tokens/s |
| 热身后 adapter 8 tokens | 1.275 秒，6.276 tokens/s |
| 热身后 adapter 32 tokens | 5.505 秒，5.813 tokens/s |
| 226-token system prompt + 5 输出 tokens | 8.06 秒，0.62 tokens/s |

FP16 在该平台上明显优于 BF16：热身后 8-token 短序列从 2.328
提升到 6.276 tokens/s，并且加载后保留了约 1.08 GiB 的空闲显存。因此兼容层默认
使用 FP16 compute dtype；BF16 作为数值稳定性优先时的可选模式。

加载耗时从 96 秒到 271 秒不等，主要受模型所在 F: 盘缓存、后台评测/系统负载
和 4 GB 显存压力影响，不能视为稳定的磁盘 I/O 基准。

#### 7.3.3 固定人设评测（项目已有评测）

项目已有的 18 个固定 probe 对基座和 v3 adapter 的评测结果如下：

| 指标 | 基座 | v3 adapter |
|---|---:|---:|
| `expected_all_rate` | 1.000 | 1.000 |
| `expected_any_rate` | 0.643 | 0.714 |
| `must_not_rate` | 1.000 | 1.000 |
| 旧名 `猫猫` 计数 | 0 | 0 |
| 自称 AI/助手计数 | 0 | 0 |
| 平均回复长度（字） | 17.6 | 12.7 |
| <=5 字回复数 | 1 | 5 |

需要注意：兼容层单次随机采样曾输出“我是猫猫w”，而项目固定评测中 adapter
在相似条件下输出“我是 catgirlw”。两者使用的是相同 adapter（checkpoint-120
与根目录 adapter 的 SHA256 完全一致），差异更可能来自采样随机性、compute
dtype 和显存压力，而不应直接解释为 adapter 失效。正式论文应使用固定 seed、
贪心解码或多次重复实验后再下结论。

### 7.4 Ollama v3 结果

实际模型：

- 标签：`maibot-catgirl-v3:latest`
- 格式：Qwen2，Q4_K_M
- 参数：约 3.1B
- 模型大小：2.0 GB
- `ollama ps`：2.4 GB，100% GPU，context 4096
- 能力：`completion`、`tools`
- adapter GGUF：约 29.9 MB

使用同一个 226-token system prompt 和用户问题“你是谁？”，通过
`/v1/chat/completions` 测量：

| 请求 | 时间 | 完成 tokens | 观测吞吐 | 说明 |
|---|---:|---:|---:|---|
| 冷启动请求 | 42.193 秒 | 8 | 0.190 tokens/s | 包含模型加载到 GPU |
| 热身后同一 prompt | 0.311 秒 | 7 | 22.514 tokens/s | 可能受益于 prompt cache |
| 热身输出文本 | — | — | — | `我是 catgirl，黑龙江高一的` |
| 热身后输出文本 | — | — | — | `我是黑龙江高一学生w` |

该表中的 22.514 tokens/s 是基于 `/v1/chat/completions` 的外层 wall-clock
观测，且可能命中 prompt cache。后续第 7.6 节使用 Ollama 原生
`eval_count / eval_duration` 得到约 59.8 decode tokens/s，因此论文应以
第 7.6 节的原生指标为正式高性能结果；本表保留作为接口层初步测量记录。

### 7.5 Transformers/NF4 优化消融

在短 prompt “你好”、FP16、`min_new_tokens=max_new_tokens=32`、无 system
prompt 的条件下，得到：

| 变体 | 32-token 时间 | 观测吞吐 |
|---|---:|---:|
| 动态 KV cache | 5.096 秒 | 6.279 tokens/s |
| 静态 KV cache | 5.178 秒 | 6.180 tokens/s |
| 静态 KV cache + Flash-SDPA | 5.173 秒 | 6.186 tokens/s |
| `torch.compile(mode="reduce-overhead")` | 5.213 秒 | 6.138 tokens/s |

静态 cache、Flash-SDPA 和 `torch.compile` 均未产生实质提升。Transformers 4.49
的静态 cache 默认会自动编译 decode loop，而 bitsandbytes 的 `Params4bit`
无法被 Dynamo 追踪；将 `generation_config.disable_compile=True` 后可运行静态
cache，但性能与动态 cache 相同。将 bitsandbytes Linear4bit 标记为 Dynamo
opaque 后再 `torch.compile`，同样没有加速。

因此该平台上的瓶颈是 bitsandbytes/Transformers 的通用 4-bit 执行路径，而不是
KV cache 分配或 Python 层缓存管理。

另外，本机预编译的 `llama-server.exe` 实际上 `--list-devices` 返回空，是无 CUDA
后端的 CPU-only 构建：纯基座 32-token 解码约 19.0 tokens/s，加载 LoRA 后约
17.8 tokens/s。它可以说明 LoRA 只带来约 7% 的额外开销，但不是 GPU 性能路径，
因此没有被集成成高性能后端。

### 7.6 双后端同条件对照

使用相同的 226-token system prompt、用户问题“你是谁？”和 seed，结果如下：

| 指标 | Transformers/NF4 | Ollama 后端 |
|---|---:|---:|
| 模型/服务加载 | 253.842 秒 | 连接 0.009 秒；首次请求 2.434 秒（页面缓存暖态；更早冷磁盘运行 43.158 秒） |
| 加载峰值 allocated | 2074.2 MiB | — |
| 加载后空闲显存 | 1.055 GiB | — |
| benchmark completion tokens | 32（强制） | 10（模型 EOS） |
| benchmark wall 时间 | 5.256 秒 | 0.286 秒 |
| 端到端 wall 吞吐 | 6.089 tokens/s | 35.005 tokens/s |
| 原生 decode 吞吐 | — | **59.781 tokens/s**（重复测量 58.8–59.8） |
| 自然采样输出 | “我是猫猫w” | “我是黑龙江高一的学生，叫 catgirl” |

Ollama 的 prompt 第二次请求显示 `prompt_eval_seconds≈0.018`，明显受益于
prompt cache；Transformers 每次请求都重新执行 prefill。即使忽略这一差异，
Ollama 的原生 decode 吞吐仍是 NF4 直载路径的约 9.8 倍。

结论应当写成：

- **系统层面**：兼容层通过 Ollama 高性能后端达到了 Ollama 性能；
- **算法/内核层面**：Transformers + PEFT + bitsandbytes NF4 直载路径仍只有
  约 6 tokens/s，静态 cache、Flash-SDPA 和 `torch.compile` 都没有改变这一点；
- 因此“达到 Ollama 性能”的准确表述是**后端委托**，不是 NF4 直载内核优化。

### 7.7 正式复测：TTFT、10 轮稳定性与接口对照

正式复测使用固定 system prompt、固定 seed、3 轮吞吐、10 轮连续对话和
3 个一致性 prompt。完整报告见
[`QWEN_COMPAT_FORMAL_RESULTS.md`](QWEN_COMPAT_FORMAL_RESULTS.md)。

| 指标 | Transformers/NF4 | Ollama Q4_K_M + LoRA |
|---|---:|---:|
| 加载耗时 | 267.613 秒 | 请求级 load 2.108 / 2.637 / 0.088 秒 |
| 加载峰值 allocated | 2074.2 MiB | — |
| 加载/运行峰值 reserved | 2332.0 MiB | 固定 2,390,300,672 bytes VRAM |
| TTFT / prompt eval 中位数 | 0.317 秒（首 token 采样） | 0.223 秒（独立请求）；0.044 秒（热态稳定轮次） |
| 正式吞吐测试 completion tokens | 15 / 15 / 15 | 8 / 7 / 8 |
| 端到端 tokens/s 中位数 | 5.554 | 3.201（含请求级模型加载） |
| decode tokens/s 中位数 | **5.859** | **53.178**（独立请求）；**55.750**（热态稳定轮次） |
| 10 轮连续对话 | 10/10 成功 | 10/10 成功 |
| 显存趋势 | reserved 在 2332 MiB 平台化 | `size_vram` 与 nvidia-smi 逐轮不变 |
| OpenAI 兼容端点 | 通过 `chat_completion` 形状验证 | `/v1/chat/completions` 成功 |

补充说明：

- Transformers 的吞吐请求在 15 个 token 时自然停止，短序列使首 token
  开销占比较高；此前固定 32-token 控制实验的 6.089 tokens/s 仍是主要
  长度固定参考。
- Ollama 独立请求的 load 时间在 0.088～2.637 秒之间变化，说明服务端缓存状态
  会影响首 token 代理指标；decode tps 使用原生 `eval_duration` 单独计算。
- 3 个固定 prompt 的双后端输出 0/3 完全一致，字符相似度为
  0.25、0.6452、0.6364；这不是语义等价性结论。
- 10 轮短对话稳定性不等于长上下文、并发或小时级压力测试的结论。

### 7.8 结论性观察

1. 兼容层可以在 4 GB RTX 3050 上真实加载 Qwen2.5-3B + LoRA 并生成文本；
2. FP16 compute dtype 比 BF16 更适合该显存配置；
3. NF4 直载路径的静态 cache、Flash-SDPA 和 compile 优化均未达到性能目标；
4. Ollama 高性能后端在系统层面达到 Ollama 的 decode 性能；
5. 兼容层的价值是“可验证的 PyTorch/NF4 直载 + 可选高性能服务后端”，
   而不是把 Transformers 的 4-bit 内核改造成 Ollama；
6. 固定 seed、TTFT、10 轮稳定性和 OpenAI 兼容接口的正式复测已完成；
   仍需补充的仅是长上下文、并发、小时级压力和人工语义质量评测。

## 8. 可复现性

### 8.1 环境准备

使用现有 Qwen 训练环境：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe --version
```

依赖：

```text
torch 2.5.1+cu121
transformers 4.49.0
peft 0.14.0
bitsandbytes 0.48.2
accelerate 1.3.0
safetensors 0.5.2
```

### 8.2 兼容层路径

```powershell
$env:PYTHONPATH = "D:\DeepInfer"
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat --help
```

### 8.3 论文中建议的复现命令

```powershell
$env:PYTHONPATH = "D:\DeepInfer"
$env:PYTHONIOENCODING = "utf-8"
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --load-in-4bit `
  --compute-dtype float16 `
  --max-new-tokens 96 `
  --temperature 0.7 `
  --top-p 0.9 `
  --seed 20261025 `
  --prompt "你是谁？"
```

### 8.4 可复现基准脚本

为避免论文数据依赖临时命令行，兼容层提供统一基准脚本。

Transformers/NF4 后端：

```powershell
$env:PYTHONPATH = "D:\DeepInfer"
$env:PYTHONIOENCODING = "utf-8"
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat.benchmark `
  --backend transformers `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --load-in-4bit --compute-dtype float16 `
  --benchmark-tokens 32 `
  --system-file F:\QwenQLoRA\system_prompt.txt `
  --output D:\DeepInfer\doc\QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json
```

Ollama 高性能后端：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe -m deepinfer_compat.benchmark `
  --backend ollama `
  --model maibot-catgirl-v3:latest `
  --benchmark-tokens 32 `
  --system-file F:\QwenQLoRA\system_prompt.txt `
  --output D:\DeepInfer\doc\QWEN_COMPAT_OLLAMA_BENCHMARK.json
```

该脚本会分别记录：

- 模型加载或服务连接耗时；
- 加载前后空闲显存；
- 加载峰值 allocated / reserved；
- warmup 和固定 32-token 基准的 tokens/s；
- Ollama 原生 `prompt_eval_seconds`、`decode_seconds` 和 decode tokens/s；
- 带 system prompt 的自然采样输出和对应 token 数。

正式稳定性与 TTFT 复测脚本：

```powershell
cd D:\DeepInfer

# 先释放 Ollama 显存，避免与 NF4 直载模型争用 4 GB GPU
F:\Ollama\app\ollama.exe stop maibot-catgirl-v3:latest

# CPU 控制流回归
F:\QwenQLoRA\venv\Scripts\python.exe scripts\smoke_qwen_cpu.py `
  --output D:\DeepInfer\doc\results\qwen_compat_cpu_smoke_20261025.json

# 3 轮吞吐 + TTFT + 10 轮稳定性 + 3 组一致性 + Ollama OpenAI 接口
F:\QwenQLoRA\venv\Scripts\python.exe scripts\bench_qwen_compat.py `
  --output D:\DeepInfer\doc\results\qwen_compat_formal_20261025.json
```

只补采 Ollama 阶段：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe scripts\bench_qwen_compat.py `
  --skip-transformers `
  --output D:\DeepInfer\doc\results\qwen_compat_formal_ollama_rerun.json
```

## 9. 局限性与有效性威胁

1. **不是原生 DeepInfer 性能**：兼容层绕过了 SM100 专用内核，不能用于证明 DeepInfer 的吞吐优势。
2. **4 GB 显存接近极限**：FP16 加载峰值 allocated 约 2074 MiB；加载后空闲约 0.64–1.06 GiB。10 轮短对话可稳定运行，但长上下文、长生成和并发仍可能触发显存压力。
3. **NF4 直载吞吐有限**：FP16 32-token 基准约 6.1–6.3 tokens/s；正式复测中位端到端 5.554、decode 5.859 tokens/s。静态 cache、Flash-SDPA 和 `torch.compile` 均未产生实质提升。
4. **高性能路径是后端委托**：约 53–60 tokens/s 来自 Ollama 后端，不是 Transformers/NF4 内核；论文必须明确区分两者。
5. **统计口径差异**：Ollama 使用原生 `eval_duration` 计算 decode；Transformers 通过首 token 计时拆出 decode 区间，但仍依赖 PyTorch 事件/同步时钟。Ollama 的独立请求还会受到模型 load 和 prompt/runner cache 状态影响。
6. **量化质量差异**：NF4 与 BF16/FP8 参考模型可能存在数值和生成质量差异；固定 prompt 上 NF4 与 Q4_K_M+LoRA 输出 0/3 完全一致。
7. **采样稳定性已部分覆盖**：正式复测使用固定 seed、贪心解码和 10 轮连续对话；但输出质量仍需更多 prompt、人工评分和语义一致性测试，不能仅凭字符相似度判断等价。
8. **CPU 冒烟测试规模小**：合成微型模型只能验证控制流，不能代表 3B 模型行为。
9. **Windows 与 Linux 差异**：当前实验在 Windows 上通过 PyTorch/bitsandbytes 运行，而 DeepInfer 原生目标平台是 Linux。
10. **工具调用**：Transformers 后端只将 tools 定义传给 ChatML 模板，不解析 `tool_calls`；Ollama 后端可透传原生 `tool_calls`。
11. **加载时间不稳定**：Transformers 后端加载耗时在 253.842～267.613 秒之间，受 F: 盘缓存、后台任务和显存压力影响，不能作为 I/O 基准。
12. **llama.cpp 预编译包为 CPU-only**：本机 `llama-server.exe --list-devices` 返回空，不能作为 GPU 高性能后端。
13. **训练数据与评估数据有限**：模型输出质量结论应结合固定 prompt 集、人工评分和验证 loss，而不能只看单次生成。
14. **正式吞吐 prompt 在 7～15 tokens 自然停止**：短序列会突出首 token 开销，固定长度吞吐应结合 32-token 控制实验阅读。
15. **稳定性范围有限**：10 轮连续对话通过不等于长上下文、并发、小时级服务或显存碎片压力测试通过。
16. **OpenAI 端点仅做功能验证**：尚未进行 OpenAI 兼容端点的并发和持续压测。

## 10. 论文可用方法表述

### 10.1 中文版本

> 为在普通消费级 GPU 上验证 Qwen2.5-3B QLoRA 检查点的推理行为，本文实现了一个独立的 DeepInfer-Qwen 兼容层。该兼容层不修改 DeepInfer 面向 SM100 GPU 的 FlashMLA、DeepGEMM 和 DeepEP 执行路径，而是提供两条共享 API 的后端：一条采用 Transformers AutoModelForCausalLM、bitsandbytes NF4 双重量化和 PEFT LoRA，实现 4 GB 显存上的直接检查点推理；另一条通过本机 Ollama 原生 `/api/chat` 委托执行，并转发 Ollama 的 `prompt_eval_duration` 和 `eval_duration` 原生计时指标。实验在 RTX 3050 Laptop GPU 上完成了 CPU 微型 Qwen2 冒烟测试、真实 Qwen2.5-3B + LoRA NF4 生成和同 prompt 双后端对照。NF4 直载后端在 32-token 生成时约为 6.1–6.3 tokens/s，静态 KV Cache、Flash-SDPA 和 `torch.compile` 均未带来实质提升；Ollama 高性能后端达到约 59.8 decode tokens/s（58.8–59.8）。正式复测进一步加入中位首 token 延迟、3 轮吞吐、10 轮连续对话和 OpenAI 兼容接口验证：NF4 直载中位端到端吞吐为 5.554 tokens/s、decode 为 5.859 tokens/s，首 token 延迟中位数为 0.317 秒，10 轮全部成功；Ollama 独立请求中位 decode 为 53.178 tokens/s，热态稳定轮次中位为 55.750 tokens/s，同样 10/10 成功。该结果在系统层面达到 Ollama 性能，但必须明确表述为后端委托，而不是 Transformers/NF4 内核本身的吞吐优化。

### 10.2 English version

> To evaluate Qwen2.5-3B QLoRA checkpoints on a consumer-grade GPU, we introduce an optional DeepInfer-Qwen compatibility layer with two interchangeable backends. The direct backend uses Transformers AutoModelForCausalLM, bitsandbytes NF4 double quantization, and PEFT LoRA adapter loading to construct a low-VRAM inference path on a 4 GB RTX 3050 Laptop GPU. The high-performance backend delegates inference to a local Ollama service through its native `/api/chat` endpoint and exposes Ollama's native `prompt_eval_duration` and `eval_duration` metrics. Experiments completed a synthetic Qwen2+LoRA smoke test, real Qwen2.5-3B+LoRA NF4 generation, and a same-prompt backend comparison. The direct NF4 backend reached approximately 6.1–6.3 tokens/s for 32-token generation; static KV cache, Flash-SDPA and `torch.compile` produced no material improvement. The Ollama backend reached approximately 59 decode tokens/s (58.8–59.8). A formal replication added median first-token latency, repeated throughput, 10-round consecutive conversations, and OpenAI-compatible endpoint validation: the direct NF4 path reached 5.554 end-to-end and 5.859 decode tokens/s with a median first-token latency of 0.317 s and 10/10 successful rounds; Ollama reached 53.178 decode tokens/s for independent requests and 55.750 tokens/s in warm stability rounds, also with 10/10 success. This achieves Ollama-level throughput at the system level, but must be reported as backend delegation rather than a native speedup of the Transformers/NF4 execution path.

## 11. 建议的论文贡献表述

可以将贡献谨慎表述为：

1. 提出一个与 DeepInfer 原生专用内核解耦的 Qwen2/Qwen2.5 双后端兼容层；
2. 给出在 4 GB 显存消费级 GPU 上运行 Qwen2.5-3B + LoRA 的 NF4 实现路径；
3. 通过静态 KV Cache、Flash-SDPA 和 `torch.compile` 消融，证明性能瓶颈位于通用 4-bit 执行路径而非缓存管理；
4. 设计共享 API 的 Ollama 高性能后端，使系统级吞吐达到 Ollama 水平；
5. 明确区分 NF4 直载兼容性、Ollama 后端委托和原生框架性能，避免将 fallback 或代理路径结果误用于宣传原生 DeepInfer 性能。

不建议表述为：

- “DeepInfer 已原生支持 Qwen2.5”；
- “在 RTX 3050 上复现了 DeepInfer 的高性能”；
- “兼容层与 FlashMLA/DeepGEMM 等价”；
- “Transformers/NF4 直载路径达到了约 59.8 tokens/s”；
- “已经证明该路径达到生产级吞吐”。

更准确的表述是：

> 兼容层通过 Ollama 性能后端在系统层面达到 Ollama 吞吐；其 Transformers/NF4 直载路径仍保持约 6 tokens/s，用于兼容性验证而非生产性能。

## 12. 参考文献与软件引用

- DeepInfer: <https://github.com/deepseek-ai/DeepInfer>
- Qwen2.5: <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>
- Transformers: Wolf et al., *Transformers: State-of-the-Art Natural Language Processing*, EMNLP 2020.
- PEFT: <https://github.com/huggingface/peft>
- bitsandbytes: <https://github.com/bitsandbytes-foundation/bitsandbytes>
- Ollama: <https://ollama.com>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>

## 附录 A：当前代码文件清单

| 文件 | 作用 |
|---|---|
| `deepinfer_compat/__init__.py` | 对外导出兼容层 API |
| `deepinfer_compat/qwen.py` | Qwen2/Qwen2.5 + PEFT + NF4 核心实现 |
| `deepinfer_compat/ollama.py` | Ollama 高性能后端、原生计时和工具调用透传 |
| `deepinfer_compat/__main__.py` | 双后端命令行入口、单轮生成、交互式 REPL |
| `deepinfer_compat/benchmark.py` | 双后端加载/显存/吞吐/人设生成基准 |
| `deepinfer_compat/README.md` | 使用说明与限制 |
| `scripts/smoke_qwen_cpu.py` | CPU 微型 Qwen2 + LoRA 正式冒烟脚本 |
| `scripts/bench_qwen_compat.py` | 正式吞吐、TTFT、10 轮稳定性、输出对照和 OpenAI 接口复测 |
| `scripts/validate_qwen_results.py` | 正式结果 JSON 的结构、稳定性、显存和接口一致性校验 |
| `requirements-qwen-compat.txt` | 可选依赖 |
| `doc/QWEN_COMPAT_METHODOLOGY_ZH.md` | 本方法文件 |
| `doc/QWEN_COMPAT_FORMAL_RESULTS.md` | 正式复测报告 |
| `doc/results/qwen_compat_formal_20261025.json` | 正式复测结构化原始结果 |
| `doc/results/qwen_compat_formal_ollama_rerun.json` | Ollama 逐轮显存补充结果 |
| `doc/results/qwen_compat_cpu_smoke_20261025.json` | CPU 冒烟结构化结果 |
| `doc/QWEN_COMPAT_BENCHMARK_RESULTS.json` | 早期 CPU/GPU/Ollama 结果 |
| `doc/QWEN_COMPAT_OPTIMIZATION_RESULTS.json` | 静态 cache/Flash-SDPA/compile 消融结果 |
| `doc/QWEN_COMPAT_TRANSFORMERS_BENCHMARK.json` | 统一 benchmark 的 Transformers 结果 |
| `doc/QWEN_COMPAT_OLLAMA_BENCHMARK.json` | 统一 benchmark 的 Ollama 后端结果 |

## 附录 B：当前结果汇总表

| 实验 | 模型/路径 | 关键显存 | 观测生成吞吐 | 是否成功 | 备注 |
|---|---|---|---:|---|---|
| CPU smoke test | tiny Qwen2 + LoRA | N/A | N/A | 是 | 控制流验证 |
| GPU NF4 BF16 | Qwen2.5-3B + ckpt-120 | peak alloc 2072.7 MiB；加载后空闲 0 MiB | 2.328 tokens/s（8 tokens） | 是 | 早期诊断 |
| GPU NF4 FP16 | Qwen2.5-3B + ckpt-120 | peak alloc 2074.2 MiB；加载后空闲 1.055 GiB | 6.089 tokens/s（32 tokens，wall） | 是 | 同条件统一 benchmark |
| 正式 NF4 复测 | Qwen2.5-3B + ckpt-120 | peak reserved 2332.0 MiB；稳定后空闲 527.5 MiB | 5.554 端到端 / 5.859 decode tokens/s；TTFT 0.317 秒 | 是 | 15 tokens 自然停止 |
| 静态 KV cache | Qwen2.5-3B + ckpt-120 | 同上 | 6.180 tokens/s | 是 | 无提升 |
| 静态 cache + Flash-SDPA | Qwen2.5-3B + ckpt-120 | 同上 | 6.186 tokens/s | 是 | 无提升 |
| `torch.compile` | Qwen2.5-3B + ckpt-120 | 同上 | 6.138 tokens/s | 是 | 无提升 |
| Ollama 后端 | Q4_K_M + GGUF LoRA | 100% GPU | **59.781 decode tokens/s** | 是 | 原生 `eval_duration` |
| 正式 Ollama 复测 | Q4_K_M + GGUF LoRA | 固定 2,390,300,672 bytes VRAM | 53.178 decode tokens/s；热态稳定轮次 55.750 | 是 | OpenAI 兼容端点通过 |
| llama.cpp CPU-only | Q4_K_M + LoRA | CPU | 17.788 tokens/s | 是 | `--list-devices` 为空 |
| 10 轮稳定性 | Transformers/NF4 | reserved 2332 MiB 平台化 | 10/10 成功 | 是 | 短对话压力 |
| 10 轮稳定性 | Ollama | VRAM 逐轮恒定 | 10/10 成功 | 是 | 热态轮次中位 55.750 decode tokens/s |
| 固定 prompt 输出对照 | NF4 vs Ollama | N/A | N/A | 完成 | 0/3 完全一致 |
