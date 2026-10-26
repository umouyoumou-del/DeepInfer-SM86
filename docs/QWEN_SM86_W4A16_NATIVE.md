# DeepInfer Qwen2.5 SM86 W4A16 原生解码路径

## 1. 结果摘要

已在 RTX 3050 Laptop（SM86，4 GB）上跑通 DeepInfer Qwen2.5-3B 的
SM86 原生 W4A16 + GQA 解码路径，并完成真实权重生成与 CUDA Graph 测速。

| 指标 | 结果 |
|---|---:|
| GPU | RTX 3050 Laptop, 4 GB, SM86 |
| 模型 | Qwen2.5-3B-Instruct + checkpoint-120 LoRA |
| 量化 | 对称 int4，group size 128 |
| 层内线性层 | DeepInfer SM86 W4A16 GEMV（CuPy/NVRTC） |
| `lm_head` | FP16 dense（未走 int4） |
| attention | DeepInfer SM86 GQA decode kernel |
| KV cache | 静态 GQA cache + CUDA Graph |
| 峰值显存 | 2.493 GiB allocated / 2.627 GiB reserved |
| prefill TTFT | 4.555 秒（30 prompt tokens） |
| CUDA Graph 捕获 | 0.234 秒 |
| Graph 首次 replay | 0.137 秒 |
| decode 平均 | **37.157 tokens/s** |
| decode 中位数 | **42.666 tokens/s** |
| 稳态均值（去掉首次 replay） | **42.721 tokens/s** |
| 稳态中位数 | **42.724 tokens/s** |
| 生成内容（EOS 前） | `你好！有什么问题请尽管告诉我，我会尽力回答的。` |

24 次 graph replay 总耗时 `0.645914` 秒。首次 replay 约 `107.5 ms`，
之后稳定在约 `23.4 ms/token`；因此同时报告平均值和稳态值，避免只挑最好数字。

## 2. 这次“原生”具体指什么

已实现的原生部分：

- DeepInfer 仓库内的 `Qwen2W4A16ForCausalLM` 和 `W4A16Linear`；
- 自定义 SM86 CUDA W4A16 GEMV kernel，CUDA 源码由 CuPy/NVRTC 编译；
- 自定义 SM86 GQA decode kernel，支持 16 Q heads / 2 KV heads、
  RoPE 后 K/V、128-slot 静态 cache 和 attention mask；
- 真实 HF 分片权重 + PEFT LoRA 合并后转换为 int4；
- 静态 GQA KV cache；
- CUDA Graph 捕获和 replay；
- 单 token decode 由 DeepInfer 模型代码直接调用上述 kernel。

校验结果：

- 原生 GQA 与 PyTorch SDPA 的 logits cosine 为 `1.0`，
  max abs diff 为 `0.015625`；
- CUDA Graph replay 与普通原生 GQA 前向的 logits max abs diff 为 `0.0`；
- W4A16 线性层 decode cosine `0.996838`，prefill cosine `0.997703`。

仍未完成的部分：

- 没有接入 DeepSeek 的 `worker/cache_engine.py`、block table、slot mapping；
- 没有接入 DeepInfer 原生 `Sampler`；
- 没有完整 engine/scheduler/worker 路径；
- 当前 GPU 测速通过 Python 3.10 CUDA 环境加载同一份 DeepInfer source；
  Python 3.12 CUDA wheel 解压在本机卡住，尚未完成 3.12 GPU 环境验证。

因此应称为 **DeepInfer model_executor 级 SM86 原生解码后端**，
不能称为完整 DeepInfer engine 已支持 SM86。

## 3. 性能解释

Ollama Q4_K_M 此前在相同 3050 上的独立请求中位数约为
`53.178 decode tokens/s`。本路径稳态约 `42.7 tokens/s`，已经进入同一数量级，
但仍低于 Ollama。主要差距来自：

- 36 层每层仍有 RMSNorm/RoPE/残差等 PyTorch 算子；
- 每层 7 个 W4A16 线性层尚未融合 q/k/v 和 gate/up；
- 当前 int4 是对称 per-group scale，未做 GPTQ/AWQ 校准；
- 还没有完整 engine 的 continuous batching 和调度优化。

## 4. 质量与限制

- 量化使用对称 int4、group size 128；短回答质量可读，但仍可能有格式退化。
- CUDA Graph capture 只记录自定义 CUDA kernel，不会执行它们；必须 replay
  一次才能得到真实输出。测速脚本已经修正这一点。
- 测速脚本没有在 EOS 处停止；EOS 后的文本不代表模型质量。
- 这是单请求、短上下文（128 slot 静态 cache）测速，不是生产吞吐。
- 0.417 tokens/s 的 CPU 结果与这个 GPU 结果不是同一路径。
- `native_runner_ready` 仍为 `False`，因为完整 CacheEngine/Sampler 尚未接入。

## 5. 复现

环境：

```text
Python 3.10.3
torch 2.5.1+cu121
CUDA 12.1
CuPy 14.2.0
RTX 3050 Laptop 4 GB
```

命令：

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen2_w4a16_gpu_harness.py `
  --max-new-tokens 24 --max-cache 128 `
  --output D:\DeepInfer\doc\results\qwen2_w4a16_native_gqa_gpu.json
```

原始结果：

- `doc/results/qwen2_w4a16_native_gqa_gpu.json`（最终原生 GQA + Graph）
- `doc/results/qwen2_w4a16_graph_gpu.json`（SDPA attention 的历史基线）
- `doc/results/qwen2_w4a16_gpu_harness.json`（非 graph 基线）
- `doc/results/qwen2_w4a16_gpu_unit.txt`（W4A16 数值校验）
- `doc/results/qwen2_w4a16_graph_tiny.txt`（GQA/Graph 等价性校验）
- `doc/results/w4a16_sm86_microbench.json`（微内核）
- `doc/results/qwen2_w4a16_sm86_summary.json`

## 6. 下一步

1. 融合 q/k/v 与 gate/up，减少每层 kernel launch 数。
2. 把静态 cache 接入 DeepInfer full-attention CacheEngine 接口。
3. 接入原生 Sampler，支持 EOS/top-k/top-p/temperature。
4. 持久化量化 checkpoint，去掉每次 5 分钟 CPU dense load。
5. 完成 Python 3.12 CUDA 环境下的真实 package 集成测试。
6. 加入 continuous batching 和生产上下文长度支持。
