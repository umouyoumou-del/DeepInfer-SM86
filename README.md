# DeepInfer-SM86

在 **RTX 3050 Laptop 4 GB（SM86 / Windows）** 上为 DeepInfer 增加 Qwen2.5 的
model_executor 级原生推理后端。

这是一个**非官方扩展仓库**，不是 DeepInfer 官方发行版，也不是完整 DeepInfer
engine。当前重点是：

- SM86 / SM86 可运行的 W4A16、W3A16 CUDA GEMV；
- 原生 GQA decode kernel；
- 静态 KV cache；
- CUDA Graph；
- Qwen2.5-3B / 7B 的量化、加载和测速；
- 4 GB 显存下的混合精度（Q4/Q3）与整体 Q4 实验数据。

---

## 结果速览

测试环境：Windows、RTX 3050 Laptop 4 GB、Python 3.10.3、torch 2.5.1+cu121、CUDA 12.1、CuPy 14.2。

| 路径 | 量化 | 显存 | decode |
|---|---|---:|---:|
| Qwen2.5-3B 兼容层（NF4） | NF4 | 2.28 GiB reserved | 5.859 tokens/s |
| Qwen2.5-3B DeepInfer SM86 原生 | W4A16 + GQA + Graph | 2.49 GiB allocated | **42.7 tokens/s（稳态）** |
| Qwen2.5-3B 常驻网关 | W4A16 + 256 context | 约 2.5 GiB | **44.773 tokens/s / 0.895 秒一次请求** |
| Qwen2.5-7B 混合精度 | 4-bit attention/embed + 3-bit MLP/head | 3.20 GiB allocated | **25.962 平均 / 26.188 中位** |
| Qwen2.5-7B 整体 Q4 | 全 4-bit | 3.92 GiB allocated | 7.942 tokens/s（持续 paging） |

> 这些结果均为 `model_executor` 级；`full_engine=false`、`native_runner_ready=false`。
> 不能把它们写成“完整 DeepInfer engine 已支持 Qwen2.5”。

## 本次更新（2026-10-26）

- 新增 3B W4A16 分层 checkpoint 导出与常驻加载：
  - `scripts/export_qwen3b_w4_checkpoint.py`
  - `Qwen2W4A16ForCausalLM.from_checkpoint(...)`
  - `D:\Qwen3B-W4A16-SM86` 约 1.912 GiB，加载约 11.75 秒
- 网关新增 `--backend w4a16` / `--backend mixed`，3B 与 7B 都可作为 OpenAI 兼容服务；
- 新增 256-slot DeepInfer SM86 native GQA kernel；
- 新增 3B / 7B / Ollama 顺序基准脚本和原始结果：
  - `scripts/bench_productivity_backends.py`
  - `docs/results/productivity_backends_combined.json`
- 修复 `.gitignore` 中 `models/` 误忽略
  `deepinfer/model_executor/models/` 源码的问题；
- 补齐 `deepinfer/model_executor/`、`config.py`、`environ.py`、engine 类型和源码测试；
- 长文拆分为两篇独立文章：
  - 上篇：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`
  - 下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 合并稿保留为：`docs/QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md`

### 7B 为什么不是纯 4-bit

Qwen2.5-7B 全 4-bit 权重约 **3.657 GiB**，但该卡实测只有 **3.237 GiB** CUDA
可用显存。纯 4-bit 会进入 WDDM paging：

- 预填充约 86 秒；
- decode 约 7.9 tokens/s。

7B 混合精度 checkpoint 约 **2.930 GiB**、有效位宽约 **3.30 bpw**，才有约
26 tokens/s 的速度；它不是 Q4，量化配置为：

```text
attention q/k/v/o : 4-bit, group 128
embedding         : 4-bit, group 128
MLP gate/up/down  : 3-bit, group 128
lm_head           : 3-bit, group 128
```

---

## 功能

- 统一 `ModelSpec` 抽象层，执行器不直接依赖 Qwen 字段；
- 已支持 Qwen2ForCausalLM + LlamaForCausalLM（标准 GQA/SwiGLU/RoPE）；
- Qwen2/Qwen2.5/Llama HF config、safetensors 分片、tied embedding；
- PEFT LoRA 合并；
- W4A16 GEMV（symmetric int4，group 128）；
- W3A16 GEMV（8 个 signed 3-bit 压进 3 bytes）；
- GQA decode kernel（支持 16/2、28/4 等 Q/KV head 组合）；
- 静态 KV cache（固定 slot、外部 mask）；
- CUDA Graph replay；
- 构造期 fused qkv / fused gate-up，避免加载后替换造成显存碎片；
- 7B 流式量化、按层 safetensors checkpoint；
- 3-bit lm_head；
- 3B/7B 测速 harness 和 CPU 复核工具；
- 本地 OpenAI 兼容网关（`/v1/models`、`/v1/chat/completions`、SSE 流式）；
- 固定槽位连续动态批处理调度器（W4/W3 batch GEMV + batch CUDA Graph）；
- 桌面剪贴板快捷操作脚本。

---

## 目录

```text
deepinfer/
  hugging_face/qwen2_config.py          Qwen2 HF config -> DeepInfer AttrDict
  hugging_face/config.py                配置分派修改
  model_config_loader.py                允许纯文本 Qwen2
  model_executor/models/qwen2.py        Qwen2 source model、GQA、静态 cache
  model_executor/models/qwen2_w4a16.py  W4A16 + GQA CUDA 后端与 3B 路径
  model_executor/models/qwen2_mixed_sm86.py
                                        4-bit/3-bit 混合模型、3-bit kernel
kernels/
  qwen2_w4a16_gemv.cu
  qwen2_w3a16_gemv.cu
  qwen2_gqa_decode.cu
  qwen2_gqa_decode_256.cu
apps/
  server.py                             OpenAI 兼容本地网关
  runtime.py                            常驻模型 + 静态 cache + Graph
  ask.py                                CLI 客户端
desktop/
  quick_ask.ps1                         剪贴板总结/改写/翻译/待办
scripts/
  run_qwen2_w4a16_gpu_harness.py        3B 原生测速入口
  export_w4_checkpoint.py               通用模型 W4A16 checkpoint
  export_qwen3b_w4_checkpoint.py        导出 3B 分层 W4A16 checkpoint
  bench_productivity_backends.py        3B/7B/Ollama 顺序基准
  run_qwen7b_mixed_gpu_harness.py       3B/7B/Q4 通用测速入口
  quantize_qwen7b_mixed.py              7B 逐层量化
  finish_qwen7b_mixed_quantize.py       分块量化 embedding/head 并补 manifest
  quantize_qwen7b_head3.py              将 lm_head 降到 3-bit
  quantize_qwen7b_q4.py                 生成全 4-bit Q4 checkpoint
  bench_w4a16_sm86.py                   W4A16 微基准
  bench_w3a16_sm86.py                   W3A16 微基准
  export_sm86_kernels.py                导出 .cu 源文件
tests/
  models/test_qwen2_source_integration.py
  models/test_qwen2_w4a16_gpu.py
docs/
  ARTICLE_1_DEEPINFER_SM86_KERNEL.md          上篇：内核与量化工程
  ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md    下篇：生产力、网关与基准
  ARTICLE_3_DEEPINFER_SM86_GENERIC.md          扩展篇：通用化与 Llama
  QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md   合并稿存档
  QWEN_SM86_W4A16_NATIVE.md
  QWEN_7B_MIXED_SM86.md
  QWEN_7B_Q4_SM86.md
  QWEN_7B_4GB_FEASIBILITY.md
  QWEN_NATIVE_MODEL_EXECUTOR.md
  GENERIC_MODEL_ADAPTER.md                     通用模型抽象层
  DYNAMIC_BATCHING_SCHEDULER.md                 动态批处理与调度器
  PERSONAL_PRODUCTIVITY.md                     从网关到桌面快捷操作
  PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md       3B/7B/Ollama 完整基准
  results/                                     选定的可复现结果
docs/results/
  qwen2_w4a16_native_gqa_gpu.json
  qwen7b_mixed_head3_final.json
  qwen7b_q4_gpu.json
  qwen7b_mixed_sm86_summary.json
  qwen7b_q4_sm86_summary.json
  ...
```

---

## 快速开始

### 1. 前置条件

- NVIDIA RTX 3050 或其它 SM86 GPU；
- Windows / Linux；
- Python 3.10（本仓库 GPU harness 的已验证环境）；
- CUDA 12.1 对应的 PyTorch；
- CuPy 含 CUDA 头文件（`cupy-cuda12x[ctk]`）；
- 一个可用的 DeepInfer checkout。

安装：

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1
pip install -r requirements-sm86.txt
```

本仓库是在 DeepInfer 基线 `8b16149` 上开发的。  
DeepInfer 主包本身可能需要 Python 3.12；本仓库的 GPU harness 已验证 Python 3.10 +
轻量 package stub 的方式。

### 2. 叠加到 DeepInfer

先备份 DeepInfer 工作区，然后运行：

```powershell
cd D:\DeepInfer-SM86
.\install-overlay.ps1 -DeepInferRoot D:\DeepInfer
```

脚本会把 `deepinfer/` 下的 overlay 文件复制到现有 DeepInfer checkout。  
也可以手工复制，但要保留备份。

> 该 overlay 会覆盖 `deepinfer/__init__.py`、`logger.py`、
> `model_config_loader.py`、`hugging_face/config.py`、
> `model_executor/models/__init__.py` 等兼容性文件。

### 3. 跑 3B

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen2_w4a16_gpu_harness.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --max-new-tokens 24 --max-cache 128 `
  --output D:\DeepInfer\doc\results\qwen2_w4a16_native_gqa_gpu.json
```

3B 会先在 CPU 合并 dense 权重，再转换到 W4A16；CPU 加载较慢，正式测速请用
已量化 checkpoint，或接受一次性的转换时间。

### 4. 跑 7B 混合精度

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --max-new-tokens 24 --warmup-replays 3 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_mixed_head3_final.json
```

### 5. 跑整体 Q4

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Q4-SM86 `
  --max-new-tokens 4 --warmup-replays 1 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_q4_gpu.json
```

4 GB 上 Q4 会持续 paging；不要把这个结果当作生产吞吐。

### 6. 启动个人生产力网关（3B / 7B）

3B W4A16 网关：

```powershell
cd D:\DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --model-id deepinfer-sm86-3b-w4a16 `
  --max-cache 256 `
  --port 18001
```

3B 动态批处理（固定 2 slot）：

```powershell
python apps\server.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --scheduler --max-batch-size 2 `
  --port 18001
```

7B 混合网关：

```powershell
cd D:\DeepInfer-SM86
python apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --backend mixed `
  --model-id deepinfer-sm86-7b-mixed `
  --max-cache 256 `
  --port 18000
```

接口（3B 使用 18001，7B 使用 18000）：

```text
GET  http://127.0.0.1:18000/health
GET  http://127.0.0.1:18000/v1/models
POST http://127.0.0.1:18000/v1/chat/completions
```

OpenAI 客户端把 Base URL 设为：

```text
http://127.0.0.1:18000/v1
```

命令行：

```powershell
python apps\ask.py --server http://127.0.0.1:18000 "用一句话解释 CUDA Graph。"
```

流式请求：

```powershell
curl.exe -N http://127.0.0.1:18000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"deepinfer-sm86-7b-mixed\",\"messages\":[{\"role\":\"user\",\"content\":\"写三句话解释 CUDA Graph\"}],\"max_tokens\":64,\"stream\":true}"
```

桌面剪贴板动作：

```powershell
.\desktop\quick_ask.ps1 -Task 总结
.\desktop\quick_ask.ps1 -Task 翻译成英文
.\desktop\quick_ask.ps1 -Task 提取待办
```

结果会写回剪贴板。建议用 PowerToys Keyboard Manager 绑定
`Win+Alt+Q` 等快捷键，详见 `desktop/README.md`。

当前网关的已验证参数：

| 指标 | 值 |
|---|---:|
| context 上限 | 256 tokens（prompt + output） |
| 3B W4A16 decode | 约 44.8 tokens/s |
| 3B 一次请求 wall | 约 0.90 秒 |
| 3B prefill | 约 0.46 秒 |
| 7B 混合 decode | 约 25.8 tokens/s（隔离环境） |
| 7B prefill | 3.06 秒（隔离）；53 秒以上（paging） |
| 并发 | 单请求串行 |

256 context 适合总结选中段落、改写、翻译、提取待办。长文档需要先分段；
后续再接入 map-reduce 和本地文件 RAG。

---

## 3B W4A16 checkpoint

3B 每次启动都重新做 dense→W4A16 会花约 5 分钟，因此先导出分层 checkpoint：

```powershell
python scripts\export_qwen3b_w4_checkpoint.py `
  --model F:\QwenQLoRA\models\Qwen2.5-3B-Instruct-smallshards `
  --adapter F:\QwenQLoRA\train\maibot-catgirl-v3\checkpoint-120 `
  --output D:\Qwen3B-W4A16-SM86
```

产物约 1.912 GiB，加载约 11.75 秒。

---

## 7B 量化流程

原始 7B BF16 checkpoint 不包含在本仓库中。

### 混合 Q4/Q3

```powershell
python scripts\quantize_qwen7b_mixed.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --output D:\Qwen7B-Mixed-SM86
```

在低内存机器上，主脚本完成 28 层后可能在 embedding/lm_head 阶段 OOM；随后执行：

```powershell
python scripts\finish_qwen7b_mixed_quantize.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --output D:\Qwen7B-Mixed-SM86
```

再把 lm_head 降到 3-bit：

```powershell
python scripts\quantize_qwen7b_head3.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --checkpoint D:\Qwen7B-Mixed-SM86
```

### 整体 Q4

```powershell
python scripts\quantize_qwen7b_q4.py `
  --source F:\QwenQLoRA\models\Qwen2.5-7B-Instruct `
  --mixed D:\Qwen7B-Mixed-SM86 `
  --output D:\Qwen7B-Q4-SM86
```

---

## 内核说明

### W4A16

- 2 个 signed int4 压入 1 byte；
- group 128；
- 每层一个 warp 处理一个输出 row；
- x 放 shared memory；
- 适合单 token batch=1 decode。

### W3A16

- 8 个 signed 3-bit 值压入 3 bytes；
- group 128；
- gate/up 使用 512 threads / 16 rows per block；
- down 使用 768 threads / 24 rows per block；
- 3-bit kernel 与显式反量化 matvec 的 cosine 为 1.0。

### GQA decode

- 一个 block 处理一个 query head；
- 最多 128 个静态 cache slot；
- 支持 16 Q / 2 KV 和 28 Q / 4 KV；
- 与 PyTorch SDPA 对照 cosine 约 1.0；
- 只用于单 token decode；prefill 仍走 PyTorch。

CUDA 源文件由：

```powershell
python scripts\export_sm86_kernels.py
```

从 Python 模块中的 NVRTC 字符串导出。

---

## 已知限制

- 这是 **model_executor 级** 后端，不是完整 engine；
- 没有接入 `worker/cache_engine.py` 的 full-attention block table/slot mapping；
- 没有接入 DeepInfer 原生 Sampler；
- 没有完整 scheduler/worker/continuous batching；
- prefill 仍是分块 dequant + cuBLAS，不是 tiled W4/W3 GEMM；
- GPU harness 在 Python 3.10 下验证；Python 3.12 GPU package 验证未完成；
- 3-bit MLP/lm_head 是有损量化，短输出可读不代表与 BF16 等价；
- 4 GB 上整体 Q4 只能作为质量/兼容实验，速度约 8 tokens/s；
- 单请求、短上下文，不是生产吞吐；
- 网关 context 上限为 256 token，尚未接入长文 map-reduce；
- 3B 与 7B 网关不能同时常驻 4 GB；
- 桌面快捷操作依赖 Windows PowerShell 剪贴板和本地 HTTP 网关。

---

## 结果与文档

- 上篇（内核/量化）：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`
- 下篇（生产力/基准）：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- 扩展篇（通用化/Llama）：`docs/ARTICLE_3_DEEPINFER_SM86_GENERIC.md`
- 合并稿存档：`docs/QWEN_DEEPINFER_4GB_ENGINEERING_ARTICLE.md`
- 3B：`docs/QWEN_SM86_W4A16_NATIVE.md`
- 7B 混合：`docs/QWEN_7B_MIXED_SM86.md`
- 7B Q4：`docs/QWEN_7B_Q4_SM86.md`
- 7B 可行性：`docs/QWEN_7B_4GB_FEASIBILITY.md`
- 个人生产力落地：`docs/PERSONAL_PRODUCTIVITY.md`
- 3B/7B/Ollama 基准：`docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md`
- 通用模型抽象层：`docs/GENERIC_MODEL_ADAPTER.md`
- 动态批处理与调度器：`docs/DYNAMIC_BATCHING_SCHEDULER.md`
- 结构化结果：`docs/results/`

---

## 许可证与来源

本项目基于 DeepInfer 的 Apache-2.0 代码与设计进行非官方扩展。

- 原始 DeepInfer：<https://gitee.com/mirrors/DeepInfer>
- 本仓库不是 DeepInfer 官方仓库；
- 原始代码和衍生代码遵循 Apache License 2.0；
- 详见 `LICENSE` 与 `NOTICE`。

如果你使用本仓库的结果，请同时说明：

1. 这是 **SM86 model_executor 级** 实现；
2. 不是完整 DeepInfer engine；
3. 4 GB 上整体 Q4 与几十 tokens/s 不能同时满足。
