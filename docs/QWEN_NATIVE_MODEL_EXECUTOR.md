# Qwen2.5 原生 model_executor 接入（源码级里程碑）

> 日期：2026-10-25
> 状态：**源码级接入与 CPU/参考路径验证完成；SM100 原生 kernel、完整引擎和性能尚未验证**
> 目标模型：`Qwen2.5-3B-Instruct` + `maibot-catgirl-v3/checkpoint-120` LoRA

## 1. 结论先说

这次改动不是 `deepinfer_compat` 外部兼容层，而是把 Qwen2 放进了
DeepInfer 自己的：

- `ModelRegistry`
- HF `config.json` 解析入口
- `deepinfer/model_executor/models/`
- `Qwen2ForCausalLM.load_model(...)`
- 原生风格的 `forward(kv_caches, input_metadata, hidden_states)` 接口
- 标准 GQA KV cache 模型接口
- PEFT LoRA 权重合并

但必须明确：

- 默认源码路径的 attention 仍是 **PyTorch SDPA 参考实现**；SM86 后端已有自定义 GQA decode kernel，见 5.2；
- DeepInfer 的 `CacheEngine` 仍不支持 full attention，Qwen 还不能通过完整 engine 路径启动；
- native `Sampler` 尚未绑定；
- 当前 RTX 3050/Windows 环境不能构建或验证 SM100 原生目标；
- 因此不能把本文结果写成“DeepInfer 原生 Qwen 性能”。

## 2. 已完成

| 项目 | 状态 | 说明 |
|---|---|---|
| Qwen2 config 解析 | 完成 | 识别 `model_type=qwen2`，生成 DeepInfer `model_cfg.gpt` |
| ModelRegistry | 完成 | `Qwen2ForCausalLM -> qwen2.Qwen2ForCausalLM` |
| HF safetensors 分片加载 | 完成 | 支持 `model.safetensors.index.json` 与单文件 |
| tied embedding / lm_head | 完成 | 支持 `tie_word_embeddings=true` |
| GQA attention | 参考实现完成 | 16 Q heads / 2 KV heads，SDPA 参考路径 |
| RoPE | 完成 | base RoPE，按 Qwen2 的 rotate-half 方式实现 |
| RMSNorm | 完成 | float32 方差计算后回原 dtype |
| SwiGLU MLP | 完成 | gate/up/down 三线性层 |
| KV cache 续算 | 完成 | prefill + decode 分段 logits 与整段 logits 对齐 |
| PEFT LoRA 合并 | 完成 | 支持 PEFT `lora_A/lora_B`、alpha/r、RSLoRA |
| HF logits 对照 | 完成 | 微型 Qwen2 与 Transformers 输出通过 2e-5 绝对误差 |
| 真实 3B config/key 覆盖 | 完成 | 3.086B 参数；434 个 HF 张量全部匹配，tied `lm_head` 除外 |
| SM86 W4A16 + GQA 原生解码 | 完成（model_executor 级） | W4A16 GEMV + GQA kernel + 静态 cache + CUDA Graph；完整 engine 未接 |
| SM86 7B 混合 W4/W3 解码 | 完成（model_executor 级） | 2.930 GiB checkpoint，4-bit attention/embed + 3-bit MLP/head，25.962 tokens/s |
| 原生 GQA FlashAttention kernel | **未完成** | 目标 SM100 上需接 FlashAttention-4 |
| full-attention CacheEngine | **未完成** | 当前 CacheEngine 明确 assert 不支持 full attention |
| Qwen InputMetadata producer | **未完成** | 需要 block table / slot mapping / standard K/V layout |
| DeepInfer Sampler 绑定 | **未完成** | `Qwen2ForCausalLM.sample()` 明确抛 `NotImplementedError` |
| SM100 构建/运行/基准 | **未完成** | 当前硬件为 Windows + RTX 3050/SM86 |

## 3. 相关文件

| 文件 | 作用 |
|---|---|
| `deepinfer/hugging_face/qwen2_config.py` | Qwen2 HF config → DeepInfer AttrDict |
| `deepinfer/hugging_face/config.py` | 在 DeepSeek V4 与 Qwen2 配置解析之间分派 |
| `deepinfer/model_config_loader.py` | 允许纯文本 Qwen2，不再强制 `vl_aligner` |
| `deepinfer/model_executor/models/__init__.py` | 注册 `Qwen2ForCausalLM` |
| `deepinfer/model_executor/models/qwen2.py` | Qwen2 模型、HF 加载、LoRA 合并、参考 GQA/SDPA、静态 KV cache |
| `deepinfer/model_executor/models/qwen2_w4a16.py` | SM86 W4A16/GQA 原生解码、静态 cache、CUDA Graph |
| `doc/QWEN_SM86_W4A16_NATIVE.md` | SM86 原生后端结果、限制与复现 |
| `tests/models/test_qwen2_source_integration.py` | CPU/HF/PEFT/KV-cache 源码级测试 |
| `doc/results/qwen2_native_source_tests_20261025.txt` | 正式测试输出记录 |
| `deepinfer/__init__.py` | 非 Linux 环境下 `os.uname()` 的可导入回退 |
| `deepinfer/logger.py` | Windows source-test 下 `ForkProcess` 的可导入回退 |

## 4. 模型接口

```python
from deepinfer.model_executor.models.qwen2 import (
    Qwen2ForCausalLM,
    Qwen2InputMetadata,
    Qwen2LayerKVCache,
)
```

原生风格的 forward：

```python
metadata = Qwen2InputMetadata(
    input_ids=token_ids,
    positions=position_ids,
    is_prompt=True,
)
output = model.forward(None, metadata)
hidden_states = output.hidden_states
```

Prefill + decode cache：

```python
cache = model.new_kv_cache()   # list[Qwen2LayerKVCache | None] * num_layers

prefix_logits = model.forward_logits(ids[:, :5], pos[:, :5], kv_cache=cache)
next_logits = model.forward_logits(ids[:, 5:], pos[:, 5:], kv_cache=cache)
```

参考约束：

- 每层 cache 的 key/value 形状为 `[batch, num_kv_heads, sequence, head_dim]`；
- 当前 full attention 使用 block size=1 的逻辑视图，未来需映射为 CacheEngine 的
  paged full-attention layout；
- `forward_logits` 是源码级验证辅助接口，不是最终 engine 的采样入口。

## 5. 验证结果

测试环境（独立，不污染原生 Linux 目标环境）：

- Windows 11
- Python 3.12.13
- torch 2.5.1+cpu
- transformers 4.50.0
- peft 0.14.0
- 不使用 CUDA / 不构建 `deepinfer._C`

测试命令：

```powershell
cd D:\DeepInfer
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest `
  -q tests/models/test_qwen2_source_integration.py
```

结果：

```text
5 passed
```

完整输出：`doc/results/qwen2_native_source_tests_20261025.txt`。
文件哈希：`doc/results/qwen2_native_source_SHA256SUMS`。

测试覆盖：

1. Qwen2 config 分派和 ModelRegistry；
2. 微型 Qwen2 的 DeepInfer logits 与 Transformers logits 对齐；
3. prefill + decode 的 KV cache 续算与整段 logits 对齐；
4. PEFT LoRA merge 后 logits 与 PEFT 包装模型对齐；
5. 真实 Qwen2.5-3B 配置参数数量与 434 个 HF checkpoint 张量名称覆盖检查。

真实 3B 检查：

```text
architecture       = Qwen2ForCausalLM
hidden_size        = 2048
num_hidden_layers  = 36
num_attention_heads = 16
num_key_value_heads = 2
head_dim           = 128
parameter_count    = 3,085,938,688 (3.086B)
HF tensors         = 434
model parameters   = 435 (including tied lm_head)
missing            = {"lm_head.weight"} only, tied to embed_tokens
```

## 5.1 真实 3B 权重 CPU 冒烟结果

使用 DeepInfer 的 `Qwen2ForCausalLM` 代码直接加载完整 Qwen2.5-3B 基底和
`checkpoint-120` PEFT adapter，在 CPU 上执行一次短生成：

| 指标 | 结果 |
|---|---:|
| 模型加载耗时 | 345.469 秒 |
| 加载后 RSS | 5.454 GiB |
| 最终 RSS | 5.785 GiB |
| 参数数量 | 3,085,938,688 |
| 参数 dtype | BF16 |
| 合并 LoRA 矩阵 | 252 |
| prompt tokens | 30 |
| 生成 tokens | 3 |
| 生成文本 | `你好呀！` |
| TTFT | 66.131 秒 |
| decode 时间 | 4.801 秒（2 个后续 token） |
| decode 速度 | **0.417 tokens/s** |

该结果证明：

- 真实 3B 权重能被 DeepInfer Qwen2 模型代码完整加载；
- HF 分片、tied embedding 和 checkpoint-120 LoRA 合并可用；
- RoPE/RMSNorm/GQA/SwiGLU/KV-cache 的参考路径能生成合理中文文本。

但这**不是可用的性能结果**：

- CPU BF16 执行没有使用 GPU kernel；
- 机器只有约 5 GiB 空闲物理内存，测试发生明显换页；
- 没有进入 DeepInfer native engine；
- 不能把 0.417 tokens/s 与 Ollama 或 GPU NF4 性能直接比较。

原始结果：

- `doc/results/qwen2_native_real_smoke.json`

复现：

```powershell
cd D:\DeepInfer
D:\QwenNativeTest-venv\Scripts\python.exe scripts\smoke_qwen2_native_real.py `
  --max-new-tokens 3 --threads 6 `
  --output D:\DeepInfer\doc\results\qwen2_native_real_smoke.json
```

## 5.2 SM86 W4A16 GPU 解码实测

在 RTX 3050 Laptop 4 GB 上新增了 model_executor 级 SM86 W4A16 后端：

- 自定义 int4 W4A16 GEMV kernel（CuPy/NVRTC 编译）；
- 自定义 SM86 GQA decode kernel；
- 真实 Qwen2.5-3B + checkpoint-120 LoRA 权重转换；
- 静态 GQA KV cache；
- CUDA Graph decode。

24 次 graph replay 结果：

| 指标 | 结果 |
|---|---:|
| 平均 decode | 37.157 tokens/s |
| 中位数 decode | 42.666 tokens/s |
| 稳态均值（去掉首次 replay） | 42.721 tokens/s |
| prefill TTFT | 4.555 秒 |
| graph capture | 0.234 秒 |
| 峰值显存 | 2.493 GiB |
| 原生 GQA vs SDPA cosine | 1.0 |
| Graph vs 原生 GQA max diff | 0.0 |

注意：这是 DeepInfer model_executor 级 SM86 后端；完整 CacheEngine/Sampler/
engine 尚未接入，不能作为完整 DeepInfer engine 的支持声明。详细报告见
`doc/QWEN_SM86_W4A16_NATIVE.md`，原始结果见
`doc/results/qwen2_w4a16_native_gqa_gpu.json`。

## 5.3 Qwen2.5-7B 混合 W4/W3 实测

7B 纯 4-bit 权重为 3.657 GiB，超过这张卡可用的 3.237 GiB，已验证不可行。
改为 4-bit attention/embed/lm_head + 3-bit MLP 后，checkpoint 为 2.994 GiB。

RTX 3050 4 GB 实测：

| 指标 | 结果 |
|---|---:|
| decode 平均 | 25.962 tokens/s |
| decode 中位 | 26.188 tokens/s |
| prefill TTFT | 3.061 秒 |
| 峰值 allocated | 3.196 GiB |
| 3-bit MLP kernel | cosine 1.0 vs 显式反量化 |
| 原生 GQA vs SDPA | cosine 1.0 |

这是 model_executor 级实现，完整 CacheEngine/Sampler/engine 仍未接入。
详细报告见 `doc/QWEN_7B_MIXED_SM86.md`。

## 5.4 整体 Q4 在 4 GB 上的实测

全 4-bit checkpoint 为 3.658 GiB，超过 3.237 GiB 可用显存；可以加载并生成，
但运行中 paging：

| 指标 | 结果 |
|---|---:|
| decode 平均 | 7.942 tokens/s |
| prefill TTFT | 86.316 秒 |
| 峰值 reserved | 4.076 GiB |
| 有效位宽 | 约 4.125 bpw |

结论：4 GB 上整体 Q4 只能作为质量/兼容验证，不是可部署速度；详细报告见
`doc/QWEN_7B_Q4_SM86.md`。

## 6. 为什么还不能直接跑完整 engine

### 6.1 CacheEngine 只实现了压缩/MLA KV

`deepinfer/worker/cache_engine.py` 对 full-attention 层会直接：

```python
raise AssertionError("KVCacheManager 不支持 full attention 层")
```

Qwen2 是标准 GQA full attention，必须有：

- 每层 `[num_blocks, block_size, num_kv_heads, head_dim]` 的 K/V cache；
- block table / slot mapping；
- prefill/decode metadata；
- KV cache 的 clone/restore/state 语义；
- 与现有 CUDA Graph/输入 metadata 的对齐。

### 6.2 现有 attention kernel 不适配

现有 DeepInfer attention 路径面向 DeepSeek MLA/MQA、压缩 KV 和 FlashMLA
SM100 内核。Qwen2 是 16 Q heads + 2 KV heads 的 standard GQA，需要用
FlashAttention-4 或新写的 GQA kernel，不能直接复用现有 MLA kernel。

### 6.3 Sampler 仍是 DeepSeek-V4 形状

当前 `Qwen2ForCausalLM.sample()` 明确失败，避免返回一个看似可运行但不兼容
的采样对象。目标环境应接入 DeepInfer 原生 `Sampler`，并验证：

- prefill logits；
- decode sampling；
- EOS / stop；
- 结构化输出；
- top-k/top-p/temperature。

## 7. 下一步（必须在 SM100/Linux/CUDA 13）

1. 在目标机安装/构建 `flash_attn`、`tile_kernels`、`deep_gemm` 和
   `deepinfer._C`；
2. 在 `qwen2.py` 增加 `Qwen2FlashAttentionBackend`，接 FlashAttention-4 的
   varlen/GQA prefill 和 paged decode；
3. 为 `worker/cache_engine.py` 增加 full-attention KV layout；
4. 新增 `Qwen2ModelMetadata.prepare_inputmetadata_dict`，产出 block table、
   slot mapping、positions 和 cu_seqlens；
5. 把 `Qwen2ForCausalLM.sample()` 绑定到 DeepInfer `Sampler`；
6. 跑同一套 3 轮吞吐、TTFT、10 轮稳定性和输出对照；
7. 只有完成以上步骤并且实际以 DeepInfer native path 运行，才能报告
   “DeepInfer 原生 Qwen2.5 性能”。

## 8. 当前表述边界

可以写：

> 已在 DeepInfer 的模型注册、HF 配置解析和 `model_executor` 模型层完成
> Qwen2.5-3B + PEFT LoRA 源码级接入，并验证了 GQA/KV cache/LoRA 的参考
> 计算正确性。

不能写：

> DeepInfer 已原生支持 Qwen2.5。
> DeepInfer 在 RTX 3050 上达到 Ollama 吞吐。
> Qwen2 已跑通 DeepInfer 原生 attention/cache/kernel 路径。
