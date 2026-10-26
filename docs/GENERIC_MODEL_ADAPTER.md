# DeepInfer SM86 通用模型抽象层

> 系列扩展篇：完整过程见 `docs/ARTICLE_3_DEEPINFER_SM86_GENERIC.md`。

> 方向一：构建模型抽象层，解耦“模型定义”与“执行逻辑”。

## 1. 为什么要做

此前 SM86 路径与 Qwen2.5 深度绑定：

- HF config 解析只认识 qwen2；
- registry 只注册 Qwen2；
- checkpoint loader 只按 Qwen2 的字段构建；
- 模型/执行器虽然已经是标准 GQA decoder，但名字和入口仍像 Qwen 专用代码。

这会导致每接入一个新模型都要改执行器。正确做法是先归一化模型规格，再让
执行器只依赖规格。

## 2. 抽象层结构

```text
HF config.json
      │
      ▼
ArchitectureAdapter
  Qwen2Adapter / LlamaAdapter
      │
      ▼
ModelSpec  ← 统一字段：层数、头数、KV 头、head_dim、激活、RoPE、norm、tie 等
      │
      ▼
ModelRegistry
      │
      ▼
Generic decoder execution
  RMSNorm + RoPE + GQA + SwiGLU
  W4A16 / W3A16 / GQA kernel
  static cache + CUDA Graph
```

核心文件：

```text
deepinfer/model_spec.py
deepinfer/hugging_face/qwen2_config.py   # 兼容包装，内部走 ModelSpec
deepinfer/model_executor/models/__init__.py
deepinfer/model_executor/models/llama.py
deepinfer/model_executor/models/llama_w4a16.py
scripts/export_w4_checkpoint.py
```

## 3. ModelSpec

`ModelSpec` 统一描述：

```text
family / architecture / model_type
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
vocab_size
max_position_embeddings
rms_norm_eps
rope_theta / rope_scaling
attention_bias / attention_dropout
hidden_act
mlp_type
tie_word_embeddings
dtype
sliding_window
bos/eos
adapter_path
```

执行器不再从 HF 原始 dict 直接取 Qwen 字段，而是读 `ModelSpec.to_gpt_config()`。

## 4. 已支持的模型族

| HF model_type | Architecture | 状态 |
|---|---|---|
| `qwen2` | `Qwen2ForCausalLM` | 已完成，保持原结果 |
| `llama` | `LlamaForCausalLM` | 已完成标准 GQA/SwiGLU/RoPE 路径 |

Llama 适配差异：

- q/k/v bias 默认 `False`；
- `head_dim` 可从 config 读，否则 `hidden_size // num_attention_heads`；
- RoPE 使用 config 的 `rope_theta`；
- tied embedding 由 config 决定；
- 权重 key 与通用 decoder 直接对齐。

当前明确拒绝：

- `rope_scaling`；
- sliding-window attention；
- MoE / MLA / 多模态；
- 非 SwiGLU MLP。

## 5. 执行器如何解耦

`Qwen2MLP` 不再硬编码 `F.silu`，而是：

```python
self.activation = _activation_fn(cfg.hidden_act)
...
self.down_proj(self.activation(gate) * up)
```

`ModelRegistry` 现在：

```text
Qwen2ForCausalLM -> qwen2.Qwen2ForCausalLM
LlamaForCausalLM -> llama.LlamaForCausalLM
```

Llama 只覆盖来源标记，模型结构复用通用 GQA decoder；W4A16 后端同理：

```text
LlamaW4A16ForCausalLM
```

## 6. 验证

### Qwen2 回归

```text
tests/models/test_qwen2_source_integration.py
6 passed
```

### Llama spec + forward

新增：

```text
tests/models/test_model_spec_llama.py
```

使用随机 tiny Llama：

```text
hidden_size = 128
layers = 2
heads = 4
kv_heads = 2
head_dim = 32
```

对比 HF Llama：

```text
shape = (1, 6, 512)
max_abs = 0.0
cosine = 1.0
```

### Llama W4A16

```text
scripts/export_w4_checkpoint.py
  --model D:\TinyLlama-SM86Test
  --output D:\TinyLlama-W4-SM86
```

已验证：

- 生成 `layer_00/01`、embed、norm、manifest；
- `LlamaW4A16ForCausalLM.from_checkpoint()` 可加载；
- forward logits shape `(1, 6, 512)`。

## 7. 接入新模型的最短路径

1. 在 `deepinfer/model_spec.py` 新增 `ArchitectureAdapter`；
2. 在 `ModelRegistry` 注册 architecture；
3. 如果结构与 Qwen2/Llama 相同，直接继承通用 decoder；
4. 如果 activation/MLP/norm 不同，在执行器增加 field dispatch；
5. 如果是新 weight naming，增加 adapter 的 key remap；
6. 用 tiny random HF model 做 logits 对齐；
7. 再走 `export_w4_checkpoint.py` 和网关。

## 8. 当前边界

- 这是“模型规格抽象层”的第一步，不是所有模型自动支持；
- 标准 GQA + RMSNorm + RoPE + SwiGLU 的模型可以较快接入；
- Llama 的 llama3 RoPE scaling 尚未实现；
- Mistral 的 sliding window 尚未实现；
- 完整 engine、CacheEngine/Sampler 仍未接入；
- 4 GB 下仍优先 3B W4A16 / 7B mixed。
## 9. 速度回归

抽象层只在模型加载/配置阶段工作；逐 token 执行仍然是同一个 W4A16/W3A16 +
GQA + CUDA Graph 路径。实测：

| 模型 | 通用化前 decode | 通用化后 decode | 变化 |
|---|---:|---:|---:|
| 3B fused W4A16 | 44.777 t/s | **45.417 t/s** | +1.43% |
| 7B mixed | 25.141 t/s | **25.497 t/s** | +1.42% |

3B wall 从 0.9009 s 到 0.8940 s。  
7B prefill 在两次测试中差异很大（53 s vs 172 s），原因是当前桌面后台显存导致的
WDDM paging，而不是抽象层增加了逐 token 开销；decode 反而略高。

原始数据：

```text
docs/results/productivity_3b_fused.json
docs/results/productivity_3b_generic.json
docs/results/qwen7b_current_paging_harness.json
docs/results/qwen7b_generic_harness.json
docs/results/generic_speed_regression.json
```

