# DeepInfer SM86 通用化：ModelSpec 抽象层与 Llama 接入

> 系列第三篇·通用化扩展  
> 时间：2026-10-26  
> 硬件：NVIDIA RTX 3050 Laptop GPU，4 GB，SM86  
> 系统：Windows

> 术语约定：本文中的 **native** 一律指 **DeepInfer SM86 native**
> （DeepInfer model_executor / W4A16 / W3A16 / GQA / CUDA Graph），
> 不是 Ollama、llama.cpp、Transformers 兼容层或第三方 native engine。

> 系列文章：
>
> - 上篇：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`，内核与量化工程；
> - 下篇：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`，3B/7B 网关、OpenAI API 与 Ollama 基准；
> - 本篇：把 Qwen 专用路径抽象成 `ModelSpec`，并接入第一个非 Qwen 模型 Llama。

---

## 0. 结论先行

这次完成的不是“支持所有模型”，而是完成了通用化的关键一步：

```text
HF config
  → ArchitectureAdapter
  → ModelSpec
  → ModelRegistry
  → 通用 GQA/SwiGLU/RoPE decoder
  → W4A16 / W3A16 / GQA / CUDA Graph
```

已经验证：

| 项目 | 结果 |
|---|---|
| Qwen2/Qwen2.5 原路径 | 保持可用 |
| Llama 标准 GQA | 已接入 |
| tiny Llama vs HF logits | max_abs 0.0，cosine 1.0 |
| 通用 W4A16 exporter | tiny Llama 导出、加载成功 |
| Qwen2 原测试 | 6 passed |
| Llama spec/forward 测试 | 7 passed |
| 3B decode 回归 | 44.777 → 45.417 t/s |
| 7B decode 回归 | 25.141 → 25.497 t/s |

结论：

> **Qwen2/Qwen2.5 的逐 token 速度没有损失；通用化只发生在加载和配置阶段。**

---

## 1. 为什么必须做抽象层

此前 SM86 路径与 Qwen2.5 深度绑定：

- HF config 解析只认识 `qwen2`；
- `ModelRegistry` 只注册 `Qwen2ForCausalLM`；
- checkpoint loader 只按 Qwen 字段构造；
- MLP 直接写死 `F.silu`；
- runtime 只认识 Qwen/mixed backend。

这会导致每接一个新模型都要改执行器。  
真正需要的是：

> 让执行器只依赖统一的模型规格，而不是依赖 Qwen 的字段和类名。

方向就是图里的：

> **构建模型抽象层，解耦“模型定义”与“执行逻辑”。**

---

## 2. 新的结构

```text
HF config.json
      │
      ▼
ArchitectureAdapter
  Qwen2Adapter
  LlamaAdapter
      │
      ▼
ModelSpec
  层数 / 头数 / KV 头 / head_dim / 激活 / RoPE / norm / tie ...
      │
      ▼
ModelRegistry
      │
      ▼
通用 decoder execution
  RMSNorm + RoPE + GQA + SwiGLU
  W4A16 / W3A16 / GQA kernel
  static KV cache + CUDA Graph
```

核心文件：

```text
deepinfer/model_spec.py
deepinfer/hugging_face/qwen2_config.py
deepinfer/model_executor/models/__init__.py
deepinfer/model_executor/models/llama.py
deepinfer/model_executor/models/llama_w4a16.py
scripts/export_w4_checkpoint.py
```

---

## 3. ModelSpec

统一字段：

```text
family
architecture
model_type
hidden_size
intermediate_size
num_hidden_layers
num_attention_heads
num_key_value_heads
head_dim
vocab_size
max_position_embeddings
rms_norm_eps
rope_theta
rope_scaling
attention_bias
attention_dropout
hidden_act
mlp_type
tie_word_embeddings
dtype
sliding_window
bos_token_id / eos_token_id
adapter_path
```

执行器不再直接读 HF 原始 dict，而是读：

```python
spec.to_gpt_config()
```

模型加载、W4A16 构造、checkpoint loader、runtime 都用同一份归一化字段。

---

## 4. ArchitectureAdapter

当前 adapter：

```text
Qwen2Adapter
  attention_bias = True
  architecture = Qwen2ForCausalLM

LlamaAdapter
  attention_bias = False
  architecture = LlamaForCausalLM
```

Llama 适配差异：

- q/k/v bias 默认 `False`；
- `head_dim = hf.head_dim or hidden_size // num_attention_heads`；
- `rope_theta` 从 config 读取；
- `tie_word_embeddings` 从 config 读取；
- 权重 key 与通用 decoder 直接对齐。

当前 adapter 明确拒绝：

- `rope_scaling`；
- sliding-window attention；
- 非 SwiGLU MLP；
- MoE / MLA / 多模态。

这是有意的：

> 不静默忽略不支持的配置，而是在 load 阶段直接报错。

---

## 5. 执行器如何解耦

### 5.1 Registry

```text
Qwen2ForCausalLM -> qwen2.Qwen2ForCausalLM
LlamaForCausalLM -> llama.LlamaForCausalLM
```

`LlamaForCausalLM` 只标记来源，结构和执行路径复用通用 GQA decoder。

### 5.2 动态激活

以前：

```python
return self.down_proj(F.silu(gate) * up)
```

现在：

```python
self.activation = _activation_fn(cfg.hidden_act)
...
return self.down_proj(self.activation(gate) * up)
```

支持：

```text
silu
gelu
gelu_new
```

### 5.3 Runtime

网关的 `--backend w4a16` 会读取模型 architecture：

```text
LlamaForCausalLM -> LlamaW4A16ForCausalLM
Qwen2ForCausalLM -> Qwen2W4A16ForCausalLM
```

因此同一套 OpenAI 网关可以加载 Llama W4A16 checkpoint。

---

## 6. 第一个非 Qwen：Llama

### 6.1 为什么选 Llama

Llama 与 Qwen2 的结构接近：

- decoder-only；
- RMSNorm；
- RoPE；
- GQA；
- SwiGLU；
- 相同 safetensors key 命名。

差异集中在 config/adapter，而不是 kernel。

### 6.2 数值对齐

使用 random tiny Llama：

```text
layers = 2
hidden = 128
heads = 4
kv_heads = 2
head_dim = 32
vocab = 512
```

对比 HF Llama：

```text
shape   (1, 6, 512)
max_abs 0.0
cosine  1.0
```

这不是“能 import”，而是逐 logits 对齐。

### 6.3 Llama W4A16

新增：

```text
scripts/export_w4_checkpoint.py
deepinfer/model_executor/models/llama_w4a16.py
```

已验证：

- tiny Llama 导出分层 W4A16；
- `LlamaW4A16ForCausalLM.from_checkpoint()` 可加载；
- forward logits shape `(1, 6, 512)`；
- 普通 Qwen2 3B checkpoint 仍可加载。

---

## 7. 通用 W4A16 exporter

用法：

```powershell
python scripts\export_w4_checkpoint.py `
  --model D:\models\SomeLlama `
  --output D:\SomeLlama-W4-SM86
```

流程：

1. 读取 HF config；
2. `ArchitectureAdapter` 解析为 `ModelSpec`；
3. Registry 找 source model class；
4. dense 模型加载；
5. 选对应 W4A16 class；
6. 逐层保存 packed/scales/bias；
7. 写 embed/norm/manifest。

manifest 新增：

```json
{
  "architecture": "LlamaForCausalLM",
  "backend": "w4a16"
}
```

这样 runtime 不再靠目录名猜模型类型。

---

## 8. 速度回归

抽象层只在加载/配置阶段工作。逐 token 仍然是同一个：

```text
W4A16 GEMV
W3A16 GEMV
256-slot GQA
static KV cache
CUDA Graph
```

### 3B Qwen2.5

| 项目 | 通用化前 | 通用化后 |
|---|---:|---:|
| decode 中位 | 44.777 t/s | **45.417 t/s** |
| prefill 中位 | 0.4551 s | 0.4566 s |
| wall 中位 | 0.9009 s | **0.8940 s** |

### 7B mixed

| 项目 | 通用化前 | 通用化后 |
|---|---:|---:|
| decode 中位 | 25.128 t/s | **25.555 t/s** |
| decode 平均 | 25.141 t/s | **25.497 t/s** |
| 峰值 allocated | 3.196 GiB | 3.196 GiB |
| 峰值 reserved | 3.357 GiB | 3.357 GiB |

7B prefill 两次测试差异很大：

```text
53.089 s -> 172.118 s
```

这是后台 GPU 占用导致的 WDDM paging，不是抽象层增加了逐 token 开销。  
decode 和峰值显存都没有退化。

---

## 9. 测试

### Qwen2 原路径

```text
tests/models/test_qwen2_source_integration.py
6 passed
```

### Llama spec + forward

```text
tests/models/test_model_spec_llama.py
```

两文件合计：

```text
7 passed
```

### 通用 W4A16 导出

```text
saved layer 1/2
saved layer 2/2
done architecture=LlamaForCausalLM
```

---

## 10. 走过的弯路

### 10.1 一开始只加 Llama 类，没有抽象字段

第一反应是：

> 复制一个 `LlamaForCausalLM`，改改名字就行。

但 config loader、checkpoint loader、runtime 仍然读 Qwen 字段。  
所以必须先做 `ModelSpec`，否则只是换了类名。

### 10.2 `ArchitectureAdapter.parse` 忘记实现

第一次跑 Qwen2 测试直接失败：

```text
NotImplementedError
```

原因是基类 `parse()` 没有调用 `_base_spec()`。  
这个小错误导致 5 个原测试同时失败：

> 抽象层必须从第一天就覆盖原有 Qwen 路径。

### 10.3 Python 3.10 下直接 import deepinfer 失败

```text
ImportError: cannot import name 'Self' from 'typing'
```

3B/7B/通用 exporter 在 Python 3.10 下都需要 harness stubs。  
通用 exporter 不能直接 `from deepinfer.hugging_face import ...`。

### 10.4 `_load_module` 返回 fake package

在 Python 3.10 harness 中：

```text
deepinfer.model_executor.models
```

已经被安装成 fake package，直接 `_load_module` 同名文件不会执行模块体，导致：

```text
module has no attribute ModelRegistry
```

修复：

- registry 使用独立模块名加载；
- 不能依赖 fake package 的属性。

### 10.5 untied Llama 的 lm_head 是 leaf Variable

通用 exporter 第一次加载 tiny Llama 报：

```text
a leaf Variable that requires grad is being used in an in-place operation
```

原因：

- Qwen2.5-3B tied embedding，跳过了 lm_head copy；
- Llama `tie_word_embeddings=false`，需要 copy untied lm_head；
- 但模型还没 `requires_grad_(False)`。

修复：

```python
model.requires_grad_(False)
```

放在 from_dense_model 的构造之后、拷贝之前。

### 10.6 `head_dim` 不是所有模型都显式提供

Llama config 可能没有 `head_dim`。  
修复：

```python
head_dim = hf.get("head_dim") or hidden_size // num_attention_heads
```

并在 `ModelSpec.validate()` 检查乘积关系。

### 10.7 Qwen 的 q/k/v bias 不能默认套给 Llama

Qwen2 attention 有 q/k/v bias；Llama 没有。  
adapter 显式设置：

```text
Qwen2Adapter.attention_bias_default = True
LlamaAdapter.attention_bias_default = False
```

否则权重 shape 能对上，但模型会多出 bias，加载失败或结果错误。

### 10.8 不是所有 Llama 都能直接支持

Llama 3 系列可能带：

```text
rope_scaling: llama3
```

当前执行器只有 base RoPE，因此 adapter 直接报错，而不是静默忽略。  
这意味着：

- 标准 Llama 2 / TinyLlama 风格可接；
- Llama 3 的 llama3 RoPE 需要额外实现；
- Mistral sliding window 需要额外 mask 逻辑。

### 10.9 7B prefill 波动被误认为抽象层开销

通用化后第一次 7B harness：

```text
prefill = 172 s
```

看起来像“抽象层变慢了”。  
对比 decode 和峰值显存后确认：

```text
decode 25.497 t/s
peak 3.196 GiB
```

kernel 和 graph 都没变；慢的是 4 GB 边界下的 paging。  
不能拿 prefill 的波动判断抽象层开销。

### 10.10 tiny Llama 只能证明数值正确，不证明质量

random tiny Llama 的 max_abs=0 说明：

- 结构、RoPE、GQA、SwiGLU、weight mapping 对齐；
- 不说明真实 Llama 权重的生成质量；
- 也不代表有生产速度。

真实模型还需要下载、量化、benchmark。

---

## 11. 支持矩阵

| 模型族 | 状态 | 说明 |
| 模型族 | 状态 | 说明 |
|---|---|---|
| Qwen2 / Qwen2.5 | 已完成 | 原 W4A16/W3A16 + 3B/7B |
| Llama 标准 GQA | 已完成 | tiny 数值对齐 + W4 导出/加载 |
| Llama 3 rope_scaling | 未完成 | 需要 llama3 RoPE |
| Mistral | 未完成 | 需要 sliding-window attention |
| Gemma / Phi | 未完成 | 需要额外 adapter/MLP/norm |
| MoE / MLA | 未完成 | 需要专门执行路径 |

---

## 12. 接入新模型的最短路径

1. 在 `deepinfer/model_spec.py` 新增 adapter；
2. 在 `ModelRegistry` 注册 architecture；
3. 如果结构与 Qwen/Llama 相同，继承通用 decoder；
4. 如果 activation / MLP / norm 不同，在执行器增加 field dispatch；
5. 如果 weight naming 不同，增加 key remap；
6. 用 tiny random HF model 做 logits 对齐；
7. 用 `scripts/export_w4_checkpoint.py` 导出；
8. 用网关跑真实 benchmark。

---

## 13. 复现

### Qwen2 回归

```powershell
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest `
  tests\models\test_qwen2_source_integration.py
```

### Llama 测试

```powershell
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest `
  tests\models\test_model_spec_llama.py
```

### 通用 W4A16 导出

```powershell
python scripts\export_w4_checkpoint.py `
  --model D:\TinyLlama-SM86Test `
  --output D:\TinyLlama-W4-SM86
```

### 3B 速度回归

```powershell
python scripts\bench_productivity_backends.py --skip-7b --skip-ollama
```

---

## 14. 当前状态

代码和结果已提交到：

```text
https://github.com/umouyoumou-del/DeepInfer-SM86
```

关键提交：

```text
283180f feat: add model spec abstraction and Llama support
1985f9a docs: record generic refactor speed regression
```

原始结果：

```text
docs/results/generic_speed_regression.json
docs/results/productivity_3b_generic.json
docs/results/qwen7b_generic_harness.json
```

---

## 系列文章

- **上篇**：`docs/ARTICLE_1_DEEPINFER_SM86_KERNEL.md`  
  DeepInfer SM86 内核、量化、显存预算和生成正确性。
- **下篇**：`docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`  
  3B/7B 常驻网关、OpenAI API、桌面快捷操作和 Ollama 基准。
- **本篇**：`docs/ARTICLE_3_DEEPINFER_SM86_GENERIC.md`  
  ModelSpec 抽象层、Llama 接入、通用 W4A16 exporter 和速度回归。

> 当前不是“支持所有模型”，而是“执行器不再绑死 Qwen”。
