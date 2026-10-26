# Qwen2.5-7B 在 RTX 3050 4 GB 上的可行性结论

## 结论

**按当前 DeepInfer SM86 4-bit 路径，Qwen2.5-7B 无法在这张 4 GB 卡上全量 GPU 运行，
也无法通过 CPU 分层 offload 达到几十 tokens/s。**

不是 kernel 写法或 CUDA Graph 的问题，而是容量和带宽的硬限制：

- 7B int4 权重本身需要 **3.657 GiB**；
- 这张卡在 Windows + CUDA context 下实测只有 **3.237 GiB** 可用；
- 还没算 KV cache、激活、CUDA Graph 私有内存；
- 超过约 3.2 GiB 后，显存开始页换入，读取带宽从 **170.6 GiB/s**
  掉到 **32.8 GiB/s**；
- CPU offload 足够层数后，理论上限只有 **0.64 tokens/s**。

因此没有继续下载完整 15 GB BF16 7B，也没有伪造 7B 生成速度。
下载在约 2.6 GB 时停止，保留用于未来 8 GB+ 环境续传。

## 后续结果：混合 W4/W3 已跑通

在上述纯 4-bit 结论之后，后续实现了：

- 4-bit attention / embed / lm_head；
- **3-bit MLP**；
- checkpoint 2.994 GiB；
- 原生 W3A16 kernel、原生 GQA、CUDA Graph。

RTX 3050 4 GB 实测平均 **25.962 tokens/s**，中位 **26.188 tokens/s**。
这验证了上一节“需要 lower-bit quantization”的判断，但不是纯 4-bit 结果。
详见 `doc/QWEN_7B_MIXED_SM86.md`。

整体 Q4（4.125 bpw）也能运行，但 4 GB 下持续 paging：decode 约
7.942 tokens/s，prefill 约 86 秒，见 `doc/QWEN_7B_Q4_SM86.md`。

## 1. 7B 权重预算

Qwen2.5-7B-Instruct 配置：

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

int4、group size 128、scale 为 FP16 时：

| 项目 | 参数量 | 量化后 |
|---|---:|---:|
| 28 层线性层 | 6,525,288,448 | 3.134 GiB |
| embedding | 544,997,376 | 0.262 GiB |
| lm_head | 544,997,376 | 0.262 GiB |
| **合计** | **7,615,283,200** | **3.657 GiB** |

`tie_word_embeddings=false`，所以 embedding 和 lm_head 不能共享。

可用显存：

```text
GPU total              = 4.000 GiB
实测 CUDA free         = 3.237 GiB
7B int4 weights        = 3.657 GiB
weights-only deficit   = 0.420 GiB
```

再加上 128 slot KV cache（约 0.007 GiB）、激活、CUDA context 和 Graph
私有内存，实际缺口约 0.6 GiB 或更多。

## 2. 显存页换入带宽

用 uint8 全量扫描测有效读取带宽：

| 张量大小 | 中位耗时 | 有效带宽 |
|---:|---:|---:|
| 2.0 GiB | 12.571 ms | 162.9 GiB/s |
| 2.8 GiB | 17.273 ms | 166.0 GiB/s |
| 3.2 GiB | 19.202 ms | 170.6 GiB/s |
| 3.5 GiB | 63.597 ms | 56.4 GiB/s |
| 3.8 GiB | 118.468 ms | 32.8 GiB/s |

3.2 GiB 还能维持接近满带宽；3.5 GiB 起明显页换入。7B 权重 3.657 GiB
已经落在这一区间。

即使假设全量页换入后仍能保持 32.8 GiB/s，单 token 读 3.657 GiB
就需要约 **114 ms**，上限约 **8.8 tokens/s**，还没有计算 attention、
RMSNorm、RoPE 和实际 kernel 效率。因此 4 GB 上不可能达到 3B 那种
40 tokens/s 量级。

## 3. CPU offload 不可行

把需要 offload 的层预先反量化成 FP16，避免每 token 重新 dequant：

| 项目 | 结果 |
|---|---:|
| 7B 单层 CPU 矩阵乘（FP16 权重，6 线程） | 194.15 ms |
| 8 层 CPU 的 tokens/s 上限 | **0.64** |
| 4 层 CPU 的 tokens/s 上限 | 1.28 |

如果为了给 GPU 腾出约 0.42 GiB 而 offload 4~6 层，CPU 部分本身就会把
上限压到约 1 tokens/s。再叠加 CPU/GPU 往返和无法整段 CUDA Graph，
实际会更低。

## 4. 可选路线

| 路线 | 可行性 | 预期 |
|---|---|---|
| 8 GB+ GPU，复用当前 4-bit W4A16/GQA/Graph 路径 | 推荐 | 有机会达到几十 tokens/s |
| 4 GB + 3-bit 全量 GPU | 理论可行 | 需新 kernel；质量和速度都需重新验证 |
| 4 GB + 显存 paging | 能跑但不稳 | 上限约 5~9 tokens/s |
| 4 GB + CPU 分层 offload | 不推荐 | 当前 CPU 路径低于 1 tokens/s |
| 继续 3B W4A16 | 当前最佳 | 实测稳态 42.7 tokens/s |

## 5. 当前状态

- 7B BF16 下载：已停止，约 2.6 GB 留在
  `F:\QwenQLoRA\models\Qwen2.5-7B-Instruct`
- 7B 生成/速度：**未执行**，因为硬件预算不满足
- 3B 原生路径：仍可用，稳态约 42.7 tokens/s
- 没有把 7B 结果写成“已支持”

## 6. 相关结果

- `doc/results/qwen7b_memory_budget.json`
- `doc/results/w4a16_cpu_layer_7b.json`
- `doc/results/qwen7b_cpu_fp16_layer.json`
- `doc/results/qwen7b_paging_bench.json`
- `doc/QWEN_SM86_W4A16_NATIVE.md`
- `doc/results/qwen2_w4a16_native_gqa_gpu.json`
