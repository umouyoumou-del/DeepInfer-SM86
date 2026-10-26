# Qwen2.5-7B：4 GB RTX 3050 混合 W4/W3 原生解码

## 1. 结果

已在 RTX 3050 Laptop 4 GB 上跑通 Qwen2.5-7B-Instruct 的 DeepInfer
model_executor 级混合精度原生路径。

| 指标 | 结果 |
|---|---:|
| GPU | RTX 3050 Laptop, 4 GB, SM86 |
| 模型 | Qwen2.5-7B-Instruct |
| attention / embed | 4-bit，group 128 |
| lm_head | **3-bit**，group 128 |
| MLP | **3-bit**，8 个值压进 3 bytes，group 128 |
| 量化 checkpoint | 2.930 GiB |
| 权重加载 | 4.445 秒 |
| prefill TTFT | 3.061 秒（30 prompt tokens） |
| CUDA Graph 捕获 | 0.121 秒 |
| Graph 首次 replay | 0.0331 秒 |
| decode 平均 | **25.962 tokens/s** |
| decode 中位 | **26.188 tokens/s** |
| 中位单 token | 38.186 ms |
| 峰值 GPU allocated | 3.196 GiB |
| 峰值 GPU reserved | 3.357 GiB |
| 生成后 allocated | 2.970 GiB |

19 次 graph replay 总耗时 `0.73184` 秒（生成到 EOS 提前停止）。

生成结果（30 tokens，未遇到 EOS）：

```text
你好！您好吗？我可以帮你回答问题或完成任务，请告诉我需要做的，请告诉我你接下来的目的或任务。
```

## 2. 为什么必须用 3-bit MLP

7B 纯 4-bit 权重预算为 **3.657 GiB**，而这张卡实测只有
**3.237 GiB** 可用。混合精度改为：

- 4-bit attention：0.395 GiB
- 4-bit embed：0.262 GiB
- 3-bit lm_head：0.198 GiB
- **3-bit MLP**：2.075 GiB
- scales 及其他：约 0.00–0.10 GiB

总计约 **2.930 GiB**，才留出 CUDA Graph、KV cache 和激活空间。

## 3. 实现路径

- `W3A16Linear`：8 个 signed 3-bit 值压进 3 bytes；
- 自定义 SM86 `qwen2_w3a16_gemv` kernel；
- `W4A16Linear`：attention / lm_head 继续使用 4-bit GEMV；
- `QuantizedEmbedding`：int4 词表按 token gather 后反量化；
- `Qwen2MixedForCausalLM`：4-bit/3-bit 混合模型执行器；
- 构造期直接生成 fused qkv / fused gate-up，避免后替换造成 allocator 碎片；
- W3 gate/up 使用 512 threads，down 使用 768 threads；
- lm_head 也已降到 3-bit，减少每 token 词表读取量；
- 64-slot 静态 cache；
- 原生 SM86 GQA decode kernel；
- 静态 GQA KV cache + CUDA Graph；
- prefill 对 head/MLP 按行 chunk 反量化，避免 4 GB 卡 OOM。

校验：

- 3-bit kernel vs 显式反量化 matvec：cosine `1.0`，max abs `0.001953`；
- 原生 GQA vs PyTorch SDPA（3B tiny）：cosine `1.0`，max diff `0.015625`；
- CUDA Graph replay vs 普通原生前向（3B tiny）：max diff `0.0`。

## 4. 质量与限制

- 3-bit MLP 和 lm_head 都是**有损量化**；当前短回答可读，但不能据此声称无损或与 BF16 等价。
- 当前只测了单请求、64-slot 静态 cache、短上下文；不是生产吞吐。
- prefill 仍走分块 dequant + cuBLAS，约 8.7 秒，尚未做 tiled W4/W3 GEMM。
- `full_engine=false`；CacheEngine、Sampler、scheduler/worker 尚未接入。
- GPU 测速通过 Python 3.10 CUDA harness 加载同一份 DeepInfer source；
  Python 3.12 package GPU 验证仍未完成。
- 峰值 allocated `3.196 GiB`、reserved `3.357 GiB`，已经接近空闲显存
  `3.237 GiB`，说明接近 WDDM 边界；这版能跑，但不应再增加 cache 或并行请求。

## 5. 和 3B / Ollama 对比

| 路径 | decode |
|---|---:|
| 3B 纯 4-bit DeepInfer SM86 | 42.721 tokens/s（稳态） |
| 7B 混合 W4/W3+W3-head DeepInfer SM86 | **25.962 tokens/s（平均）** |
| 7B 混合中位 | 26.188 tokens/s |
| 整体 Q4（4 GB paging） | 7.942 tokens/s |
| Ollama 7B Q4_K_M（4 GB，需 offload） | 未在本机复测 |

7B 混合路径达到“二十 tokens/s”量级，但仍低于 3B，这是 7B 权重和
3-bit MLP 计算量共同决定的。

## 6. 复现

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --max-new-tokens 24 --warmup-replays 3 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_mixed_head3_final.json
```

量化 checkpoint：

```text
D:\Qwen7B-Mixed-SM86
```

结果文件：

- `doc/results/qwen7b_mixed_head3_final.json`
- `doc/results/qwen7b_mixed_sm86_summary.json`
- `doc/results/w3a16_sm86_microbench.json`
- `doc/results/w3a16_sm86_microbench_v16.log`
- `doc/results/qwen7b_mixed_finish.log`
- `doc/results/qwen7b_mixed_manifest.json`

## 7. 下一步

1. 继续优化 prefill 和 3-bit down，目标稳定 25–30 tokens/s。
2. 把 prefill 改成 tiled W4/W3 GEMM，降低约 8.7 秒 TTFT。
3. 接入 DeepInfer CacheEngine/Sampler。
4. 用 BF16 7B 或 Ollama 对照 3-bit 质量。
