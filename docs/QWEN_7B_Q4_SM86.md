# Qwen2.5-7B 真正 Q4 在 RTX 3050 4 GB 上的实测

## 结论

整体 Q4 **能加载并生成**，但在这张 4 GB 卡上无法全量驻留显存：

- 全 4-bit checkpoint 为 **3.658 GiB**；
- 实测 CUDA 可用显存约 **3.237 GiB**；
- 运行中持续 WDDM paging；
- decode 只有 **7.942 tokens/s（平均）**；
- prefill TTFT 约 **86.3 秒**。

因此 Q4 路径是“质量优先、速度不可用”的实验结果，不是可部署路径。

## 配置

| 项目 | 结果 |
|---|---:|
| 模型 | Qwen2.5-7B-Instruct |
| attention | 4-bit |
| MLP | 4-bit |
| embedding | 4-bit |
| lm_head | 4-bit |
| group size | 128 |
| 有效位宽（含 scales） | 约 4.125 bpw |

## 实测

| 指标 | 结果 |
|---|---:|
| checkpoint | 3.658 GiB |
| 权重加载 | 7.793 秒 |
| prefill TTFT | 86.316 秒 |
| Graph 捕获 | 0.243 秒 |
| 首次 replay | 0.135 秒 |
| warm eager decode | 2.412 秒 |
| decode 平均 | **7.942 tokens/s** |
| decode 中位 | **7.939 tokens/s** |
| 中位单 token | 125.955 ms |
| 峰值 allocated | 3.918 GiB |
| 峰值 reserved | 4.076 GiB |
| 生成后 allocated | 3.693 GiB |

生成结果：

```text
你好！很高兴见到你。你可以叫我
```

## 与混合 Q4/Q3 对比

| 路径 | 有效位宽 | checkpoint | decode |
|---|---:|---:|---:|
| 混合 Q4/Q3 + 3-bit head | 3.30 bpw | 2.930 GiB | **25.962 tokens/s** |
| 整体 Q4（本页） | 4.125 bpw | 3.658 GiB | 7.942 tokens/s |

整体 Q4 的质量更高，但 4 GB 显存不足导致：
- prefill 慢约 10 倍；
- decode 慢约 3.2 倍；
- 运行中 reserved 达到 4.076 GiB，超过物理显存。

## 复现

```powershell
cd D:\DeepInfer
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\run_qwen7b_mixed_gpu_harness.py `
  --checkpoint D:\Qwen7B-Q4-SM86 `
  --max-new-tokens 4 --warmup-replays 1 --max-cache 64 `
  --output D:\DeepInfer\doc\results\qwen7b_q4_gpu.json
```

Q4 checkpoint：`D:\Qwen7B-Q4-SM86`

## 结果文件

- `doc/results/qwen7b_q4_gpu.json`
- `doc/results/qwen7b_q4_sm86_summary.json`
- `doc/results/qwen7b_q4_manifest.json`

## 建议

- 要兼顾 Q4 和几十 tokens/s：至少 8 GB 显存。
- 要继续用这张 4 GB 卡：混合 Q4/Q3 是目前唯一能到 25 tokens/s 的路径。
- 4 GB 全 Q4 只能作为质量/兼容性验证，不建议作为生产运行方式。
