# 从 SDPA 到 Native Paged GQA：RTX 3050 4 GB 上的下一阶段

> 日期：2026-10-27
> 硬件：RTX 3050 Laptop 4 GB、SM86、Windows
> 软件：Python 3.10.3、torch 2.5.1+cu121、CUDA 12.1、CuPy 14.2
> 模型：Qwen2.5-3B W4A16
> 代码版本：本篇文章对应的 native kernel / benchmark 提交

## 1. 这一阶段到底要解决什么

上一阶段已经把 paged KV、连续批处理、chunked prefill、4 GB KV 预算、OpenAI
completions API 和租户 prefix cache 做出来了。但还有两个关键缺口：

1. paged attention 仍然走 `block gather -> dense K/V -> SDPA`；
2. 没有真实 3050 上的 TTFT/TPOT/并发吞吐与长文档插入抖动数据。

所以这一阶段的目标不是继续堆 API，而是：

- 写 native paged GQA decode kernel；
- 把 batch>1 的 W4A16 GEMV 从“每个 batch 重复读权重”改成真正共享权重；
- 在真实 4 GB 显卡上做并发 sweep；
- 测“长文档插入时，已有流 TPOT 是否被打爆”；
- 把失败路径、数字和仍然没做到的事情写清楚。

结果先说结论：

- native paged GQA 数值可用，单次 decode 约快 26%～31%；
- batched W4 GEMV 让 512x256 层的 batch 1/4/8 延迟几乎不变；
- 3B W4A16 在 4 GB 上，native+batched 路径 c1 8.592 tokens/s、c4
  14.736 tokens/s；
- `max_num_seqs=8` 能跑，但 TTFT p50 约 5.24 s，吞吐只到 15.931 tokens/s；
- 长文档插入的 TPOT 抖动没有“免费解决”：chunk 8 抖动低但 TPOT 被抬高，
  chunk 32/64 中位 TPOT 正常但出现约 0.5 s 尖峰；
- 这不是完整 DeepInfer engine，仍然只是 model_executor + engine-core 原型。

## 2. 弯路一：我们一开始以为没有 CUDA 环境

前一轮所有测试都跑在 `D:\QwenNativeGPU-venv`，这个环境里的 torch 是 CPU
构建，`torch.cuda.is_available()` 返回 `False`。因此上一轮很自然地得出
“没有 CUDA 测试环境，不应该把 native paged kernel 放进主链路”的结论。

这个结论在当时是对的，但环境判断不完整。后来检查发现：

```text
nvidia-smi                    -> RTX 3050 Laptop, 4096 MiB
F:\QwenQLoRA\venv\python.exe  -> torch 2.5.1+cu121, CUDA available
```

也就是说，机器一直有 GPU，只是默认测试 venv 是 CPU 版。于是这一阶段把真实
GPU 验证切到 `F:\QwenQLoRA\venv`，CPU 回归继续用原来的 venv。

这件事的教训不是“应该早点猜对”，而是：

> 性能结论必须绑定明确的解释器、torch build、CUDA 版本和驱动；只写
> “没有 GPU”是不够的。

后文所有数字都来自 `F:\QwenQLoRA\venv` 的 cu121 构建，不是 CPU 模拟。

## 3. 弯路二：Python 3.10 兼容性

GPU venv 是 Python 3.10，而仓库里有一部分代码是按 Python 3.12 写的。

第一个错误：

```text
ImportError: cannot import name 'Self' from 'typing'
```

出现在 `deepinfer/attrdict.py` 和 `deepinfer/model_executor/model_output.py`。
`typing.Self` 是 3.11 才进入标准库的，所以改成：

```python
from typing_extensions import Self  # noqa: UP035
```

第二个错误：

```text
class EnvVar[T]:
          ^
SyntaxError: expected ':'
```

这是 PEP 695 泛型类语法，3.12 才有。改成 3.10 可用的：

```python
T = TypeVar("T")

class EnvVar(Generic[T]):
    ...
```

第三个坑是 Ruff 的 `UP035`：它看到 `typing_extensions.Self` 就建议改成
`typing.Self`，但那样会在 3.10 上直接 ImportError。最后只能显式
`# noqa: UP035`，并把 `typing-extensions>=4.8` 写进 requirements。

这类问题不影响算法，但会让“在 3.10 上能启动”变成假象。修完之后，GPU
验证脚本才真正跑起来。

## 4. Native paged GQA decode kernel 的设计

新增的文件是 `deepinfer/engine/paged_attention.py`。decode 阶段输入：

```text
query        [B, H, 1, D]
key pool     [num_blocks, KVH, block_size, D]
value pool   [num_blocks, KVH, block_size, D]
block_table  [B, max_blocks]
valid_lengths[B]
```

不再做：

```text
block table -> gather dense K/V -> SDPA
```

而是在 kernel 中直接完成：

```text
每个 (batch, head) 一个 CUDA block
  -> 按 block_table 找到每个 token 的 K
  -> 计算 q·k / softmax / value 加权
  -> 写回 [B, H, 1, D]
```

支持：

- 变长 batch，每条序列不同 `valid_lengths`；
- GQA，多个 query head 共享 KV head；
- sliding window；
- Gemma2 的 attention logit softcap；
- FP16 KV；
- INT8 KV + FP16 scale sidecar；
- `head_dim <= 128`、`max_length <= 1024`。

不满足这些条件时自动回退 SDPA，包括：

- prefill `T > 1`；
- 有 external attention mask；
- 未支持的 dtype；
- head_dim / max_length 超出 envelope。

环境变量 `DEEPINFER_DISABLE_NATIVE_PAGED_ATTENTION=1` 可以强制回退，便于对比。

## 5. 弯路三：第一版 kernel cosine 只有 0.73，softcap 还是 NaN

第一版 direct test 的结果很难看：

```text
no window, no softcap : cosine 0.7296
window 3              : cosine 0.7987
softcap 5.0           : NaN
window + softcap      : NaN
```

一开始怀疑了三件事：

1. block table 的 batch stride 写错了；
2. shared-memory softmax reduction 有 race；
3. PyTorch 和 CuPy 不在同一个 CUDA stream。

于是分别做了这些检查：

- 用 `[10,20] / [30,40] / [50,60]` 这种手工 V 值验证输出；
- 用所有 K 相同、Q 相同的输入验证 softmax 是否退化成均匀分布；
- 单独写 debug kernel，把 q、k、dot、block_id、kv_offset 都写出来；
- 显式 `torch.cuda.synchronize()` 排除 stream 竞争；
- 把 shared reduction 换成一个最小 kernel 验证。

这些检查里有一个很迷惑的现象：手工 V 值的测试“看起来是对的”，但随机
K/V 的测试 cosine 只有 0.7。后来才意识到，手工测试的 V 值刚好让
“正确 softmax”和“均匀 softmax”产生相近的结果，属于假阳性。

真正的根因是：**CuPy RawKernel 的标量参数类型。**

kernel 声明是：

```c
float scale,
float softcap
```

Python 调用时传的是：

```python
float(scale),
float(softcap)
```

在 CuPy/NVRTC 的参数布局里，这些 Python float 没有按 kernel 的 `float`
参数严格对齐。结果是 kernel 里读到的 `scale` 变成了 0，`softcap` 变成
了错误值。`scale=0` 让所有 score 变成 0，softmax 退化成均匀分布；
`softcap` 的错误值则在 tanh 路径上产生 NaN。

修复很小：

```python
cp.float32(scale),
cp.float32(softcap or 0.0),
```

修完以后：

```text
B=2,H=4,KVH=2,D=16, window+softcap:
cosine 0.99999988
max abs error 0.0009765625
```

这个坑的教训：

> CUDA kernel 的标量参数不要依赖语言层的隐式类型推断。整数、float32、
> float64 在参数布局上不是一回事；混合 int/float 参数时尤其要显式转换。

## 6. Batched W4 GEMV：旧 kernel 每个 batch 都在重复读权重

旧 W4A16 GEMV 的 grid 是：

```text
grid = (output_rows / rows_per_block, batch)
```

每个 `grid.y` 对应一个 batch token。也就是说，batch=4 时同一份 packed weight
会被四个 block 组分别读取。虽然 L2 可能帮忙，但这不是真正的 batched GEMV。

新 kernel 改成 batch tile：

```text
grid = (output_rows / rows_per_block, ceil(batch / 8))
```

一个 warp 负责一个输出行，在寄存器里维护 8 个 batch 的 accumulator：

```text
for k0:
  load packed weight word once
  load scale once
  for j in 4:
    for b in 0..7:
      acc[b] += x[b, kk] * scale * int4_value
```

这样 packed weight 在一个 batch tile 内只读一次。batch=8 时，token 吞吐接近
线性增长；batch=1 仍走原来的 shared-x kernel，避免回退。

微基准（2048 -> 1024，group 128，RTX 3050）：

| batch | 单次耗时 | tokens/s | cosine |
|---:|---:|---:|---:|
| 1 | 0.1001 ms | 9987.9 | 1.0 |
| 2 | 0.0945 ms | 21174.2 | 0.99999994 |
| 4 | 0.0936 ms | 42749.0 | 0.99999994 |
| 8 | 0.1010 ms | 79213.8 | 0.99999988 |

注意这不是完整模型 tokens/s，只是单层 2048x1024 的 kernel 微基准；它说明
batch 维度的权重复用生效了。

## 7. Native kernel 数值验证

三条验证：

1. synthetic paged attention vs PyTorch reference；
2. synthetic batched W4 GEMV vs dequant reference；
3. 真实 3B W4A16 模型上，native paged decode logits vs SDPA fallback。

结果：

| 项目 | cosine | max abs error |
|---|---:|---:|
| paged attention FP16 | 1.0 | 0.0009765625 |
| paged attention INT8 | 1.0 | 0.0009765625 |
| batched W4 GEMV | 0.99999994 | 0.0009765625 |
| 3B end-to-end logits | 0.99999881 | 0.03515625 |

3B 的 argmax 两边一致。单次 decode：

```text
native paged : 0.0659 s
SDPA fallback: 0.0863 s
speedup      : 1.31x
```

FP16 和 INT8 paged attention 都通过了 synthetic reference；INT8 的误差没有
明显比 FP16 差，因为 scale 是每个 token/head 单独存的 FP16。

复现命令：

```powershell
python scripts\validate_sm86_kernels.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --output docs\results\sm86_kernel_validation_3b.json
```

## 8. 真实 4 GB 并发 sweep

测试条件：

```text
checkpoint: D:\Qwen3B-W4A16-SM86
max_cache: 128
block_size: 16
prefill_chunk_tokens: 32
max_new_tokens: 16
prefix cache: off
gpu_memory_utilization: 0.80
reserve_memory_mb: 384
```

启动后的预算：

```text
total GPU memory      4096 MiB
free before KV pool   1361 MiB
model parameters       594 MiB
model buffers         1364 MiB
KV budget              914 MiB
KV block                576 KiB
```

实际只用了 32 个 block（约 18 MiB），因为 `max_cache=128` 和
`max_num_seqs=4` 先限制了逻辑容量。4 GB 的主要压力不是 KV，而是 W4 权重、
FP16 buffer 和桌面占用。

### 8.1 并发 1 / 2 / 4

| 路径 | c | tokens/s | TTFT p50 | TPOT p50 | TPOT p95-p50 | max decode batch |
|---|---:|---:|---:|---:|---:|---:|
| SDPA reference | 1 | 6.906 | 0.946 s | 0.0914 s | 0.0066 s | 1 |
| SDPA reference | 2 | 9.845 | 1.486 s | 0.0865 s | 0.0114 s | 2 |
| SDPA reference | 4 | 12.049 | 3.510 s | 0.1398 s | 0.3735 s | 4 |
| native paged | 1 | 8.144 | 0.924 s | 0.0693 s | 0.0036 s | 1 |
| native paged | 2 | 10.709 | 1.500 s | 0.0695 s | 0.0159 s | 2 |
| native paged | 4 | 12.466 | 3.475 s | 0.1299 s | 0.3546 s | 4 |
| + batched W4 | 1 | 8.592 | 0.712 s | 0.0766 s | 0.0123 s | 1 |
| + batched W4 | 2 | 12.227 | 1.159 s | 0.0724 s | 0.0057 s | 2 |
| + batched W4 | 4 | 14.736 | 2.746 s | 0.1214 s | 0.2903 s | 4 |

从 SDPA 到 native+batched：

- c1：6.906 -> 8.592 tokens/s，约 +24.4%；
- c4：12.049 -> 14.736 tokens/s，约 +22.3%；
- c4 TTFT p50：3.510 -> 2.746 s，约 -21.8%；
- c1 TPOT p50 从 0.0914 s 降到 0.0766 s；native paged alone 更低
  （0.0693 s），batched W4 的 c1 数字受本轮采样波动影响。

### 8.2 max_num_seqs=8

把 `max_cache` 加到 256，跑 1/2/4/8：

| c | tokens/s | TTFT p50 | TPOT p50 | max decode batch | failed |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.897 | 0.715 s | 0.0722 s | 1 | 0 |
| 2 | 12.479 | 1.136 s | 0.0711 s | 2 | 0 |
| 4 | 14.756 | 2.720 s | 0.1222 s | 4 | 0 |
| 8 | 15.931 | 5.243 s | 0.1910 s | 8 | 0 |

完整数据在 `docs/results/paged_concurrency_3b_native_batched_c8.json`。

8 路没有 OOM，但 TTFT 和 TPOT 都明显变差。原因是当前调度循环仍然是：

```text
decode 一次
  -> prefill 一个 chunk
  -> 下一轮
```

prefill 和 decode 串行占 GPU。并发越高，prefill 队列越长，decode 等得越久。
这不是 4 GB 显存不够，而是执行结构没有真正 overlap。

### 8.3 INT8 KV 的真实收益和代价

同一条件，`--kv-cache-dtype int8`：

| 路径 | c | tokens/s | TTFT p50 | TPOT p50 | block bytes |
|---|---:|---:|---:|---:|---:|
| native + batched FP16 KV | 4 | 14.736 | 2.746 s | 0.1214 s | 589824 |
| native + batched INT8 KV | 4 | 13.868 | 2.831 s | 0.1324 s | 299520 |

INT8 block 从 576 KiB 降到 292.5 KiB，几乎减半。但 c4 吞吐下降约 6%，因为
kernel 每个 K/V 元素多了 scale load 和反量化。当前 3B 场景 KV 只占 18 MiB，
所以 INT8 的主要价值是容量，不是速度。

### 8.4 为什么 batch 4 不是 4 倍

有三层原因：

1. 权重虽然通过 batch tile 复用了，但 attention、norm、采样和 Python 调度
   仍逐层执行；
2. prefill 与 decode 串行，batch 4 的四个请求都要等 prefill chunk；
3. decode 每次只做 1 个 token，kernel launch 和 Python 层数占比高。

下一阶段如果要做动态 batch CUDA Graph，需要先解决可变 block table 和
ragged KV length 的图捕获问题，而不是直接把当前 scheduler 套 graph。

## 9. 长文档插入时，已有流的 TPOT 会不会被打爆

这是 P0 验收里最容易被忽略的问题。只测 aggregate tokens/s 不够，因为长
prompt 的 prefill 可能把正在 decode 的流卡住 0.5 秒，而平均吞吐看起来还行。

### 9.1 测法

1. 先启动一个短 prompt 的流式请求，`max_new_tokens=24`；
2. 收到第一个 stream delta 后，提交一个 189-token 的长 prompt；
3. 长请求 `max_new_tokens=2`；
4. 读完整条短流，取出 `GenerationResult.inter_token_seconds`；
5. 对比“没有插入”和“有插入”的逐 token interval。

测试用的 chunk 是 8 / 16 / 32 / 64。

### 9.2 结果

| chunk | 基线 TPOT p50 | 插入后 TPOT p50 | 插入后 ITL p95 | ITL p95-p50 | 最大 ITL | 长 prompt TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0706 s | 0.1885 s | 0.1981 s | 0.0095 s | 0.1992 s | 4.481 s |
| 16 | 0.0706 s | 0.2659 s | 0.2943 s | 0.0285 s | 0.2959 s | 3.465 s |
| 32 | 0.0705 s | 0.0692 s | 0.4926 s | 0.4234 s | 0.4950 s | 2.927 s |
| 64 | 0.0726 s | 0.0722 s | 0.5336 s | 0.4615 s | 0.5368 s | 1.606 s |

这组数字非常反直觉：

- chunk 8 的 p95-p50 抖动只有 9.5 ms，看起来最好；
- 但插入期间 TPOT 从 0.074 s 被抬到约 0.185 s，等于持续慢 2.5 倍；
- chunk 32/64 的 TPOT 中位数没变，但每隔一个 prefill chunk 就出现约
  0.5 s 的尖峰；
- chunk 16 的 p50 甚至比 chunk 8 更差，因为每个 prefill chunk 本身更重。

### 9.3 结论

当前调度器是“每次 decode 后串行做一个 prefill chunk”。在这个结构下：

- 想让每个 token 的延迟都稳定 -> 只能把 chunk 压得很小，长 TTFT 变差；
- 想让长 prompt 快 -> 只能加大 chunk，已有流出现周期性尖峰；
- 两者不能靠调一个固定 chunk 同时解决。

所以“长文档插入时 TPOT 抖动不超过阈值”这个验收，当前只能在给定 SLA 下
**部分满足**：

- 如果 SLA 是 p95-p50 抖动小于 30 ms，chunk 8/16 可以；
- 如果 SLA 是最大 ITL 小于 300 ms，chunk 16 可以，chunk 32/64 不行；
- 如果 SLA 是插入期间 TPOT 中位数不超过 100 ms，chunk 32/64 可以，chunk 8/16
  不行。

真正解决需要：

1. prefill 和 decode 放到不同 CUDA stream，做有限并发；
2. 或者做 time-sliced prefill，按毫秒预算切 chunk；
3. 或者做单次 forward 内的 prefill+decode mixed batch；
4. 同时用 tiled W4/W3 prefill GEMM 降低 prefill 成本。

这一阶段没有把这三件事伪装成“已完成”。

## 10. 仍然没做到的事情

- prefill + decode 在单次 forward 内真正混合；
- prefill 独立 CUDA stream 并发；
- 抖动闭环自动调节 chunk size；
- dynamic batch CUDA Graph；
- KV FP8、CPU cold-block offload；
- W4/W3 tiled prefill GEMM；
- 7B 在 4 GB 下的系统 sweep；
- pip / Docker / 生产部署。

其中优先级最高的是 prefill 和 decode 的 overlap，因为它同时影响 TTFT、
TPOT 和 c8 扩展性。native paged kernel 本身已经不是瓶颈。

## 11. 复现实验

CPU 回归：

```powershell
D:\QwenNativeGPU-venv\Scripts\python.exe -m pytest -q tests
```

GPU kernel 校验：

```powershell
python scripts\validate_sm86_kernels.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --output docs\results\sm86_kernel_validation_3b.json
```

并发 sweep：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-new-tokens 16 `
  --max-cache 128 `
  --prefill-chunk-tokens 32 `
  --concurrency 1,2,4 `
  --reserve-memory-mb 384 `
  --no-prefix-cache `
  --output docs\results\paged_concurrency_3b_native_batched_fp16.json
```

长文档插入抖动：

```powershell
python scripts\bench_tpot_jitter.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --max-cache 256 `
  --max-new-tokens 24 `
  --chunk-sizes 8,16,32,64 `
  --output docs\results\tpot_jitter_3b.json
```

W4 GEMV 微基准：

```powershell
python scripts\bench_w4a16_batched_gemv.py `
  --in-features 2048 --out-features 1024 --group-size 128 `
  --batches 1,2,4,8 --iterations 200 `
  --output docs\results\w4_batched_gemv_microbench.json
```

## 12. 最后的判断

这一阶段最大的成果不是“吞吐又涨了几个点”，而是把 paged scheduler 从
“功能上能并发”推进到了“有 native kernel、有真实 4 GB 数据、知道瓶颈在哪”。

最诚实的结论是：

- native paged GQA 和 batched W4 GEMV 都值得做，二者叠加让 3B 在 4 GB 上
  的 c4 吞吐从 12.0 提到 14.7 tokens/s；
- 8 路能跑，但当前串行 prefill/decode 结构会把 TTFT 和 TPOT 拉坏；
- 长文档插入抖动不是调一个 chunk size 就能解决的，必须做 prefill/decode
  overlap 或 mixed forward；
- INT8 KV 已经能 native 跑，但要接受约 6% 的 decode 代价，换来约一半 KV
  容量；
- 没有真实数据支撑的“优化”不应该写进 README。

下一步如果继续，应该优先做 prefill/decode 并发和 tiled prefill GEMM，而不是
继续在 API 层加功能。
