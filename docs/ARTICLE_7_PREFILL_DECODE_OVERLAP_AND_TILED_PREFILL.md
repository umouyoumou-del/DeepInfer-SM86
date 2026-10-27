# DeepInfer-SM86 第七篇：prefill/decode 并发与 tiled W4/W3 prefill GEMM

> 系列第七篇·engine-core 并发调度与 prefill 内核  
> 时间：2026-10-27  
> 硬件：NVIDIA RTX 3050 Laptop GPU，4 GB，SM86  
> 系统：Windows + WDDM  
> 软件：PyTorch 2.5.1+cu121、CUDA 12.1、CuPy 14.2  
> 仓库：`D:\DeepInfer-SM86`

---

## 0. 结论先行

上一篇 `ARTICLE_6_NATIVE_PAGED_KERNEL_AND_4GB_BENCH.md` 的结尾把下一步写得非常明确：

1. 做真正的 prefill/decode stream overlap；
2. 做 tiled W4/W3 prefill GEMM；
3. 不要继续在 API 层堆功能。

这篇文章记录的就是这两项工作的完整实现过程、测量方法和走过的弯路。

最终完成的内容：

| 项目 | 状态 |
|---|---|
| W4A16 tiled prefill GEMM | ✅ |
| W3A16 tiled prefill GEMM | ✅ |
| W3A16 batch-tile GEMV 实验 | ⚠️ 正确但未启用（更慢） |
| paged scheduler 的 prefill/decode 双 CUDA stream 并发 | ✅ |
| chunked prefill + batched decode 同周期调度 | ✅ |
| prefill/decode 逻辑 overlap 与实际 stream overlap 分开计指标 | ✅ |
| CPU 回归 | ✅ 42 passed, 7 skipped |
| GPU 数值/并发测试 | ✅ 6 项通过 |
| 真实 3B W4A16 paged 并发 benchmark | ✅ |

关键结论：

- tiled prefill GEMM 是这次**最有确定性的收益**；
- W4 tiled GEMM 相对旧 dequantize + cuBLAS fallback 为 **2.24×–2.87×**；
- W3 tiled GEMM 为 **7.71×–9.45×**；
- 真实 3B W4A16 paged 路径相对上一版串行 prefill 基线：
  - c1: 8.592 → **11.189 tokens/s**
  - c2: 12.227 → **17.484 tokens/s**
  - c4: 14.736 → **22.779 tokens/s**
- prefill/decode 双 stream overlap 在真实 GPU 上确实发生；
- 但在 4 GB / 内存带宽受限卡上，overlap 对总吞吐的提升有限，不能把它写成
  “免费翻倍”；
- c2 场景 overlap 对 TPOT/jitter 有小幅改善；chunk 128 时长文档插入的 max ITL
  从 0.6605 s 降到 0.6489 s，收益真实但不大；
- 真正的长文档低抖动仍需更激进的策略：动态 chunk、decode 高优先级流、
  mixed forward 或 time-sliced prefill。

---

## 1. 起点：上一篇留下的具体问题

在 `ARTICLE_6` 结束时，paged scheduler 已经具备：

- paged / elastic KV block allocator；
- chunked prefill；
- 可变 batch decode；
- native paged GQA decode；
- batched W4A16 GEMV；
- KV memory budget；
- request-level OOM 隔离；
- prefix cache；
- TTFT/TPOT metrics；
- OpenAI `/v1/completions`。

但 scheduler 的执行顺序本质上是：

```text
admit queued requests
decode all DECODE requests
prefill one PREFILL chunk
```

这不是“同一次 forward 内 mixed batch”，也不是“prefill 和 decode 在 GPU 上同时执行”。
它只是**同一个调度周期内先做 decode，再做 prefill**。

### 1.1 旧 prefill 的计算路径

旧的 `W4A16Linear.forward`：

```python
if flat.shape[0] <= 32 and flat.is_cuda:
    output = self._forward_gemv(flat)
else:
    for output_row_chunk in range(...):
        dense_chunk = self._dequantize_rows(start, end)
        out[:, start:end] = F.linear(flat, dense_chunk)
```

这有两个问题。

第一，M=16/32 的 prefill chunk 原先会走小 batch GEMV。W4 的 batch-tile 能复用
权重，但矩阵乘仍然没有 tiled GEMM 的 2D tile 复用；W3 的 `grid.y=batch` 更会
让每个 batch row 重复读 packed weight。

第二，M 很大或 CPU fallback 时会走 dequantize + cuBLAS，临时生成
`[chunk_rows, K]` 的完整 FP16 权重块。对于 7B 的 W3 MLP，这一步尤其重。

### 1.2 旧 `_run()` 的序列结构

```python
decode_worked = self._decode_step()
prefill_worked = self._prefill_step()
```

即使 decode 和 prefill 在“同一轮 scheduler 循环”中都有工作，它们仍然共用一个
默认 CUDA stream，GPU 上是前后串行的。

上一篇记录过：

- chunk 小，decode 抖动小，但长 prefill 的 TTFT 高；
- chunk 大，TTFT 好一点，但 decode 的 max ITL 会出现明显尖峰；
- 单靠固定 chunk size 无法同时满足 TTFT 和 TPOT。

所以这次必须从执行结构上改，而不是继续调一个 magic number。

---

## 2. Tiled W4A16 prefill GEMM

### 2.1 目标性质

不是简单地写一个“量化 GEMM”，而是先定义必须满足的工程性质：

1. 不物化完整的 `[N_chunk, K]` FP16 权重；
2. 任意时刻临时权重存储只和 tile 大小有关；
3. 输出 FP16，FP32 累加；
4. CPU / 非 FP16 有明确 fallback；
5. 不改变已有单请求和 batched decode 路径；
6. 可以单独做数值测试和微基准。

### 2.2 Tiling 参数

最终选择：

```text
CTA output tile:   BM=32, BN=64
K tile:            BK=32
threads:           256
thread output tile: 2 × 4
accum:             float
```

CTA 负责 `C[m0:m0+32, n0:n0+64]`。每轮 K：

```text
shared_a[32][32]     = 2.0 KiB
shared_b[32][65]     ≈ 4.1 KiB
```

每个线程负责：

```text
行: 2 * (tid >> 4), 2 * (tid >> 4) + 1
列: 4 * (tid & 15) ... +3
```

也就是每个线程 8 个 accumulator。

对每个 K 值：

- 一个 A 值被 4 列复用；
- 一个 B 值被 2 行复用；
- 8 个 FMA 只读 6 个 shared 元素。

### 2.3 W4 shared B tile 的解包

W4 packed 布局：

```text
packed[N, K/2]
每个 byte:
  低 4 bit  = even K
  高 4 bit  = odd K
scales[N, K/group_size]
```

一个 warp 处理两行 B tile：

- lane 0–15 处理第一行的 16 个 packed bytes；
- lane 16–31 处理第二行的 16 个 packed bytes；
- 每个 lane 展开两个 K 值，写入 `shared_b[k][n]`。

这样 global packed weight 访问沿 K 方向连续，同时 shared B 又能被 16 个
`col_group` 的线程复用。

核心逻辑：

```cuda
byte = packed[global_n * (K / 2) + k0 / 2 + byte_in_tile];
scale0 = scales[global_n * groups_per_row + global_k / group_size];
v0 = int4(byte & 0x0f) * scale0;
v1 = int4(byte >> 4) * scale1;
shared_b[local_k][local_n] = half(v0);
shared_b[local_k + 1][local_n] = half(v1);
```

### 2.4 W3 shared B tile 的解包

W3 每 8 个 signed 3-bit 值压成 3 bytes：

```text
q0 = b0        & 0x07
q1 = (b0 >> 3) & 0x07
q2 = ((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)
q3 = (b1 >> 1) & 0x07
q4 = (b1 >> 4) & 0x07
q5 = ((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)
q6 = (b2 >> 2) & 0x07
q7 = (b2 >> 5) & 0x07
```

然后：

```text
if q >= 4: q -= 8
```

W3 的线程映射与 W4 不同。一个 warp 一次处理 8 个 B 行：

- `lane >> 2` 选择行 0..7；
- `lane & 3` 选择这行的第几个 8-value group。

因此 8 个 warp 一轮就覆盖 64 个 B 行，每个 lane 处理 3 bytes / 8 values。
这比“一个 warp 一行、只有 4 个 lane 有活干”的写法更均衡。

### 2.5 后端选择

最终选择：

| 输入行数 | CUDA FP16 路径 |
|---:|---|
| 1 | 单行 GEMV |
| 2–8 | 小 batch GEMV（W4 使用 weight-reuse batch tile；W3 保持 `grid.y=batch`） |
| >8 | tiled W4/W3 GEMM |
| CPU / 非 FP16 | dequantize + cuBLAS fallback |

为什么不是所有 M 都走 tiled GEMM：

- M=1 时 32 行 tile 只有 1 行有效；
- M=2–8 时小 batch GEMV 的 tile 利用率更高；
- W4 的 batch-tile GEMV 能复用权重，实测比 tiled GEMM 更合适；
- W3 保持原 `grid.y=batch`，因为后文会看到 batch-tile 版本反而更慢。

为什么不是所有 M 都走 batched GEMV：

- W4 batched GEMV 每次最多复用 8 个 batch；
- W3 原 GEMV 每个 batch row 都要独立读权重；
- M=16/32 的 prefill chunk 继续用 GEMV 会把权重读多次。

### 2.6 数值正确性

测试方式：

- 构造真实 quantize 后的 `W4A16Linear` / `W3A16Linear`；
- M=37 强制走 tiled GEMM；
- 用同一模块的 `_dequantize()` 结果作为参考；
- 比较 cosine 和 mean abs diff。

结果：

- W4 cosine ≈ 1.0；
- W3 cosine ≈ 1.0；
- 误差来自 FP16 shared tile 和不同累加顺序，不是布局错误。

GPU 测试：

```text
test_w4a16_tiled_prefill_matches_dequantized_reference
test_w3a16_tiled_prefill_matches_dequantized_reference
```

### 2.7 Tiled GEMM 微基准

合成 packed 权重，M=30，group size=128；warmup 3 次，20 次平均。
对比对象是**同一模块的真实旧 fallback** `_forward_dequant_cublas()`，不是只和
dense FP16 cuBLAS 比。

W4A16：

| shape (M×N×K) | tiled | old fallback | speedup |
|---|---:|---:|---:|
| 30×5632×2048 | 0.736 ms | 2.109 ms | 2.87× |
| 30×2048×5632 | 0.834 ms | 1.884 ms | 2.26× |
| 30×3584×18944 | 4.632 ms | 10.365 ms | 2.24× |

W3A16：

| shape (M×N×K) | tiled | old fallback | speedup |
|---|---:|---:|---:|
| 30×37888×3584 | 6.081 ms | 57.467 ms | 9.45× |
| 30×3584×18944 | 3.553 ms | 27.400 ms | 7.71× |

原始结果：

- `docs/results/qwen_sm86_tiled_prefill.json`
- 复现脚本：`scripts/bench_sm86_tiled_prefill.py`

W3 的收益显著大于 W4，原因是旧 W3 fallback 的 PyTorch 位运算 dequantize 成本更高。
这再次说明 7B 的 prefill 瓶颈主要在 3-bit MLP，而不是 attention 的小投影。

---

## 3. 小 batch 路径与一次失败的 W3 batch-tile 实验

这一项不是最初计划的核心，但实现 overlap 时做了一个值得记录的负优化实验。

### 3.1 原状

- W4A16 已经有 `DP_BATCH_TILE=8` 的 weight-reuse batched GEMV；
- W3A16 的旧 GEMV 使用 `grid.y = batch`，每个 batch row 独立读同一份 packed weight；
- 从理论上说，W3 也应该能像 W4 一样把 8 个 batch 的 accumulator 放在一个 block 里，
  让 packed weight 只读一次。

### 3.2 实现与测量

最后确实写了一个 W3 batch-tile GEMV：

- 每个 block 负责一个输出行组；
- 一次处理最多 8 个 batch row；
- 权重行只读一次；
- 8 组 accumulator 分别累加。

在 30×5632×2048 形状上对比原来的 `grid.y=batch`：

| batch | 原 `grid.y=batch` | 新 batch-tile | 结果 |
|---:|---:|---:|---:|
| 2 | 0.1306 ms | 0.2716 ms | 0.48× |
| 4 | 0.2543 ms | 0.5162 ms | 0.49× |
| 8 | 0.5099 ms | 0.9583 ms | 0.53× |

结论是反直觉的：

> 权重少读了，但 8 组 accumulator 显著降低了 occupancy，最终整体更慢。

### 3.3 最终选择

- W3 batch-tile kernel 保留为实验代码和 correctness test；
- 但**不启用为默认 decode 路径**；
- 默认仍使用原 `grid.y=batch` GEMV 处理小 batch；
- M>8 才切 tiled W3 GEMM。

这件事的教训：

> 不能把“减少 global memory 读取”直接等同于更快；register pressure、occupancy
> 和 kernel 占用率必须实测。

## 4. 真正的 prefill/decode overlap

### 4.1 设计目标

目标不是“把两个函数放进两个线程”，而是：

1. decode batch 和 prefill chunk 各自独立准备；
2. decode forward 和 prefill forward 分别在两个 CUDA stream 上 launch；
3. 两者完成后统一 join；
4. join 之后才做 sampling、token accept、request state 更新；
5. 仍然保留 CPU 和禁用 overlap 时的串行 fallback；
6. 失败/OOM 只影响相关请求，不让 scheduler 线程整体崩溃。

### 4.2 为什么要先拆 `_decode_step` / `_prefill_step`

原来的两个方法都同时做：

```text
选请求 → 准备 views/input/mask → forward → synchronize
→ sample → state update → metrics
```

如果直接把这两个方法放到两个线程里，会同时修改：

- `self.running`；
- block allocator；
- prefix cache；
- request state；
- metrics。

这些都需要锁和复杂的状态一致性。

最终采用“三段式”重构：

```text
_prepare_prefill_work()
    - 选择 round-robin prefill 请求
    - 计算 chunk start/end
    - 构造 input_ids / positions / paged views
    - 不启动模型

_launch_prefill_work(work, stream)
    - 在指定 stream 上 forward
    - 记录 CUDA event
    - 不修改请求状态

_finish_prefill_work(work, logits)
    - sample first token
    - 处理 prefix cache
    - 状态 PREFILL -> DECODE
    - metrics / request finish
```

decode 侧完全对称：

```text
_prepare_decode_work()
_launch_decode_work()
_finish_decode_work()
```

这两个 work 对象就是并发调度的最小单元：

```python
@dataclass
class _PrefillWork:
    request: PagedScheduledRequest
    start: int
    end: int
    chunk_length: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    views: list[Any]
    started_event: Any | None
    finished_event: Any | None


@dataclass
class _DecodeWork:
    requests: list[PagedScheduledRequest]
    input_ids: torch.Tensor
    positions: torch.Tensor
    views: list[Any]
    started_event: Any | None
    finished_event: Any | None
```

### 4.3 双 stream 调度伪代码

```python
prefill_work, _ = self._prepare_prefill_work()
decode_work, _ = self._prepare_decode_work()

if prefill_work and decode_work and self.prefill_stream:
    decode_logits = self._launch_decode_work(
        decode_work, self.decode_stream
    )
    prefill_logits = self._launch_prefill_work(
        prefill_work, self.prefill_stream
    )
    self._join_overlap_streams()
    self._finish_prefill_work(prefill_work, prefill_logits)
    self._finish_decode_work(decode_work, decode_logits)
```

`_join_overlap_streams()`：

```python
current = torch.cuda.current_stream(self.runtime.device)
current.wait_stream(self.prefill_stream)
current.wait_stream(self.decode_stream)
current.synchronize()
```

### 4.4 为什么 wait 关系必须成对

准备阶段会在默认 stream 上做：

- `input_ids` / `positions` 的 `.to(device)`；
- paged view 的 block table；
- valid length tensor；
- 可能还有新 block 分配后的零初始化。

两个工作 stream 在 launch 前都必须等默认 stream：

```python
stream.wait_stream(torch.cuda.current_stream(device))
```

否则可能出现：

- decode kernel 已经读 block table，而 block table 还没写完；
- prefill kernel 已经写 KV block，而 block allocator 的零初始化还没完成；
- 上一步 `_finish_*` 还在改 mask 或 prefix cache，下一步已经开跑。

step 结束时 default stream 又必须等两个工作 stream：

```python
current.wait_stream(prefill_stream)
current.wait_stream(decode_stream)
```

之后才允许 CPU sampling 和 finish。这个双向依赖是**并发正确性的核心**，
不是可选的性能优化。

### 4.5 计时事件

没有把两个 forward 的耗时都记成“整个 overlap 周期”，而是每个 work 各自记录：

```python
work.started_event.record()
logits = self.model.forward_logits(...)
work.finished_event.record()
```

join 后：

```python
elapsed = work.started_event.elapsed_time(work.finished_event) / 1000
```

这样：

- `prefill_chunk_seconds` 只记录 prefill 自己的 stream 时间；
- `decode_step_seconds` 只记录 decode 自己的 stream 时间；
- 不会因为 overlap 把同一个墙钟时间重复算两次。

这里后来踩了一个坑：**serial fallback 下事件对象存在但没有 record**，
`elapsed_time()` 会直接抛：

```text
RuntimeError: Both events must be recorded before calculating elapsed time.
```

修复方式：

- 只有真正在 stream 上 launch 时才创建并记录 event；
- serial launch 时把 event 置空，改用 monotonic timer。

### 4.6 CPU / 禁用重叠的 fallback

CPU 没有 CUDA stream。设为 `enable_prefill_decode_overlap=False` 时也必须工作。

fallback 顺序：

```text
decode () -> prefill ()
```

保留原有 decode 优先的顺序，避免在 CPU 测试中改变 TPOT 行为。
同时仍然增加逻辑 overlap metric：

```python
self.metrics.inc("prefill_decode_overlap_steps")
```

但实际 stream overlap metric 只有在真正用两个 stream 时才增加：

```python
self.metrics.inc("prefill_decode_stream_overlap_steps")
```

这两个指标分开很重要：

- `prefill_decode_overlap_steps`：同一 scheduler cycle 内既有 prefill 又有 decode；
- `prefill_decode_stream_overlap_steps`：确实在两条 CUDA stream 上 launch。

这样 CPU 回归不会被迫 mock CUDA，同时也不会把“同周期调度”伪造成“物理并发”。

### 4.7 与 paged KV / block table 的配合

这次没有另造 static cache，而是直接复用 `PagedKVCacheManager.make_views()`：

- prefill view 写到当前 prompt 的 block；
- decode view 只读自己的历史 block，并写当前 token slot；
- 两个请求可以共享 prefix 的只读 block；
- prefill 和 decode 不写同一个请求的状态；
- block allocator 仍然只在 scheduler 线程中操作。

因此 stream overlap 不需要改 KV 布局，也不需要新增 per-request dense cache。

### 4.8 状态输出

`PagedBatchScheduler.status()` 新增：

```json
{
  "prefill_decode_overlap": true,
  "prefill_decode_stream_overlap_steps": 3
}
```

benchmark 的单次 run 也记录：

- `prefill_decode_overlap_steps`；
- `prefill_decode_stream_overlap_steps`。

### 4.9 CLI 开关

服务端新增：

```text
--no-prefill-decode-overlap
```

默认启用；需要调试、对比或排查显存时可以关闭。

benchmark 同样支持：

```powershell
python scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --prefill-chunk-tokens 32 `
  --no-prefill-decode-overlap
```

---

## 5. 真实测量

### 5.1 Tiled prefill 端到端收益

旧基线来自本轮修改前的同仓库结果：

`docs/results/paged_concurrency_3b_native_batched_fp16.json`

相同设置：

```text
Qwen2.5-3B W4A16
max_cache=128
prefill_chunk_tokens=32
max_new_tokens=16
no prefix cache
```

新结果：

`docs/results/paged_concurrency_3b_tiled_overlap.json`

| 并发 | 旧 tokens/s | 新 tokens/s | 提升 |
|---:|---:|---:|---:|
| 1 | 8.592 | **11.189** | 1.30× |
| 2 | 12.227 | **17.484** | 1.43× |
| 4 | 14.736 | **22.779** | 1.55× |

TTFT p50 也明显改善：

| 并发 | 旧 TTFT p50 | 新 TTFT p50 |
|---:|---:|---:|
| 1 | 0.712 s | **0.339 s** |
| 2 | 1.159 s | **0.515 s** |
| 4 | 2.746 s | **1.337 s** |

这说明 prefill 的主要收益来自 tiled GEMM，而不是 CUDA stream overlap。

### 5.2 overlap 打开 / 关闭对比

同一台机器、同一 checkpoint、同一 3B 命令，只切换：

```text
--no-prefill-decode-overlap
```

在新 tiled prefill 已经启用的情况下：

| 并发 | no-overlap tokens/s | overlap tokens/s | no-overlap TPOT p50 | overlap TPOT p50 | no-overlap jitter | overlap jitter | 实际 overlap steps |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.153 | 11.189 | 0.0728 | 0.0727 | 0.00725 | 0.00842 | 0 |
| 2 | 16.481 | **17.484** | 0.0796 | **0.0717** | 0.0332 | **0.0087** | 1 |
| 4 | 22.604 | 22.779 | 0.1056 | 0.1045 | 0.1583 | 0.1570 | 3 |

解读：

- c2 上有小幅吞吐和明显 TPOT/jitter 改善；
- c4 上吞吐几乎不变，说明 decode 与 prefill 都在争用同一份内存带宽；
- c1 没有 overlap，符合预期；
- `prefill_decode_stream_overlap_steps` 明确大于 0，证明不是“假 overlap”。

### 5.3 长文档插入的 ITL

使用 `scripts/bench_tpot_jitter.py`：

- 3B W4A16；
- long prompt 189 tokens；
- chunk size 128；
- max_new_tokens=12。

| 指标 | no-overlap | overlap |
|---|---:|---:|
| 插入期间 p95-p50 | 0.5819 s | **0.5707 s** |
| max ITL | 0.6605 s | **0.6489 s** |
| long prompt TTFT | 1.0299 s | **1.0195 s** |

只是小幅改善，不是数量级变化。

### 5.4 为什么 overlap 没有带来大收益

在 4 GB RTX 3050 上：

- 模型权重已经从显存读取；
- W4/W3 prefill 和 decode 都是 memory-bandwidth-bound；
- 两个 stream 同时运行时，带宽是共享资源；
- decode batch 只有 1–4，计算量很小；
- prefill 仍需读大量量化权重；
- 因此 overlap 更容易改善“等待/调度空窗”，但很难让总带宽翻倍。

这不是实现失败，而是硬件/工作负载边界。文章里不能写“overlap 后吞吐翻倍”。

真正的下一步应该是：

1. decode 使用高优先级 stream；
2. 有 decode 活跃时动态缩小 prefill chunk；
3. time-sliced prefill；
4. 单次 forward 内 prefill + decode mixed batch；
5. 对 prefill 做 Tensor Core MMA + cp.async pipeline，降低带宽/占用。

---

## 6. 正确性验证

### 6.1 CPU 全套

```powershell
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest -q tests
```

结果：

```text
42 passed, 7 skipped
```

这是本文当时的 CPU 结果；后续 ARTICLE_8 又加入了 FP8 和 CPU offload 测试，
当前 QwenNativeGPU CPU 回归为 `53 passed, 6 skipped`。

覆盖：

- paged KV block 分配、prefix cache、INT8 KV；
- chunked prefill / decode / OOM 隔离；
- CPU fallback 下 prefill+decode 同 cycle 逻辑；
- W4/W3 fallback；
- 通用模型 config / forward / cache；
- quantization 转换。

### 6.2 GPU

```powershell
python -m pytest -q tests\models\test_qwen2_w4a16_gpu.py `
  tests\apps\test_paged_scheduler_gpu.py
```

本次在 RTX 3050 CUDA 解释器中手动执行 6 项：

1. W4A16 prefill/decode vs dense；
2. W4A16 tiled GEMM vs dequantize 参考；
3. W4A16 batched GEMV vs 单行 GEMV；
4. W3A16 tiled GEMM vs dequantize 参考；
5. W3A16 batch-tile GEMV vs 单行 GEMV（实验路径，不作为默认）；
6. paged scheduler 真实双 stream overlap。

结果：全部通过。

### 6.3 新增关键测试

`test_prefill_decode_stream_overlap_runs_both_forwards` 断言：

```python
assert scheduler.metrics.get("prefill_steps_total") == 12
assert scheduler.metrics.get("prefill_decode_overlap_steps") > 0
assert scheduler.metrics.get("prefill_decode_stream_overlap_steps") > 0
assert status["prefill_decode_overlap"] is True
```

这个测试不是只检查“调度器还能生成”，而是检查实际 overlap 计数器。

---

## 7. 弯路与踩坑

### 7.1 最大的一次弯路：先在错误的仓库里实现

这次工程最严重的弯路不在 kernel，而在工作目录：

- 一开始改了 `D:\DeepInfer`；
- 正确的仓库实际是 `D:\DeepInfer-SM86`；
- 两者都有 Qwen2/DeepInfer/SM86 文件，但架构阶段不同；
- 正确仓库已经有 paged scheduler、block cache、prefix cache、metrics 等大量代码。

结果是：

- 第一版实现了一个 model_executor 级 static-cache 并发 engine；
- 它能工作，但和正确仓库已有的 paged continuous scheduler 架构重叠；
- 如果直接搬过去，会形成两套 scheduler，而不是增强现有 engine-core。

最终处理：

1. 承认目录错误；
2. 回到 `D:\DeepInfer-SM86`；
3. 重新按正确仓库的架构实现；
4. tiled kernel 可以复用，但并发调度必须改成 paged scheduler 内的 stream overlap；
5. 文章也重写，不再以错误仓库为准。

这个弯路的教训：

> 在大仓库里做性能改动前，先检查 Git remote、工作目录、当前已实现的 scheduler 边界。
> “同类代码”不等于“正确代码基”。

### 7.2 弯路：把 static-cache 并发 engine 当模板

错误仓库里做的第一版是：

```text
per-request dynamic KV
→ decode 时复制到 static batched KV
→ 两个 stream 做 prefill/decode
```

正确仓库已经有：

```text
paged KV manager
→ make_views(block_tables, seq_lens)
→ native paged GQA / SDPA
```

如果直接搬 static-cache 方案：

- 会重复管理 KV；
- 会绕过 block allocator / prefix cache；
- 会重新引入 `max_batch × max_cache` 预分配；
- 无法和 PagedScheduler 的 OOM/prefix 指标兼容。

最终没有使用 static cache，而是直接对 `_PrefillWork` / `_DecodeWork` 的
paged views 做双 stream launch。

### 7.3 弯路：CUDA event 创建了但没 record

在并发路径中，`_prepare_*_work` 只在 `enable_prefill_decode_overlap=True`
时创建 TimingEvent。

但当一个 step 只有 prefill 或只有 decode 时，会走 serial launch。
serial launch 不记录 event，`_finish_*` 却调用：

```python
started_event.elapsed_time(finished_event)
```

于是 GPU 测试报：

```text
RuntimeError: Both events must be recorded before calculating elapsed time.
```

修复：

- serial launch 前将 event 设为 None；
- `_work_elapsed()` 根据 event 是否存在选择事件计时或 monotonic timer。

### 7.4 弯路：把“同周期调度”误认为“物理 overlap”

旧代码里有：

```python
if decode_worked and any(prefill state):
    metrics.inc("prefill_decode_overlap_steps")
```

严格来说，这只是一个 scheduling-cycle 指标，不是 CUDA stream overlap。
直接把它改名为 stream overlap 会导致：

- CPU 测试看起来通过了；
- 但 GPU 上其实没有并发；
- 文章会写出虚假结论。

最终拆成两个指标：

```text
prefill_decode_overlap_steps
    同周期有 prefill + decode 工作
prefill_decode_stream_overlap_steps
    真正在两个 CUDA stream 上 launch
```

### 7.5 弯路：CPU fallback 需要保留原有测试语义

现有 CPU 测试断言：

```python
scheduler.metrics.get("prefill_decode_overlap_steps") > 0
```

如果改成只有 CUDA stream 才增加这个指标，CPU 回归会全部失败；
但如果不改，又无法区分真实并发。

最终方案：

- CPU 串行 fallback 继续增加逻辑 overlap metric；
- 只有 CUDA 双 stream 才增加 `prefill_decode_stream_overlap_steps`；
- GPU 测试断言两者；
- 文档明确两者语义。

### 7.6 弯路：W3 的 CPU fallback 一开始仍然会碰 CuPy

W3 的 `load_from_packed()` 原来无条件：

```python
self._bind_cupy()
```

在 CPU 环境会直接抛 `CuPy is unavailable`。

修复：

```python
if cp is not None and self.packed.is_cuda:
    self._bind_cupy()
```

这个错误不影响 GPU，但说明所谓 fallback 必须从权重加载阶段就完整可用。

### 7.7 弯路：tiled GEMM 不是所有 M 都更快

一开始很自然地把 M>1 全部交给 tiled GEMM。
测量后发现小 batch decode 时 tiled 的 32 行 tile 大量空转。

最终改成：

```text
M=1            单行 GEMV
M=2..8         W4: batch-tile GEMV / W3: grid.y=batch GEMV
M>8            tiled GEMM
```

这不是为了“看起来优雅”，而是为了符合 GPU 上真实的最优路径。

### 7.8 弯路：只和 dense cuBLAS 比较会得到错误收益

如果 benchmark 只测：

```text
tiled W4 GEMM
vs
dense FP16 cuBLAS
```

会严重低估旧 fallback 的成本，因为旧路径真正的成本是 dequantize +
临时权重分配。

最终始终比较：

```text
native tiled GEMM
vs
同一模块的 _forward_dequant_cublas()
```

这样得到的 2.24×–2.87× 才是替换后线性层的真实收益。

### 7.9 弯路：以为 overlap 会显著提升 throughput

双 stream 实现完成后，第一轮 benchmark 显示 c4 吞吐几乎不变：

- c4 no-overlap: 22.604 tokens/s；
- c4 overlap: 22.779 tokens/s。

一开始怀疑 stream 没有真正并行，于是：

1. 检查每个 RawKernel 都使用 `current_stream`；
2. 检查 `torch.cuda.stream` context；
3. 检查 `wait_stream` 依赖；
4. 增加 actual overlap metrics；
5. 用长 prompt chunk 128 测 max ITL。

最终确认：

- launch 确实在两个 stream；
- overlap metrics >0；
- 但 4 GB 卡上 prefill/decode 都受同一内存带宽限制；
- 所以互斥资源没有变多，overlap 不是“免费提吞吐”。

这个结论比“看代码觉得应该快”更重要。

### 7.10 弯路：单次 TPOT 数字波动很大

在 chunk 32 的第一次实验中，overlap 的 max ITL 甚至比 no-overlap 更差。
重新跑后 c2 又显示 overlap 明显更好。

原因：

- 4 GB WDDM 显存压力；
- 短生成只有 12 个 token；
- 单个 prefill chunk 的持续时间接近 decode 间隔；
- 系统调度、页面迁移、GPU 频率波动会放大单次结果。

所以这次文章使用多组指标：

- aggregate throughput；
- TTFT p50/p95；
- TPOT p50/p95；
- inter-token jitter；
- actual overlap steps；
- 原始 JSON。

并且明确标注：单次 smoke 不等于生产 SLA。

### 7.11 弯路：Benchmark 必须把 overlap 是否开启写进结果

否则很容易出现：

- 一次 run 默认 overlap 打开；
- 另一次 run 代码变更后默认关闭；
- 两次 JSON 看起来字段一样，实际调度路径不同。

现在：

- `bench_paged_scheduler.py` 支持 `--no-prefill-decode-overlap`；
- JSON 记录 `prefill_decode_overlap`；
- 单次 run 记录 logical/stream overlap step 增量；
- server status 也暴露 overlap 状态。

---

## 8. 当前限制

必须继续明确：

- 不是完整 DeepInfer engine；
- 不是单次 forward 内的 prefill + decode mixed batch；
- 每个 scheduler cycle 仍只推进一个 prefill chunk；
- stream overlap 在 4 GB 带宽受限卡上收益有限；
- prefill 仍然有 Python/CuPy launch 开销；
- tiled GEMM 目前是 CUDA core + FP32 累加，不是 Tensor Core；
- W3/W4 tiled kernel 没有做 cp.async / software pipeline；
- decode batch CUDA Graph 尚未实现；
- 长上下文仍受 4 GB 物理显存和激活峰值限制；
- 7B mixed 路径能跑 paged smoke，但 4 GB 下性能不稳定，不应作为生产吞吐结论；
- 输出质量仍取决于 W4/W3 量化，不是无损。

---

## 9. 下一步

优先级建议：

1. decode 高优先级 stream + prefill 低优先级 stream；
2. decode 活跃时动态缩小 prefill chunk；
3. time-sliced prefill（按毫秒预算而不是固定 token 数）；
4. tiled GEMM Tensor Core 化：`mma.sync.aligned.m16n8k16`；
5. cp.async + double buffering；
6. 单次 forward 的 prefill/decode mixed batch；
7. dynamic batch CUDA Graph；
8. 长文档插入 SLA：max ITL、p95 TPOT、TTFT 同时约束；
9. 7B 4 GB 的 paged 稳定性 sweep；
10. 把 stream overlap、chunk size、priority 做成自适应策略。

---

## 10. 复现

### 10.1 CPU 全套

```powershell
cd D:\DeepInfer-SM86
D:\QwenNativeTest-venv\Scripts\python.exe -m pytest -q tests
```

### 10.2 Tiled GEMM 微基准

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_sm86_tiled_prefill.py `
  --iterations 20 `
  --output docs\results\qwen_sm86_tiled_prefill.json
```

### 10.3 3B paged 并发 + overlap

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-new-tokens 16 `
  --max-cache 128 `
  --prefill-chunk-tokens 32 `
  --concurrency 1,2,4 `
  --reserve-memory-mb 384 `
  --no-prefix-cache `
  --output docs\results\paged_concurrency_3b_tiled_overlap.json
```

关闭 overlap 的对照：

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_paged_scheduler.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-new-tokens 16 `
  --max-cache 128 `
  --prefill-chunk-tokens 32 `
  --concurrency 1,2,4 `
  --reserve-memory-mb 384 `
  --no-prefix-cache `
  --no-prefill-decode-overlap `
  --output docs\results\paged_concurrency_3b_tiled_no_overlap.json
```

### 10.4 长文档插入 ITL

```powershell
F:\QwenQLoRA\venv\Scripts\python.exe `
  scripts\bench_tpot_jitter.py `
  --checkpoint D:\Qwen3B-W4A16-SM86 `
  --backend w4a16 `
  --max-cache 256 `
  --max-new-tokens 12 `
  --chunk-sizes 128 `
  --long-repeat 8 `
  --output docs\results\tpot_jitter_3b_chunk128_overlap.json
```

---

## 11. 文件地图

| 文件 | 作用 |
|---|---|
| `deepinfer/model_executor/models/qwen2_w4a16.py` | W4 tiled GEMM、batched GEMV、paged GQA |
| `deepinfer/model_executor/models/qwen2_mixed_sm86.py` | W3 tiled GEMM、W3 batch-tile 实验内核、混合 7B |
| `apps/paged_scheduler.py` | prefill/decode work 拆分、双 stream overlap、fallback、metrics |
| `deepinfer/engine/metrics.py` | logical / actual stream overlap 指标 |
| `apps/server.py` | `--no-prefill-decode-overlap` 开关 |
| `tests/models/test_qwen2_w4a16_gpu.py` | W4/W3 tiled 与 batched kernel 正确性 |
| `tests/apps/test_paged_scheduler_gpu.py` | 真实双 stream overlap 测试 |
| `tests/apps/test_paged_scheduler.py` | CPU 串行 fallback 与调度回归 |
| `scripts/bench_sm86_tiled_prefill.py` | tiled GEMM 微基准 |
| `scripts/bench_paged_scheduler.py` | 并发 sweep + overlap 开关 |
| `scripts/bench_tpot_jitter.py` | 长 prompt 插入 TPOT/ITL |
| `docs/results/qwen_sm86_tiled_prefill.json` | tiled GEMM 原始数据 |
| `docs/results/paged_concurrency_3b_tiled_overlap.json` | 新并发 + overlap 原始数据 |
| `docs/results/paged_concurrency_3b_tiled_no_overlap.json` | overlap 关闭对照 |
| `docs/results/tpot_jitter_3b_chunk128_overlap.json` | 长文档插入 overlap |
| `docs/results/tpot_jitter_3b_chunk128_no_overlap.json` | 长文档插入对照 |

---

## 12. 一句话总结

这次把 DeepInfer-SM86 从：

```text
串行 decode → 串行 prefill chunk
prefill 走 dequantize + cuBLAS
```

推进到了：

```text
paged scheduler + 双 CUDA stream prefill/decode overlap
+ tiled W4/W3 prefill GEMM
+ 真实 overlap 指标与可复现 benchmark
```

但同样要说清楚：

> 在 4 GB 带宽受限的 RTX 3050 上，stream overlap 是正确性和延迟结构上的进步，
> 不是吞吐翻倍按钮；这次真正的吞吐收益主要来自 tiled prefill GEMM。

后续：`docs/ARTICLE_8_KV_FP8_AND_CPU_OFFLOAD.md`，记录 KV FP8 native decode 与 CPU cold-block offload。
