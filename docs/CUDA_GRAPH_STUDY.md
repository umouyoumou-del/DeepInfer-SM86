# 让 decode 可被 CUDA Graph 捕获：现状与改造方案

> 状态：**调研完成，未实施**
> 触发点：`torch.profiler` 显示 pin-all 之后 GPU 只忙 29.8%，
> 墙钟 318 ms/token 里 109 ms 是 CPU op、95 ms 是 kernel、约 114 ms 是空档。

---

## 1. 一个有利的发现：decode 的形状已经是固定的

`_fused_expert_chunk` 的入参形状由三件事决定：

| 量 | 值 | 每 token 是否变化 |
|---|---|---|
| `n_experts`（chunk 内专家数） | 6（= top_k） | **否** |
| `max_count`（每专家最大 token 数） | 1（decode 只有 1 个 token） | **否** |
| `counts` | `[1,1,1,1,1,1]` | **否** |
| 各槽位指向哪个专家的指针 | 动态 | **是** |

原因：decode 时 `token_count == 1`，routing 恰好产生 `top_k` 个不同专家、
每个 count 为 1；而 `expert_chunk_size` 默认 8 ≥ 6，所以
**每层只有一个 chunk，且形状恒定**。

`fused_gate_up_gemm_ptr_grouped` / `fused_down_gemm_ptr_grouped`
本来就接受**指针数组**作为输入（这正是 pointer-grouped 的含义），
所以"动态路由"并没有破坏形状——它只改变指针数组的内容。

**结论：不需要"固定形状 + 掩码"那套改造，形状本来就固定。**

---

## 2. 真正阻塞捕获的是什么

按阻碍程度排列：

### 2.1 指针数组每步重新分配

```python
pointers = tuple(
    torch.tensor([entry[projection].data_ptr() for entry in resolved],
                 dtype=torch.int64, device=self.device)
    for projection in range(3)
)
```

每 token 每层新建 3 个小 GPU 张量。Graph replay 要求缓冲区地址固定，
所以必须改成**预分配的固定 buffer + 原地写入**：

```python
self._ptr_buffers[layer][projection][slot] = ptr   # in-place, 不重新分配
```

### 2.2 ECS 热度更新在 Python 里逐 token 执行

```python
if self.ecs_controller is not None:
    self.ecs_controller.observe_routes(...)   # 每个 token、每层都调
```

这是纯 Python + 数据结构更新，**无法被 CUDA Graph 捕获**。
必须移到 graph 之外（例如每个请求结束时批量更新），或在该模式下关闭。

### 2.3 控制流依赖张量的值

`_fused_expert_chunk_on_device` 里有多处 `if`：
`gpu_cache_enabled()`、`target == self.device`、cpu/gpu expert 集合分支。
这些本身是 host 侧布尔值，只要在捕获期间取值恒定就没问题——
但**必须保证捕获期与 replay 期的分支完全一致**。

### 2.4 attention 路径

MLA 的 `DeepSeekPagedCache` 会按 `seq_lens` 动态决定 mask 与 position，
其中 `past_length = int(seq_lens.max().item())` 是**一次 D2H 同步**。
Graph 要求 `total_length` 恒定，因此需要：
- 固定 `max_cache`，把 `past_length` 作为常量参数传入，或
- 为每个 `past_length` 分桶捕获（实践中一般按 block 分桶）

### 2.5 采样与循环

`generate()` 的 for 循环每步都做 `next_token = sample(logits)`（含 `.item()`），
这决定了下一步的输入——**不能放进 graph**。
标准做法：graph 覆盖"一次完整 forward"，采样留在外面。

---

## 3. 改造清单

| # | 改动 | 影响面 | 估计 |
|---|---|---|---|
| 1 | 指针数组改为预分配 buffer + 原地写 | `load_pointer_experts` | 0.5 天 |
| 2 | ECS 热度更新移出 decode 循环 | dsv2 runtime | 0.5 天 |
| 3 | `past_length` 固定或分桶 | attention 路径 | 1 天 |
| 4 | 捕获一层 → 验证数值一致性 | 新增 capture 工具 | 1 天 |
| 5 | 捕获整个 decode step + replay | 新增 | 1 天 |
| 6 | 回退开关与测试 | | 0.5 天 |
| | **合计** | | **4.5 天** |

---

## 4. 预期收益

按 §5.7 第五步的分解（pin-all 下）：

```
墙钟 318 ms/token = CPU 109 + GPU 95 + 空档 114
```

Graph 只能消掉**空档**，不能消掉 CPU op 时间（路由与采样仍在 Python）。

| 情形 | 墙钟 | 倍数 |
|---|---:|---:|
| 现在 | 318 ms | 1× |
| 空档全消 | 110–130 ms | 2.4–2.9× |
| 现实（部分消除） | 160–210 ms | 1.5–2× |

**要突破 2×，必须同时削减每 token 的 Python 工作量**——
而那需要把路由、分块、gather/scatter 下沉到 C++/CUDA，
是比 CUDA Graph 大得多的工程。

---

## 5. 建议

**先做 1 和 2**（各半天），它们不需要 CUDA Graph 就能生效：

- 1 让指针数组不再每 token 重新分配（减少 allocator 压力与 Python 开销）
- 2 让 ECS 热度更新从每 token 每层降到每请求一次（直接减少 CPU op 时间）

做完这两项再复测 profile，如果 GPU 利用率仍低于 50%，再投入 3–6 的 Graph 工作。
**在没测过 1、2 的收益之前就做 Graph，等于又一次不分解就归因。**

---

## 6. 逐项 profile：那 109 ms 到底是什么

`torch.profiler` 逐项排序（pin-all，12 token）：

```
wall=2.58s  215.0 ms/token   cpu=0.901s   cuda=1.147s
```

### CPU 侧（self time）

| 操作 | 总时长 | 调用次数 | 每 token |
|---|---:|---:|---:|
| **cudaLaunchKernel** | 186.8 ms | **38,865** | 15.6 ms |
| **cudaStreamSynchronize** | 97.7 ms | **3,039** | 8.1 ms |
| aten::mul | 51.0 ms | 6,608 | 4.3 ms |
| aten::mm | 49.3 ms | 2,591 | 4.1 ms |
| aten::copy_ | 48.3 ms | 11,147 | 4.0 ms |
| aten::empty | 38.5 ms | 11,183 | 3.2 ms |

### CUDA 侧

| 操作 | 总时长 | 调用次数 | 每 token |
|---|---:|---:|---:|
| aten::_to_copy | 193.7 ms | 7,369 | 16.1 ms |
| **Memcpy HtoD (Pinned→Device)** | 184.1 ms | **仅 174** | **15.3 ms** |
| dp_ptr_grouped_gate_up_gemm | 39.0 ms | 286 | 3.2 ms |

**总 op 数 271,152 / 12 token = 每 token 22,596 个操作。**

### 三个真问题

**1. 每 token 3,239 次 kernel launch、253 次同步。**

这是 Python 逐层编排的直接后果。launch 的 CPU 开销 15.6 ms/token、
同步 8.1 ms/token，合计约 24 ms/token——**CUDA Graph 正好针对这一项**。

**2. H2D 拷贝在 pin-all 下依然存在（15.3 ms/token）。**

只有 174 次调用（每 token 约 14.5 次），单次约 1 MB。说明**仍有约 14 MB/token
的 pinned→device 拷贝没有被驻留覆盖**。来源未定位：可能是 shared expert、
attention 权重转换，或 `_to_copy` 引起的 dtype 转换。**这是下一步该查的。**

**3. `aten::empty` 11,183 次 / 12 token ≈ 932 次/token。**

张量分配次数极高，但上一轮实测"预分配指针 buffer"并没有收益，
说明分配本身便宜，**贵的是 launch 与同步**。

### 修正后的优先级

| 手段 | 针对 | 量级 |
|---|---|---|
| **CUDA Graph** | 3,239 launch + 253 sync | 约 24 ms/token 的 CPU 开销 + 空档 |
| 定位并消除残留 H2D | 15.3 ms/token | 15.3 ms/token |
| 减少 `_to_copy`（7,369 次） | 16.1 ms/token CUDA | 未量化 |

注意：本次 wall 为 215 ms/token，而前两次测量分别是 318 与 353 ms/token——
**同机波动接近 65%**，所以任何小于 30% 的改动都无法用单次测量判定。
后续实验必须多次重复取中位数。

---

## 7. 五次重复的中位数：pin-all 其实是 2 倍提速

§6 发现同机波动 65%，于是把基准改成**5 次重复取中位数**
（`scripts/bench_decode_repeats.py`，含 1 次预热）。

| 配置 | 逐次 ms/token | **中位数** | tok/s |
|---|---|---:|---:|
| 默认 | 243.7 / 170.4 / 158.2 / 154.9 / 154.6 | **158.2** | 6.32 |
| **pin-all** | 104.6 / 79.7 / 66.6 / 67.0 / 87.9 | **79.7** | **12.54** |

**pin-all 是 2.0 倍提速（中位数），稳态是 2.3 倍（154.6 → 67.0）。**

### 这推翻了 §5.7 第三步的结论

论文 §5.7 写"驻留之后吞吐没有变化（4.68 vs 5.41 tok/s）"——
那是**单次短跑**，且**没有预热**。热态后：

```
3050（NVMe 流式）      2.45 tok/s
4080S 默认（流式）      6.32 tok/s
4080S + pin-all（驻留） 12.54 tok/s   ← 比 3050 快 5.1 倍
```

### 方法论教训（应写进 §7）

1. **预热必须单独丢弃**：第 1 次比稳态慢 57%（243.7 vs 154.6），
   单次测量会把预热成本算进结果；
2. **至少 5 次取中位数**：同配置极差可达 56%；
3. 今天有**三个结论**（驻留无效、分配是瓶颈、94% 主机开销）
   都是被单次未预热的测量误导的。

### 对 CUDA Graph 的影响

pin-all 稳态 67.0 ms/token。§6 测得的 launch + 同步开销是
24 ms/token（占 36%），这是 Graph 能直接吃掉的量。

所以 Graph 的合理预期是 **67 → 约 45–55 ms/token，即 1.2–1.5 倍**，
而不是更早估计的 2–3 倍——**因为 pin-all 已经吃掉了拷贝与相当一部分空档。**

---

## 8. 探针实验：空的图，以及修好之后的真实数字

`scripts/probe_cuda_graph.py` 捕获单个固定形状的 fused expert chunk
（6 专家，max_count=1，gate/up Q4_K + down Q8_0），对比 eager 与 graph replay。

### 第一次：捕获到空图（结果作废）

```
eager 312.4 us   graph 29.5 us   "10.59x"
```

PyTorch 警告 `The CUDA Graph is empty`。原因是探针**直接调用**
`fused_*_ptr_grouped`，而 kernel 是 `cp.RawKernel`，不指定 stream 时
落在 **CuPy 的 null stream** 上；`torch.cuda.graph()` 只记录 torch stream 上的操作。
29.5 µs 是 replay 空图的开销。

**运行时本身是对的**——它用 `cp.cuda.ExternalStream(stream.cuda_stream)`
把 kernel 绑到 torch stream，只有探针漏了这层。

### 第二次：补上 stream 绑定

```python
with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
    fused_gate_up_gemm_ptr_grouped(...)
```

```
eager   gate_up 213.4 us   down 140.0 us   sum 353.4 us
graph   gate_up+down replayed together:   198.4 us
per-chunk speedup: 1.78x
x27 layers: 9.5 -> 5.4 ms/token
```

**1.78 倍，无警告，是真图。**

### 但这个 1.78 倍换算到端到端只有约 6%

```
MoE chunk 每 token   9.5 ms（eager） → 5.4 ms（graph）   省 4.1 ms
pin-all 稳态每 token 79.7 ms（中位数）
```

省 4.1 / 79.7 ≈ **5%**。

原因是 §6 的 3,239 次 launch 里，**大部分不在 MoE chunk 内**——
attention、elementwise、`_to_copy` 都在别处。只对 MoE 做 Graph 收益有限。

### 端到端若能全捕获

§6 测得 launch 15.6 ms + 同步 8.1 ms = **24 ms/token**（占 79.7 的 30%）。
若全部消除：79.7 → 约 56 ms/token，即 **1.4 倍**。

这与 §7 修正后的估计（1.2–1.5×）一致。

### 结论

| 层 | eager | graph | 倍数 | 端到端贡献 |
|---|---:|---:|---:|---:|
| 单个 expert chunk | 353 µs | 198 µs | **1.78×** | ~5% |
| 端到端（若全捕获） | 79.7 ms | ~56 ms | 1.4× | 30% |

**Graph 可做，但天花板是 1.4 倍，不是数量级。**
真正想要数量级，必须先解决 §6 里那 3,239 次 launch 的**来源**——
把逐层 Python 编排整体下沉，而不是给现有编排套一层图。

---

## 9. 3,239 次 launch 的来源（按调用次数排序）

pin-all，6 token，总计 **130,661 个 op = 每 token 21,777 个**：

| 操作 | 次数 | 每 token | CPU | CUDA |
|---|---:|---:|---:|---:|
| **cudaLaunchKernel** | 17,967 | **2,994** | 156.5 ms | — |
| aten::as_strided | 11,581 | 1,930 | 9.6 ms | — |
| aten::copy_ | 5,375 | 896 | 42.1 ms | 187.7 ms |
| aten::empty | 5,321 | 887 | 30.7 ms | — |
| aten::slice | 5,302 | 884 | 8.9 ms | — |
| aten::to | 5,055 | 842 | 6.8 ms | 185.3 ms |
| aten::view | 4,716 | 786 | 4.9 ms | — |
| aten::_to_copy | 3,511 | 585 | 15.8 ms | 185.3 ms |
| aten::mul | 2,888 | 481 | 38.0 ms | 3.5 ms |

### 结论：这是**未融合的 eager PyTorch**

27 层 / token，却要发起约 **2,994 次 kernel launch**——平均**每层 111 次**。

原因很清楚：代码用的是逐算子的 eager PyTorch，
没有把 elementwise 与 reduce 融合成少数几个 kernel。
`as_strided`/`slice`/`view`/`reshape` 这类视图操作本身不发 kernel，
但每个真正计算的算子都发一个。

按每次 launch 约 8 µs 计：

```
2,994 x 8 us = 24 ms/token     （中位数 79.7 ms 的 30%）
```

与 §6 实测的 "launch 15.6 + sync 8.1 = 24 ms" 完全吻合。

### 这决定了优化的真实上限

| 手段 | 能消掉的 | 端到端 |
|---|---|---:|
| CUDA Graph 全捕获 | 2,994 次 launch 的开销（约 24 ms） | 79.7 → ~56 ms（**1.4×**） |
| 手工融合 elementwise | 算子数量本身 | 需要重写计算 kernel |
| 把逐层循环下沉到 C++/CUDA | Python 侧 op 调度 | 与上一条同量级 |

**注意 `aten::copy_` 仍有 896 次/token、187.7 ms CUDA 时间。**
pin-all 之后专家权重已经常驻，这些拷贝是**别的东西**——
候选：shared expert、attention 的 KV 写入、`_to_copy` 的 dtype 转换。
**这一项尚未定位，但它在 CUDA 侧占比很大，可能比 launch 更值得查。**

### 建议的下一步顺序

1. **先查清 896 次 copy / 187.7 ms 是什么**——它比 launch 开销更值钱，
   而且可能是一个简单的冗余（比如每步重复转换同一批权重）；
2. 再决定是否投入 CUDA Graph（天花板 1.4×）；
3. 想要数量级，只能走算子融合或把整个 forward 写成少数几个大 kernel——
   那是与"在 DeepInfer 之上做扩展"不同的工程量级。
