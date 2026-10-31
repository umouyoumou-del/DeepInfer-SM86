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

---

## 10. 那 896 次 copy 是谁：pin-all 挡不住的 ~500 MB/token

用 `record_shapes=True` 按输入形状分组（pin-all，4 token）：

| 操作 | 次数 | 每 token | CUDA | 形状（字节） |
|---|---:|---:|---:|---|
| **Memcpy HtoD (Pinned→Device)** | 174 | **44** | **180.8 ms** | — |
| `_to_copy` + `copy_` | 84 | 21 | 84.0 ms | **12,976,128 = 12.4 MB** |
| `_to_copy` + `copy_` | 19 | 5 | 36.0 ms | **24,510,464 = 23.4 MB** |
| `_to_copy` + `copy_` | 23 | 6 | 28.4 ms | **15,859,712 = 15.1 MB** |
| `_to_copy` + `copy_` | 12 | 3 | 10.5 ms | 11,354,112 = 10.8 MB |

### 算术

```
每 token 拷贝量 ≈ 21x12.4 + 5x23.4 + 6x15.1 + 3x10.8 ≈ 500 MB
每 token 拷贝耗时 ≈ 37 ms（4 次运行共 148 ms）
→ 有效带宽 ≈ 500 MB / 37 ms ≈ 13.5 GB/s
```

**13.5 GB/s 是 PCIe 的速度，不是 D2D 的速度（后者应在 TB/s 级）。**
配合 174 次 `Memcpy HtoD (Pinned→Device)`，结论明确：

> **即使专家已全部驻留显存，每 token 仍有约 500 MB 的权重在从主机重新拷进显存。**

### 候选对象

单个 MoE 层的 **shared expert**（`ffn_gate_shexp` / `up` / `down`，各 2816×2048 Q6_K）
合计约 12–14 MB，与观测到的 12.4 MB / 15.1 MB 尺寸吻合；
每 token 约 21 次也接近 26 个 MoE 层的数量级。

也就是说：**shared expert 的权重没有被缓存到显存，而是每次 forward 重新 H2D 一次。**

### 为什么这比 CUDA Graph 更值得做

| 手段 | 收益 | 成本 |
|---|---|---|
| CUDA Graph 全捕获 | 约 24 ms/token（1.4×） | 4.5 天 |
| **消除这 ~500 MB/token 的重复 H2D** | **约 37 ms/token，且可能更多**（移除后 PCIe 不再参与稳态） | 取决于定位，预计 0.5–1 天 |

而且两者的收益可以叠加。

### 下一步（未做）

1. 在 `_embed_tokens` / `shared_experts` 路径上确认哪些权重是 mmap 直读而非常驻；
2. 把 shared expert（及 attention）权重在 load 时一次性 `to(device)` 并保留；
3. 用 `bench_decode_repeats.py` 复测（5 次中位数），确认 ≥30% 的变化才可信。

**这一项是目前唯一还没被证伪、且量级与 launch 开销相当的方向。**

---

## 11. 更正：shared expert 的猜测不成立

§10 推测那 ~500 MB/token 的 H2D 来自 shared expert 权重被重复上传。
**这个推测是错的。**

运行日志里已经写明了落位：

```
loaded: GPU free 29698.6 MiB, shared=cuda, lm_head=cuda, fused_moe=True
```

`_load_weights` 在加载阶段就把 shared expert 放到了 CUDA
（`_choose_optional_device(..., required_mib=1150)` 通过），
所以它们**本来就在显存里**，每 token 不会被重新上传。

因此 12.4 MB × 21 次/token 的来源**仍未定位**。

### 已知的排除项

| 候选 | 状态 |
|---|---|
| 专家权重（`ffn_*_exps`） | 已排除——pin-all 驻留 1664 个 |
| shared expert（`ffn_*_shexp`） | **已排除——日志显示 shared=cuda** |
| lm_head | 已排除——日志显示 lm_head=cuda |
| attention 权重 | 尚未核对；单层约 7.7 MB，与 12.4 MB 不符 |
| KV cache 写入 | 量级不符（54 KB/token） |

### 下一步只剩机械核对

不要再猜。可行做法：

1. 用 `with_stack=True` 抓 `_to_copy` 的 **Python 调用栈**，直接看到是哪个函数；
2. 或给 `GGUFIndex` 的 tensor 读取加一次性日志，看重放期间谁还在读文件。

**这是唯一能终结猜测的手段。** 今天已经有五次归因被后续测量推翻
（缓存、驻留、94%、分配、shared expert），共同点都是**没有直接观测就下结论**。

### 今天的结论沉淀

```
optimization_order.md 的核心教训：

1. 先确认测量本身可信（预热、重复、分母、工具是否静默失败）
2. 再确认归因有直接观测支撑，而不是从总量倒推
3. 最后才改代码
```

违反第 1 条，得到的是噪声；
违反第 2 条，得到的是错误方向——而错误方向今天出现了五次。

---

## 12. 更正：那 2.2 GB 是 **prefill**，不是 decode

§10 与 §11 把 H2D 量按生成的 token 数相除，得出"每 token 743 MB"。
**这个归一化是错的。**

用 monkeypatch 拦截 `torch.Tensor.to`（CPU→CUDA 且 >1 MB），
对同一个 prompt 只改生成长度：

| max_new_tokens | H2D 总量 | 调用次数 |
|---:|---:|---:|
| 4 | **2230.4 MB** | 174 |
| 32 | **2230.4 MB** | 174 |

**与生成长度完全无关** → 这 2.2 GB **全部发生在 prefill**，decode 阶段是 **0**。

### 结论：pin-all 在 decode 上完全成功

- prefill 期间缓存被刻意停用（§5.7 第二步），走 `load_packed_experts`，
  每个请求把约 2.2 GB 专家权重搬一次——**这是一次性成本，不是每 token**；
- decode 期间 0 MB 搬运，说明驻留**彻底生效**。

§10 写的"每 token 743 MB 走 PCIe"和 §11 的"仍未定位"**均作废**。
12,976,128 字节那些 copy 也来自 prefill。

### 这解释了 2× 提速的来源

之前 pin-all 的 2×（158.2 → 79.7 ms/token）并非"显存压力缓解"，
而是**decode 真的不再搬运**。只是因为基准里 prefill 被摊到 12 个 token 上，
所以看起来提升只有 2×；生成长度越长，提升会越接近比值本身。

### 修正后的成本结构

| 阶段 | 成本 | 性质 |
|---|---|---|
| prefill | 约 2.2 GB H2D ≈ 165 ms | 每请求一次 |
| **decode** | **0 MB 搬运**，79.7 ms/token 中主要是计算与 launch | 每 token |

**因此下一阶段的优化重点应该转向 prefill**：
每个请求一次性搬 2.2 GB，对短请求（如 12 token）占总时间的一半以上。
把 prefill 也改成走常驻缓存（而不是停用），是收益最大的一步。

### 方法论补充（第六次）

**归一化前必须确认被归一化的量是否随分母变化。**
本节与 §10 的错误都是"用生成 token 数去除一个与生成无关的量"。
一个 30 秒的实验（改 `max_new_tokens` 对比总量）就能证伪——
而这个实验在写下 §10 之前就该做。

---

## 13. 让 prefill 走常驻缓存：中位数 79.7 → 47.7 ms/token

§12 指出 prefill 每请求仍要搬约 2.2 GB。修改只有一处：
pin-all 模式下**不再在 prefill 前停用缓存**。

```python
if not self.pin_all_experts:
    expert_store.clear_gpu_cache()
    expert_store.set_gpu_cache_active(False)   # 仅默认路径
# pin-all: 缓存保持启用，prefill 也走 load_pointer_experts
```

### 实测（`bench_decode_repeats.py`，5 次取中位数，12 token）

| 配置 | 逐次 ms/token | 中位数 | tok/s |
|---|---|---:|---:|
| 默认 | 171.9 / 168.3 / 165.4 / 167.7（+2 次） | 168.3 | 5.94 |
| **pin-all（改前，§7）** | 104.6 / 79.7 / 66.6 / 67.0 / 87.9 | 79.7 | 12.54 |
| **pin-all（改后）** | 89.8 / 79.1 / **47.7 / 47.4 / 47.7** | **47.7** | **20.96** |

**提升 1.67 倍**（79.7 → 47.7）。稳态（第 3–5 次）稳定在 **47.4–47.7**，
极差仅 0.3 ms；前两次是预热，说明预热成本依然显著。

### 累计进展

| 配置 | tok/s | 相对 3050 |
|---|---:|---:|
| RTX 3050 4 GB（NVMe 流式） | 2.45 | 1× |
| 4080 SUPER 默认（流式） | 5.94 | 2.4× |
| 4080 SUPER + pin-all | 12.54 | 5.1× |
| **4080 SUPER + pin-all + prefill 走缓存** | **20.96** | **8.6×** |

### 为什么这次生效

prefill 阶段原本要处理的专家数远超 decode（一个 prompt 的每个 token 都要过 top-6，
大量不同专家被触及），因此 staging 量最大。专家已全部驻留时，
这条路径只剩"取指针"，那 2.2 GB 自然消失。

### 仍需注意

- 默认路径不受影响（`pin_all_experts=False` 时行为与之前完全一致）；
- 默认路径本次测得 168.3，高于 §7 的 158.2，属同机波动（本次极差仅 5.9%，更稳定）；
- 预热成本仍大（第 1 次 89.8 vs 稳态 47.7），后续基准应丢弃前两次。

---

## 14. pin-all 改为自动：零参数下 3.2 倍

§13 的收益藏在 `--pin-all-experts` 后面。现在改为**自动探测**：
加载后计算全部专家所需字节，若能在**剩余显存的 70%** 之内，就自动配置缓存预算并预热。

### 实测（`bench_decode_repeats.py`，5 次，12 token，无任何 flag）

| 配置 | 逐次 ms/token | 中位数 | tok/s |
|---|---|---:|---:|
| 默认（自动未生效前） | 175.4 / 178.3 / 173.9 / 178.0 / 173.4 | 175.4 | 5.70 |
| **自动 pin-all** | 74.9 / 75.0 / **54.5 / 47.6 / 48.0** | **54.5** | **18.35** |

**3.2 倍**，且稳态（第 4–5 次）是 47.6 / 48.0，与手动开启时的 47.7 一致。

### 实现要点

```
required = Σ(bytes_per_expert × expert_count)  over all MoE layers
free     = cuda.mem_get_info()[0]
if required <= free * 0.7:  启用缓存预算 + preload
```

启用的判定值实测：`required=9086 MiB, free=29699 MiB, ratio=0.31` → 通过。

### 两个实现上的坑（都已修）

1. `self.pin_all_experts = bool(pin_all_experts)` 把 `None`（auto）变成了 `False`，
   导致自动分支永不进入；必须保留 `None`。
2. 自动模式必须先设置 `_gpu_cache_limit_bytes` 与 `_gpu_cache_per_layer`，
   否则 `preload_gpu_experts()` 会因"缓存未配置"直接抛错。

### 副作用

- `pin_active` 是解析后的真实布尔值，`generate()` 的两个分支都改用它，
  避免"用户没传 flag 但专家其实已驻留"时走进默认（停用缓存）路径；
- `--pin-all-experts` 仍可强制开启（即使判定为放不下）；
- 未提供关闭开关——需要时传 `pin_all_experts=False`。

### 累计

| 配置 | tok/s | 相对 3050 |
|---|---:|---:|
| RTX 3050 4 GB（NVMe 流式） | 2.45 | 1× |
| 4080 SUPER 默认（改前） | 5.70 | 2.3× |
| **4080 SUPER 自动 pin-all（当前默认）** | **18.35**（稳态约 21） | **7.5×** |

---

## 15. 加载后自动预热：首请求 74.9 → 48.4 ms/token，极差 50% → 1.1%

§14 的自动 pin-all 仍有一个问题：**前两次调用明显更慢**（74.9 / 75.0 vs 稳态 47.6）。
JIT 编译与显存分配器增长的成本落在了第一个提问的人身上。

修法：`pin_active` 后执行一次极小的 `generate(prompt="hi", max_new_tokens=2)`
并 `torch.cuda.synchronize()`。失败被吞掉，绝不影响加载
（可用 `DEEPINFER_SKIP_WARMUP=1` 关闭）。

### 实测（5 次，12 token，无 flag）

| 版本 | 逐次 ms/token | 中位数 | 极差 |
|---|---|---:|---:|
| 无预热（§14） | 74.9 / 75.0 / 54.5 / 47.6 / 48.0 | 54.5 | **50.2%** |
| **有预热** | **48.4 / 48.9 / 48.5 / 49.0 / 48.5** | **48.5** | **1.1%** |

两个收益：

1. **首请求 74.9 → 48.4 ms/token**（快 35%），用户体验上"第一次就正常"；
2. **测量极差从 50.2% 降到 1.1%**——预热同时是一个测量问题，
   之前所有"单次测量不可信"的困扰有相当一部分就是预热造成的。

### 最终数字

| 配置 | tok/s | 相对 3050 |
|---|---:|---:|
| RTX 3050 4 GB（NVMe 流式） | 2.45 | 1× |
| 4080 SUPER（改前默认） | 5.70 | 2.3× |
| **4080 SUPER（当前默认：自动 pin-all + 预热）** | **20.61** | **8.4×** |

极差 1.1% 意味着：**此后小于 5% 的改动都可以用单次中位数判定**，
不再需要为噪声问题反复重测。

### 今天到此的完整改动

| 改动 | 文件 | 效果 |
|---|---|---|
| `drop_blocks` / `evict_policy` | kv_cache / scheduler | 基础设施 |
| `--pin-all-experts` | deepseek_v2_runtime | 驻留专家 |
| prefill 不禁用缓存（pin 时） | deepseek_v2_runtime | 79.7 → 47.7 |
| 自动探测显存 | deepseek_v2_runtime | 零参数 3.2× |
| 加载后预热 | deepseek_v2_runtime | 首请求 −35%，噪声 1.1% |
| `bench_decode_repeats.py` | scripts | 中位数基准 |
| `probe_cuda_graph.py` | scripts | Graph 可行性探针 |

---

## 16. 一个真实回归：自动探测在 4 GiB 卡上把速度砍了 3 倍

§14 的自动探测原始写法把"配置缓存预算"放在了容量检查**之前**：

```python
store._gpu_cache_limit_bytes = required   # 9086 MiB
store.set_gpu_cache_active(True)          # ← 此时还没检查放不放得下
if required > free * 0.7:
    return 0                              # 已经迟了
```

在 4 GiB 的 3050 上后果是：**缓存被激活并配了 9 GB 预算**，
于是每步都在显存里反复挤进挤出。

| 指标 | 改动前 | 回归 | 修复后 |
|---|---:|---:|---:|
| decode | 2.45 tok/s | **0.824 tok/s** | **2.542 tok/s** |
| 峰值显存 | 2253.7 MiB | **4329.5 MiB** | **2277.6 MiB** |

修复：把容量检查移到配置之前，并在跳过时打印原因：

```
auto pin-all skipped: experts need 9086.0 MiB, only 733.9 MiB of 1048.5 MiB free fits
```

### 教训

**"探测"和"生效"必须分离。** 探测阶段的任何副作用都会在最坏的情况下生效——
这里正是如此：判定为"放不下"，但配置已经写进去了。

而如果没有跑 3050 的回归验证，这个 bug 会随默认开启推到所有小显存用户身上，
而大显存上的 8.4 倍会让它看起来完全正常。

---

## 17. 修复后的最终状态与新的成本分布

### 两台机器都验证过

| 机器 | decode | 峰值显存 | 极差 |
|---|---:|---:|---:|
| RTX 3050 4 GB | 2.542 tok/s | 2277.6 MiB | — |
| **RTX 4080 SUPER 32 GB** | **20.92 tok/s（47.8 ms/token）** | — | **0.4%** |

3050 与改动前（2.45 / 2253.7 MiB）一致，确认回归修复无副作用。

**极差 0.4%** 意味着此后 5% 量级的改动都能用单次中位数判定。

### 当前的成本分布（profiler，含采样开销，只看比例）

实际 GPU 时间约 **34 ms/token**（0.41 s / 12 token），
而墙钟 47.8 ms → **GPU 利用率约 71%**（此前默认路径只有 29.8%）。

> **注意：`aten::matmul` / `aten::linear` / `aten::mm` 是相互包含的**
> （linear 内部调 mm，matmul 再分派到它们）。**不能相加**——
> 本节初稿把三者相加得到 135 ms / 33%，那是重复计数，已更正如下。

按不重叠的口径：

| 项 | 每 12 token | 占 CUDA |
|---|---:|---:|
| MoE `gate_up` + `down` grouped | 74.8 ms | **18%** |
| matmul 家族（三者取其一，约） | 约 46 ms | 11% |
| 其余（elementwise、norm、reduce 等） | 约 289 ms | **71%** |

调用次数 `344 / 12 = 28.7` 次/token ≈ 27 层 + 1，确认每层每 token 只发一次
`gate_up`，指针路径已经在用，没有多余 staging。

### 结论：优化重心已经转移

- **MoE grouped kernel 占 18%**——之前所有围绕专家搬运的优化已经做完；
- **matmul 家族约 11%**，不是最大的单项；
- **"其余"占 71%**——这是大量小算子（elementwise、norm、reduce）的集合，
  正是 §9 里 2,994 次 launch 的来源；
- **GPU 仍空闲 29%**（约 14 ms/token），是 launch/同步的残余。

因此下一步按收益排序：

1. **减少小算子数量与 launch 开销**——"其余 71%" 与 29% 空闲合起来是绝大多数时间，
   CUDA Graph 或算子融合都指向这里，上限约 1.4×；
2. MoE kernel 本身（18%）；
3. attention matmul（11%）。

**初稿把优先级排在 attention 上是错的**，原因就是那个重复计数。

**注意：本轮 profiler 的 255 ms/token 不能与基准的 47.8 ms 直接比较**，
profiler 会给 25 万个 op 加采样开销。比例可信，绝对值不可信。
