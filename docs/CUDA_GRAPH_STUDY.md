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
