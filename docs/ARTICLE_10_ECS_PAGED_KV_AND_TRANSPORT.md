# 第十篇：把 ECS 推进到「二卡定版」——paged KV、前缀共享，以及三次押错方向

> 项目：DeepInfer-SM86（非官方扩展，Apache-2.0）
> 硬件：2× RTX 3080 Ti 12 GB（无 P2P）、Xeon Silver 4214R、440 GiB RAM、Ubuntu 22.04
> 模型：DeepSeek-V2-Lite-Chat Q4_K_M
> 基线：单远端 expert worker + 点对点 request/reply

这篇文章记录的是把 ECS 从「连续批处理能跑但吞吐低」推到「二卡定版」的这一轮。
里面有三处我押错了方向，其中一处还被我写进了文档当成结论。都原样记下来。

---

## 0. 起点：一个卡死的终端

上一轮会话的终端卡死了，只剩一份 8.4 MB 的 `session.v4.jsonl.zstd`。
解出来 10461 条记录，断点很干净：停在 turn 49，刚读完 `paged_adapter.py`，
正准备给 `ContinuousBatchEngine` 上 **paged KV slot allocator**。

于是这一轮从这个断点开始。

---

## 1. Paged KV slot allocator

### 动机

原来的 dense slot 引擎把整批活跃行放在一个 `DynamicCache` 里：

- 新请求进来，要把**所有活跃行重新打包**；新 prompt 更长时还要整体右移；
- 请求结束，要把剩下的行**再 gather 一遍**。

单测和 doc 都指向同一个猜测：这些 padding/selection 拷贝是 4.2 tok/s 平台的瓶颈。

### 实现

`deepinfer/moe/paged_continuous_batching.py`：

- 一个 `PagedKVCacheManager` 持有物理 KV 池；
- 每个请求持有自己的 `block_ids`，**按需逐块增长**（17-token prompt 不会占住 128 token 的容量）；
- decode 每步只重建一张 `[batch, max_blocks]` block table，交给 native paged MLA kernel
  直接从池子里读；
- 请求结束，`allocator.free` 归还，**行永远不搬**。

两种 prefill 策略：`batched`（左填充 dense 前向 + 一次性分组写回 paged）和
`paged`（直接走 `forward_logits`，不建 dense cache）。

### 数据（同机、同 prompt、配对比较）

| 负载 | dense | paged |
|---|---|---|
| 4 prompt × 17 tok | 5.127 s / 5.071 t/s | 5.133 s / 5.065 t/s |
| 8 prompt × 17 tok | 5.609 s / 9.093 t/s | 5.657 s / 9.016 t/s |
| 4 prompt × **44 tok** | 7.004 s / 3.855 t/s | **5.965 s / 4.526 t/s** |

### 第一次押错

**结论和猜测相反**：短 prompt 下两者没有差别，只有 44-token 才快 1.17×。

也就是说，之前文档里那句「cache padding/selection 主导了 4.2 tok/s 平台」
**在小规模上不成立**。拷贝 `[rows, heads, tokens, dim]` 这件事，只有在 `tokens`
变大时才是真活。

真正的差距在别处：admission 同步，以及**每个 scheduling iteration 只推进一步
decode**。

顺手修了一个真问题：写回是逐请求×逐层的，27 层 × 8 请求 = 216 次小 kernel launch。
改成**按等长 prompt 分组**批量写，一次 layer 一次写。b8 从 5.807 s 降到 5.657 s。

---

## 2. 前缀块共享

`PrefixCache` 本来就存在（block hash + retain/LRU），一直没接到 paged 引擎上。

### 实现

1. 把 prompt 按 block 切块，累积 hash；
2. `match` 命中最长前缀 → 这些 block 被 **retain**，直接挂到请求上；
3. 只对**剩余后缀**做前向，并且**从真实位置开始**（不是从 0 重算）；
4. 每次 admission 结束后，把完整 prompt block `register` 回 LRU。

两个安全细节，都不是可选的：

- **block 对齐的 prompt 绝不交还最后一块**。那一块还要写本行尾部，交出去就是把
  共享块写脏。所以 `start >= length` 时退回一个 block。
- 复用到的 block 必须**不可变**，写只能落在新分配的 block 上。

### 数据

4 个请求共享 89-token 前缀（5 个 16-token block 可复用）：

| 配置 | wall | aggregate | 每请求 prefill |
|---|---|---|---|
| paged，无前缀缓存 | 8.359 s | 1.555 t/s | 2.52 / 1.17 / 1.22 / 1.21 s |
| **paged + prefix cache** | **6.151 s** | **2.114 t/s** | 2.47 / **0.58 / 0.52 / 0.54** s |
| dense 引擎同 prompt | 6.299 s | 2.064 t/s | 2.52 / 1.42 / 1.42 / 1.42 s |

**wall −26%，单请求 prefill −53%**，追平 dense。第一个请求仍付全价，所以收益随
复用次数放大，不是单请求能拿到的。

限制说清楚：**只在 `prefill_mode="paged"` 生效**。batched dense prefill 从位置 0
写整行，没法从偏移开始。

---

## 3. Admission 合并：一个自己挖的 0.5 秒坑

把调度循环改成「先 decode 再 admit」之后，b8 从 5.657 s 掉到 8.327 s——**慢了 47%**。

原因是我在 `_collect_pending` 里这么写：

```python
first = self.queue.get(timeout=0.5) if idle else self.queue.get_nowait()
```

看着没问题，但只要**没有待入队的请求**，`get_nowait` 抛 `Empty` 就走正常路径了……
问题在于我第一版写的是**无条件阻塞**版本，每轮 decode 之后都要等下一个请求到达，
最多等 0.5 s。8 个请求跑完多花了 2.7 s。

修法很简单：**有活跃行时严格非阻塞**，只有空闲引擎才等。

然后加了 `--admission-grace-ms`：空闲期等一小段，把一串到达合成一次 prefill
（一次 17-token prefill 在双卡上约 0.86 s，八次单独 admission 就要付八次）。

| 轮次 | dense | paged grace 0 | paged grace 300 ms |
|---|---|---|---|
| 1 | 6.328 s / 8.059 t/s | 6.762 s / 7.543 t/s | 6.457 s / 7.898 t/s |
| 2 | 5.830 s / 8.748 t/s | 6.090 s / 8.374 t/s | 5.961 s / 8.556 t/s |

grace 稳定赢 2–5%，但**没追平 dense**。而且这台机器同配置重复跑的波动已经有约
8%（6.328 vs 5.830），5% 以内我不当成真实差异。

---

## 4. Prefill/decode overlap：撞上 NCCL 的通信流模型

剩下的结构性开销很明确：paged 引擎的 prefill 还在**主 stream** 上，每次 admission
都把正在 decode 的行冻住。

### 第一步：per-thread expert staging

`GGUFExpertStore` 原来用**进程级**单调计数器轮转 2 个 pinned slot，pinned buffer /
copy event / copy stream 也都是全局字典。两个 forward 并发时会各自推进计数器、
步调错开，A 还在读的 slot 被 B 覆盖。

改成按线程隔离：`_stage_tag()` / `_stage_counter()` / `_stage_streams()` /
`_stage_key()`，槽位计数器、buffer、event、stream 全部 per-thread。单线程行为不变。

这一步是对的，也是任何并发方案的前提。

### 第二步：实现、然后 SIGABRT

admission 丢进单 worker executor + 独立 CUDA stream，decode 主循环继续走，
prefill 完成后用跨流 `wait_stream` 同步。双卡上一跑：

```
exit code -6, Signal 6 (SIGABRT)
```

NCCL watchdog 直接 abort。

### 第三步：多路复用客户端

原始协议是**严格 FIFO、无请求 id**，worker 按到达顺序回包。所以任何多线程发消息
都是协议违规。

于是把 `NCCLExpertClient` 改成**单 I/O 线程**：`submit_chunk()` 只入队，
`_io_loop` 线程独占协议收发，调用方各自等自己的 `threading.Event`。

序列化路径验证通过：输出正确、heartbeat healthy、全量测试绿。

### 第四步：overlap 还是 SIGABRT

**发送端交织不是根因。** 真正的嫌疑是 NCCL 本身：

> 一个 `ProcessGroupNCCL` 的点对点通信跑在**一条内部 comm stream** 上。
> 两个 forward 同时要 expert 交换，不管哪个线程发起调用，都抢同一条流。

这不是「客户端加个队列」能解决的。

### 第二次押错（这次更严重）

上一步之前，我给 overlap 加了护栏：

```python
raise ValueError("overlap_prefill is not supported with a remote expert client")
```

然后我跑了测试，看到 `ChildFailedError`，就把「加了多路复用客户端仍然失败」写进了文档，
还给了根因分析。

**那次 run 根本没跑起来**——护栏在 `__init__` 里就抛了 `ValueError`，torchrun 只是
把它包成了 `ChildFailedError`。

也就是说：我拿一个**自己写的护栏**当成了实验结果。后来把护栏换成能力协商
（`transport.supports_concurrent_submit`）、真正跑了一次，才拿到真的 SIGABRT。

代价是文档里多了一段错误结论，还得再写一段更正。

---

## 5. Worker 池化：分发器 + 一个真 bug

单 worker 串行服务所有 chunk，要突破只能复制 worker。

`PooledExpertClient`：

- 按 **in-flight 最少**分发，平手轮转；
- 保持 `submit_chunk`/`wait_chunk` 契约；
- 回包按 `PendingRemoteChunk.owner` **回到提交它的 worker**；
- 聚合每 worker 计数 + `imbalance`（单 worker 时根本看不到的指标）。

第一版 `wait_chunk` 是**按 client 顺序猜 owner** 的，回包会从错误的 worker 收。
fake transport 的单测立刻暴露了。这种 bug 在真机上会表现成随机数值错误，很难查。

**但它只有单测。** 池化要 1 协调 + 2 worker，这台机器只有 2 张卡，远端 worker
复制不出来。所以它**从未承载真实流量**。

---

## 6. 定版：二卡形态

最后把范围钉死，写进 `docs/ECS_TWO_GPU_RELEASE.md`：

**已验证**：expert 分片 + token 精确、连续批处理 + paged KV、前缀共享、
故障监督重启、KV 池完整归还。

**不在本版本**：worker 池化（未接真流量）、prefill/decode overlap（被 NCCL
通信流卡住）、collective all-to-all（只有单独 smoke）、in-flight 续跑（崩了整请求重来）。

一句话：二卡形态是**能跑、可复现、token 精确**的分布式 MoE runtime，
**不是吞吐机器**。tok/s 上打不过 llama.cpp，这点不该粉饰。

---

## 7. 三次押错的共同点

回头看，这轮的三次误判不是同一类错误：

1. **paged KV**：假设「拷贝是瓶颈」。测出来发现只有长上下文才成立。
   → 教训：**先测再改**。这个假设如果先花十分钟量一次 KV 拷贝占比，能省两天。

2. **0.5 s 阻塞**：实现了「先 decode 后 admit」，但 `_collect_pending` 在无请求时
   阻塞，把收益吃成了 47% 回退。
   → 教训：**性能改动必须配对测量**。5.657 → 8.327 这个数字如果只在最后跑一次
   全量，很可能被归因成「机器波动」。

3. **把护栏当成失败**：最严重。用自己写的保护代码的报错，当成了实验结论。
   → 教训：**看到异常先读异常**。`ChildFailedError` 里明明包着 `ValueError`，
   我抓到的是 `tail -c 1200` 截断后的尾巴，然后就开始推理根因。

第 3 条其实还有一个更隐蔽的版本：我当时确信「加了多路复用就能修好」，于是在
看到失败时，第一反应是去解释「为什么没修好」，而不是去确认「它到底跑没跑」。

---

## 8. 现在的状态

- 全量测试 **127 passed**
- 二卡形态定版，范围写在 `docs/ECS_TWO_GPU_RELEASE.md`
- 完整实验日志（含失败实验）在 `docs/REMOTE_ECS_2X3080TI.md`
- 结果 JSON 在 `docs/results/`

多卡版本的升级路径（4 步，按依赖排序）：

1. 复制并监督 worker（**需要 ≥3 卡**，唯一硬门槛）
2. 激活按 expert ownership 分片 + 每 worker 路由表
3. collective all-to-all 替代点对点（这也是 overlap 的前置）
4. in-flight request/KV 续跑

第 1 步没硬件就只能写纸面代码——这一轮已经吃过亏，不再重复。
