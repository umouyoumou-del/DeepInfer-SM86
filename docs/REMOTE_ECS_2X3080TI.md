# ECS on 2×RTX 3080 Ti

This document records the first real multi-GPU ECS validation. The host is a
temporary two-GPU substitute for the unavailable 8×P100/NCCL machine.

## Hardware and software

| Item | Value |
|---|---|
| GPU | 2× NVIDIA GeForce RTX 3080 Ti, 12 GB, SM 8.6 |
| CPU | Intel Xeon Silver 4214R, 48 hardware threads |
| RAM | 440 GiB |
| OS | Ubuntu 22.04.4 |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime | 12.1 |
| CuPy | 14.2.0 |
| Transformers | 4.49.0 |
| Model | DeepSeek-V2-Lite-Chat Q4_K_M, 10,364,416,320 bytes |
| Model SHA-256 | `30b4fb4ab1fbe1c6a827303ea898296c9faf8f54d6cdf8b9fbdda7a7ebfb292a` |

The two GPUs do not expose peer-to-peer access (`can_device_access_peer=False`
in both directions), so cross-device hidden states and packet exchange go
through host/PCIe/NCCL as appropriate.

## Environment bootstrap

```bash
python3.12 -m venv /root/autodl-tmp/ecs-venv
/root/autodl-tmp/ecs-venv/bin/pip install --upgrade pip wheel setuptools
# torch 2.5.1+cu121 (use the cached/system CUDA libraries available on host)
/root/autodl-tmp/ecs-venv/bin/pip install \
  torch==2.5.1 typing_extensions filelock fsspec jinja2 networkx sympy
/root/autodl-tmp/ecs-venv/bin/pip install \
  cupy-cuda12x==14.2.0 transformers==4.49.0 accelerate gguf safetensors \
  numpy pytest psutil
apt-get install -y build-essential aria2
```

The GGUF was downloaded from the ModelScope mirror because direct
Hugging Face access timed out from the host:

```bash
aria2c -c -x16 -s16 -k1M --file-allocation=none \
  -o DeepSeek-V2-Lite-Chat-Q4_K_M.gguf \
  https://modelscope.cn/models/second-state/DeepSeek-V2-Lite-Chat-GGUF/resolve/master/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
```

## Results

Single GPU baseline:

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Prompt tokens | 17 |
| Completion tokens | 7 |
| Prefill | 0.90 s |
| Decode | 1.43 s |
| Decode rate | 4.20 tok/s |
| Peak GPU allocated | 2.20 GiB |

Two-stage layer pipeline, split after layer 12:

- GPU0 executes embedding + layers 0–12.
- GPU1 executes layers 13–26 + final norm + LM head.
- Hidden state is copied between GPUs once per stage per token.
- Exact token parity with the single-GPU baseline:
  `[207, 20, 12, 17, 28, 18, 100001]`.

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Prefill | 1.61 s |
| Decode | 1.26 s |
| Decode rate | 4.76 tok/s |
| Peak GPU0 / GPU1 | 2.22 / 2.22 GiB |
| Load time (both runtimes) | 32.89 s |

The current layer-pipeline smoke loads the full runtime on both GPUs and
executes only the assigned layer ranges. It validates two-GPU execution,
cross-GPU hidden-state transfer, and token parity, but it is not yet a
memory-optimized layer-sharded loader.

Dual-GPU ECS worker pool (one independent runtime per GPU, queue-depth
assignment):

| Metric | Value |
|---|---:|
| Two concurrent requests | `5-2=3`, `2+3=5` |
| Assigned devices | `[0, 1]` |
| Wall time | 4.77 s |
| Per-stream decode | 2.88 / 2.82 tok/s |
| Aggregate output | both correct |

NCCL ECS packet All-to-All:

- `torch.distributed` backend: NCCL.
- Hot expert 7 was token-sharded 2/2 across the two GPUs.
- Expert 8 was routed to rank 1.
- Rank 0 sent 3 packets; rank 1 received 2 packets
  (`[101, 102]` for expert 7 and `[201, 202]` for expert 8).
- Rank 0 received its local shard `[103, 104]`.

## Expert-sharded single-model execution

The runtime now supports `--expert-devices 0 1`.  With ECS enabled, routed
experts receive an ECS route-placement owner on GPU0 or GPU1.  Each expert
chunk is split by owner, the owner stages its packed expert bytes and runs the
grouped gate/up/down kernels on that GPU, and the chunk output is copied back
to the coordinator GPU.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/python -u   scripts/smoke_deepseek_v2_lite_moe.py   --model /root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-GGUF/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf   --meta-dir /root/autodl-tmp/meta   --device cuda:0 --expert-devices 0 1 --ecs   --prompt '只回答数字：5-2=?' --max-new-tokens 8 --warmup-tokens 2   --max-cache 128 --output /root/autodl-tmp/results/ecs_expert_shard.json
```

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token IDs | `[207, 20, 12, 17, 28, 18, 100001]` |
| Token parity vs single GPU | exact |
| Expert dispatches on GPU0 / GPU1 | 1416 / 1415 |
| Expert chunk dispatches | 834 |
| Prefill | 1.42 s |
| Decode | 3.25 tok/s |
| Coordinator peak | 2.24 GiB |

This is slower than the single-GPU 4.20 tok/s because every expert chunk now
has cross-PCIe activation/output copies and the coordinator still runs all
attention/shared layers.  It proves correctness and balanced expert ownership;
it is not yet a throughput win.

The current path uses direct `Tensor.to()` copies between the two GPUs.  The
NCCL ExpertPacket All-to-All above is the transport path for the next revision
of in-model activation exchange.

## NCCL expert-parallel runtime

The next revision removes the duplicated non-routed weights from the remote
rank.  Rank 0 keeps attention, shared experts and the LM head; rank 1 is a
lightweight routed-expert executor that only opens the GGUF expert bytes.
Requests and activation tensors travel over NCCL send/receive.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/torchrun   --standalone --nproc_per_node=2   scripts/bench_ecs_expert_parallel_nccl.py   --max-new-tokens 8 --max-cache 128   --output /root/autodl-tmp/results/ecs_expert_parallel_nccl.json
```

| Metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token IDs | `[207, 20, 12, 17, 28, 18, 100001]` |
| Token parity vs single GPU | exact |
| Remote expert requests | 294 |
| Expert chunk dispatches | 591 |
| GPU0 / GPU1 expert dispatches | 996 / 984 |
| NCCL bytes each way | 21,344,256 |
| NCCL request/reply latency | 0.375 s total |
| Prefill | 2.62 s |
| Decode | 3.44 tok/s |
| Coordinator peak | 2.21 GiB |
| Remote worker peak | **37.7 MiB** |

The remote rank does not load the transformer model: its peak allocation is
only the expert staging buffer and CUDA context.  This addresses the previous
duplicate-non-routed-weight problem.

The first version used blocking NCCL request/reply and decoded at 3.44 tok/s.
The asynchronous revision submits the remote chunk with `isend`/`irecv`, then
executes local expert groups while rank 1 computes, and waits only before
merging the output.

| Async NCCL metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token parity | exact |
| Prefill | 2.02 s |
| Decode | **4.90 tok/s** |
| Single-GPU baseline | 4.20 tok/s |
| Async speedup | 1.17× |
| Remote requests | 294 |
| Remote active sends at finish | 0 |
| Accumulated request latency | 1.48 s |
| Remote worker peak | **37.7 MiB** |

This is the first dual-GPU path that beats the single-GPU decode baseline.
The accumulated request latency is larger than wall decode time because the
requests overlap local compute; wall time is what matters.

The current hybrid revision batches all remote expert groups of a prefill
layer, then runs local expert groups while the remote rank works; decode keeps
the lower-latency per-chunk async path.  The 8-token smoke becomes:

| Hybrid NCCL metric | Value |
|---|---:|
| Output | `5-2=3` |
| Token parity | exact |
| Prefill | 1.74 s |
| Decode | **4.71 tok/s** |
| Single-GPU decode | 4.20 tok/s |
| Hybrid speedup | 1.12× |
| Remote requests | 232 |
| Remote worker peak | 50.4 MiB |

Per-chunk async decode alone measured 4.90 tok/s, but its prefill was 2.02 s.
The hybrid keeps the better prefill batching and the better decode strategy.

The transport is still point-to-point request/reply rather than a collective
all-to-all.

### Expert ownership balance

The hybrid path defaults to 50% remote experts.  A 67% remote experiment
(1313 remote vs 667 local expert dispatches) overloaded the single remote
worker:

| Remote fraction | Prefill | Decode |
|---:|---:|---:|
| 50% (default) | 1.74 s | 4.71 tok/s |
| 67% | 7.79 s | 2.74 tok/s |

The remote rank is one executor, not a pool; moving more experts to it just
serializes the work.  Keep 50/50 until remote worker replication exists.

### Multi-request and dead-rank timeout

The same NCCL worker was served two sequential prompts with
`--requests 2`:

| Request | Output | Prefill | Decode |
|---|---|---:|---:|
| `5-2?` | `5-2` (4 generated) | 2.13 s | 4.14 tok/s |
| `2+3?` | `2+3` (4 generated) | 0.97 s | 4.03 tok/s |

Both finished with `active_sends=0`, so the worker queue drained and the
worker handled more than one request lifetime.  This is sequential queue
validation, not concurrent request batching.

`--timeout-seconds` is now passed to the NCCL process group.  A dead remote
rank therefore produces a timeout instead of hanging forever, but automatic
worker restart is not implemented.

A bounded `NCCLRequestService` now accepts concurrent API submissions and
drains them through the single coordinator runtime.  The remote worker writes
a heartbeat file that the coordinator monitors.

| Concurrent service metric | Value |
|---|---:|
| Submitted concurrently | 2 |
| Completed / failed | 2 / 0 |
| Queue depth after drain | 0 |
| Active after drain | 0 |
| Remote requests | 304 |
| Heartbeat age | 0.44 s |
| Heartbeat healthy | yes |

Request-level outputs:

| Request | Output | Prefill | Decode |
|---|---|---:|---:|
| `5-2?` | `5-2` (4 generated) | 1.81 s | 4.67 tok/s |
| `2+3?` | `2+3` (4 generated) | 0.71 s | 5.59 tok/s |

Important limitation: the service provides concurrent **submission and
queueing**, but the coordinator runtime still executes one request at a time.
This is not continuous batching.

### Fixed-batch generation

`generate_batch()` now steps several sequences through the coordinator and
NCCL expert path together.  This is fixed-size batch generation, not
continuous batching: all rows share the same number of decode steps and
finished rows remain as padded rows.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/torchrun   --standalone --nproc_per_node=2 scripts/bench_ecs_batch_generate.py   --batch-size 2 --max-new-tokens 8 --max-cache 128   --output /root/autodl-tmp/results/ecs_batch_generate.json
```

Batch scaling on the same two GPUs:

| Batch | Prefill | Decode wall | Aggregate decode | Speedup vs single | Remote peak |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.81 s | 2.06 s | 6.32 tok/s | 1.50× | 52.0 MiB |
| 4 | 1.95 s | 1.45 s | **16.60 tok/s** | 3.95× | 55.6 MiB |
| 8 | 1.89 s | 2.01 s | **25.90 tok/s** | 6.17× | 61.7 MiB |

Aggregate decode counts all generated tokens across rows; per-row throughput is
lower.  Batch 8 preserves the `5-2` token IDs exactly:
`[207, 20, 12, 17, 28, 18, 100001]`.  Some duplicate prompts in larger batches
render spacing differently (`2+3=5` vs `2 + 3 = 5`) while remaining
semantically correct.

All rows are fixed-size: they start together and use equal decode steps.
Finished rows remain padded; this is not continuous batching.

### Continuous batching

`deepinfer/moe/continuous_batching.py` now implements a slot-based continuous
scheduler:

- up to `max_batch_size` active rows;
- staggered requests are admitted into free slots;
- each new request is prefetched through the same NCCL expert worker;
- KV caches are left-padded, appended, and row-selected as requests finish;
- completed futures are returned independently.

Smoke command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/torchrun   --standalone --nproc_per_node=2 scripts/bench_ecs_continuous_batch.py   --max-batch-size 4 --max-new-tokens 8 --arrival-delay 0.15
```

| Continuous metric | Value |
|---|---:|
| Capacity | 4 |
| Staggered arrivals | 4 |
| Completed / failed | 4 / 0 |
| Wall time | 5.94 s |
| Aggregate decode | 4.21 tok/s |
| Remote requests | 478 |
| Heartbeat | healthy |

All four outputs were correct:
`5-2=3`, `2+3=5`, `9-4=5`, `1+1=2。`.

A batched-admission revision groups queued prefills into one forward.  It
reduced remote requests from 478 to 391, but did not improve wall time:

| Continuous revision | Wall | Aggregate decode | Remote requests |
|---|---:|---:|---:|
| Individual admission | 5.94 s | 4.21 tok/s | 478 |
| Batched admission | 6.13 s | 4.24 tok/s | 391 |

The scheduler is functionally correct, but aggregate throughput remains below
fixed batch (`6.32–25.90 tok/s`) because per-row cache padding/selection and
admission synchronization dominate.  A paged KV slot allocator and batched
decode admission are the next performance tasks.

### Supervised restart

`scripts/run_ecs_expert_parallel_supervised.py` launches the coordinator and
remote rank itself, monitors process exit and the heartbeat file, and restarts
both ranks after a failure.  A simulated first-attempt fault was injected with
`DEEPINFER_REMOTE_FAULT_AFTER=2` (rank 1 exits with code 42):

| Supervised restart metric | Value |
|---|---:|
| Attempts | 2 |
| Attempt 0 | `worker_exit_42` |
| Attempt 1 worker / coordinator | 0 / 0 |
| Recovered | yes |
| Output | `5-2` (4 generated) |
| Token IDs | `[207, 20, 12, 17]` |
| Prefill / decode | 1.88 s / 4.63 tok/s |
| Heartbeat age | 0.004 s, healthy |

This is whole-request restart: both ranks are killed/recreated, the model is
reloaded, and the request is retried.  In-flight KV/cache resume is not
implemented, and model reload adds startup cost.

Long-context smoke (45 prompt tokens, 4 generated tokens):

| Metric | Single GPU | Async NCCL expert |
|---|---:|---:|
| Token IDs | `[207, 28272, 10900, 39322]` | same |
| Prefill | 1.76 s | 2.71 s |
| Decode | 4.50 tok/s | 4.28 tok/s |
| Peak GPU | 2.23 GiB | 2.28 GiB coordinator |

The 45-token case still has exact token parity but loses the short-decode
throughput win.  Multi-request NCCL scheduling and remote-worker
heartbeat/restart are still unmeasured.

### Paged KV slot allocator

The slot engine above keeps every active row in one dense `DynamicCache`.
Admission re-packs all live rows (and shifts them right when the newcomer is
longer) and completion gathers the survivors.  `deepinfer/moe/paged_continuous_batching.py`
replaces that with a real block allocator:

- one `PagedKVCacheManager` pool owns the physical KV blocks;
- each request holds its own `block_ids` and grows them one block at a time as
  it decodes (`PagedSlotAllocator.reserve`), so a 17-token prompt never pins
  128 tokens of cache;
- a decode step builds one `[batch, max_blocks]` block table and lets the
  native paged MLA kernel read straight out of the pool -- no row is ever
  re-packed, shifted or re-indexed;
- completion returns the blocks to the allocator and nothing else happens.

Two prefill strategies are available.  `prefill_mode="batched"` (default)
prefills the queued requests in one left-padded dense forward, then copies each
row's valid suffix into its own blocks once, grouped by equal prompt length.
`prefill_mode="paged"` skips the dense cache entirely and prefills each request
through `runtime.forward_logits` with `seq_lens=0`.

Command:

```bash
PYTHONPATH=$PWD /root/autodl-tmp/ecs-venv/bin/torchrun   --standalone --nproc_per_node=2   scripts/bench_ecs_paged_continuous_batch.py   --max-batch-size 4 --max-new-tokens 8 --arrival-delay 0.15 --max-cache 256   --output /root/autodl-tmp/results/ecs_paged_paged_b4.json
```

Same host, same prompts, dense slot engine versus the paged engine:

| Workload | Dense wall | Paged wall | Dense agg. | Paged agg. |
|---|---:|---:|---:|---:|
| 4 prompts, 17 tok, 8 new | 5.127 s | 5.133 s | 5.071 tok/s | 5.065 tok/s |
| 8 prompts, 17 tok, 16 new | 5.609 s | 5.657 s | 9.093 tok/s | 9.016 tok/s |
| 4 prompts, 44 tok, 8 new | 7.004 s | **5.965 s** | 3.855 tok/s | **4.526 tok/s** |
| 4 prompts, capacity 1 (parity) | -- | 8.505 s | -- | 2.940 tok/s |
| 4 prompts, paged prefill | -- | 6.369 s | -- | 3.925 tok/s |

All runs completed every request with zero failures, and every run ends with
`used_blocks = 0`, i.e. the pool is fully returned.

What this does and does not show:

- Paging pays off exactly when the KV is long: at 44-token prompts the paged
  engine is 1.17x faster (7.004 s to 5.965 s).  Copying
  `[rows, heads, tokens, dim]` per layer per step is real work only once
  `tokens` is large.
- At 17-token prompts the two engines are indistinguishable (5.127 s vs
  5.133 s).  The earlier hypothesis in this document -- that cache padding and
  row selection dominate the 4.2 tok/s plateau -- is **not** supported at that
  scale.  The remaining gap to fixed-batch (9.09 tok/s at batch 8, 25.90 tok/s
  in the fixed-batch harness) sits in admission synchronization and one decode
  step per scheduling iteration, not in KV copies.
- Paged prefill (6.369 s) is 24% slower than batched admission prefill
  (5.133 s), so keeping the grouped dense prefill and writing it through once
  is the right default.
- Grouping the write-through by prompt length mattered: before grouping the
  batch-8 run was 5.807 s, after it 5.657 s.

Numerical parity: the capacity-1 run reproduces the single-request outputs
exactly (`5-2=3`, `2+3=5`, `9-4=5`, `1+1=2。`) and the `5-2` token IDs stay
`[207, 20, 12, 17, 28, 18, 100001]` at every batch size.  The spaced variant
`2 + 3 = 5` appears only once rows are batched, and it appears identically in
the dense engine, so it is a batching-order effect in the expert path rather
than something the paged KV introduced.

Known costs and limits:

- the MLA pool pads the key head dimension up to `max(key_dim, value_dim)`
  (192 to 512 here), so key storage carries a 2.7x overhead;
- a 16-token block across 27 layers, 16 heads and 512 dims is 14.2 MB, so the
  32-block default pool is ~453 MB;
- the pool is sized from free memory at build time
  (`--kv-memory-utilization`, default 0.30) and capped at
  `max_batch_size x ceil(max_cache / block_size)`;
- blocks are private per request: prefix sharing and block reuse across
  requests are still not wired into this engine;
- no in-flight KV resume after a worker restart.

### Prefix block sharing and admission coalescing

Two follow-ups to the paged allocator.

**Shared prompt blocks.** The engine now hashes its prompt in block-sized
chunks (`PrefixCache.block_hashes`), retains the longest cached prefix, and
forwards only the remaining suffix at its true position instead of restarting
from zero.  Blocks reused this way stay immutable: the matcher never hands back
the final block of an exactly block-aligned prompt, because that block still
has to receive the row's tail.  Complete prompt blocks are published back to
the LRU after every admission, so a second request with the same preamble pays
only for its question.

Four requests sharing an 89-token preamble (5 full 16-token blocks reusable),
`--prefill-mode paged`, same host:

| Configuration | Wall | Aggregate | Prefill per request |
|---|---:|---:|---|
| paged, no prefix cache | 8.359 s | 1.555 tok/s | 2.52 / 1.17 / 1.22 / 1.21 s |
| paged + `--prefix-cache-blocks 64` | **6.151 s** | **2.114 tok/s** | 2.47 / **0.58 / 0.52 / 0.54** s |
| dense slot engine, same prompts | 6.299 s | 2.064 tok/s | 2.52 / 1.42 / 1.42 / 1.42 s |
| paged + prefix cache, batched prefill (no reuse) | 6.087 s | 2.136 tok/s | 2.30 / 1.42 / 1.42 / 1.42 s |

Prefix reuse cuts wall time by 26% and per-request prefill by 53% (1.17 s to
0.55 s), and it brings the paged path level with the dense engine on this
workload.  The first request still pays the full prefill, so the win scales
with reuse count, not with a single prompt.

Caveat: reuse only applies in `prefill_mode="paged"`.  The batched dense
prefill writes every row from position 0, so it cannot start at an offset; the
last row of the table above is that mode with the cache configured but unused.

**Admission coalescing.** The scheduler loop now decodes before admitting, and
an idle engine waits out a short `--admission-grace-ms` window so that a burst
of arrivals becomes one prefill instead of N.  Only the idle engine waits:
blocking for arrivals while rows are decoding stalls the batch, which cost
~2.7 s on the batch-8 workload during development.

Batch 8, 0.15 s arrival spacing, two paired repeats per configuration:

| Run | dense | paged, grace 0 | paged, grace 300 ms |
|---|---:|---:|---:|
| 1 | 6.328 s / 8.059 tok/s | 6.762 s / 7.543 tok/s | 6.457 s / 7.898 tok/s |
| 2 | 5.830 s / 8.748 tok/s | 6.090 s / 8.374 tok/s | 5.961 s / 8.556 tok/s |

The grace window is a consistent 2-5% win over grace 0, but it does not close
the gap to the dense engine, and the host's own run-to-run spread is now about
8% (6.328 s vs 5.830 s for the identical dense configuration), so differences
below ~5% should not be read as real.

Postscript: the paged engine still *prefills on the main stream*, so every
admission blocks the rows that are already decoding.  At 0.86 s per 17-token
prefill that is the remaining structural cost, and it needs prefill/decode
stream overlap rather than more allocator work.

### Prefill/decode overlap: blocked by the expert transport

The remaining structural cost after the paged allocator is that a prefill runs
on the scheduler's main stream, so every admission freezes the rows that are
already decoding (about 0.86 s per 17-token prefill on this host).  The
follow-up attempt was to move admission prefill into its own thread on its own
CUDA stream while the main stream keeps decoding.

Two changes were made:

1. **Per-thread expert staging.** `GGUFExpertStore` rotated through two pinned
   slots with a single process-wide counter, so two concurrent forwards would
   pick slots out of step with each other and overwrite a pinned buffer that
   another stream was still reading.  Slot counters, pinned buffers, copy
   events and copy streams are now per thread (`_stage_tag`,
   `_stage_counter`, `_stage_streams`, `_stage_key`).  Single-threaded
   behaviour is unchanged and the full suite still passes.
2. **Overlapped admission** (`--overlap-prefill`): admission runs in a
   one-worker executor on a dedicated stream, the decode loop keeps stepping,
   and finished prefills are adopted with a cross-stream `wait_stream` before
   their rows are decoded.

On the two-GPU NCCL host this **aborts the process**: the run dies with
`SIGABRT` from the NCCL watchdog (exit code -6) instead of producing output.
The expert transport is a single-producer request/reply channel -- one remote
worker serves one chunk at a time, and two threads issuing sends on the same
process group interleave into a protocol violation.

The engine therefore refuses the combination up front:

```python
if overlap_prefill and runtime.remote_expert_client is not None:
    raise ValueError(
        "overlap_prefill is not supported with a remote expert client: "
        "the NCCL expert transport is single-producer"
    )
```

A first transport change landed after that: `NCCLExpertClient` now queues
submissions and drains them on one dedicated I/O thread (`_TransportRequest`,
`_io_loop`), so the wire protocol still sees a single ordered producer while
callers may submit from any thread and wait on their own event.  The serial
path is unaffected and still produces correct output with a healthy heartbeat.

Overlap still fails with the multiplexed client: the run now dies with a
`ChildFailedError` from `torchrun` rather than the NCCL watchdog `SIGABRT`,
i.e. the send-side interleaving is no longer the trigger, and the underlying
exception has not been isolated yet.  Treat `--overlap-prefill` as experimental
and unsupported on the NCCL path until that is diagnosed.  The likely remaining
suspects are the coordinator's own expert staging (now per thread, so probably
not), the shared ECS heat/LRU controller under two threads, and the single
remote worker serialising both streams anyway -- in which case overlap would
buy little even once it is correct.

What that means for the roadmap: prefill/decode overlap is not a scheduler
change on this architecture.  It needs one of

- a multiplexed transport (request ids, out-of-order replies, many in flight),
- one request channel per producer stream, or
- a single remote-facing queue that both streams feed and one thread drains.

Until then the honest statement is that continuous batching on this host is
bounded by admission prefill blocking decode, and the per-thread staging work
is a prerequisite that is already in place.

## What is and is not validated

Validated on hardware:

- real dual-GPU PyTorch/CuPy execution;
- real C++ ECS core on Linux (`cpp` backend);
- real NCCL ExpertPacket all-to-all;
- hot-expert token sharding across two physical GPUs;
- real cross-GPU layer pipeline with exact token parity;
- queue-depth worker-pool request assignment.

Still not validated:

- NVLink/P2P (this host reports no P2P);
- 8×P100 or any multi-node fabric;
- a collective All-to-All as the in-model activation exchange (the async NCCL
  path uses point-to-point isend/irecv; the separate ExpertPacket all-to-all
  smoke did pass);

- in-flight request/KV resume after a restart (whole-request retry is
  implemented by the supervisor);
- production heartbeat/worker restart for the dual-GPU pool.

Next: share KV blocks between requests (prefix cache on the paged engine),
support in-flight request resume, validate a collective all-to-all, and cut the
per-iteration admission synchronization that still separates continuous
batching from the fixed-batch throughput.  Fixed batch generation reaches 25.90 aggregate tok/s at
batch 8 (6.17x single GPU), the hybrid single-request path reaches 4.71 tok/s,
and the supervisor recovers from a simulated rank-1 crash by whole-request
retry.
