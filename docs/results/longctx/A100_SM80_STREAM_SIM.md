# A100 / SM80 4M streaming user simulation

Date: 2026-11-04

This run exercises the **streaming-ingest** path, not a monolithic prefill:

- `deepinfer-server` is started with a 4,259,840-token capacity.
- `scripts/longctx_user_sim.py` uploads 866 files through `POST /ingest` in file/chunk
  order, 16K tokens per encode chunk, and asks questions while the queue is still
  being read.
- Questions are answered between encode chunks; the server keeps reading afterwards.

## Setup

```text
GPU:              A100-SXM4-80GB (SM80)
capacity:         4,259,840 tokens
GPU-resident KV:  3,830,167 tokens
workspace:        866 files, 4,126,773 tokens (1,252,848 project)
target:           4,150,000 tokens
```

Command shape:

```bash
python apps/longctx_chat.py \
  --capacity 4259840 --key "$KEY" \
  --models-dir /root/models --meta-root /root \
  --check-copies --self-check

python scripts/longctx_user_sim.py run \
  --url http://127.0.0.1:6006 --key "$KEY" \
  --project /root/DeepInfer-SM86 --meta /root/meta_coder \
  --target 4150000 --out /root/di_bench/sim_4m_a100.json
```

## Timeline

```text
0.12 min   project files queued
0.37 min   stdlib files queued behind the project
21.65 min  all read: 4,126,960 tokens, queued 0
45.9 min   all 21 simulated turns done
```

## Actual user input/output samples

### Asked while the context was still streaming in

At `context = 1,310,724`, `queued = 2,832,437`:

```text
User:
In deepinfer/moe/longctx_service.py, what is the default value of the
parameter `chunk` of `build_session`?

Assistant:
16384
```

Verdict: RIGHT, 34.14 s, 2 rounds.

### Asked after all 4.13M tokens had been read

At `context = 4,127,006`, `queued = 0`:

```text
User:
In deepinfer/moe/latent_mla_triton.py, what is the default value of the
parameter `block_n` of `decode_partial`?

Assistant:
32
```

Verdict: RIGHT, 111.49 s, 2 rounds.

## Result

```text
checked turns: 16/20 right
default:        7/8
constant:       3/5
docs:           1/1
hand:           5/6
memory turn:    unchecked
```

The four failures were retrieval misses on code constants/locations, not
streaming or SM80 failures.  The final memory turn returned `16384` instead of
the first question; it is recorded but not scored.

Full result: [`sim_4m_a100_stream.json`](sim_4m_a100_stream.json)

SHA-256:

```text
e22444e7a983c8da7426af7411e7870b31790c5c9d6892ec19553298f345f61e
```
