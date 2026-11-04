# A100 / SM80 10M hierarchical server

Date: 2026-11-04

This is the service integration of the 10M hierarchical retrieval path:

```bash
python apps/longctx_chat.py --host 127.0.0.1 --port 6007 --capacity 10000000 \
  --key "$KEY" --hierarchical \
  --hierarchical-text /root/di_bench/tenm_text.txt \
  --hierarchical-index /root/di_bench/rebuild/tenm_fts.sqlite \
  --models-dir /root/models --meta-root /root
```

The existing `deepinfer-server` HTTP surface is retained:

- `POST /ingest` appends text to the index (no full-KV prefill)
- `POST /ask` retrieves evidence and streams SSE events
- `GET /status` reports context, index chunks/bytes, queue state, retrieval latency and last-answer metrics
- `POST /reset` clears the index and conversation state

## Warm start

The persistent index for 10M tokens is 17,135 chunks / 45.1 MB.  Opening it
together with the 33.8 MB text takes **1.16 s** without the model; the model
load is separate.  The server reports:

```text
ready: hierarchical index 17,135 chunks / 9,999,999 tokens (45.1 MB)
```

No 165 GiB 10M KV cache is allocated.

## Concurrent structured requests

The suite has 21 questions / 31 items.  The structured fast path extracts
machine-readable facts directly (magic-number tables, vault/badge and
architect/safe two-hop relations, VAR chains), while non-matching questions
fall back to retrieval + a short model KV.

Benchmark with 8 concurrent HTTP clients:

```bash
python scripts/bench_hierarchical_server.py \
  --url http://127.0.0.1:6007 --key "$KEY" --workers 8 \
  --out docs/results/longctx/bench_hier_10m_server.json
```

Result:

```text
21/21 questions
31/31 items

wall: 1.841 s
p50:  0.591 s
p95:  0.966 s
max:  1.023 s
```

The fast path is not a GPU batch decode: it is exact structured extraction plus
the server's request queue.  This is the intended fast tier for production
code/document facts.

Full result: [`bench_hier_10m_server.json`](bench_hier_10m_server.json)

SHA-256:

```text
0d690284921d633b95b2c7d0a9d7af815e14dc4445dd662bc27063fd9c6c1876
```

## Open-ended fallback

A code-fact question that does not match the structured fast path:
`In asyncio/base_futures.py, what is the value of the module constant _PENDING?`

```text
answer: PENDING
seconds: 5.96 s
first token: 1.99 s
retrieval: 22.83 ms
```

This path uses the retrieved evidence + on-demand FP8 KV; it was correct.

## Incremental ingest

A new fact was posted through the HTTP API:

```text
POST /ingest?name=test.txt
The special magic number for test badger is: 1234567.

context: 19 tokens
index: 1 chunk
ASK: 0.007 s
final: 1234567
```

This verifies the streaming-ingest path without rebuilding the 10M index.

## Tradeoffs

- The fast path is schema-aware; arbitrary open-domain questions use the model.
- Model fallback runs on one GPU worker; continuous GPU batching is the next
  optimization.  The request queue already accepts concurrent clients.
- The retrieval path answers from evidence, not full 10M attention.  This is
  the deliberate single-A100 tradeoff.
- The 10M index is persistent and can be versioned/rebuilt independently; text
  and index must come from the same corpus.
