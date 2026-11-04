# Production tiering for 3M / 4M / 10M

Date: 2026-11-04

The production server now selects the context tier automatically:

```text
ctx_size <= 4,194,304    -> full-KV attention (previous 3M/4M behavior)
ctx_size >  4,194,304    -> hierarchical retrieval + on-demand KV
```

The mode is also explicit:

```bash
deepinfer-server --mode auto    # default: size-based selection
deepinfer-server --mode full
deepinfer-server --mode hierarchical
```

`--hierarchical` remains a shortcut for `--mode hierarchical`.

## Full-KV tier (3M / 4M)

- Preserves the previous full-attention behavior.
- `--load-cache` starts from a saved `DeepSeekLongContext` cache.
- Global count/summary/aggregation questions are now handled by the same
  full-document `global_answer` path used by the 10M server, before locate.
- For a cold 4M run, prefill was 18.4 min and 21/21, 31/31 was verified.
- 3M was previously verified at 21/21, 31/31; 4M A100 at 21/21, 31/31.
- The 4M full KV is ~69 GiB; the 3M full KV is ~52 GiB. Production warm starts
  should use `--load-cache`; caches are produced by the existing
  `DeepSeekLongContext.save` / `needle_complex --save-cache` path.

## Hierarchical tier (above 4M)

- Text + SQLite FTS5 index; only retrieved 13-17K evidence is encoded.
- Global count/summary/aggregation still scan the full text.
- 10M suite: 21/21 questions, 31/31 items.
- 10M 100-needle run: 100/100.
- 10M global run: count 137/137, summary 5/5 themes, aggregation all records.

## Global-query validation at 3M / 4M

The same global benchmark was run at both sizes:

```text
3M:  build 12.42 s; count 0.004 s; summary 1.200 s; compare 0.004 s
4M:  build 16.11 s; count 0.005 s; summary 1.507 s; compare 0.006 s

checks for both:
  count:   true  (137/137)
  summary: true  (all 5 themes, 73 units / 16 depth buckets)
  compare: true  (100 records, all category totals)
```

Summary selection is line-level TF-IDF with:

- one uppercase-heading line per depth bucket;
- one rarest-term line per bucket;
- information-density fill lines.

The heading/rare slots are what keep short themes and unique constants from
being outvoted by long code lines.

Result files:

- [`global3m_results.json`](global3m_results.json)
- [`global4m_results.json`](global4m_results.json)
- [`global10m_results.json`](global10m_results.json)

SHA-256:

```text
3m:  cd3ad63cdb28532f14eb12bcd7bab7d9191ce1f1614bf8ebea856cdd6f3c9f27
4m:  d7acb968ea3bf8368154654baea737ee867f9d332e4ebec2668cf01f3e4cf609
10m: 1b8d58611f7e32a9e98de3b58fd1ce8202ae4392d12f9dddd904f2a26becd508
```

## Tests

- mode routing: full through 4M, hierarchical above 4M
- full-KV global count routing
- hybrid structured fast path and global count/summary/compare
- retrieval/count/aggregation/summary unit tests

## Tradeoffs

- Full KV is semantically the strongest, but 3M/4M cold prefill and cache sizes
  are large. Warm cache is the production path.
- Hierarchical tier is much faster and scales beyond 4M, but is not full
  attention.
- Global summaries are extractive and full-document; an optional model can
  synthesize from the selected depth-balanced passages.
