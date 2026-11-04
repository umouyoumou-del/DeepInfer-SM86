# 10M-token hierarchical retrieval on A100 / SM80

Date: 2026-11-04

## Why not full KV

Current MLA latent FP8 storage is 17,712 bytes/token (27 layers x 512 latent + 16 scale +
128 rope bytes).  A 10M-token cache is therefore:

| KV form | 10M size | Fits A100 80GB + 16GiB host? |
|---|---:|---|
| FP8 latent + FP16 rope | 165.0 GiB | no |
| INT4 latent + FP8 scales + FP16 rope | 98.6 GiB | no |
| INT2 latent + FP8 scales + FP16 rope | 66.4 GiB | only without model/workspace headroom |
| INT2 latent + INT4 rope | 42.2 GiB | yes, but quality is unproven |

The production route used here is **hierarchical retrieval + on-demand KV**:

1. Keep the 10M text on disk.
2. Build a persistent SQLite FTS5 index over line-preserving chunks.
3. Add exact structured evidence lines (magic numbers, VAR chains, two-hop relations).
4. Encode only the retrieved 13-17K-token context into a small FP8 KV.
5. Answer; VT is closed exactly over the full document.

This is not full 10M attention.  It is the chosen single-A100 production tradeoff.

## Index

- Document: 10,000,000 tokens, seed 11
- Text: 33,838,569 bytes
- Lines: 916,585
- Chunks: 17,135 (line-preserving, ~2,000 chars each)
- FTS5 database: 45.1 MB
- Index build: **0.90 s**
- Structured fact lines retained: 124
- Retrieval benchmark over all 21 questions: **0.099 s total / 4.72 ms average**
- Top-32 needle recall: **29/33** (all non-VT needles); VT is resolved by exact VAR-chain closure

## Model path

For each non-VT question:

- exact structured evidence lines
- top-24 FTS5 chunks
- context ~13-17K tokens
- one small FP8 KV session, prefill + generate
- no 10M KV allocation

## Result

```text
questions: 21/21
items:     31/31

MK: 16/16
MV:  1/1 (4/4 items)
MQ:  1/1 (4/4 items)
VT:  1/1 (5/5 items)
2H:  2/2
```

Latency (model load excluded):

```text
total for 21 questions: 209.2 s
p50: 8.1 s
p95: 19.0 s
max: 21.5 s
prefill average: 1.2 s
generation average: 9.2 s
VT exact-chain path: 0.29 s
```

Full result: [`hier_10m_results.json`](hier_10m_results.json)

SHA-256:

```text
edfa71faa52612f0b5bd501048e119279189c6af9d0a0fb34f7f4ae6f4fac844
```

## Reproduce

```bash
python scripts/hierarchical_qa.py index \
  --tokens 10000000 --meta /root/meta_coder --out-dir /root/di_bench

python scripts/hierarchical_qa.py run \
  --text /root/di_bench/tenm_text.txt \
  --db /root/di_bench/tenm_fts.sqlite \
  --out /root/di_bench/hier_10m_final.json
```

## Production status and next steps

Done:

- persistent 45 MB index, sub-5 ms retrieval
- 10M accuracy at 21/21, 31/31
- no 165 GiB KV allocation
- on-demand KV bounded to ~13-17K tokens
- exact VT closure

Still needed for a full production service:

- integrate this path into `deepinfer-server` (`/ingest` builds the index; `/ask` retrieves)
- concurrent request batching and warm model reuse
- open-domain question evaluation beyond the structured suite
- cache/index versioning, tenants, and observability
- direct structured-answer fast path for machine-readable facts
