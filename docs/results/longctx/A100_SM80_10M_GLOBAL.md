# 10M global queries: count, summary, aggregation

Date: 2026-11-04

Top-k retrieval is not a global reader.  This benchmark covers the three
query families that need the whole 10M-token document:

1. exact literal count;
2. summary with whole-document depth coverage;
3. aggregation over every structured record.

## Corpus

```text
tokens:        10,000,000
text:          33,834,980 bytes
markers:       137 occurrences of GLOBAL_MARKER_X
metrics:       100 GLOBAL METRIC records across 5 categories
themes:        5 unique theme lines, spread across depth buckets
```

## Results

```text
count:  137/137        0.013 s   exact full-text scan
summary: all 5 themes  3.217 s   64 selected units / 16 depth buckets
compare: all totals    0.013 s   all 100 records aggregated
build:                  40.38 s
```

All checks are true:

```json
{"count": true, "summary": true, "compare": true}
```

Metric totals:

```text
alpha=618, beta=594, gamma=570, delta=587, epsilon=604
largest: alpha
```

## How each family is handled

### Count

The question is classified as a count query, the literal term is extracted
from backticks/quotes/uppercase identifiers, and every occurrence is counted
with `re.finditer` over the complete text.  No retrieval window is involved.

### Summary

The text is split into line-sized units, scored with TF-IDF, and selected with:

- top information-density lines per depth bucket;
- one extra rarest-term line per depth bucket.

That second slot is what keeps short, high-value theme/constant lines from
being outvoted by long code lines.  The result includes a depth citation for
every selected unit and covers all 16 buckets.

This is an **extractive** summary; an optional model can synthesize from the
selected full-coverage passages.  It is not a model-generated abstract.

### Compare / aggregate

Every `GLOBAL METRIC category=... value=...` line is parsed and summed.  The
answer reports all category totals and the largest category.  No top-k sampling
is involved.

## Integration

`deepinfer-server --hierarchical` routes global questions before the normal
retrieval/model path:

- `global_answer(question, text, facts)` returns count, summary, or aggregation;
- count/compare are deterministic and sub-20 ms;
- summary is extractive and full-document;
- non-global questions still use FTS retrieval + model fallback.

Code:

- `deepinfer/retrieval/global_queries.py`
- `deepinfer/moe/hierarchical_service.py`
- `scripts/global_10m.py`

Full result: [`global10m_results.json`](global10m_results.json)

SHA-256:

```text
1b8d58611f7e32a9e98de3b58fd1ce8202ae4392d12f9dddd904f2a26becd508
```

## Limits

- Summary is extractive and depth-balanced, not an abstractive understanding of
  every paragraph.
- Arbitrary semantic global patterns that are not literal counts or structured
  records still need a map-reduce extraction pass; that is the next extension.
- The current production fast path is CPU-side for these three query families; it
  does not allocate a 10M KV cache.
