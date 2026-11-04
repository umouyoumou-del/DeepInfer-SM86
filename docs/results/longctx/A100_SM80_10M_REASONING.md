# 10M complex global reasoning: fact graph + SQL joins

Date: 2026-11-04

Top-k retrieval can miss records needed for global questions.  The new global
fact-graph tier scans the complete 10M text once, builds a SQLite graph of
metrics, entities and directed relations, and executes aggregation / distinct
counts / filters / two-hop joins over every record.

## Corpus

```text
tokens:        10,000,000
text:          33,667,316 bytes
metrics:       2,000 GLOBAL METRIC records
entities:      2,000 ENTITY records
relations:     3,998 directed RELATION records
build:         41.01 s
graph build:    0.33 s
```

## Questions and checks

| Question | Expected | Check | Query time |
|---|---|---|---|
| largest metric category | delta | true | 0.328 s |
| top 3 categories | delta=20927, epsilon=20694, beta=20521 | true | 0.331 s |
| category alpha total | 19909 | true | 0.391 s |
| alpha total, values >= 50 | 15130 | true | 0.331 s |
| distinct subjects with relation=owns | 1999 | true | 0.328 s |
| entities type=person, group=alpha | 334 | true | 0.337 s |
| e0001 --owns--> ? --located_in--> ? | e0003 | true | 0.327 s |
| e0001 --owns--> ? --located_in--> ? --owns--> ? | e0004 | true | 0.330 s |

All eight checks are true:

```json
{"largest": true, "top3": true, "alpha_total": true, "alpha_ge50": true,
 "owns_subjects": true, "person_alpha": true, "two_hop": true, "three_hop": true}
```

## Why this is different from top-k RAG

The fact graph is built from **all lines**, not from a retrieved window.  A
question such as "which category has the largest total" or "how many distinct
subjects own something" is executed as SQL:

```sql
SELECT category, sum(value) FROM metrics GROUP BY category;
SELECT count(DISTINCT subject) FROM relations WHERE relation='owns';
SELECT r2.object FROM relations r1
JOIN relations r2 ON r1.object = r2.subject
WHERE r1.subject='e0001' AND r1.relation='owns' AND r2.relation='located_in';
```

The multi-hop query uses an arbitrary relation walk, not a fixed two-hop
special case:

```python
graph.walk("e0001", ["owns", "located_in", "owns"])  # -> ["e0004"]
```

The model is not asked to infer a global count from a sample.  This removes
evidence loss for the supported structured-global families.

## Integration

`global_answer()` tries `answer_reasoning()` before the simple count/compare
classifiers.  The full-KV and hierarchical services both use this path, so 3M,
4M and 10M get the same global fact operations.

Code:

- `deepinfer/retrieval/global_reasoning.py`
- `deepinfer/retrieval/global_queries.py`
- `scripts/global_reasoning_10m.py`

Full result: [`global_reasoning10m_results.json`](global_reasoning10m_results.json)

SHA-256:

```text
a7f1344a00798db03195e064ceda6f926bf6e1b928f08c3f0d727c38ff75a05f
```

## Limits

- The graph covers the explicit `GLOBAL METRIC`, `ENTITY` and `RELATION`
  formats.  Arbitrary unstructured semantic global reasoning still needs a
  map-reduce extraction pass.
- Open-ended questions that require interpretation across unlabeled prose are
  not equivalent to full 10M attention.
- For the structured-global families above, there is no top-k evidence loss:
  aggregation, filtering, distinct counts, ranking and two-hop joins are exact.
