# 10M unlabeled prose: map-reduce facts into the reasoning graph

Date: 2026-11-04

The structured fact graph handles explicit `GLOBAL METRIC`, `ENTITY` and
`RELATION` records.  This extension adds a map-reduce prose layer:

```text
map:    split the document into bounded passages; extract subject/relation/object
reduce: normalize relations, normalize numeric attributes, deduplicate triples
merge:  FactGraph.add_triples() into the same SQLite graph used by SQL and walks
query:  same graph query / inverse query / relation walk
```

## Two map backends

- **Rules**: deterministic, full-corpus safe; reads explicit prose patterns such
  as `X owns Y`, `X is located in Y`, `X works for Y`, `X has N employees`.
- **Model**: the local DeepSeek runtime; prompts each passage for JSONL triples.
  This is the offline semantic enrichment backend.

Both backends emit the same triple format and pass through the same reduce and
merge functions.

## 10M full-corpus validation (rules)

```text
tokens:              10,000,000
text:                33,711,891 bytes
prose fact lines:     4,000
map-reduce triples:   4,000
graph relations:      4,000
prose triples:        4,000
document build:       38.40 s
map-reduce extraction: 6.23 s
graph build:           0.33 s
query after merge:     0.001 s each
```

Questions and checks:

| Question | Answer | Check |
|---|---|---|
| Who owns Beta0001? | Acme0001 | true |
| Where is Beta0001 located? | Berlin0001 | true |
| Who works for Acme0001? | Alice0001 | true |
| How many employees does Acme0001 have? | 1201 | true |

All checks are true:

```json
{"owns": true, "located": true, "works": true, "employees": true}
```

The graph cache means extraction happens once; subsequent prose queries reuse
the merged graph.

## Model-backed map phase validation

A short prose passage was extracted by the real local model:

```text
model extraction: 8.60 s
triples:          6
checks:           owns=true, located=true, works_for=true, employees=true
```

The model output included:

```text
Acme Corporation | has_employees | 1200
Acme Corporation | owns          | Beta Labs
Alice Chen       | works_for     | Acme Corporation
Beta Labs        | located_in    | Berlin
```

The reduce phase normalizes model phrasings such as `has 1,200 employees` into
`has_employees=1200`, so the same graph query works for both backends.

## Persisted graph and server integration

The reduce output can be persisted as a SQLite graph:

```bash
python scripts/build_fact_graph.py   --text /root/di_bench/prose10m/haystack.txt   --out /root/di_bench/prose10m/facts.sqlite   --backend rules
```

For the 10M prose corpus this produced:

```text
triples:       4000
graph size:    476 KB
build time:    6.54 s
```

The hierarchical server then loads it directly:

```bash
python -m deepinfer.cli.main server   -m /root/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf   --meta /root/meta_coder -c 10000000 --mode hierarchical   --hierarchical-text /root/di_bench/prose10m/haystack.txt   --hierarchical-index /root/di_bench/prose10m/fts.sqlite   --fact-graph /root/di_bench/prose10m/facts.sqlite   --host 127.0.0.1 --port 6191
```

`GET /status` reported:

```text
mode:         hierarchical
context:       9,999,999 tokens
index chunks:  17,071
fact_triples:  4,000
```

Four prose questions through `POST /ask`:

| Question | Final | Seconds |
|---|---|---|
| Who owns Beta0001? | Acme0001 | 0.086 |
| Where is Beta0001 located? | Berlin0001 | 0.047 |
| Who works for Acme0001? | Alice0001 | 0.043 |
| How many employees does Acme0001 have? | 1201 | 0.049 |

All four returned `global_kind=graph`; no model generation was used for these
answers.

- `global_answer()` tries the fact graph before ordinary count/compare paths.
- Prose questions trigger the rules map-reduce layer only once per cached text.
- The model backend can run offline and its reduced triples are merged through
  the same `FactGraph.add_triples()` path.
- Existing structured metric/entity/relation reasoning and multi-hop walks are
  unchanged and share the same graph.

## Limits

- The full 10M corpus was processed with the deterministic rules backend.  The
  model backend was validated on a representative short passage, not on all 10M
  tokens in this run; full model extraction is an offline batch job.
- Rules only capture explicit patterns.  The model backend is the path for
  implicit prose relations.
- A 10M model map is expensive: with 2K-token passages it is thousands of model
  calls; production should run it offline, use larger legal chunks, and cache
  the reduced graph.

## Artifacts

- `deepinfer/retrieval/fact_extraction.py`
- `deepinfer/retrieval/global_reasoning.py`
- `scripts/build_fact_graph.py`
- `scripts/prose_facts_10m.py`
- `scripts/model_fact_extraction_demo.py`
- [`prose10m_results.json`](prose10m_results.json)
- [`model_fact_extraction.json`](model_fact_extraction.json)

SHA-256:

```text
prose10m:      46c88b69b11fd9c1924dd7092113bf344a9bcc65aeacee3ef565a067c8ecb0e5
model demo:    0b164ba559b98cbdf8f776479757e55465a044734d6a69236ed9ca8de130605b
```
