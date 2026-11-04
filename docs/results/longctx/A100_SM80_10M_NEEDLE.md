# 10M-token needle-in-a-haystack: 100/100

Date: 2026-11-04

## Design

- Haystack: Python standard library, exactly **10,000,000 tokens**
- 100 unique needles, each a different 7-digit value
- Needle template:
  ```text
  The special magic number for needle NNN is: <7 digits>.
  ```
- Depths are spread across the whole document, from **1% to 99%**
- One question per needle:
  ```text
  What is the special magic number for needle NNN?
  ```
- Production retrieval path: FTS5 top-8 chunks + exact fact extraction
- An answer counts only if the needle line was actually retrieved **and** its
  value was extracted correctly from those retrieved chunks

## Corpus/index

```text
tokens:        10,000,000
text:          33,835,320 bytes
chunks:        17,133
index:         45,129,728 bytes (45.1 MB)
build:         39.29 s (document + tokenizer)
index build:   0.904 s
```

## Result

```text
needles:        100
correct:        100
retrieval hit:  100/100
top_k:          8
retrieval:      0.405 s for all 100 questions
answer extract: 0.008 s for all 100 questions
```

Every needle was retrieved in the top 8 chunks and every answer was exactly the
inserted value.  The executable check is:

```python
correct = retrieved and answer == item["value"]
```

No misses were reported; the script raises a non-zero exit status if any needle
fails.

Full result: [`needle10m_results.json`](needle10m_results.json)

SHA-256:

```text
91325aa869528ad0fd960c39498b813ff284abe19932f40b7d1fb7eb126e10fb
```

## Reproduce

```bash
python scripts/needle_10m.py \
  --tokens 10000000 --needles 100 --seed 7 --top-k 8 \
  --out-dir /root/di_bench/needle10m \
  --out /root/di_bench/needle10m_results.json
```

## Scope

This is the production **structured-fact fast path** over the 10M hierarchical
retriever.  It does not use a 10M full-KV attention pass; that is physically
impossible on this single A100 under the 80GB GPU / 16GiB host limit.  The same
10M index/model fallback was separately validated on the 21-question suite at
21/21 and 31/31.
