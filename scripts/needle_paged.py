"""Recall through the paged serve path, checked against the dense reference.

This is the experiment in docs/HANDOFF.md section 5 with its harness bug fixed.
The first version (/tmp/needle.py on the 4080 SUPER host) allocated fresh blocks
for every prefill chunk and again for the question, so each view's block table
held only the current chunk and the model never saw more than the last 512
tokens.  Here a sequence owns one block list sized for everything it will hold,
and every prefill chunk and decode step passes that list with its true past
length.

Configurations are ``<kv>:<attention>``: kv is ``dense`` (the model's own
attention over a transformers DynamicCache - an independent implementation, the
reference), ``fp16`` or ``int8`` (paged pool, native paged MLA kernels);
attention is ``full`` or ``w<window>s<sink>``.

Tasks:

* ``each`` (default): five distinct facts ("The codeword for the east gate is
  MORROW.") at 10/30/50/70/90 % depth; the document is prefilled once and each
  question branches off it, so every depth gets its own single-answer question.
* ``list``: the first harness's task - "The secret codeword is X." five times and
  one question asking for all of them.

Self-checks (the run is only evidence if both pass):

* chunking - NLL and last logits of a 2K prefix must not depend on the prefill
  chunk size; the old per-chunk allocation is run on the same prefix to show
  what failing looks like;
* reference - paged fp16 full attention must agree with dense on the document
  (mean NLL within 0.02 nats, top-1 agreement at least 0.97 - the same tolerance
  the chunking check shows for fp16 + MoE routing noise).

Recall under full attention is reported separately: if dense itself misses a
fact, that is the model at this length, not the harness.

    python scripts/needle_paged.py --configs dense:full,fp16:full,int8:full,int8:w1024s4,int8:w4096s4 \\
        --out docs/results/needle_paged/8k_each.json
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
import time

import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepinfer.engine.kv_cache import PagedKVCacheManager
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime

BLOCK = 16
DEPTHS = (0.10, 0.30, 0.50, 0.70, 0.90)
GATES = ("north", "east", "south", "west", "central")
WORDS = ("KESTREL", "MORROW", "TALLOW", "VESSEL", "HARROW")
LIST_QUESTION = "List every secret codeword that appeared in the document above, in order. Output only the words."


def facts_for(task: str) -> list[dict]:
    if task == "list":
        return [
            {
                "depth": d,
                "word": w,
                "text": f"\n\nThe secret codeword is {w}.\n\n",
                "question": LIST_QUESTION,
            }
            for d, w in zip(DEPTHS, WORDS, strict=True)
        ]
    return [
        {
            "depth": d,
            "word": w,
            "text": f"\n\nThe codeword for the {g} gate is {w}.\n\n",
            "question": f"What is the codeword for the {g} gate? Answer with the codeword only.",
        }
        for d, g, w in zip(DEPTHS, GATES, WORDS, strict=True)
    ]


def parse_config(spec: str) -> dict:
    kv, _, attention = spec.partition(":")
    if kv not in ("dense", "fp16", "int8"):
        raise SystemExit(f"bad KV in {spec!r}: use dense, fp16 or int8")
    if attention in ("", "full"):
        return {"name": spec, "kv": kv, "window": None, "sink": None}
    match = re.fullmatch(r"w(\d+)s(\d+)", attention)
    if not match or kv == "dense":
        raise SystemExit(
            f"bad attention in {spec!r}: full, or w<window>s<sink> for a paged pool"
        )
    return {
        "name": spec,
        "kv": kv,
        "window": int(match.group(1)),
        "sink": int(match.group(2)),
    }


class PagedSequence:
    """One sequence on its own pool: a single block list that the whole sequence writes into."""

    def __init__(
        self, rt: DeepSeekV2LiteGGUFRuntime, kv: str, capacity: int, device: str
    ) -> None:
        cfg = rt.config
        key_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)
        self.mgr = PagedKVCacheManager(
            num_layers=int(cfg.num_hidden_layers),
            num_kv_heads=int(cfg.num_attention_heads),
            head_dim=max(key_dim, int(cfg.v_head_dim)),
            block_size=BLOCK,
            num_blocks=(capacity + BLOCK - 1) // BLOCK + 2,
            dtype=torch.float16,
            kv_cache_dtype=kv,
            device=device,
        )
        self.blocks = self.mgr.allocate_tokens(capacity)
        self.capacity = len(self.blocks) * BLOCK
        self.rt = rt
        self.device = device
        self.length = 0

    def memory_gib(self) -> float:
        return self.mgr.memory_bytes / 2**30

    def forward(self, ids: list[int]) -> torch.Tensor:
        n = len(ids)
        if self.length + n > self.capacity:
            raise ValueError(f"sequence capacity {self.capacity} exceeded")
        views = self.mgr.make_views(
            [self.blocks], [self.length], max_length=self.capacity, new_tokens=n
        )
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        pos = torch.arange(self.length, self.length + n, device=self.device).unsqueeze(
            0
        )
        logits = self.rt.forward_logits(x, pos, kv_cache=views)
        self.length += n
        return logits[0]

    def truncate(self, length: int) -> None:
        self.length = length  # later writes overwrite the slots; lengths mask the rest

    def release(self) -> None:
        del self.mgr
        torch.cuda.empty_cache()


class DenseSequence:
    """The reference: the model's own attention over a DynamicCache, no paged code involved."""

    def __init__(self, rt: DeepSeekV2LiteGGUFRuntime, device: str) -> None:
        self.rt = rt
        self.device = device
        self.cache = DynamicCache()
        self.length = 0

    def memory_gib(self) -> float:
        return 0.0

    @torch.inference_mode()
    def forward(self, ids: list[int]) -> torch.Tensor:
        n = len(ids)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.rt.model.model(
            inputs_embeds=self.rt._embed_tokens(x),
            attention_mask=torch.ones(
                (1, self.length + n), dtype=torch.long, device=self.device
            ),
            position_ids=torch.arange(
                self.length, self.length + n, device=self.device
            ).unsqueeze(0),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        self.cache = out.past_key_values
        self.length += n
        return self.rt._all_logits(out.last_hidden_state)[0]

    def truncate(self, length: int) -> None:
        self.cache.crop(length)
        self.length = length

    def release(self) -> None:
        self.cache = None
        torch.cuda.empty_cache()


def prefill(seq, ids: list[int], chunk: int):
    """Feed ids in chunks.  Returns seconds, per-position NLL of the next token, per-position argmax, last logits."""
    count = len(ids)
    nll = torch.full((count,), float("nan"))
    top1 = torch.full((count,), -1, dtype=torch.long)
    last = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, count, chunk):
        end = min(start + chunk, count)
        logits = seq.forward(ids[start:end])
        stop = min(end, count - 1)  # position i predicts token i + 1
        if stop > start:
            logp = torch.log_softmax(logits[: stop - start].float(), dim=-1)
            target = torch.tensor(ids[start + 1 : stop + 1], device=logp.device)
            nll[start:stop] = -logp.gather(1, target[:, None])[:, 0].cpu()
            top1[start:stop] = logp.argmax(-1).cpu()
            del logp
        last = logits[-1].float().clone()
        del logits
    torch.cuda.synchronize()
    return time.perf_counter() - t0, nll, top1, last


def generate(seq, logits: torch.Tensor, max_new: int, eos: int):
    out: list[int] = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max_new):
        nxt = int(logits.argmax())
        out.append(nxt)
        if nxt == eos:
            break
        logits = seq.forward([nxt])[-1]
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0 / max(1, len(out))


def old_allocation_nll(rt, ids: list[int], chunk: int, device: str) -> float:
    """The first harness's prefill: fresh blocks per chunk, so no chunk sees the ones before it."""
    cfg = rt.config
    key_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)
    n = len(ids)
    mgr = PagedKVCacheManager(
        num_layers=int(cfg.num_hidden_layers),
        num_kv_heads=int(cfg.num_attention_heads),
        head_dim=max(key_dim, int(cfg.v_head_dim)),
        block_size=BLOCK,
        num_blocks=(n + 64) // BLOCK + 2,
        dtype=torch.float16,
        kv_cache_dtype="fp16",
        device=device,
    )
    nll = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        blk = mgr.allocate_tokens(end - start)
        views = mgr.make_views(
            [blk], [start], max_length=n + 64, new_tokens=end - start
        )
        x = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
        pos = torch.arange(start, end, device=device).unsqueeze(0)
        logits = rt.forward_logits(x, pos, kv_cache=views)[0]
        stop = min(end, n - 1)
        if stop > start:
            logp = torch.log_softmax(logits[: stop - start].float(), dim=-1)
            target = torch.tensor(ids[start + 1 : stop + 1], device=logp.device)
            nll.append(-logp.gather(1, target[:, None])[:, 0].cpu())
    del mgr
    torch.cuda.empty_cache()
    return float(torch.cat(nll).mean())


def chat_wrap(tok) -> tuple[list[int], list[int], str]:
    placeholder = "⁣DOC⁣"
    try:
        text = tok.apply_chat_template(
            [{"role": "user", "content": placeholder}],
            tokenize=False,
            add_generation_prompt=True,
        )
        source = "tokenizer chat template"
    except (ValueError, AttributeError):  # no template shipped with the tokenizer files
        text = (tok.bos_token or "") + "User: " + placeholder + "\n\nAssistant:"
        source = "DeepSeek-V2 chat format (fallback)"
    pre, post = text.split(placeholder)
    return (
        tok(pre, add_special_tokens=False)["input_ids"],
        tok(post, add_special_tokens=False)["input_ids"],
        source,
    )


def haystack(tok, corpus: str, target: int) -> list[int]:
    texts = []
    for path in sorted(glob.glob(corpus)):
        with open(path, encoding="utf-8") as fh:
            texts.append(fh.read())
    ids = tok("\n\n".join(texts), add_special_tokens=False)["input_ids"][:target]
    if len(ids) < target:
        raise SystemExit(f"corpus {corpus!r} has only {len(ids)} tokens, need {target}")
    return ids


def insert_facts(tok, ids: list[int], facts: list[dict]) -> list[int]:
    pieces: list[int] = []
    last = 0
    for fact in facts:
        at = int(len(ids) * fact["depth"])
        pieces += ids[last:at]
        fact["token_offset"] = len(pieces)
        pieces += tok(fact["text"], add_special_tokens=False)["input_ids"]
        last = at
    return pieces + ids[last:]


def deciles(nll: torch.Tensor) -> list[float]:
    edges = [int(len(nll) * k / 10) for k in range(11)]
    return [round(float(nll[a:b].nanmean()), 4) for a, b in itertools.pairwise(edges)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="/root/autodl-tmp/models/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
    )
    parser.add_argument("--meta", default="/root/autodl-tmp/meta")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--corpus", default=os.path.join(root, "docs", "*.md"))
    parser.add_argument(
        "--tokens",
        type=int,
        default=8000,
        help="haystack tokens before the facts are inserted",
    )
    parser.add_argument("--task", choices=("each", "list"), default="each")
    parser.add_argument(
        "--configs", default="dense:full,fp16:full,int8:full,int8:w1024s4,int8:w4096s4"
    )
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument(
        "--check-tokens",
        type=int,
        default=2048,
        help="prefix for the chunking self-check (0 = skip)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cache", type=int, default=12288)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    configs = [parse_config(s) for s in args.configs.split(",") if s]

    rt = DeepSeekV2LiteGGUFRuntime(
        args.model,
        meta_dir=args.meta,
        device=args.device,
        max_cache=args.max_cache,
        verbose=False,
    )
    tok = rt.tokenizer
    eos = int(tok.eos_token_id)
    facts = facts_for(args.task)
    doc = insert_facts(tok, haystack(tok, args.corpus, args.tokens), facts)
    pre, post, wrap_source = chat_wrap(tok)
    context = pre + doc  # shared prefix; each question branches off its end
    questions = (
        [facts[0]["question"]]
        if args.task == "list"
        else [f["question"] for f in facts]
    )
    tails = [
        tok("\n\n" + q, add_special_tokens=False)["input_ids"] + post for q in questions
    ]
    capacity = len(context) + max(len(t) for t in tails) + args.max_new + BLOCK
    doc_span = slice(len(pre), len(context) - 1)
    print(
        f"task {args.task}: document {len(doc)} tokens (haystack {args.tokens} + facts), "
        f"context {len(context)}, {len(questions)} question(s), wrap: {wrap_source}",
        flush=True,
    )

    result = {
        "harness": "scripts/needle_paged.py",
        "model": os.path.basename(args.model),
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "task": args.task,
        "haystack_tokens": args.tokens,
        "document_tokens": len(doc),
        "context_tokens": len(context),
        "format": wrap_source,
        "facts": facts,
        "chunk": args.chunk,
        "self_check": {},
        "runs": [],
    }

    if args.check_tokens:
        n = min(args.check_tokens, len(context))
        rt.paged_sliding_window, rt.paged_attention_sink = None, None
        stats = []
        for chunk in (n, args.chunk):
            seq = PagedSequence(rt, "fp16", n + BLOCK, args.device)
            _, nll, top1, last = prefill(seq, context[:n], chunk)
            stats.append((nll, top1, last))
            seq.release()
        (nll_a, top_a, last_a), (nll_b, top_b, last_b) = stats
        mask = ~torch.isnan(nll_a)
        check = {
            "tokens": n,
            "chunks": [n, args.chunk],
            "nll_mean": [
                round(float(nll_a[mask].mean()), 4),
                round(float(nll_b[mask].mean()), 4),
            ],
            "top1_agreement": round(
                float((top_a[mask] == top_b[mask]).float().mean()), 4
            ),
            "last_logits_max_abs_diff": round(float((last_a - last_b).abs().max()), 4),
            "last_argmax_equal": bool(int(last_a.argmax()) == int(last_b.argmax())),
            "old_allocation_nll_mean": round(
                old_allocation_nll(rt, context[:n], args.chunk, args.device), 4
            ),
        }
        check["ok"] = bool(
            check["last_argmax_equal"]
            and check["top1_agreement"] >= 0.97
            and abs(check["nll_mean"][0] - check["nll_mean"][1]) < 0.02
        )
        result["self_check"]["chunking"] = check
        print(
            f"self-check chunking: {'PASS' if check['ok'] else 'FAIL'} {check}",
            flush=True,
        )

    reference = None
    for cfg in configs:
        rt.paged_sliding_window, rt.paged_attention_sink = cfg["window"], cfg["sink"]
        if cfg["kv"] == "dense":
            seq = DenseSequence(rt, args.device)
        else:
            seq = PagedSequence(rt, cfg["kv"], capacity, args.device)
        seconds, nll, top1, _ = prefill(seq, context, args.chunk)
        answers = []
        for q, tail in zip(questions, tails, strict=True):
            seq.truncate(len(context))
            t0 = time.perf_counter()
            logits = seq.forward(tail)[-1].float()
            tail_s = time.perf_counter() - t0
            out, decode_ms = generate(seq, logits, args.max_new, eos)
            answers.append(
                {
                    "question": q,
                    "answer": tok.decode(out, skip_special_tokens=True),
                    "tail_prefill_s": round(tail_s, 3),
                    "decode_ms_per_token": round(decode_ms, 1),
                }
            )
        pool = seq.memory_gib()
        seq.release()
        if args.task == "list":
            text = answers[0]["answer"].lower()
            hits = [f["word"] for f in facts if f["word"].lower() in text]
        else:
            hits = [
                f["word"]
                for f, a in zip(facts, answers, strict=True)
                if f["word"].lower() in a["answer"].lower()
            ]
            for f, a in zip(facts, answers, strict=True):
                a["depth"], a["expected"], a["correct"] = (
                    f["depth"],
                    f["word"],
                    f["word"] in hits,
                )
        run = {
            **cfg,
            "pool_gib": round(pool, 3),
            "prefill_s": round(seconds, 2),
            "answers": answers,
            "found": hits,
            "recall": f"{len(hits)}/{len(facts)}",
            "doc_nll_mean": round(float(nll[doc_span].nanmean()), 4),
            "doc_nll_by_decile": deciles(nll[doc_span]),
            "decode_ms_per_token": round(
                sum(a["decode_ms_per_token"] for a in answers) / len(answers), 1
            ),
        }
        if reference is None:
            reference = (cfg["name"], nll, top1)
        else:
            ref_name, ref_nll, ref_top1 = reference
            run["vs"] = ref_name
            run["doc_nll_delta"] = round(
                float(nll[doc_span].nanmean() - ref_nll[doc_span].nanmean()), 4
            )
            run["doc_top1_agreement"] = round(
                float((top1[doc_span] == ref_top1[doc_span]).float().mean()), 4
            )
        result["runs"].append(run)
        detail = (
            f" (d {run['doc_nll_delta']:+.4f}, top1 {run['doc_top1_agreement']:.4f} vs {run['vs']})"
            if "vs" in run
            else ""
        )
        print(
            f"{cfg['name']:<14} pool {pool:5.2f} GiB  prefill {seconds:6.1f} s  decode {run['decode_ms_per_token']:6.1f} ms/tok  "
            f"recall {run['recall']} {hits}  nll {run['doc_nll_mean']:.3f}{detail}",
            flush=True,
        )
        for a in answers:
            print(
                f"    {a.get('expected', '')!s:<8} -> {a['answer'][:90]!r}", flush=True
            )

    by_name = {r["name"]: r for r in result["runs"]}
    if "dense:full" in by_name and "fp16:full" in by_name:
        paged = by_name["fp16:full"]
        if paged.get("vs") == "dense:full":
            agree = {
                "nll_delta": paged["doc_nll_delta"],
                "top1_agreement": paged["doc_top1_agreement"],
                "same_answers": [a["answer"] for a in paged["answers"]]
                == [a["answer"] for a in by_name["dense:full"]["answers"]],
            }
            agree["ok"] = bool(
                abs(agree["nll_delta"]) < 0.02 and agree["top1_agreement"] >= 0.97
            )
            result["self_check"]["reference"] = agree
            print(
                f"self-check reference (paged fp16 vs dense): {'PASS' if agree['ok'] else 'FAIL'} {agree}",
                flush=True,
            )
    result["harness_ok"] = bool(result["self_check"]) and all(
        v.get("ok", False) for v in result["self_check"].values()
    )
    result["full_attention_recall"] = {
        r["name"]: r["recall"] for r in result["runs"] if r["window"] is None
    }
    print(
        (
            "HARNESS OK"
            if result["harness_ok"]
            else "HARNESS SELF-CHECK FAILED - the numbers below are not evidence"
        )
        + f"; full-attention recall {result['full_attention_recall']}",
        flush=True,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
