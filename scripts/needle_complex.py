"""Complex needle-in-a-haystack (RULER-style) over one long document.

Every needle is in the document at once - one prefill, then all questions branch
off the same cache - so each needle is a distractor for the others:

* MK  multi-key: 16 facts of one template, "The special magic number for
      <key> is: <7 digits>.", keys sharing words (crimson falcon / crimson otter /
      silver falcon ...); each key is asked.
* MV  multi-value: one key holds four numbers at four depths ("One of the special
      magic numbers for golden viper is ..."), next to near-miss keys (golden
      walrus, azure viper); asked for all four.
* MQ  multi-query: one question asks for four of the MK keys at once.
* VT  variable tracking: "VAR QPZRT = 58213", "VAR MLKWE = VAR QPZRT", ... five
      links in shuffled order plus a four-link distractor chain with another
      value; asked for every variable that ends up holding the value.
* 2H  two-hop: "The vault key of the Orrin observatory is held by Kaila Brenn."
      and, elsewhere, "The employee badge number of Kaila Brenn is ...", next to
      the badge numbers of two other people; asked for the badge number of the
      key holder.  Two such questions.

21 questions, 31 expected items.  Single-answer questions must return exactly
the expected code; list questions must contain every expected item (extras and
distractor-chain names are reported).  Answers come from locate + re-read over
the latent KV library (``--mode single``) or with answer feedback (``--mode
feedback``: each round's answer joins the question for the next locate).  Blocks
are scored by the sharpest attention row minus a neutral prompt's (``--score``).

    python scripts/needle_complex.py --tokens 2097152 --positions dca --kv fp8 \\
        --encode-window 98304 --out docs/results/longctx/complex_2m.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import string
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from needle_longctx import (
    MODELS,
    document_hash,
    haystack_tokens,
    header_ids,
    line_start,
)

from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_mla import DualChunkPositions, LatentKVCache
from deepinfer.moe.latent_reread import (
    Refine,
    feedback_locate_and_reread,
    final_answer,
    locate_and_reread,
    neutral_baseline,
    reread,
)

_GIB = 2**30
MK_KEYS = [
    "crimson falcon", "crimson otter", "silver falcon", "silver otter",
    "amber lynx", "amber heron", "cobalt lynx", "cobalt heron",
    "ivory mantis", "ivory panther", "jade mantis", "jade panther",
    "scarlet ibis", "scarlet badger", "onyx ibis", "onyx badger",
]
MV_KEY = "golden viper"
MV_NEAR = ["golden walrus", "azure viper"]
MQ_KEYS = ["crimson falcon", "amber heron", "jade mantis", "onyx badger"]
TWO_HOP = [
    {
        "bridge": "The vault key of the Orrin observatory is held by {person}.",
        "target": "The employee badge number of {person} is {code}.",
        "question": "What is the employee badge number of the person who holds the vault key of the Orrin observatory?",
        "person": "Kaila Brenn",
        "others": ["Talia Rusk", "Joren Vale"],
    },
    {
        "bridge": "The Teal Harbor lighthouse was designed by the architect {person}.",
        "target": "The personal safe of {person} opens with the combination {code}.",
        "question": "Which combination opens the personal safe of the architect who designed the Teal Harbor lighthouse?",
        "person": "Marek Ostrov",
        "others": ["Ilse Varga", "Tomas Reed"],
    },
]
# One instruction for every question: copy the evidence, reason, then answer.  Copying
# makes the model match the exact key / name before it commits to a number; reasoning
# step by step is what lets it follow a chain (probe: asked directly it stops after one
# hop, traced step by step it gets all five); the answer feeds the next round's locate.
QUOTE_FIRST = (
    "First copy, word for word, every sentence or line of the text above that is needed to answer "
    "the question. Then reason step by step. Finally write 'Answer:' followed by {what}."
)
CODE_ONLY = QUOTE_FIRST.format(what="the 7-digit number only")


def build_suite(seed: int) -> tuple[list[dict], list[dict]]:
    """(needles, questions); needles carry their text, questions their expected items."""

    rng = random.Random(seed)
    used: set[str] = set()

    def code() -> str:
        while True:
            value = str(rng.randrange(1_000_000, 10_000_000))
            if value not in used:
                used.add(value)
                return value

    def name() -> str:
        while True:
            value = "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
            if value not in used:
                used.add(value)
                return value

    needles: list[dict] = []
    questions: list[dict] = []

    def needle(text: str, group: str) -> int:
        needles.append({"text": text, "group": group})
        return len(needles) - 1

    mk = {}
    for key in MK_KEYS:
        value = code()
        mk[key] = (value, needle(f"The special magic number for {key} is: {value}.", "MK"))
    for key in MK_KEYS:
        value, index = mk[key]
        questions.append(
            {"kind": "MK", "question": f"What is the special magic number for {key}?",
             "instruction": CODE_ONLY, "expected": [value], "exact": True, "needles": [index]}
        )

    mv_values, mv_needles = [], []
    for _ in range(4):
        value = code()
        mv_values.append(value)
        mv_needles.append(needle(f"One of the special magic numbers for {MV_KEY} is: {value}.", "MV"))
    for key in MV_NEAR:
        needle(f"One of the special magic numbers for {key} is: {code()}.", "MV-near")
    questions.append(
        {"kind": "MV", "question": f"What are all the special magic numbers for {MV_KEY}?",
         "instruction": QUOTE_FIRST.format(what="all of the numbers, separated by commas"),
         "expected": mv_values, "exact": False, "needles": mv_needles}
    )

    questions.append(
        {"kind": "MQ",
         "question": "What are the special magic numbers for " + ", ".join(MQ_KEYS[:-1]) + f" and {MQ_KEYS[-1]}?",
         "instruction": QUOTE_FIRST.format(what="one line per key in the form key: number"),
         "expected": [mk[key][0] for key in MQ_KEYS], "exact": False,
         "needles": [mk[key][1] for key in MQ_KEYS]}
    )

    def chain(links: int, group: str) -> tuple[str, list[str], list[int]]:
        value = str(rng.randrange(10_000, 100_000))
        names = [name() for _ in range(links)]
        indices = [needle(f"VAR {names[0]} = {value}", group)]
        for previous, current in itertools.pairwise(names):
            indices.append(needle(f"VAR {current} = VAR {previous}", group))
        return value, names, indices

    value, names, indices = chain(5, "VT")
    _, decoy_names, _ = chain(4, "VT-decoy")
    # Asked to chain backwards ("which variables get the value"), this model drops the
    # directly assigned one or wanders into the decoy chain - on nine clean lines, with
    # no long context involved.  Asked to run the lines forward and keep a table of
    # values (what the question means), its table is right every time; only the last
    # step, picking the rows with the value, drops a name now and then (a different
    # one from run to run, never a decoy: those are computed as the other value).  So
    # the final pick is asked three ways and the names are united (self-consistency).
    # The re-read lines are in document order, which is the order the program runs in.
    # At 2M the re-read lines are mostly real Python about assignments and chains (the
    # haystack is the standard library), and the model starts explaining that code; so
    # the instruction says what the text is, as RULER's VT template does ("track the
    # chain(s) of variable assignment hidden in the following text").
    simulate = (
        "The text above is unrelated program code with a few lines of the form VAR X = ... mixed in; only those "
        "VAR lines matter, ignore everything else. Go through the VAR lines in order and keep track of every "
        "variable's value: a line VAR X = <number> sets X to that number, and a line VAR B = VAR A sets B to the "
        "current value of A. "
    )
    questions.append(
        {"kind": "VT",
         "question": f"Find all variables that are assigned the value {value}, directly or through a chain of "
                     "assignments (VAR B = VAR A gives B the value of A).",
         "instruction": simulate + f"After the last line, list every variable whose value is {value}. Finally write "
                        "'Answer:' followed by their names, separated by commas.",
         "variants": [
             simulate + "After the last line, write out the table of every variable and its value, one row per "
                        f"variable, then copy every row whose value is {value}. Finally write 'Answer:' followed by "
                        "the names in the copied rows, separated by commas.",
             simulate + f"Each time a variable gets the value {value}, write it down at once. Finally write "
                        "'Answer:' followed by every variable you wrote down, separated by commas.",
         ],
         "expected": names, "forbidden": decoy_names, "exact": False, "needles": indices}
    )

    for item in TWO_HOP:
        value = code()
        bridge = needle(item["bridge"].format(person=item["person"]), "2H")
        target = needle(item["target"].format(person=item["person"], code=value), "2H")
        for other in item["others"]:
            needle(item["target"].format(person=other, code=code()), "2H-decoy")
        questions.append(
            {"kind": "2H", "question": item["question"], "instruction": CODE_ONLY,
             "expected": [value], "exact": True, "needles": [bridge, target]}
        )

    # spread the needles over the document; the shuffle puts keys, values, bridges
    # and targets in arbitrary order and far apart
    depths = [0.02 + 0.96 * i / (len(needles) - 1) for i in range(len(needles))]
    rng.shuffle(depths)
    for item, depth in zip(needles, depths, strict=True):
        item["depth"] = round(depth, 4)
    # a chain is a program: each link comes after the one it reads (at random,
    # spread-out depths), so "assigned through a chain" has one meaning
    for group in ("VT", "VT-decoy"):
        links = [item for item in needles if item["group"] == group]
        for item, depth in zip(links, sorted(item["depth"] for item in links), strict=True):
            item["depth"] = depth
    return needles, questions


def build_document(tok, target: int, needles: list[dict], root: str) -> list[int]:
    """``[BOS] User: <haystack with every needle on its own line>``, exactly ``target`` tokens."""

    header = header_ids(tok)
    ids = [tok.encode(f"\n{item['text']}\n", add_special_tokens=False) for item in needles]
    body_length = target - len(header) - sum(len(x) for x in ids)
    body = haystack_tokens(tok, root, body_length)
    pieces = tok.convert_ids_to_tokens(body)
    cuts = sorted(
        ((line_start(pieces, item["depth"], body_length), item, x) for item, x in zip(needles, ids, strict=True)),
        key=lambda cut: cut[0],
    )
    document = list(header)
    previous = 0
    for cut, item, x in cuts:
        document.extend(body[previous:cut])
        item["position"] = len(document)
        item["length"] = len(x)
        document.extend(x)
        previous = cut
    document.extend(body[previous:body_length])
    return document


def score(question: dict, answer: str) -> dict:
    """Score the final answer - the text after the last ``Answer:`` - not the copied sentences."""

    answer = final_answer(answer)
    if question["kind"] == "VT":
        found = set(re.findall(r"\b[A-Z]{5}\b", answer))
    else:
        found = set(re.findall(r"(?<!\d)\d{7}(?!\d)", answer))
    expected = question["expected"]
    hits = [item for item in expected if item in found]
    extras = sorted(found - set(expected))
    forbidden = sorted(found & set(question.get("forbidden", [])))
    full = len(hits) == len(expected) and (not question["exact"] or not extras) and not forbidden
    return {"hits": hits, "extras": extras, "forbidden": forbidden, "full": full}


NEUTRAL_PROMPT = "\n\nSummarize the text above in one sentence.\n\nAssistant:"


def rank_report(
    session, tok, questions: list[dict], needles: list[dict], *, sink: int = 64, near_window: int = 98304
) -> dict:
    """Rank of each question's needles among all blocks under several locate scorings.

    ``*-far`` ranks only among blocks before the last ``near_window`` tokens (what a
    near-region quota leaves for the rest of the sequence)."""

    from deepinfer.moe.latent_reread import BLOCK, contrast, locate

    neutral = tok.encode(NEUTRAL_PROMPT, add_special_tokens=False)
    base = {reduce: locate(session, neutral, reduce=reduce) for reduce in ("sum", "max")}
    rows = []
    for question in questions:
        ids = tok.encode(f"\n\n{question['question']} {question['instruction']}\n\nAssistant:", add_special_tokens=False)
        mass = {reduce: locate(session, ids, reduce=reduce) for reduce in ("sum", "max")}
        variants = {
            "sum": mass["sum"], "max": mass["max"],
            "sum-contrast": contrast(mass["sum"], base["sum"]),
            "max-contrast": contrast(mass["max"], base["max"]),
            "max-ratio": contrast(mass["max"], base["max"], "ratio"),
        }
        far = max(0, session.length - near_window) // BLOCK
        for name in ("max-contrast", "max-ratio"):
            masked = variants[name].clone()
            masked[far:] = float("-inf")
            variants[name + "-far"] = masked
        ranks = {}
        for name, scores in variants.items():
            scores = scores.clone()
            scores[: sink // BLOCK] = float("-inf")
            order = torch.argsort(scores, descending=True)
            rank = torch.empty_like(order)
            rank[order] = torch.arange(order.numel(), device=order.device)
            ranks[name] = [
                min(
                    int(rank[b])
                    for b in range(
                        needles[i]["position"] // BLOCK,
                        (needles[i]["position"] + needles[i]["length"] - 1) // BLOCK + 1,
                    )
                )
                for i in question["needles"]
            ]
        rows.append({"kind": question["kind"], "question": question["question"], "ranks": ranks})
        print(f"  {question['kind']:<2} {question['question'][:58]:58s} " + "  ".join(f"{k} {v}" for k, v in ranks.items()), flush=True)
    summary = {}
    for name in ("sum", "max", "sum-contrast", "max-contrast", "max-ratio", "max-contrast-far", "max-ratio-far"):
        flat = [r for row in rows for r in row["ranks"][name]]
        summary[name] = {f"top{k}": f"{sum(r < k for r in flat)}/{len(flat)}" for k in (32, 64, 128)}
        print(f"  {name:<13} needles within " + ", ".join(f"{k} {v}" for k, v in summary[name].items()), flush=True)
    return {"rows": rows, "summary": summary}


def answer_questions(session, tok, questions: list[dict], needles: list[dict], args, *, baseline=None):
    """Answer ``questions`` against a ready session; ``args`` carries the answering options of the CLI.

    Returns ``(results, summary)``.  Also used on a session kept alive by
    ``scripts/longctx_workbench.py``, where a SimpleNamespace stands in for ``args``.
    """

    header = header_ids(tok)
    separator = tok.encode("\n...\n", add_special_tokens=False)
    stop = [tok.eos_token_id]
    reduce = args.score.split("-")[0]
    contrast_mode = "ratio" if args.score.endswith("ratio") else "difference"
    if baseline is None and args.score.endswith(("contrast", "ratio")):
        baseline = neutral_baseline(session, tok.encode(NEUTRAL_PROMPT, add_special_tokens=False), reduce=reduce)
    modes = ["single", "feedback"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        for number, question in enumerate(questions):
            started = time.perf_counter()
            bare = tok.encode(f"\n\n{question['question']}\n\nAssistant:", add_special_tokens=False)
            refine = None
            if args.refine:
                refine = Refine(
                    tok, bare, tok.encode(NEUTRAL_PROMPT, add_special_tokens=False), keep=args.refine, order=args.order
                )
            if mode == "single":
                q_ids = tok.encode(
                    f"\n\n{question['question']} {question['instruction']}\n\nAssistant:", add_special_tokens=False
                )
                produced, spans = locate_and_reread(
                    session, q_ids, header_ids=header, separator_ids=separator, top=args.top,
                    neighbors=args.neighbors, max_new_tokens=args.max_new, stop_ids=stop,
                    reduce=reduce, baseline=baseline, locate_ids=bare, refine=refine,
                )
                detail = {"spans": spans}
            else:
                detail = feedback_locate_and_reread(
                    session, tok, question["question"], answer_instruction=question["instruction"],
                    header_ids=header, separator_ids=separator, baseline=baseline, reduce=reduce,
                    top=args.top, follow_top=args.follow_top, neighbors=args.neighbors,
                    rounds=args.rounds, max_new_tokens=args.max_new, stop_ids=stop, refine=refine,
                    contrast_mode=contrast_mode, near_window=args.near_window, follow_final=args.follow_final,
                    final_with_question=getattr(args, "final_with_question", False),
                    final_entity_rows=getattr(args, "final_entity_rows", False),
                )
                produced, spans = detail.pop("answer_ids"), detail["spans"]
                variants = []
                for instruction in question.get("variants", []):
                    # the same converged context, the final pick asked another way
                    extra = reread(
                        session, spans,
                        tok.encode(f"\n\n{question['question']} {instruction}\n\nAssistant:", add_special_tokens=False),
                        header_ids=header, separator_ids=separator, max_new_tokens=args.max_new, stop_ids=stop,
                        refine=refine, refine_query=detail["refine_query"],
                    )
                    variants.append(tok.decode(extra, skip_special_tokens=True).strip())
                if variants:
                    detail["variant_answers"] = variants
            seconds = time.perf_counter() - started
            answer = tok.decode(produced, skip_special_tokens=True).strip()
            if detail.get("variant_answers"):
                # union of the names the three phrasings pick
                picked = set()
                for text in [answer, *detail["variant_answers"]]:
                    picked |= set(re.findall(r"\b[A-Z]{5}\b", final_answer(text)))
                answer = answer + "\n\n[union of 3 phrasings]\nAnswer: " + ", ".join(sorted(picked))
            result = score(question, answer)
            located = [
                any(lo < needles[i]["position"] + needles[i]["length"] and needles[i]["position"] < hi for lo, hi in spans)
                for i in question["needles"]
            ]
            record = {
                "mode": mode, "kind": question["kind"], "question": question["question"],
                "expected": question["expected"], "answer": answer, **result,
                "located": located, "seconds": round(seconds, 1),
                "rounds": len(detail.get("rounds", [])) or None,
                "trace": detail.get("rounds"), "variant_answers": detail.get("variant_answers"),
                "reread_tokens": detail.get("reread_tokens", sum(hi - lo for lo, hi in spans)),
            }
            results.append(record)
            print(
                f"[{mode}] {number + 1:>2} {question['kind']:<2} {'FULL' if result['full'] else 'MISS'} "
                f"{len(result['hits'])}/{len(question['expected'])}"
                + (f" extras {result['extras']}" if result["extras"] else "")
                + (f" DECOY {result['forbidden']}" if result["forbidden"] else "")
                + f" located {sum(located)}/{len(located)}"
                + (f" rounds {record['rounds']}" if record["rounds"] else "")
                + f" ({seconds:.0f} s) | {question['question'][:60]} -> {answer[:120]!r}",
                flush=True,
            )
    summary = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {
            "questions_full": f"{sum(r['full'] for r in rows)}/{len(rows)}",
            "items": f"{sum(len(r['hits']) for r in rows)}/{sum(len(r['expected']) for r in rows)}",
            "by_kind": {
                kind: f"{sum(r['full'] for r in rows if r['kind'] == kind)}/{sum(1 for r in rows if r['kind'] == kind)}"
                for kind in dict.fromkeys(r["kind"] for r in rows)
            },
        }
        print(f"[{mode}] questions fully right {summary[mode]['questions_full']}, items {summary[mode]['items']}, "
              f"by kind {summary[mode]['by_kind']}", flush=True)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="coder")
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--tokens", type=int, default=262144)
    parser.add_argument("--positions", choices=("plain", "dca"), default="dca")
    parser.add_argument("--far", type=int, default=-1)
    parser.add_argument("--kv", choices=("fp16", "fp8"), default="fp8")
    parser.add_argument("--encode-window", type=int, default=98304)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--tile", type=int, default=32768)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--unpin", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=("feedback", "single", "both"), default="feedback")
    parser.add_argument(
        "--score", choices=("sum", "max", "sum-contrast", "max-contrast", "max-ratio"), default="max-contrast",
        help="block scoring for locate: mass summed or sharpest row, optionally minus a neutral prompt's",
    )
    parser.add_argument("--top", type=int, default=32, help="blocks re-read in the first round")
    parser.add_argument("--follow-top", type=int, default=8, help="new blocks per later round")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument(
        "--follow-final", type=int, default=32,
        help="later rounds also read the N best blocks for the final answer alone (0 = off)",
    )
    parser.add_argument("--final-with-question", action="store_true", help="that locate also carries the question")
    parser.add_argument("--final-entity-rows", action="store_true", help="score it with the answer's tokens only")
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--order", choices=("score", "document"), default="document", help="order of the refined lines")
    parser.add_argument(
        "--refine", type=int, default=12,
        help="keep only the N best lines of the re-read text (line-level locate in the short context); 0 = off",
    )
    parser.add_argument("--kinds", default="MK,MV,MQ,VT,2H")
    parser.add_argument("--max-new", type=int, default=1024, help="room for copied lines and step-by-step reasoning")
    parser.add_argument(
        "--near-window", type=int, default=0,
        help="give the last N tokens a share of the located blocks in proportion to their length (0 = off)",
    )
    parser.add_argument("--save-cache", default="", help="after prefilling, save the latent cache here")
    parser.add_argument(
        "--save-spill", default="",
        help="comma-separated directories (other disks) for the layers that do not fit in --save-cache",
    )
    parser.add_argument("--load-cache", default="", help="load a cache saved by --save-cache instead of prefilling")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--diagnose", action="store_true", help="rank needles under each locate scoring, then stop")
    parser.add_argument("--then-answer", action="store_true", help="with --diagnose: answer the questions afterwards")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gguf, meta, trained = MODELS[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.meta_root, meta), trust_remote_code=True)
    needles, questions = build_suite(args.seed)
    kinds = set(args.kinds.split(","))
    questions = [q for q in questions if q["kind"] in kinds]
    started = time.perf_counter()
    document = build_document(tok, args.tokens, needles, args.haystack_root)
    print(
        f"document {len(document):,} tokens with {len(needles)} needles ({time.perf_counter() - started:.0f} s); "
        f"{len(questions)} questions, {sum(len(q['expected']) for q in questions)} expected items",
        flush=True,
    )
    per_token = LatentKVCache.bytes_per_token(27, args.kv)
    capacity = len(document) + 2048  # questions and fed-back answers are written past the document while located
    if args.plan_only:
        for item in sorted(needles, key=lambda n: n["position"]):
            print(f"  {item['position']:>9,} {item['group']:<8} {item['text']}")
        return

    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf), meta_dir=os.path.join(args.meta_root, meta),
        device="cuda:0", verbose=False,
    )
    free = torch.cuda.mem_get_info()[0]
    unpin = args.unpin == "yes" or (args.unpin == "auto" and capacity * per_token + args.workspace_gib * _GIB > free)
    if unpin and rt.expert_store is not None:
        rt.expert_store.clear_gpu_cache()
        rt.expert_store.set_gpu_cache_active(False)
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
    gpu_tokens = min(capacity, int(max(0, free - args.workspace_gib * _GIB) // per_token))
    print(
        f"memory: kv {args.kv} {capacity * per_token / _GIB:.2f} GiB; experts {'streamed' if unpin else 'pinned'}; "
        f"gpu holds {gpu_tokens:,} tokens, host {capacity - gpu_tokens:,}",
        flush=True,
    )
    positions = (
        DualChunkPositions.plain() if args.positions == "plain"
        else DualChunkPositions(trained, trained * 3 // 4, args.far)
    )
    # what a saved cache must have been built with to stand in for this run's prefill
    built = {
        "model": args.model, "document_sha256": document_hash(document), "tokens": len(document),
        "encode_window": args.encode_window or None, "trained": positions.trained, "chunk": positions.chunk,
    }
    prefill_seconds = load_seconds = save_seconds = None
    if args.load_cache:
        started = time.perf_counter()
        session, saved = DeepSeekLongContext.load(
            rt, args.load_cache, capacity=capacity, gpu_tokens=gpu_tokens, far=positions.far,
            prefill_chunk=args.chunk, tile_tokens=args.tile, encode_window=args.encode_window or None,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        mismatched = {key: (saved.get(key), value) for key, value in built.items() if saved.get(key) != value}
        if mismatched or session.cache.kv_dtype != args.kv:
            raise SystemExit(f"the saved cache was built differently (saved, wanted): {mismatched}")
        print(f"loaded {session.length:,}-token cache in {load_seconds:.1f} s (prefill skipped)", flush=True)
    else:
        session = DeepSeekLongContext(
            rt, capacity=capacity, gpu_tokens=gpu_tokens, kv_dtype=args.kv, positions=positions,
            prefill_chunk=args.chunk, tile_tokens=args.tile, encode_window=args.encode_window or None,
        )
        last = [0.0]

        def progress(done: int, total: int, elapsed: float) -> None:
            if elapsed - last[0] >= 120 or done == total:
                last[0] = elapsed
                print(f"  prefill {done:,}/{total:,} tokens, {elapsed / 60:.1f} min", flush=True)

        started = time.perf_counter()
        session.prefill(document, progress=progress)
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - started
        print(f"prefill {session.length:,} tokens in {prefill_seconds / 60:.1f} min", flush=True)
        if args.save_cache:
            started = time.perf_counter()
            spill = [folder for folder in args.save_spill.split(",") if folder]
            session.save(args.save_cache, meta=built, spill=spill)
            save_seconds = time.perf_counter() - started
            print(f"saved the cache to {[args.save_cache, *spill]} in {save_seconds:.0f} s", flush=True)
    if args.diagnose:
        report = rank_report(session, tok, questions, needles, near_window=args.encode_window or 98304)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            diagnose_out = args.out.replace(".json", "_diagnose.json")
            with open(diagnose_out, "w", encoding="utf-8") as handle:
                json.dump({"tokens": len(document), "seed": args.seed, **report}, handle, indent=1)
            print(f"wrote {diagnose_out}", flush=True)
        if not args.then_answer:
            return

    results, summary = answer_questions(session, tok, questions, needles, args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": args.model, "tokens": len(document), "document_sha256": document_hash(document),
                    "positions": args.positions, "trained": trained if args.positions == "dca" else None,
                    "kv": args.kv, "encode_window": args.encode_window or None, "gpu_tokens": gpu_tokens,
                    "experts": "streamed" if unpin else "pinned",
                    "prefill_seconds": None if prefill_seconds is None else round(prefill_seconds, 1),
                    "load_seconds": None if load_seconds is None else round(load_seconds, 1),
                    "save_seconds": None if save_seconds is None else round(save_seconds, 1),
                    "score": args.score, "near_window": args.near_window, "refine": args.refine, "order": args.order, "top": args.top, "follow_top": args.follow_top, "follow_final": args.follow_final, "rounds": args.rounds,
                    "neighbors": args.neighbors, "seed": args.seed, "summary": summary,
                    "needles": needles, "results": results,
                },
                handle, indent=1, ensure_ascii=False,
            )
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
