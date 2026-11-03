"""The answer step alone: the model answering over re-read lines, with and without the checks.

The long-context pipeline ends with the model answering over the ~12 lines the line-level
refine kept.  At 4M two of 21 answers went wrong there with every needle located: a
similar key's number copied, a variable-chain link dropped in the final pick
(docs/LONG_CONTEXT_1M.md §4.12).  This replays answer contexts of that shape and compares

* ``base``   - one answer, as the pipeline gives it (variable chains: three phrasings, united);
* ``quote``  - if the answer miscopies (:func:`deepinfer.moe.answer_check.miscopies`), ask once
  more with a note saying which copy is not in the text, and keep the answer that copies right;
* ``check``  - a second turn in which the model checks its final answer against its own steps;
* ``both``   - ``quote``, then ``check``.

Contexts come from two places: the saved runs (``--replay``: every distinct round context of
docs/results/longctx/complex_*.json, rebuilt from the kept lines) and new suites
(``--synthetic N``: N seeds; for each, contexts built the way the refine leaves them - the
needed lines among other needles of the same suite and a few lines of standard-library code,
in document order).

    python scripts/answer_bench.py --replay docs/results/longctx --synthetic 30 --out answer_bench.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from needle_longctx import MODELS

from deepinfer.longctx_suite import build_suite, header_ids, score
from deepinfer.moe.answer_check import miscopies, retry_note, self_check_prompt
from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_reread import final_answer

METHODS = ("base", "quote", "check", "both")


def code_lines(root: str, count: int, rng: random.Random) -> list[str]:
    """``count`` non-trivial lines of standard-library code, the kind the refine keeps next to the needles."""

    files = sorted(p for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True) if "site-packages" not in p)
    picked: list[str] = []
    while len(picked) < count:
        with open(rng.choice(files), encoding="utf-8", errors="ignore") as handle:
            lines = [line.rstrip() for line in handle if 20 <= len(line.strip()) <= 100]
        if lines:
            picked.append(rng.choice(lines))
    return picked


def synthetic_cases(seeds: list[int], haystack_root: str, keep: int = 12) -> list[dict]:
    """Answer contexts shaped like the refine's output, from fresh suites."""

    cases = []
    for seed in seeds:
        needles, questions = build_suite(seed)
        rng = random.Random(1000 + seed)
        decoys = [n["text"] for n in needles if n["group"] == "VT-decoy"]
        decoy_names = sorted({w for t in decoys for w in re.findall(r"\b[A-Z]{5}\b", t)})
        picks = [q for q in questions if q["kind"] == "MK"]
        chosen = rng.sample(picks, 4) + [q for q in questions if q["kind"] != "MK"]
        for question in chosen:
            needed = list(question["needles"])
            if question["kind"] == "VT":
                needed += [i for i, n in enumerate(needles) if n["group"] == "VT-decoy"]
            others = [i for i in range(len(needles)) if i not in needed and needles[i]["group"] not in ("VT", "VT-decoy")]
            room = max(0, keep - len(needed))
            code_count = min(room, rng.randint(1, 3))
            extra = rng.sample(others, min(len(others), room - code_count))
            # document order: needles by depth, code lines in between at random places
            items = [(needles[i]["depth"], needles[i]["text"]) for i in needed + extra]
            items += [(rng.random(), line) for line in code_lines(haystack_root, code_count, rng)]
            lines = [text for _, text in sorted(items)]
            cases.append({
                "source": f"synthetic seed {seed}", "kind": question["kind"], "question": question["question"],
                "instruction": question["instruction"], "variants": question.get("variants", []),
                "expected": question["expected"], "exact": question["exact"],
                "forbidden": decoy_names if question["kind"] == "VT" else [], "lines": lines,
            })
    return cases


def replay_cases(directory: str, per_kind: int = 40) -> list[dict]:
    """Distinct round contexts of the saved complex-needle runs, with their questions: every
    round whose answer was wrong or miscopied with what it needed in front of it, and of the
    final rounds that were right, at most ``per_kind`` per kind (a fixed random sample)."""

    cases, seen = [], set()
    for path in sorted(glob.glob(os.path.join(directory, "complex_*.json"))):
        with open(path, encoding="utf-8") as handle:
            run = json.load(handle)
        seed = run.get("seed", 11)
        _, questions = build_suite(seed)
        by_text = {q["question"]: q for q in questions}
        decoy_names = sorted({
            w for n in run.get("needles", []) if n.get("group") == "VT-decoy" for w in re.findall(r"\b[A-Z]{5}\b", n["text"])
        })
        for result in run.get("results", []):
            question = by_text.get(result["question"])
            if question is None or question["expected"] != result["expected"]:
                continue
            traces = [t for t in result.get("trace") or [] if isinstance(t, dict) and t.get("lines")]
            for number, trace in enumerate(traces):
                case = {
                    "source": f"{os.path.basename(path)} round {trace['round']}", "kind": result["kind"],
                    "question": question["question"], "instruction": question["instruction"],
                    "variants": question.get("variants", []), "expected": question["expected"],
                    "exact": question["exact"], "forbidden": decoy_names if result["kind"] == "VT" else [],
                    "lines": trace["lines"],
                }
                answer = trace.get("answer", "")
                text = "\n".join(trace["lines"])
                # an earlier round counts when its answer went wrong with everything it needed in front of it
                answerable = all(item in text for item in question["expected"])
                if number < len(traces) - 1 and not miscopies(answer, trace["lines"]) and (
                    score(case, answer)["full"] or not answerable
                ):
                    continue
                key = (result["question"], tuple(trace["lines"]))
                if key not in seen:
                    seen.add(key)
                    case["recorded_right"] = score(case, answer)["full"] and not miscopies(answer, trace["lines"])
                    cases.append(case)
    rng = random.Random(7)
    kept = [case for case in cases if not case["recorded_right"]]
    for kind in sorted({case["kind"] for case in cases}):
        right = [case for case in cases if case["recorded_right"] and case["kind"] == kind]
        kept += right if len(right) <= per_kind else rng.sample(right, per_kind)
    return kept


class Answerer:
    def __init__(self, rt, tok, max_new: int) -> None:
        self.rt, self.tok, self.max_new = rt, tok, max_new
        self.header = header_ids(tok)
        self.stop = tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def generate(self, context: list[int], prompt: list[int]) -> list[int]:
        short = DeepSeekLongContext(self.rt, capacity=len(context) + len(prompt) + self.max_new + 8, kv_dtype="fp16")
        short.prefill(context)
        produced = short.generate(prompt, max_new_tokens=self.max_new, stop_ids=[self.stop])
        del short
        return produced

    def text(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def checked(self, context: list[int], prompt: list[int], ids: list[int]) -> tuple[list[int], str]:
        """The self-check turn on an answer: (the ids to keep, the turn's text)."""

        turn = self.generate(context + prompt, self_check_prompt(self.tok, ids, self.stop))
        text = self.text(turn)
        keep = turn if "answer:" in text.lower() and final_answer(text).strip() else ids
        return keep, text

    def phrasing(self, case: dict, instruction: str, methods: list[str]) -> dict[str, dict]:
        """One phrasing of one case under every method, sharing the generations they have in common."""

        context = self.header + self.encode("\n".join(case["lines"]) + "\n")
        prompt = self.encode(f"\n\n{case['question']} {instruction}\n\nAssistant:")
        first = self.generate(context, prompt)
        found = miscopies(self.text(first), case["lines"])
        out = {"base": {"final": self.text(first)}}
        quoted, quoted_prompt, note = first, prompt, {"miscopies": [item["copy"] for item in found]}
        if found and {"quote", "both"} & set(methods):
            again_prompt = self.encode(f"\n\n{case['question']} {instruction}\n{retry_note(found)}\n\nAssistant:")
            again = self.generate(context, again_prompt)
            note["retry"] = self.text(again)
            if len(miscopies(note["retry"], case["lines"])) < len(found):
                quoted, quoted_prompt = again, again_prompt
        out["quote"] = {"final": self.text(quoted), **note}
        if {"check", "both"} & set(methods):
            kept, text = self.checked(context, prompt, first)
            out["check"] = {"final": self.text(kept), "checked": text}
            if quoted is first:
                out["both"] = {**out["check"], **note}
            else:
                kept, text = self.checked(context, quoted_prompt, quoted)
                out["both"] = {"final": self.text(kept), "checked": text, **note}
        return {method: out[method] for method in methods}

    def run(self, case: dict, methods: list[str]) -> dict[str, dict]:
        phrasings = [self.phrasing(case, case["instruction"], methods)]
        phrasings += [self.phrasing(case, variant, methods) for variant in case["variants"]]
        results = {}
        for method in methods:
            answer = phrasings[0][method]["final"]
            if len(phrasings) > 1:
                picked = set()
                for record in phrasings:
                    picked |= set(re.findall(r"\b[A-Z]{5}\b", final_answer(record[method]["final"])))
                answer = answer + "\n\n[union of 3 phrasings]\nAnswer: " + ", ".join(sorted(picked))
            results[method] = {"answer": answer, "phrasings": [p[method] for p in phrasings], **score(case, answer)}
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="coder")
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--replay", default="", help="directory of saved complex_*.json runs")
    parser.add_argument("--synthetic", type=int, default=0, help="number of fresh suites (seeds 100, 101, ...)")
    parser.add_argument("--kinds", default="MK,MV,MQ,VT,2H")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=0, help="at most this many cases (0 = all)")
    parser.add_argument(
        "--flagged-from", default="",
        help="an earlier run's output (same case options): keep only the cases whose answer was miscopied there",
    )
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gguf, meta, _ = MODELS[args.model]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.meta_root, meta), trust_remote_code=True)
    kinds = set(args.kinds.split(","))
    cases = replay_cases(args.replay) if args.replay else []
    cases += synthetic_cases(list(range(100, 100 + args.synthetic)), args.haystack_root)
    cases = [case for case in cases if case["kind"] in kinds]
    if args.flagged_from:
        with open(args.flagged_from, encoding="utf-8") as handle:
            earlier = json.load(handle)["rows"]
        flagged = {
            (row["source"], row["question"]) for row in earlier
            if any(phrasing.get("miscopies") for phrasing in row["quote"]["phrasings"])
        }
        cases = [case for case in cases if (case["source"], case["question"]) in flagged]
    if args.limit:
        cases = cases[: args.limit]
    methods = args.methods.split(",")
    print(f"{len(cases)} cases ({sum(c['source'].startswith('synthetic') for c in cases)} synthetic), methods {methods}", flush=True)
    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf), meta_dir=os.path.join(args.meta_root, meta), device="cuda:0", verbose=False
    )
    answerer = Answerer(rt, tok, args.max_new)
    rows = []
    started = time.perf_counter()
    for number, case in enumerate(cases):
        row = {key: case[key] for key in ("source", "kind", "question", "expected", "lines")}
        with torch.inference_mode():
            row.update(answerer.run(case, methods))
        rows.append(row)
        marks = " ".join(f"{m}={'Y' if row[m]['full'] else 'n'}" for m in methods)
        print(f"{number + 1:>4}/{len(cases)} {case['kind']:<2} {marks}  {case['source']}  ({time.perf_counter() - started:.0f} s)", flush=True)
    summary = {}
    for scope in ("replay", "synthetic", "all"):
        chosen = [r for r in rows if scope == "all" or r["source"].startswith("synthetic") == (scope == "synthetic")]
        if not chosen:
            continue
        summary[scope] = {
            method: {
                "full": f"{sum(r[method]['full'] for r in chosen)}/{len(chosen)}",
                "by_kind": {
                    kind: f"{sum(r[method]['full'] for r in chosen if r['kind'] == kind)}/{sum(r['kind'] == kind for r in chosen)}"
                    for kind in sorted({r["kind"] for r in chosen})
                },
                "fixed": sum(r[method]["full"] and not r["base"]["full"] for r in chosen) if "base" in methods else None,
                "broke": sum(r["base"]["full"] and not r[method]["full"] for r in chosen) if "base" in methods else None,
            }
            for method in methods
        }
    print(json.dumps(summary, indent=1), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"methods": methods, "summary": summary, "rows": rows}, handle, indent=1, ensure_ascii=False)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
