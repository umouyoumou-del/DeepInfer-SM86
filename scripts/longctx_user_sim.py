"""A simulated user working with a 4M-token context through deepinfer-server.

The user is a developer who loads a whole workspace - this repository (code and docs) and,
after it, the Python standard library - file by file through the server's HTTP API, the way
the web page uploads, asks questions while it is still loading and more once it is done,
and refers back to earlier turns.  The model has never seen this repository (it postdates
the model), so questions about it can only be answered from the loaded text.

Questions with a checkable answer are generated from the loaded files themselves: the
default value of a parameter of a function, the value of a module constant (names that
occur once in the workspace, so the answer is unambiguous).  A few more are written by hand
about the docs (in Chinese, like the docs), about where things are used, and about the
conversation; those are checked by keywords and also kept for reading.

    python scripts/longctx_user_sim.py plan --project . --target 4150000
    python scripts/longctx_user_sim.py run --url http://127.0.0.1:6006 --key K --project . --out sim_4m.json
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request

# this script holds the hand-written questions and their keywords: never part of the context
PROJECT_SKIP = ("docs/results/", "docs/figures/", "logs/", "desktop/", ".git/", "scripts/longctx_user_sim.py")
PROJECT_SUFFIXES = (".py", ".md", ".sh", ".toml", ".txt", ".ps1", ".html", ".js", ".css", ".cu", ".cuh", ".c", ".h", ".cpp")

# hand-written: (question, keywords that a right answer contains, all of them; "any:" = one of them)
HAND = [
    ("在这个项目的 4M 测试里，读入 4,194,304 个 token 用了多少分钟？", ["any:18.4", "any:33.8", "any:33.9", "any:34"]),
    ("latent fp8 缓存每个 token 占多少 KiB？", ["17.3"]),
    ("和 llama.cpp 的对比里，llama.cpp 读入 1M token 用了多少分钟？", ["99.3"]),
    ("vLLM 0.7.3 为 DeepSeek 预先算的旋转位置表有多少行？", ["any:163,840", "any:163840"]),
    ("Which files call drop_page_cache?", ["needle_complex", "longctx_service"]),
    ("What does pinned_empty do, and why does the cache not just use torch.empty with pin_memory=True?",
     ["any:cudaHostRegister", "any:power of two", "any:power-of-two", "any:2 的幂"]),
    ("在 answer_check.py 里，判断一段话是不是抄错，相似度的阈值默认是多少？", ["0.8"]),
]
MEMORY = ("我在这次对话里问的第一个问题是什么？", [])
GUESSABLE = {"None", "True", "False", "0", "1", "-1", "0.0", "1.0", "''", '""'}


def project_files(root: str) -> list[str]:
    names = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        relative = os.path.relpath(path, root).replace("\\", "/")
        if os.path.isfile(path) and relative.endswith(PROJECT_SUFFIXES) and not relative.startswith(PROJECT_SKIP):
            names.append(relative)
    return names


def stdlib_files(root: str) -> list[str]:
    return [
        os.path.relpath(p, root).replace("\\", "/")
        for p in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
        if "site-packages" not in p
    ]


def read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def file_text(label: str, text: str) -> str:
    """What one upload adds to the context: a header naming the file, then the file."""

    return f"\n\n===== {label} =====\n{text}"


def plan(project: str, stdlib: str, target: int, tok) -> list[dict]:
    """The workspace in upload order with token counts: the whole project, then the standard
    library until ``target`` tokens."""

    files, total = [], 0
    for relative in project_files(project):
        text = file_text(f"DeepInfer-SM86/{relative}", read(os.path.join(project, relative)))
        count = len(tok.encode(text, add_special_tokens=False))
        files.append({"label": f"DeepInfer-SM86/{relative}", "path": os.path.join(project, relative), "tokens": count, "part": "project"})
        total += count
    for relative in stdlib_files(stdlib):
        text = file_text(f"python3.12/{relative}", read(os.path.join(stdlib, relative)))
        count = len(tok.encode(text, add_special_tokens=False))
        if total + count > target:
            break
        files.append({"label": f"python3.12/{relative}", "path": os.path.join(stdlib, relative), "tokens": count, "part": "stdlib"})
        total += count
    return files


def _literal(node: ast.AST) -> str | None:
    """Source text of a simple constant (number, string, None/True/False, small tuple), else None."""

    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    if isinstance(value, (bool, int, float, str, type(None))) and len(repr(value)) <= 30:
        return ast.unparse(node)
    return None


def code_facts(files: list[dict]) -> list[dict]:
    """Checkable facts in the loaded Python files: parameter defaults and module constants whose
    names occur in one place only."""

    facts, counts = [], {}
    for item in files:
        if not item["path"].endswith(".py"):
            continue
        source = read(item["path"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts[node.name] = counts.get(node.name, 0) + 1
                positional = node.args.posonlyargs + node.args.args
                args = positional + node.args.kwonlyargs
                defaults = [None] * (len(positional) - len(node.args.defaults)) + node.args.defaults + node.args.kw_defaults
                for arg, default in zip(args, defaults, strict=True):
                    value = None if default is None else _literal(default)
                    if value is not None and arg.arg not in ("self", "cls"):
                        facts.append({"kind": "default", "file": item["label"], "part": item["part"], "function": node.name,
                                      "name": arg.arg, "value": value, "written": ast.get_source_segment(source, default)})
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                value = _literal(node.value)
                if name.isupper() and len(name) > 3 and value is not None:
                    counts["const:" + name] = counts.get("const:" + name, 0) + 1
                    facts.append({"kind": "constant", "file": item["label"], "part": item["part"], "name": name, "value": value,
                                  "written": ast.get_source_segment(source, node.value)})
    unique = []
    for fact in facts:
        key = fact["function"] if fact["kind"] == "default" else "const:" + fact["name"]
        # a value one could guess without reading (None, a flag, 0/1, the name itself) checks nothing
        guessable = fact["value"] in GUESSABLE or fact["value"].strip("'\"") == fact["name"]
        if counts.get(key) == 1 and not guessable:
            unique.append(fact)
    return unique


def fact_question(fact: dict) -> str:
    short = fact["file"].split("/", 1)[1]
    if fact["kind"] == "default":
        return f"In {short}, what is the default value of the parameter `{fact['name']}` of `{fact['function']}`?"
    return f"In {short}, what is the value of the module constant `{fact['name']}`?"


def expected_forms(value: str, written: str | None = None) -> list[str]:
    """Ways a right answer may write a literal: normalized or as the file writes it (``0o777`` is
    ``511``), and a string also without its quotes."""

    forms = set()
    for form in (value, written):
        if form:
            forms.add(form)
            if len(form) >= 2 and form[0] in "'\"" and form[-1] == form[0]:
                forms.add(form[1:-1])
    return sorted(forms)


def right(answer: str, keywords: list[str]) -> bool | None:
    if not keywords:
        return None
    final = answer.rsplit("Answer:", 1)[-1] if "Answer:" in answer else answer
    plain = [k for k in keywords if not k.startswith("any:")]
    either = [k[4:] for k in keywords if k.startswith("any:")]
    whole = answer.replace("`", "")
    return all(k in whole for k in plain) and (not either or any(k in whole for k in either)) and bool(final.strip())


def literal_right(answer: str, forms: list[str]) -> bool:
    final = (answer.rsplit("Answer:", 1)[-1] if "Answer:" in answer else answer).replace("`", "").strip()
    return any(re.search(rf"(?<![\w.]){re.escape(form)}(?![\w.])", final) for form in forms)


# ----------------------------------------------------------------- the server


def call(base: str, key: str, path: str, data: bytes | None = None, extra: str = ""):
    url = f"{base.rstrip('/')}{path}?key={urllib.parse.quote(key)}{extra}"
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    return urllib.request.urlopen(request, timeout=1800)


def status(base: str, key: str) -> dict:
    with call(base, key, "/status") as response:
        return json.loads(response.read())


def ask(base: str, key: str, question: str) -> dict:
    """One question, streamed like the page shows it; returns what happened."""

    body = json.dumps({"question": question, "deep": True}).encode("utf-8")
    began = time.perf_counter()
    rounds, evidence, done = [], [], {}
    with call(base, key, "/ask", body) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["event"] == "located":
                rounds.append({"round": event["round"], "blocks": len(event["positions"]), "seconds": event["seconds"]})
            elif event["event"] == "answer":
                evidence = event["evidence"]
            elif event["event"] == "done":
                done = event
            elif event["event"] == "error":
                done = {"error": event["message"]}
    return {
        "answer": done.get("answer", ""), "final": done.get("final", ""), "first_token": done.get("first_token"),
        "seconds": done.get("seconds"), "round_trip": round(time.perf_counter() - began, 1), "rounds": rounds,
        "context": done.get("context"), "error": done.get("error"),
        "evidence": [{"source": e.get("source"), "position": e["position"], "line": e["line"][:160]} for e in evidence[:12]],
    }


def run(args, tok) -> None:
    files = plan(args.project, args.stdlib, args.target, tok)
    facts = code_facts(files)
    rng = random.Random(args.seed)
    project_facts = [f for f in facts if f["part"] == "project"]
    stdlib_facts = [f for f in facts if f["part"] == "stdlib"]
    chosen_project = rng.sample(project_facts, min(args.project_questions, len(project_facts)))
    chosen_stdlib = rng.sample(stdlib_facts, min(args.stdlib_questions, len(stdlib_facts)))
    project_tokens = sum(f["tokens"] for f in files if f["part"] == "project")
    total = sum(f["tokens"] for f in files)
    print(f"workspace: {len(files)} files, {total:,} tokens ({project_tokens:,} project)", flush=True)

    # when to ask what: two project questions once the project is in, one stdlib question
    # about a module already loaded halfway through the library, the rest at the end
    early = chosen_project[:2]
    stdlib_loaded_early = [f for f in chosen_stdlib if files.index(next(x for x in files if x["label"] == f["file"])) < len(files) // 2]
    middle = stdlib_loaded_early[:1]
    late = [f for f in chosen_project[2:] + chosen_stdlib if f not in middle]
    log = {"files": len(files), "tokens": total, "project_tokens": project_tokens, "turns": [], "ingest": []}
    started = time.perf_counter()

    def turn(question: str, check, meta: dict) -> None:
        state = status(args.url, args.key)
        print(f"\n[{(time.perf_counter() - started) / 60:.1f} min, context {state['context']:,}, queued {state['queued']:,}] {question}", flush=True)
        reply = ask(args.url, args.key, question)
        verdict = check(reply["answer"]) if check else None
        print(f"  -> {reply['final'][:200]!r}  [{'RIGHT' if verdict else 'WRONG' if verdict is False else 'read it'}; "
              f"first token {reply['first_token']} s, {reply['seconds']} s, rounds {len(reply['rounds'])}]", flush=True)
        for item in reply["evidence"][:3]:
            print(f"     evidence {item['source']} @ {item['position']:,}: {item['line'][:100]}", flush=True)
        log["turns"].append({"question": question, "right": verdict, "queued_at_ask": state["queued"],
                             "context_at_ask": state["context"], **meta, **reply})

    def wait_until(tokens: int) -> None:
        while True:
            state = status(args.url, args.key)
            if state["context"] >= tokens or not state["queued"]:
                return
            time.sleep(15)

    def upload(items: list[dict]) -> None:
        for item in items:
            text = file_text(item["label"], read(item["path"]))
            name = urllib.parse.quote(item["label"])
            with call(args.url, args.key, "/ingest", text.encode("utf-8"), f"&name={name}") as response:
                json.loads(response.read())

    def fact_turn(fact: dict) -> None:
        forms = expected_forms(fact["value"], fact.get("written"))
        turn(fact_question(fact), lambda answer: literal_right(answer, forms), {"kind": fact["kind"], "part": fact["part"], "expected": forms, "file": fact["file"]})

    project = [f for f in files if f["part"] == "project"]
    stdlib = [f for f in files if f["part"] == "stdlib"]
    upload(project)
    log["ingest"].append({"event": "project queued", "minutes": round((time.perf_counter() - started) / 60, 2)})
    upload(stdlib)  # queued behind the project; the server reads it while we ask
    log["ingest"].append({"event": "stdlib queued", "minutes": round((time.perf_counter() - started) / 60, 2)})
    wait_until(project_tokens + 2048)
    for fact in early:
        fact_turn(fact)
    turn(HAND[0][0], lambda answer: right(answer, HAND[0][1]), {"kind": "docs", "part": "project"})
    if middle:
        target = sum(f["tokens"] for f in files[: files.index(next(x for x in files if x["label"] == middle[0]["file"])) + 1])
        wait_until(target + 4096)
        fact_turn(middle[0])
    while status(args.url, args.key)["queued"]:
        time.sleep(15)
    log["ingest"].append({"event": "all read", "minutes": round((time.perf_counter() - started) / 60, 2),
                          "context": status(args.url, args.key)["context"]})
    print(f"\nall read: {status(args.url, args.key)['context']:,} tokens after {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    for fact in late:
        fact_turn(fact)
    for question, keywords in HAND[1:]:
        turn(question, lambda answer, k=keywords: right(answer, k), {"kind": "hand", "part": "project"})
    turn(MEMORY[0], None, {"kind": "memory"})
    checked = [t for t in log["turns"] if t["right"] is not None]
    log["summary"] = {
        "right": f"{sum(bool(t['right']) for t in checked)}/{len(checked)}",
        "by_kind": {
            kind: f"{sum(bool(t['right']) for t in checked if t['kind'] == kind)}/{sum(t['kind'] == kind for t in checked)}"
            for kind in sorted({t["kind"] for t in checked})
        },
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    print(json.dumps(log["summary"], ensure_ascii=False, indent=1), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(log, handle, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--url", default="http://127.0.0.1:6006")
    parser.add_argument("--key", default="")
    parser.add_argument("--project", default=".")
    parser.add_argument("--stdlib", default=os.path.dirname(os.__file__))
    parser.add_argument("--meta", default="/root/autodl-tmp/meta_coder")
    parser.add_argument("--target", type=int, default=4_150_000, help="tokens of files to load")
    parser.add_argument("--project-questions", type=int, default=8)
    parser.add_argument("--stdlib-questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.meta, trust_remote_code=True)
    if args.command == "plan":
        files = plan(args.project, args.stdlib, args.target, tok)
        facts = code_facts(files)
        project_tokens = sum(f["tokens"] for f in files if f["part"] == "project")
        print(f"{len(files)} files, {sum(f['tokens'] for f in files):,} tokens: project {project_tokens:,} "
              f"({sum(f['part'] == 'project' for f in files)} files), stdlib {sum(f['tokens'] for f in files) - project_tokens:,}")
        print(f"checkable facts: {sum(f['part'] == 'project' for f in facts)} in the project, {sum(f['part'] == 'stdlib' for f in facts)} in the library")
        rng = random.Random(args.seed)
        for fact in rng.sample([f for f in facts if f["part"] == "project"], 5) + rng.sample([f for f in facts if f["part"] == "stdlib"], 3):
            print(f"  {fact_question(fact)}  ->  {fact['value']}")
        return
    run(args, tok)


if __name__ == "__main__":
    sys.exit(main())
