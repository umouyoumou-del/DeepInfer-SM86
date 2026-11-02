"""Command-line client for deepinfer-server: stream files in, ask, watch the answer arrive.

    python scripts/longctx_client.py --url http://127.0.0.1:8080 status   # deepinfer-server on this machine
    python scripts/longctx_client.py --url http://127.0.0.1:6006 --key K status
    python scripts/longctx_client.py --url ... --key K ingest book.txt notes.md
    python scripts/longctx_client.py --url ... --key K demo --tokens 3145728
    python scripts/longctx_client.py --url ... --key K wait
    python scripts/longctx_client.py --url ... --key K ask "What is the special magic number for onyx badger?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


def call(base: str, key: str, path: str, data: bytes | None = None, extra: str = ""):
    url = f"{base.rstrip('/')}{path}?key={urllib.parse.quote(key)}{extra}"
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    return urllib.request.urlopen(request, timeout=900)


def status(base: str, key: str) -> dict:
    with call(base, key, "/status") as response:
        return json.loads(response.read())


def ingest(base: str, key: str, paths: list[str], piece: int = 400_000) -> None:
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        sent = 0
        while sent < len(text):
            cut = min(len(text), sent + piece)
            if cut < len(text):
                newline = text.rfind("\n", sent, cut)
                cut = newline + 1 if newline > sent else cut
            name = urllib.parse.quote(path.replace("\\", "/").split("/")[-1])
            with call(base, key, "/ingest", text[sent:cut].encode("utf-8"), f"&name={name}") as response:
                reply = json.loads(response.read())
            sent = cut
            print(f"{path}: {100 * sent / len(text):5.1f}% sent, {reply['tokens']:,} tokens queued", flush=True)


def wait(base: str, key: str) -> None:
    while True:
        state = status(base, key)
        print(f"context {state['context']:,} / {state['capacity']:,}  queued {state['queued']:,}  "
              f"{state['rate']:,} tok/s  eta {state['eta_seconds']} s", flush=True)
        if not state["queued"]:
            return
        time.sleep(10)


def ask(base: str, key: str, question: str, deep: bool) -> dict:
    body = json.dumps({"question": question, "deep": deep}).encode("utf-8")
    began = time.perf_counter()
    done: dict = {}
    with call(base, key, "/ask", body) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event["event"]
            if kind == "located":
                first, last = min(event["positions"]), max(event["positions"])
                print(f"\n[round {event['round']}: {len(event['positions'])} blocks between token {first:,} and "
                      f"{last:,} of {event['context']:,}, {event['seconds']} s]", flush=True)
            elif kind == "text":
                sys.stdout.write(event["text"])
                sys.stdout.flush()
            elif kind == "answer":
                # the lines the answer quotes first, then the rest the refine kept
                quoted = [item for item in event["evidence"] if item["line"][:40] in event["text"]]
                others = [item for item in event["evidence"] if item not in quoted]
                for item in (quoted + others)[:4]:
                    print(f"\n  evidence @ token {item['position']:,} ({100 * item['depth']:.1f}%): {item['line'][:100]}")
            elif kind == "union":
                print(f"\n[{event['phrasings']} phrasings, union] Answer: {event['final']}", flush=True)
            elif kind == "done":
                done = event
            elif kind == "error":
                raise SystemExit(event["message"])
    print(f"\n[first token {done.get('first_token')} s, total {done.get('seconds')} s "
          f"({time.perf_counter() - began:.1f} s round trip), {done.get('tokens')} tokens, "
          f"context {done.get('context', 0):,}]")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:6006")
    parser.add_argument("--key", default="", help="the key the server printed (none for one on 127.0.0.1)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    files = sub.add_parser("ingest")
    files.add_argument("paths", nargs="+")
    demo = sub.add_parser("demo")
    demo.add_argument("--tokens", type=int, default=3145728)
    sub.add_parser("wait")
    question = sub.add_parser("ask")
    question.add_argument("question")
    question.add_argument("--fast", action="store_true", help="one round, no confirming locate")
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(args.url, args.key), ensure_ascii=False, indent=1))
    elif args.command == "ingest":
        ingest(args.url, args.key, args.paths)
    elif args.command == "demo":
        with call(args.url, args.key, "/demo", json.dumps({"tokens": args.tokens}).encode("utf-8")) as response:
            reply = json.loads(response.read())
        print(f"queued {reply['tokens']:,} tokens: {reply['name']}")
    elif args.command == "wait":
        wait(args.url, args.key)
    elif args.command == "ask":
        ask(args.url, args.key, args.question, not args.fast)


if __name__ == "__main__":
    main()
