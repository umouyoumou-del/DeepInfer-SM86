"""Needle retrieval over a real-text haystack, up to millions of tokens.

The haystack is CPython's own standard library source, read in sorted order,
so it is real, varied text and the run is deterministic.  Needles are short
facts - "Dr. Amber Lynx keeps her research notes in locker number 4829157." -
spliced in at fixed depths, each at the start of a line.

Two modes:

* multi (default): all needles in one document, prefilled once; each question
  branches off the same cache (truncate, append the question, decode).
* --single: one document per needle, so no needle can be mistaken for another.
  Multi-needle recall falls with length on this model even when nothing is
  lost (it answers with a *different* needle's code), so --single is the test
  of whether a position can be reached at all.
* --insert: prefill the bare haystack once, then for each needle insert it
  into the cached sequence at its position (DeepSeekLongContext.insert: it
  attends to the real prefix and sits at its logical position) and ask.  One
  long prefill serves every depth; the approximation is that haystack tokens
  after the needle never saw it, which --single at the same depths measures.

Token counts and the memory plan are printed before anything is allocated.

    python scripts/needle_longctx.py --tokens 131072 --positions plain
    python scripts/needle_longctx.py --tokens 1048576 --positions dca --kv fp8 --single
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
from deepinfer.moe.latent_mla import DualChunkPositions, LatentKVCache
from deepinfer.moe.latent_reread import locate_and_reread
from deepinfer.moe.latent_retrieval import RetrievalConfig

MODELS = {
    "coder": ("DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", "meta_coder", 131072),
    "chat": ("DeepSeek-V2-Lite-Chat-Q4_K_M.gguf", "meta", 32768),
}
AGENTS = [
    "Crimson Falcon", "Silver Otter", "Amber Lynx", "Cobalt Heron", "Ivory Mantis",
    "Jade Panther", "Scarlet Ibis", "Onyx Badger", "Golden Viper", "Azure Walrus",
    "Copper Stag", "Violet Kestrel",
]
# Same-template needles ("agents") turn out to test disambiguation, not reach:
# at 32K both this runtime and llama.cpp give the same wrong code for the same
# three agents.  "distinct" states a different kind of fact each time.
FACTS = [
    ("The secret code of agent Crimson Falcon is {code}.",
     "What is the secret code of agent Crimson Falcon?"),
    ("The vault in the Silver Otter building opens with the combination {code}.",
     "Which combination opens the vault in the Silver Otter building?"),
    ("Dr. Amber Lynx keeps her research notes in locker number {code}.",
     "In which locker number does Dr. Amber Lynx keep her research notes?"),
    ("The shipping container of the Cobalt Heron expedition is labeled {code}.",
     "What label is on the shipping container of the Cobalt Heron expedition?"),
    ("The library card issued to Ivory Mantis carries the number {code}.",
     "What number is on the library card issued to Ivory Mantis?"),
    ("The parcel sent to Jade Panther has the tracking number {code}.",
     "What is the tracking number of the parcel sent to Jade Panther?"),
    ("The train ticket bought by Scarlet Ibis has the booking reference {code}.",
     "What is the booking reference of the train ticket bought by Scarlet Ibis?"),
    ("The telescope at the Onyx Badger observatory has the serial number {code}.",
     "What is the serial number of the telescope at the Onyx Badger observatory?"),
    ("The emergency hotline of the Golden Viper hotel has the extension {code}.",
     "What is the extension of the emergency hotline of the Golden Viper hotel?"),
    ("The greenhouse of the Azure Walrus farm is unlocked with the PIN {code}.",
     "Which PIN unlocks the greenhouse of the Azure Walrus farm?"),
    ("The bank account of the Copper Stag foundation ends in the digits {code}.",
     "Which digits does the bank account of the Copper Stag foundation end in?"),
    ("The museum exhibit donated by Violet Kestrel has the catalogue number {code}.",
     "What is the catalogue number of the museum exhibit donated by Violet Kestrel?"),
]
_GIB = 2**30


def needle_text(needle: dict) -> str:
    return f"\n{needle['statement']}\n"


def question_text(needle: dict) -> str:
    return f"\n\n{needle['question']} Answer with the 7-digit number only.\n\nAssistant:"


def make_needles(depths: list[float], seed: int, style: str) -> list[dict]:
    rng = random.Random(seed)
    needles = []
    for index, depth in enumerate(depths):
        code = str(rng.randrange(1_000_000, 10_000_000))
        if style == "agents":
            agent = AGENTS[index % len(AGENTS)]
            statement = f"The secret code of agent {agent} is {code}."
            question = f"What is the secret code of agent {agent}?"
        else:
            template, question = FACTS[index % len(FACTS)]
            statement = template.format(code=code)
        needles.append(
            {"statement": statement, "question": question, "code": code, "depth": depth}
        )
    return needles


def haystack_tokens(tok, root: str, count: int) -> list[int]:
    """The first ``count`` tokens of the concatenated standard library."""

    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    body: list[int] = []
    for path in files:
        if "site-packages" in path:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        ids = tok.encode(f"\n# {os.path.relpath(path, root)}\n{text}", add_special_tokens=False)
        body.extend(ids[: count - len(body)])
        if len(body) >= count:
            return body
    raise RuntimeError(f"haystack has only {len(body):,} tokens, {count:,} needed")


def line_start(pieces: list[str], depth: float, length: int) -> int:
    """Index in the first ``length`` body tokens at ``depth``, moved to a line start."""

    cut = min(int(depth * length), length)
    limit = min(length, cut + 4096)
    while 0 < cut < limit and not pieces[cut - 1].endswith("Ċ"):
        cut += 1
    return cut


def header_ids(tok) -> list[int]:
    return [tok.bos_token_id] + tok.encode("User: ", add_special_tokens=False)


def build_document(tok, target: int, needles: list[dict], body: list[int], pieces: list[str]):
    """``[BOS] User: <haystack with needles>``, exactly ``target`` tokens long.

    Sets each needle's ``position``; ``pieces`` are the body's token strings,
    used to move a cut forward to the start of a line.
    """

    header = header_ids(tok)
    needle_ids = [tok.encode(needle_text(needle), add_special_tokens=False) for needle in needles]
    body_length = target - len(header) - sum(len(ids) for ids in needle_ids)
    cuts = []
    for needle, ids in zip(needles, needle_ids, strict=True):
        cuts.append((line_start(pieces, needle["depth"], body_length), needle, ids))
    document = list(header)
    previous = 0
    for cut, needle, ids in sorted(cuts, key=lambda item: item[0]):
        document.extend(body[previous:cut])
        needle["position"] = len(document)
        document.extend(ids)
        previous = cut
    document.extend(body[previous:body_length])
    return document


def document_hash(ids: list[int]) -> str:
    return hashlib.sha256(" ".join(map(str, ids)).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="coder")
    parser.add_argument("--models-dir", default="/root/autodl-tmp/models")
    parser.add_argument("--meta-root", default="/root/autodl-tmp")
    parser.add_argument("--haystack-root", default=os.path.dirname(os.__file__))
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--positions", choices=("plain", "dca"), default="plain")
    parser.add_argument("--trained", type=int, default=0, help="trained length for dca (default: model's)")
    parser.add_argument("--far", type=int, default=-1, help="dca: distance cap for old keys (default trained-1)")
    parser.add_argument("--kv", choices=("fp16", "fp8"), default="fp16")
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--tile", type=int, default=32768)
    parser.add_argument("--workspace-gib", type=float, default=4.0)
    parser.add_argument("--unpin", choices=("auto", "yes", "no"), default="auto")
    parser.add_argument("--depths", default="0,0.1,0.25,0.4,0.5,0.6,0.75,0.9,0.99")
    parser.add_argument("--single", action="store_true", help="one document per needle")
    parser.add_argument("--insert", action="store_true", help="prefill once, insert each needle")
    parser.add_argument(
        "--retrieval", default="",
        help="block retrieval for question and answer: top_blocks[,block,window,sink]",
    )
    parser.add_argument(
        "--encode-window", type=int, default=0,
        help="prefill/insert attend only to a 64-token sink and this many earlier tokens",
    )
    parser.add_argument(
        "--reread", type=int, default=0,
        help="answer by locating the N heaviest blocks in the KV library and re-reading their text",
    )
    parser.add_argument(
        "--reposition", action="store_true",
        help="with --retrieval: move retrieved blocks right in front of the window",
    )
    parser.add_argument(
        "--logn", action="store_true",
        help="scale logits of queries past the trained length by log(pos)/log(trained)",
    )
    parser.add_argument("--save-cache", default="", help="after prefilling, save the latent cache here")
    parser.add_argument("--load-cache", default="", help="load a saved cache instead of prefilling")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--needle-style", choices=("distinct", "agents"), default="distinct")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dump-prompts", default="", help="write each full prompt as text here and exit")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    gguf, meta, trained_default = MODELS[args.model]
    trained = args.trained or trained_default
    depths = [float(value) for value in args.depths.split(",")]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.path.join(args.meta_root, meta), trust_remote_code=True
    )
    started = time.perf_counter()
    body = haystack_tokens(tok, args.haystack_root, args.tokens)
    pieces = tok.convert_ids_to_tokens(body)
    needles = make_needles(depths, args.seed, args.needle_style)
    if args.insert:
        # the bare haystack; each needle goes in later at the same place
        # build_document would have put it in a single-needle document
        trials = [needles]
        documents = [build_document(tok, args.tokens, [], body, pieces)]
        for needle in needles:
            ids = tok.encode(needle_text(needle), add_special_tokens=False)
            length = args.tokens - len(header_ids(tok)) - len(ids)
            needle["position"] = len(header_ids(tok)) + line_start(pieces, needle["depth"], length)
            needle["ids"] = ids
    else:
        trials = [[needle] for needle in needles] if args.single else [needles]
        documents = [build_document(tok, args.tokens, trial, body, pieces) for trial in trials]
    print(
        f"haystack: {len(body):,} tokens ({time.perf_counter() - started:.1f} s to tokenize); "
        f"{len(documents)} document(s) of {len(documents[0]):,} tokens; "
        f"needles at {[needle['position'] for needle in needles]}",
        flush=True,
    )
    per_token = LatentKVCache.bytes_per_token(27, args.kv)
    capacity = max(len(document) for document in documents) + 256
    print(
        f"plan: kv {args.kv} {per_token / 1024:.1f} KiB/token -> {capacity * per_token / _GIB:.2f} GiB "
        f"for {capacity:,} tokens; positions {args.positions}"
        + (f" (trained {trained}, chunk {trained * 3 // 4})" if args.positions == "dca" else ""),
        flush=True,
    )
    if args.plan_only:
        return
    if args.dump_prompts:
        # for an independent engine (llama.cpp adds its own BOS)
        os.makedirs(args.dump_prompts, exist_ok=True)
        index = 0
        for trial, document in zip(trials, documents, strict=True):
            text = tok.decode(document[1:])
            for needle in trial:
                with open(os.path.join(args.dump_prompts, f"q{index}.txt"), "w", encoding="utf-8") as handle:
                    handle.write(text + question_text(needle))
                index += 1
        with open(os.path.join(args.dump_prompts, "needles.json"), "w", encoding="utf-8") as handle:
            json.dump([{k: v for k, v in n.items() if k != "ids"} for n in needles], handle, indent=2)
        return

    rt = DeepSeekV2LiteGGUFRuntime(
        os.path.join(args.models_dir, gguf),
        meta_dir=os.path.join(args.meta_root, meta),
        device="cuda:0",
        verbose=False,
    )
    free = torch.cuda.mem_get_info()[0]
    need = capacity * per_token + args.workspace_gib * _GIB
    unpin = args.unpin == "yes" or (args.unpin == "auto" and need > free)
    if unpin and rt.expert_store is not None:
        rt.expert_store.clear_gpu_cache()
        rt.expert_store.set_gpu_cache_active(False)
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
    room = int(max(0, free - args.workspace_gib * _GIB) // per_token)
    gpu_tokens = min(capacity, room)
    print(
        f"memory: free {free / _GIB:.2f} GiB after load (experts {'streamed' if unpin else 'pinned'}); "
        f"gpu holds {gpu_tokens:,} tokens, host {capacity - gpu_tokens:,}",
        flush=True,
    )
    positions = (
        DualChunkPositions.plain()
        if args.positions == "plain"
        else DualChunkPositions(trained, trained * 3 // 4, args.far)
    )
    retrieval = None
    if args.retrieval:
        fields = [int(value) for value in args.retrieval.split(",")]
        names = ("top_blocks", "block", "window", "sink")
        retrieval = RetrievalConfig(
            **dict(zip(names, fields, strict=False)), reposition=bool(args.reposition)
        )
        print(f"retrieval: {retrieval}", flush=True)
    if args.load_cache:
        if len(documents) != 1:
            raise SystemExit("--load-cache needs one document (multi or insert mode)")
        started = time.perf_counter()
        session, saved = DeepSeekLongContext.load(
            rt,
            args.load_cache,
            capacity=capacity,
            gpu_tokens=gpu_tokens,
            far=None if args.far < 0 else args.far,
            prefill_chunk=args.chunk,
            tile_tokens=args.tile,
            retrieval=retrieval,
            logn_trained=trained if args.logn else None,
            encode_window=args.encode_window or None,
        )
        if saved.get("document_sha256") != document_hash(documents[0]):
            raise SystemExit("the saved cache was built from a different document")
        load_seconds = time.perf_counter() - started
        if not session.slot_tokens:  # caches saved before token records existed
            session.slot_tokens = list(documents[0])
        print(f"loaded {session.length:,}-token cache in {load_seconds:.1f} s (prefill skipped)", flush=True)
    else:
        load_seconds = None
        session = DeepSeekLongContext(
            rt,
            capacity=capacity,
            gpu_tokens=gpu_tokens,
            kv_dtype=args.kv,
            positions=positions,
            prefill_chunk=args.chunk,
            tile_tokens=args.tile,
            retrieval=retrieval,
            logn_trained=trained if args.logn else None,
            encode_window=args.encode_window or None,
        )
    last_report = [0.0]

    def progress(done: int, total: int, elapsed: float) -> None:
        if elapsed - last_report[0] >= 60 or done == total:
            last_report[0] = elapsed
            rate = done / max(elapsed, 1e-9)
            # attention grows linearly with position, so extrapolate quadratically
            eta = elapsed * ((total / done) ** 2 - 1) if done else 0.0
            print(
                f"  prefill {done:,}/{total:,} tokens, {elapsed:7.1f} s, {rate:7.0f} tok/s, "
                f"eta ~{eta / 60:.1f} min, attn {session.timings.get('prefill_attention', 0):.0f} s "
                f"mlp {session.timings.get('prefill_mlp', 0):.0f} s",
                flush=True,
            )

    results = []
    prefill_seconds = []
    torch.cuda.reset_peak_memory_stats()
    for trial, document in zip(trials, documents, strict=True):
        if load_seconds is None:
            session.truncate(0)
            last_report[0] = 0.0
            started = time.perf_counter()
            session.prefill(document, progress=progress)
            torch.cuda.synchronize()
            prefill_seconds.append(time.perf_counter() - started)
            print(
                f"prefill {session.length:,} tokens in {prefill_seconds[-1] / 60:.1f} min "
                f"({session.length / prefill_seconds[-1]:.0f} tok/s); "
                f"peak {torch.cuda.max_memory_allocated() / _GIB:.2f} GiB",
                flush=True,
            )
            if args.save_cache:
                started = time.perf_counter()
                session.save(
                    args.save_cache,
                    meta={
                        "model": args.model,
                        "document_sha256": document_hash(document),
                        "tokens": len(document),
                    },
                )
                print(f"saved cache to {args.save_cache} in {time.perf_counter() - started:.1f} s", flush=True)
        state = session.snapshot()
        for needle in trial:
            session.restore(state)
            if args.insert:
                session.insert(needle["ids"], needle["position"])
            q_ids = tok.encode(question_text(needle), add_special_tokens=False)
            started = time.perf_counter()
            decode_before = session.timings.get("decode_attention", 0.0) + session.timings.get("decode_mlp", 0.0)
            located = None
            if args.reread:
                slot = session.length - len(needle["ids"]) if args.insert else needle["position"]
                width = len(needle["ids"]) if args.insert else len(
                    tok.encode(needle_text(needle), add_special_tokens=False)
                )
                produced, spans = locate_and_reread(
                    session,
                    q_ids,
                    header_ids=header_ids(tok),
                    separator_ids=tok.encode("\n...\n", add_special_tokens=False),
                    top=args.reread,
                    max_new_tokens=args.max_new,
                    stop_ids=[tok.eos_token_id],
                )
                located = any(lo < slot + width and slot < hi for lo, hi in spans)
            else:
                produced = session.generate(q_ids, max_new_tokens=args.max_new, stop_ids=[tok.eos_token_id])
            seconds = time.perf_counter() - started
            decode_seconds = (
                session.timings.get("decode_attention", 0.0)
                + session.timings.get("decode_mlp", 0.0)
                - decode_before
            )
            answer = tok.decode(produced, skip_special_tokens=True)
            digits = re.findall(r"\d{7}", answer.replace(" ", ""))
            hit = needle["code"] in digits
            record = {key: value for key, value in needle.items() if key != "ids"}
            results.append(
                {**record, "answer": answer, "hit": hit, "located": located, "seconds": round(seconds, 2)}
            )
            steps = max(1, len(produced) - 1)
            print(
                f"  depth {needle['depth']:.2f} pos {needle['position']:>9,} {needle['statement'][:34]:34s} "
                f"want {needle['code']} got {answer.strip()[-40:]!r:42s} {'HIT' if hit else 'MISS'} "
                + ("" if located is None else f"[located {'yes' if located else 'no'}] ")
                + f"({seconds:.1f} s, decode {decode_seconds / steps * 1000:.0f} ms/token)",
                flush=True,
            )
    hits = sum(result["hit"] for result in results)
    mode = "insert" if args.insert else "single" if args.single else "multi"
    print(f"recall {hits}/{len(results)} at {len(documents[0]):,} tokens ({mode})", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "model": args.model,
                    "tokens": len(documents[0]),
                    "mode": mode,
                    "positions": args.positions,
                    "trained": trained if args.positions == "dca" else None,
                    "far": session.positions.far if args.positions == "dca" else None,
                    "kv": args.kv,
                    "retrieval": None if retrieval is None else vars(retrieval),
                    "logn": bool(args.logn),
                    "reread": args.reread or None,
                    "encode_window": args.encode_window or None,
                    "gpu_tokens": gpu_tokens,
                    "experts": "streamed" if unpin else "pinned",
                    "prefill_seconds": [round(value, 1) for value in prefill_seconds],
                    "cache_load_seconds": None if load_seconds is None else round(load_seconds, 1),
                    "timings": {k: round(v, 1) for k, v in session.timings.items()},
                    "peak_gib": round(torch.cuda.max_memory_allocated() / _GIB, 2),
                    "recall": f"{hits}/{len(results)}",
                    "needles": results,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
