"""deepinfer-cli: chat with a GGUF model in the terminal, or answer one prompt (``-p``).

Each turn renders the conversation with the model's chat template, prefills it into a short
latent-cache context and streams the greedy answer (one CUDA graph per decoding step where the
GPU allows it).  ``/reset`` forgets the conversation, ``/exit`` or Ctrl-D quits.
"""

from __future__ import annotations

import sys
import time


def add_arguments(parser) -> None:
    parser.add_argument("-p", "--prompt", default="", help="answer this prompt and exit")
    parser.add_argument("-n", "--n-predict", type=int, default=1024, help="most tokens per answer (default %(default)s)")
    parser.add_argument("-c", "--ctx-size", type=int, default=32768,
                        help="tokens of conversation kept; older turns are dropped (default %(default)s)")
    parser.add_argument("-sys", "--system", default="", help="system prompt")


def _prompt_ids(tok, messages: list[dict], limit: int) -> tuple[list[int], list[dict]]:
    """The rendered conversation, dropping the oldest turns until it fits in ``limit`` tokens."""

    while True:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True)
        if len(ids) <= limit or len(messages) <= 2:
            return list(ids), messages
        start = 1 if messages[0]["role"] == "system" else 0
        messages = messages[:start] + messages[start + 2:]


def run(args) -> None:
    from transformers import AutoTokenizer

    from deepinfer.cli.bench import steady_rate
    from deepinfer.cli.models import resolve
    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
    from deepinfer.moe.deepseek_v2_runtime import DeepSeekV2LiteGGUFRuntime
    from deepinfer.moe.longctx_service import TextStream

    model = resolve(args.model, args.hf, args.meta, args.meta_hf)
    print(f"loading {model.gguf.name} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(str(model.meta), trust_remote_code=True)
    rt = DeepSeekV2LiteGGUFRuntime(model.gguf, meta_dir=model.meta, device=args.device, verbose=False)
    system = [{"role": "system", "content": args.system}] if args.system else []
    history: list[dict] = []

    def answer(text: str) -> str:
        nonlocal history
        ids, kept = _prompt_ids(tok, system + history + [{"role": "user", "content": text}],
                                max(256, args.ctx_size - args.n_predict))
        history = kept[len(system):-1]
        session = DeepSeekLongContext(rt, capacity=len(ids) + args.n_predict + 8, kv_dtype="fp16")
        stream = TextStream(tok)
        began = time.perf_counter()
        marks: list[tuple[float, int]] = []
        shown = False  # until the first visible text, whitespace is dropped

        def show(new: list[int]) -> None:
            nonlocal shown
            marks.append((time.perf_counter(), len(new)))
            text = stream.feed(new)
            if not shown:
                text = text.lstrip()
                shown = bool(text)
            sys.stdout.write(text)
            sys.stdout.flush()

        produced = session.generate(ids, max_new_tokens=args.n_predict, stop_ids=[tok.eos_token_id], on_tokens=show)
        reply = tok.decode(produced, skip_special_tokens=True).strip()
        first = marks[0][0] - began if marks else 0.0
        rate = steady_rate(marks)
        print(f"\n\n[prompt {len(ids)} tokens, first token {first:.2f} s, {len(produced)} tokens, {rate:.1f} tok/s]",
              file=sys.stderr, flush=True)
        history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return reply

    if args.prompt:
        answer(args.prompt)
        return
    print(f"{model.title}: type a message; /reset forgets the conversation, /exit or Ctrl-D quits",
          file=sys.stderr, flush=True)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/reset":
            history = []
            print("(conversation cleared)", file=sys.stderr)
            continue
        if text:
            answer(text)
