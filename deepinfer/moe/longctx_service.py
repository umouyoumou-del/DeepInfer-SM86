"""A long context you keep feeding and asking: streamed ingest, located answers, streamed tokens.

One worker thread owns the GPU.  Text is tokenized when it arrives and queued; the worker
encodes it ``chunk`` tokens at a time (bounded encoding, as for a document's prefill) and,
between two chunks, answers any question that is waiting - so the context can be asked about
while it is still growing.  An answer is located over everything read so far, re-read and
generated in a short context (:func:`~deepinfer.moe.latent_reread.feedback_locate_and_reread`)
and streamed token by token; the question and its answer are then appended to the context, so
a later question can find an earlier turn.

The pieces that do not need a model - the job queue, where a position falls among the sources,
turning streamed ids into text - are separate so they can be tested on their own.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

# the instruction the 3M suite was answered with (scripts/needle_complex.py), {what} = "the answer"
QUOTE_FIRST = (
    "First copy, word for word, every sentence or line of the text above that is needed to answer "
    "the question. Then reason step by step. Finally write 'Answer:' followed by the answer."
)
NEUTRAL_PROMPT = "\n\nSummarize the text above in one sentence.\n\nAssistant:"
CONTROL, ASK, INGEST = 0, 1, 2  # job priorities: a reset first, a waiting question before the next chunk


# ------------------------------------------------------------------ helpers


@dataclass
class Source:
    """Something fed into the context; ``segments`` are its ``(start, length)`` stretches."""

    name: str
    tokens: int
    kind: str = "text"
    done: int = 0
    segments: list[list[int]] = field(default_factory=list)

    def add(self, start: int, length: int) -> None:
        if self.segments and self.segments[-1][0] + self.segments[-1][1] == start:
            self.segments[-1][1] += length
        else:
            self.segments.append([start, length])


def locate_source(sources: Sequence[Source], position: int) -> tuple[str, int] | None:
    """``(name, offset within that source)`` for a logical position, or ``None`` (header, unknown)."""

    for source in sources:
        before = 0
        for start, length in source.segments:
            if start <= position < start + length:
                return source.name, before + position - start
            before += length
    return None


class TextStream:
    """Ids arriving in pieces -> the text they add; holds back an incomplete UTF-8 tail.

    A character split across tokens decodes to replacement marks until its last
    byte arrives, so trailing marks are held back.  If decoding the longer list
    rewrites text already sent (a tokenizer's space clean-up), only what lies past
    it is sent; the round's final answer event carries the exact text.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def feed(self, ids: Sequence[int]) -> str:
        self.ids.extend(int(i) for i in ids)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True).rstrip("�")
        added = text[len(self.text):] if len(text) > len(self.text) else ""
        self.text = text
        return added


class JobQueue:
    """Priority jobs; an ingest job hands back its remainder under its original sequence number."""

    def __init__(self) -> None:
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()

    def put(self, priority: int, job, seq: int | None = None) -> int:
        seq = next(self._seq) if seq is None else seq
        self._queue.put((priority, seq, job))
        return seq

    def get(self, timeout: float | None = None):
        return self._queue.get(timeout=timeout)

    def take_ingest(self):
        """The next job if it is an ingest job - nothing more urgent waiting - else ``None``."""

        try:
            item = self._queue.get_nowait()
        except queue.Empty:
            return None
        if item[0] != INGEST:
            self._queue.put(item)
            return None
        return item

    def empty(self) -> bool:
        return self._queue.empty()


# ------------------------------------------------------------------ service


class LongContextChat:
    """The long context, its sources and the worker that grows it and answers questions about it."""

    def __init__(
        self,
        rt,
        tokenizer,
        session,
        *,
        header_ids: Sequence[int],
        chunk: int = 16384,
        reserve: int = 8192,
        near_window: int = 98304,
        check_copies: bool = False,
        self_check: bool = False,
    ) -> None:
        self.rt = rt
        self.tok = tokenizer
        self.session = session
        self.header_ids = list(header_ids)
        self.separator_ids = tokenizer.encode("\n...\n", add_special_tokens=False)
        self.chunk = int(chunk)
        self.reserve = int(reserve)  # room kept for located queries, conversation turns
        self.near_window = int(near_window)
        # answer checks (deepinfer.moe.answer_check): ask again on a miscopied line, self-check the final answer
        self.check_copies = bool(check_copies)
        self.self_check = bool(self_check)
        self.sources: list[Source] = []
        self.jobs = JobQueue()
        self.lock = threading.Lock()  # guards the bookkeeping below, not the GPU
        self.queued_tokens = 0
        self.state = "idle"
        self.rate = 0.0  # tokens / s of the last chunks
        self.turns = 0
        self.history: list[tuple[str, str]] = []  # (question, final answer), all turns
        self.last: dict = {}
        self._baseline = None
        self._baseline_length = -1
        if session.length == 0:
            session.prefill(self.header_ids)
        self.worker = threading.Thread(target=self._work, name="longctx-worker", daemon=True)
        self.worker.start()

    # ------------------------------------------------------------- requests

    def room(self) -> int:
        with self.lock:
            return self.session.cache.capacity - self.session.length - self.queued_tokens - self.reserve

    def add_text(self, text: str, name: str, kind: str = "text") -> Source:
        """Queue ``text`` for encoding; raises ``ValueError`` when the context has no room for it."""

        ids = self.tok.encode(text, add_special_tokens=False)
        return self.add_ids(ids, name, kind)

    def add_ids(self, ids: Sequence[int], name: str, kind: str = "text") -> Source:
        ids = list(ids)
        if not ids:
            raise ValueError("nothing to add")
        if len(ids) > self.room():
            raise ValueError(f"{len(ids):,} tokens do not fit: {max(0, self.room()):,} left")
        source = Source(name=name, tokens=len(ids), kind=kind)
        with self.lock:
            self.sources.append(source)
            self.queued_tokens += len(ids)
        self.jobs.put(INGEST, ("ingest", source, ids))
        return source

    def ask(
        self,
        question: str,
        *,
        deep: bool = False,
        instruction: str | None = None,
        variants: Sequence[str] = (),
        rounds: int | None = None,
    ) -> queue.Queue:
        """Queue a question; events (dicts) arrive on the returned queue, ``done`` or ``error`` last.

        ``instruction`` replaces the default quote-then-answer one, and ``variants`` -
        more phrasings of the final pick, asked over the converged reading - add their
        codes and names to the answer: how the demo's questions were evaluated.  A deep
        search runs ``rounds`` rounds at most (default 4).
        """

        events: queue.Queue = queue.Queue()
        self.jobs.put(ASK, ("ask", question, deep, events, instruction, tuple(variants), rounds))
        return events

    def reset(self) -> None:
        """Drop everything but the header (queued text included)."""

        done = threading.Event()
        self.jobs.put(CONTROL, ("reset", done))
        done.wait()

    def status(self) -> dict:
        session = self.session
        with self.lock:
            sources = [
                {"name": s.name, "kind": s.kind, "tokens": s.tokens, "done": s.done, "segments": s.segments}
                for s in self.sources
            ]
            return {
                "context": session.length, "capacity": session.cache.capacity, "queued": self.queued_tokens,
                "gpu_tokens": session.cache.gpu_tokens, "state": self.state, "rate": round(self.rate),
                "eta_seconds": round(self.queued_tokens / self.rate) if self.rate and self.queued_tokens else 0,
                "turns": self.turns, "sources": sources, "last": self.last,
            }

    # --------------------------------------------------------------- worker

    def _work(self) -> None:
        while True:
            _priority, seq, job = self.jobs.get()
            try:
                if job[0] == "ingest":
                    self._ingest(seq, job[1], job[2])
                elif job[0] == "ask":
                    self._answer(*job[1:])
                elif job[0] == "reset":
                    self._reset(job[1])
            except Exception as error:  # noqa: BLE001 - the worker must survive a failed job
                if job[0] == "ask":
                    job[3].put({"event": "error", "message": f"{type(error).__name__}: {error}"})
                with self.lock:
                    self.last = {"error": f"{type(error).__name__}: {error}"}
            finally:
                with self.lock:
                    if self.jobs.empty():
                        self.state = "idle"

    def _ingest(self, seq: int, source: Source, ids: list[int]) -> None:
        """Encode one chunk: this source's next piece, topped up with the sources queued after it.

        Every prefill call pays a fixed cost (each layer dequantizes the experts it uses), so a
        workspace of many small files read one file per call ran at 733 tok/s against about
        2,000 for one big document.  Filling the chunk from the queue reads them as fast; a
        question or a reset still goes first, between two chunks.
        """

        import torch

        if source.tokens == -1:  # dropped by a reset while queued
            return
        parts = [(source, ids[: self.chunk], seq, ids[self.chunk :])]
        room = self.chunk - len(parts[0][1])
        while room > 0:
            item = self.jobs.take_ingest()
            if item is None:
                break
            _priority, more_seq, job = item
            more, more_ids = job[1], job[2]
            if more.tokens == -1:
                continue
            parts.append((more, more_ids[:room], more_seq, more_ids[room:]))
            room -= len(parts[-1][1])
        batch = [token for _, piece, _, _ in parts for token in piece]
        with self.lock:
            self.state = "reading"
        began = time.perf_counter()
        start = self.session.next_position
        self.session.prefill(batch)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        with self.lock:
            for part, piece, _, _ in parts:
                part.add(start, len(piece))
                part.done += len(piece)
                start += len(piece)
            self.queued_tokens -= len(batch)
            current = len(batch) / max(seconds, 1e-6)
            self.rate = current if not self.rate else 0.7 * self.rate + 0.3 * current
        self._baseline_length = -1
        for part, _, part_seq, rest in parts:
            if rest:
                self.jobs.put(INGEST, ("ingest", part, rest), seq=part_seq)

    def _baseline_now(self):
        from deepinfer.moe.latent_reread import neutral_baseline

        if self._baseline_length != self.session.length:
            neutral = self.tok.encode(NEUTRAL_PROMPT, add_special_tokens=False)
            self._baseline = neutral_baseline(self.session, neutral, reduce="max")
            self._baseline_length = self.session.length
        return self._baseline

    def _answer(
        self,
        question: str,
        deep: bool,
        events: queue.Queue,
        instruction: str | None = None,
        variants: Sequence[str] = (),
        rounds: int | None = None,
    ) -> None:
        from deepinfer.moe.answer_check import self_check_prompt
        from deepinfer.moe.latent_reread import (
            BLOCK,
            Refine,
            answer_items,
            answer_over,
            feedback_locate_and_reread,
            final_answer,
            reread,
        )

        with self.lock:
            self.state = "answering"
        session = self.session
        length = session.length
        began = time.perf_counter()
        events.put({"event": "start", "question": question, "context": length, "deep": deep})
        baseline = self._baseline_now()
        events.put({"event": "baseline", "seconds": round(time.perf_counter() - began, 2)})
        bare = self.tok.encode(f"\n\n{question}\n\nAssistant:", add_special_tokens=False)
        neutral = self.tok.encode(NEUTRAL_PROMPT, add_special_tokens=False)
        refine = Refine(self.tok, bare, neutral, keep=12, order="document")
        streams: dict[int, TextStream] = {}
        first_token = [None]

        def where(position: int) -> dict:
            found = locate_source(self.sources, position)
            return {
                "position": position, "depth": round(position / max(1, length), 4),
                "source": None if found is None else found[0], "offset": None if found is None else found[1],
            }

        def relay(event: dict) -> None:
            kind = event["event"]
            if kind == "located":
                events.put({
                    "event": "located", "round": event["round"], "seconds": event["seconds"],
                    "positions": [block * BLOCK for block in event["blocks"]], "context": length,
                })
            elif kind == "tokens":
                stream = streams.setdefault(event["round"], TextStream(self.tok))
                text = stream.feed(event["ids"])
                if first_token[0] is None:
                    first_token[0] = time.perf_counter() - began
                if text:
                    events.put({"event": "text", "round": event["round"], "text": text})
            elif kind == "answer":
                evidence = [
                    {"line": line, **where(position)}
                    for line, position in zip(event["lines"], event["line_positions"], strict=False)
                    if position >= 0
                ]
                events.put({
                    "event": "answer", "round": event["round"], "text": event["text"],
                    "final": final_answer(event["text"]), "evidence": evidence, "stop": event["stop"],
                })
            elif kind == "check":
                events.put({"event": "check", "round": event["round"], "kind": event["kind"]})

        detail = feedback_locate_and_reread(
            session, self.tok, question, answer_instruction=instruction or QUOTE_FIRST, preamble=self._recent(),
            header_ids=self.header_ids, separator_ids=self.separator_ids, baseline=baseline, reduce="max",
            top=32, follow_top=8, neighbors=1, rounds=(rounds or 4) if deep else 1, max_new_tokens=1024,
            stop_ids=[self.tok.eos_token_id], refine=refine, near_window=self.near_window, follow_final=32,
            on_event=relay, check_copies=self.check_copies, self_check=self.self_check,
        )
        answer = self.tok.decode(detail["answer_ids"], skip_special_tokens=True).strip()
        if variants:
            events.put({"event": "variants", "count": len(variants)})
            texts = [answer]
            for variant in variants:
                prompt = self.tok.encode(f"\n\n{question} {variant}\n\nAssistant:", add_special_tokens=False)
                document: list[int] = []
                ids = reread(
                    session, detail["spans"], prompt, header_ids=self.header_ids, separator_ids=self.separator_ids,
                    max_new_tokens=1024, stop_ids=[self.tok.eos_token_id], refine=refine,
                    refine_query=detail["refine_query"], document_out=document,
                )
                if self.self_check:
                    checked = answer_over(
                        self.rt, document + prompt, self_check_prompt(self.tok, ids, self.tok.eos_token_id),
                        max_new_tokens=1024, stop_ids=[self.tok.eos_token_id],
                    )
                    text = self.tok.decode(checked, skip_special_tokens=True).strip()
                    if "answer:" in text.lower() and final_answer(text).strip():
                        ids = checked
                texts.append(self.tok.decode(ids, skip_special_tokens=True).strip())
            given = set(answer_items(question))  # the value asked about is not an answer
            union = sorted({item for text in texts for item in answer_items(text)} - given)
            if union:
                answer = f"{answer}\n\n[{len(texts)} phrasings, union]\nAnswer: {', '.join(union)}"
                events.put({"event": "union", "final": ", ".join(union), "phrasings": len(texts)})
        seconds = time.perf_counter() - began
        # the turn joins the context, so a later question can find it
        turn = self.tok.encode(f"\n\n[Q] {question}\n[A] {final_answer(answer)}\n", add_special_tokens=False)
        start = session.next_position
        if len(turn) <= session.cache.capacity - session.length - self.reserve // 2:
            session.prefill(turn)
            self._baseline_length = -1
            with self.lock:
                conversation = next((s for s in self.sources if s.kind == "conversation"), None)
                if conversation is None:
                    conversation = Source(name="对话", tokens=0, kind="conversation")
                    self.sources.append(conversation)
                conversation.add(start, len(turn))
                conversation.tokens += len(turn)
                conversation.done += len(turn)
        summary = {
            "question": question, "seconds": round(seconds, 2),
            "first_token": None if first_token[0] is None else round(first_token[0], 2),
            "rounds": len(detail["rounds"]), "tokens": len(detail["answer_ids"]), "context": length,
            "read_tokens": detail["reread_tokens"],
        }
        with self.lock:
            self.turns += 1
            self.history.append((question, final_answer(answer)))
            self.last = summary
        events.put({"event": "done", "answer": answer, "final": final_answer(answer), **summary})

    def _recent(self, turns: int = 16, answer_chars: int = 300, opening: int = 2) -> str:
        """Turns for the answer prompt, numbered from the start: the first ``opening`` and the
        latest ones, ``turns`` in all.  Located text alone does not say which question came
        first, and at 4M, 20 turns in, "what did I ask first" was not found by locating."""

        if not self.history:
            return ""
        numbered = list(enumerate(self.history, start=1))
        if len(numbered) > turns:
            numbered = numbered[:opening] + [None] + numbered[-(turns - opening):]
        lines = ["Earlier in this conversation:"]
        for item in numbered:
            if item is None:
                lines.append("...")
                continue
            number, (question, answer) = item
            lines.append(f"Q{number}: {question}")
            lines.append(f"A{number}: {answer[:answer_chars]}")
        return "\n".join(lines)

    def _reset(self, done: threading.Event) -> None:
        with self.lock:
            for source in self.sources:
                source.tokens = -1 if source.done < source.tokens else source.tokens
            self.sources = []
            self.queued_tokens = 0
            self.turns = 0
            self.history = []
            self.last = {}
        self.session.truncate(len(self.header_ids))
        self._baseline_length = -1
        done.set()


def build_session(rt, *, capacity: int, trained: int, encode_window: int, kv_dtype: str = "fp8",
                  workspace_gib: float = 4.0, chunk: int = 16384):
    """A DeepSeekLongContext sized like the harness's: as much of the KV in VRAM as fits, the rest pinned."""

    import torch

    from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
    from deepinfer.moe.latent_mla import (
        DualChunkPositions,
        LatentKVCache,
        drop_page_cache,
    )

    per_token = LatentKVCache.bytes_per_token(len(rt.model.model.layers), kv_dtype)
    free = torch.cuda.mem_get_info()[0]
    gpu_tokens = min(capacity, int(max(0, free - workspace_gib * 2**30) // per_token))
    if gpu_tokens < capacity and getattr(rt, "pin_active", False):
        drop_page_cache(str(rt.gguf_path))  # the experts are in VRAM; the pinned host tier needs the room
    return DeepSeekLongContext(
        rt, capacity=capacity, gpu_tokens=gpu_tokens, kv_dtype=kv_dtype,
        positions=DualChunkPositions(trained, trained * 3 // 4, -1), prefill_chunk=chunk, encode_window=encode_window,
    )

