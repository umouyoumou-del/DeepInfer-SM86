"""Production-oriented long-context service over a persistent retrieval index.

Unlike :class:`~deepinfer.moe.longctx_service.LongContextChat`, this service does
not hold a multi-million-token KV cache.  Text is indexed into SQLite FTS5 (with
an exact-fact side list); each question retrieves a 13-17K-token evidence window
and runs one short FP8 KV session.  Warm start therefore means opening a 45 MB
index, not loading a 165 GiB 10M cache.
"""

from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from deepinfer.moe.answer_check import var_chain_answer
from deepinfer.moe.latent_reread import final_answer
from deepinfer.moe.longctx_service import QUOTE_FIRST, Source
from deepinfer.retrieval.global_queries import global_answer
from deepinfer.retrieval.hybrid import (
    chunk_text,
    exact_lines,
    facts_from_lines,
    retrieve,
    structured_answer,
)


class HierarchicalChat:
    """One GPU worker, persistent text + FTS5 index, and a small on-demand KV per answer."""

    def __init__(
        self,
        runtime,
        tokenizer,
        *,
        text_path: str | os.PathLike[str] | None = None,
        index_path: str | os.PathLike[str] | None = None,
        capacity: int = 10_000_000,
        top_k: int = 24,
        max_new: int = 768,
        chunk_chars: int = 2000,
        max_context: int = 32768,
        trained: int = 131072,
    ) -> None:
        self.rt = runtime
        self.tok = tokenizer
        self.capacity = int(capacity)
        self.top_k = int(top_k)
        self.max_new = int(max_new)
        self.chunk_chars = int(chunk_chars)
        self.max_context = int(max_context)
        self.trained = int(trained)
        self.index_path = None if index_path is None else str(index_path)
        self.lock = threading.RLock()
        self.db_lock = threading.RLock()
        self.text_parts: list[str] = []
        self.text = ""
        self.facts: list[str] = []
        self.sources: list[Source] = []
        self.queued_tokens = 0
        self.turns = 0
        self.history: list[tuple[str, str]] = []
        self.last: dict = {}
        self.state = "idle"
        self.rate = 0.0
        self.retrieval_ms = 0.0
        self.session = SimpleNamespace(length=0)
        self.queued = 0
        self.jobs: queue.Queue = queue.Queue()
        if self.index_path is None:
            self.con = sqlite3.connect(":memory:", check_same_thread=False)
            self._create_schema(self.con)
        else:
            path = Path(self.index_path)
            if path.exists():
                self.con = sqlite3.connect(path, check_same_thread=False)
                self._ensure_schema(self.con)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.con = sqlite3.connect(path, check_same_thread=False)
                self._create_schema(self.con)
        if text_path is not None and Path(text_path).exists():
            self._load_text(Path(text_path))
        self._next_cid = self._max_cid() + 1
        self._update_session()
        self.worker = threading.Thread(target=self._work, name="hierarchical-worker", daemon=True)
        self.worker.start()

    # --------------------------------------------------------------- schema

    @staticmethod
    def _create_schema(con: sqlite3.Connection) -> None:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(body, cid UNINDEXED)")
        con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        con.commit()

    @staticmethod
    def _ensure_schema(con: sqlite3.Connection) -> None:
        try:
            con.execute("SELECT count(*) FROM chunks")
        except sqlite3.OperationalError:
            con.execute("CREATE VIRTUAL TABLE chunks USING fts5(body, cid UNINDEXED)")
        try:
            con.execute("SELECT count(*) FROM meta")
        except sqlite3.OperationalError:
            con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        con.commit()

    def _max_cid(self) -> int:
        with self.db_lock:
            row = self.con.execute("SELECT coalesce(max(cid), -1) FROM chunks").fetchone()
        return int(row[0]) if row else -1

    def _meta(self, key: str, default: str = "") -> str:
        with self.db_lock:
            row = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        with self.db_lock:
            self.con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", (key, value))
            self.con.commit()

    def _update_session(self) -> None:
        tokens = self._meta("tokens", "")
        if tokens:
            self.session.length = int(tokens)

    # ------------------------------------------------------------- ingestion

    def _load_text(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        self.text_parts = [text]
        self.text = text
        self.facts = facts_from_lines(text.splitlines())
        tokens = int(self._meta("tokens", "0"))
        if not tokens:
            tokens = len(self.tok.encode(text, add_special_tokens=False))
            self._set_meta("tokens", str(tokens))
        source = Source(name=f"index:{path.name}", tokens=tokens, kind="index", done=tokens)
        source.add(0, tokens)
        self.sources = [source]
        self.session.length = tokens
        if self._count_chunks() == 0:
            chunks = chunk_text(text, self.chunk_chars)
            with self.db_lock:
                self.con.executemany(
                    "INSERT INTO chunks(cid, body) VALUES (?,?)",
                    list(enumerate(chunks)),
                )
                self.con.commit()
        self._set_meta("chunks", str(self._count_chunks()))

    def _count_chunks(self) -> int:
        with self.db_lock:
            row = self.con.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def room(self) -> int:
        with self.lock:
            return max(0, self.capacity - int(self.session.length))

    def add_text(self, text: str, name: str, kind: str = "text") -> Source:
        if not text:
            raise ValueError("nothing to add")
        tokens = len(self.tok.encode(text, add_special_tokens=False))
        if tokens > self.room():
            raise ValueError(f"{tokens:,} tokens do not fit: {self.room():,} left")
        chunks = chunk_text(text, self.chunk_chars)
        start = int(self.session.length)
        with self.db_lock:
            next_cid = self._next_cid
            self.con.executemany(
                "INSERT INTO chunks(cid, body) VALUES (?,?)",
                [(next_cid + i, chunk) for i, chunk in enumerate(chunks)],
            )
            self._next_cid += len(chunks)
            self.con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('tokens',?)", (str(start + tokens),))
            self.con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES ('chunks',?)", (str(self._count_chunks()),))
            self.con.commit()
        source = Source(name=name, tokens=tokens, kind=kind, done=tokens)
        source.add(start, tokens)
        with self.lock:
            self.text_parts.append(text)
            self.text += text
            self.facts.extend(facts_from_lines(text.splitlines()))
            self.sources.append(source)
            self.session.length = start + tokens
        return source

    def add_ids(self, ids: Sequence[int], name: str, kind: str = "text") -> Source:
        return self.add_text(self.tok.decode(ids, skip_special_tokens=True), name, kind)

    # --------------------------------------------------------------- requests

    def ask(
        self,
        question: str,
        *,
        deep: bool = False,
        instruction: str | None = None,
        variants: Sequence[str] = (),
        rounds: int | None = None,
    ) -> queue.Queue:
        events: queue.Queue = queue.Queue()
        with self.lock:
            self.queued += 1
        self.jobs.put((question, events, instruction or QUOTE_FIRST))
        return events


    def reset(self) -> None:
        with self.db_lock:
            self.con.execute("DELETE FROM chunks")
            self.con.execute("DELETE FROM meta")
            self.con.commit()
        with self.lock:
            self.text_parts, self.text, self.facts, self.sources = [], "", [], []
            self.turns, self.history, self.last = 0, [], {}
            self._next_cid = 0
            self.session.length = 0
            self.state = "idle"

    def status(self) -> dict:
        with self.lock:
            sources = [
                {"name": source.name, "kind": source.kind, "tokens": source.tokens, "done": source.done,
                 "segments": source.segments}
                for source in self.sources
            ]
            chunks = self._count_chunks()
            return {
                "mode": "hierarchical", "context": int(self.session.length), "capacity": self.capacity,
                "queued": self.queued, "gpu_tokens": 0, "state": self.state, "rate": round(self.rate),
                "eta_seconds": 0, "turns": self.turns, "sources": sources, "last": self.last,
                "index_chunks": chunks, "index_bytes": 0 if self.index_path is None else os.path.getsize(self.index_path),
                "retrieval_ms": round(self.retrieval_ms, 2),
            }

    # ----------------------------------------------------------------- worker

    def _work(self) -> None:
        while True:
            question, events, instruction = self.jobs.get()
            with self.lock:
                self.queued = max(0, self.queued - 1)
                self.state = "answering"
            try:
                self._answer(question, events, instruction)
            except Exception as error:  # noqa: BLE001 - a bad request must not kill the server
                events.put({"event": "error", "message": f"{type(error).__name__}: {error}"})
            finally:
                with self.lock:
                    self.state = "idle"

    # ---------------------------------------------------------------- answer

    def _answer(self, question: str, events: queue.Queue, instruction: str) -> None:
        from deepinfer.moe.deepseek_longctx import DeepSeekLongContext
        from deepinfer.moe.latent_mla import DualChunkPositions

        began = time.perf_counter()
        context_length = int(self.session.length)
        events.put({"event": "start", "question": question, "context": context_length, "deep": True})
        with self.lock:
            text = self.text
            facts = list(self.facts)
        t = time.perf_counter()
        evidence = exact_lines(facts, question) if facts else []
        with self.db_lock:
            rows = retrieve(self.con, question, self.top_k)
        retrieval = time.perf_counter() - t
        with self.lock:
            self.retrieval_ms = retrieval * 1000
        context = "Exact evidence lines:" + chr(10) + chr(10).join(evidence)
        if rows:
            context += chr(10) + chr(10) + "Retrieved excerpts:" + chr(10) + chr(10) + chr(10).join(
                body for _, body in rows
            )
        context_ids = self.tok.encode(context, add_special_tokens=False)
        if len(context_ids) > self.max_context:
            context_ids = context_ids[-self.max_context :]
        prompt_ids = self.tok.encode(
            chr(10) + chr(10) + question + " " + instruction + chr(10) + chr(10) + "Assistant:",
            add_special_tokens=False,
        )
        events.put({"event": "located", "round": 1, "seconds": round(retrieval, 3),
                    "positions": [], "context": context_length})
        # Global count/summary/aggregation queries need every token, not top-k.
        first_token = None
        tokens = 0
        global_result = global_answer(question, text, facts)
        if global_result is not None:
            answer = global_result.answer
            global_kind = global_result.kind
            globals_evidence = global_result.evidence
            global_metrics = global_result.metrics
        else:
            global_kind = None
            globals_evidence = []
            global_metrics = {}
            structured = structured_answer(question, facts, text)
            if structured:
                answer = "Answer: " + structured
            elif text and "variable" in question.lower() and (names := var_chain_answer(question, text)):
                answer = "Answer: " + ", ".join(names)
            else:
                capacity = len(context_ids) + len(prompt_ids) + 1024
                positions = DualChunkPositions(self.trained, self.trained * 3 // 4, -1)
                session = DeepSeekLongContext(
                    self.rt, capacity=capacity, gpu_tokens=capacity, kv_dtype="fp8", positions=positions,
                    prefill_chunk=16384, tile_tokens=32768, encode_window=98304,
                )
                session.prefill(context_ids)
                first = [None]

                def on_tokens(_ids: list[int]) -> None:
                    if first[0] is None:
                        first[0] = time.perf_counter() - began

                out = session.generate(prompt_ids, max_new_tokens=self.max_new,
                                       stop_ids=[self.tok.eos_token_id], on_tokens=on_tokens)
                first_token = first[0]
                tokens = len(out)
                answer = self.tok.decode(out, skip_special_tokens=True).strip()
        final = final_answer(answer)
        ev = globals_evidence or [
            {"line": line, "position": 0, "depth": 0.0, "source": "exact", "offset": 0}
            for line in evidence[:8]
        ]
        events.put({"event": "answer", "round": 1, "text": answer, "final": final,
                    "evidence": ev, "stop": True})
        seconds = time.perf_counter() - began
        summary = {
            "question": question, "seconds": round(seconds, 2),
            "first_token": None if first_token is None else round(first_token, 2),
            "rounds": 1, "tokens": tokens, "context": context_length, "read_tokens": len(context_ids),
            "retrieval_ms": round(retrieval * 1000, 2), "global_kind": global_kind,
            "global_metrics": global_metrics,
        }
        with self.lock:
            self.turns += 1
            self.history.append((question, final))
            self.last = summary
        events.put({"event": "done", "answer": answer, "final": final, **summary})


__all__ = ["HierarchicalChat"]
