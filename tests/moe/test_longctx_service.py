"""The model-free parts of the long-context chat service."""

from __future__ import annotations

import threading

import pytest

from deepinfer.moe.longctx_service import (
    ASK,
    CONTROL,
    INGEST,
    JobQueue,
    LongContextChat,
    Source,
    TextStream,
    locate_source,
)


def test_a_question_goes_before_the_rest_of_an_ingest_and_ingests_keep_their_order() -> None:
    jobs = JobQueue()
    first = jobs.put(INGEST, "book, chunk 1")
    jobs.put(INGEST, "notes, chunk 1")
    _, seq, job = jobs.get()
    assert job == "book, chunk 1" and seq == first
    jobs.put(INGEST, "book, chunk 2", seq=seq)  # the remainder keeps its place
    jobs.put(ASK, "question")
    jobs.put(CONTROL, "reset")
    order = [jobs.get()[2] for _ in range(4)]
    assert order == ["reset", "question", "book, chunk 2", "notes, chunk 1"]
    assert jobs.empty()


def test_take_ingest_only_takes_an_ingest_job_at_the_front() -> None:
    jobs = JobQueue()
    assert jobs.take_ingest() is None
    jobs.put(INGEST, "notes")
    jobs.put(ASK, "question")
    assert jobs.take_ingest() is None  # the question is first and stays queued
    assert jobs.get()[2] == "question"
    assert jobs.take_ingest()[2] == "notes"
    assert jobs.empty()


class _Session:
    def __init__(self) -> None:
        self.next_position = 64
        self.calls: list[list[int]] = []

    def prefill(self, ids) -> None:
        self.calls.append(list(ids))
        self.next_position += len(ids)


def _chat(chunk: int) -> LongContextChat:
    chat = object.__new__(LongContextChat)  # the bookkeeping only: no model, no worker thread
    chat.session, chat.chunk, chat.jobs, chat.lock = _Session(), chunk, JobQueue(), threading.Lock()
    chat.queued_tokens, chat.rate, chat.state, chat._baseline_length = 0, 0.0, "idle", 0
    return chat


def _queue(chat: LongContextChat, name: str, count: int) -> Source:
    source = Source(name=name, tokens=count)
    chat.queued_tokens += count
    chat.jobs.put(INGEST, ("ingest", source, list(range(count))))
    return source


def test_small_sources_share_one_chunk_and_keep_their_positions(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    chat = _chat(chunk=10)
    a, b, c = _queue(chat, "a.py", 4), _queue(chat, "b.py", 3), _queue(chat, "c.py", 8)
    _, seq, job = chat.jobs.get()
    chat._ingest(seq, job[1], job[2])
    assert [len(call) for call in chat.session.calls] == [10]  # a, b and the first 3 of c in one call
    assert (a.segments, b.segments, c.segments) == ([[64, 4]], [[68, 3]], [[71, 3]])
    assert chat.queued_tokens == 5
    _, seq, job = chat.jobs.get()  # the rest of c, next in line
    assert job[1] is c and len(job[2]) == 5
    chat._ingest(seq, job[1], job[2])
    assert c.segments == [[71, 8]] and c.done == 8 and chat.queued_tokens == 0


def test_a_waiting_question_stops_the_chunk_from_filling(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    chat = _chat(chunk=10)
    _queue(chat, "a.py", 4)
    b = _queue(chat, "b.py", 3)
    _, seq, job = chat.jobs.get()
    chat.jobs.put(ASK, "question")
    chat._ingest(seq, job[1], job[2])
    assert [len(call) for call in chat.session.calls] == [4] and not b.segments
    assert chat.jobs.get()[2] == "question"


def test_the_prompt_keeps_the_first_turns_and_the_latest() -> None:
    chat = _chat(chunk=10)
    chat.history = [(f"question {n}", f"answer {n}") for n in range(1, 4)]
    assert chat._recent(turns=16).splitlines()[1:3] == ["Q1: question 1", "A1: answer 1"]
    chat.history = [(f"question {n}", f"answer {n}") for n in range(1, 21)]
    lines = chat._recent(turns=16).splitlines()
    questions = [line for line in lines if line.startswith("Q")]
    assert questions[:2] == ["Q1: question 1", "Q2: question 2"]  # "what did I ask first" stays answerable
    assert questions[2:] == [f"Q{n}: question {n}" for n in range(7, 21)] and "..." in lines
    assert len(questions) == 16


def test_positions_map_to_sources_across_interleaved_segments() -> None:
    book = Source(name="book", tokens=300)
    turns = Source(name="对话", tokens=20, kind="conversation")
    book.add(10, 100)
    book.add(110, 100)  # contiguous: merged
    turns.add(210, 20)  # a question answered while the book was still being read
    book.add(230, 100)
    assert book.segments == [[10, 200], [230, 100]]
    sources = [book, turns]
    assert locate_source(sources, 5) is None  # the header
    assert locate_source(sources, 10) == ("book", 0)
    assert locate_source(sources, 215) == ("对话", 5)
    assert locate_source(sources, 230) == ("book", 200)
    assert locate_source(sources, 329) == ("book", 299)
    assert locate_source(sources, 330) is None


class _Bytes:
    """Each id is one UTF-8 byte; an incomplete character decodes to the replacement mark."""

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="replace")


def test_text_stream_holds_back_a_character_split_across_tokens() -> None:
    stream = TextStream(_Bytes())
    encoded = list("答案：7".encode())
    assert stream.feed(encoded[:2]) == ""  # half of 答
    assert stream.feed(encoded[2:4]) == "答"  # 答 completed, half of 案 held back
    assert stream.feed(encoded[4:]) == "案：7"
    assert stream.text == "答案：7"
