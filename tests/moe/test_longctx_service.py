"""The model-free parts of the long-context chat service."""

from __future__ import annotations

from deepinfer.moe.longctx_service import (
    ASK,
    CONTROL,
    INGEST,
    JobQueue,
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
