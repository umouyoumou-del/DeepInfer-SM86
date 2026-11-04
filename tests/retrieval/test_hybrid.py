"""Line-preserving FTS5 retrieval and exact structured evidence."""

from __future__ import annotations

import sqlite3

from deepinfer.retrieval.hybrid import (
    build_index,
    chunk_text,
    exact_lines,
    facts_from_lines,
    retrieve,
    structured_answer,
)

TEXT = """def unrelated():
    pass

The special magic number for onyx badger is: 7646461.
The vault key of the Orrin observatory is held by Kaila Brenn.
The employee badge number of Kaila Brenn is 1877827.

The special magic number for onyx ibis is: 1702635.
"""


def test_chunk_text_keeps_lines_intact() -> None:
    chunks = chunk_text("a\nb\nc\nd", target_chars=5)
    assert chunks == ["a\nb", "c\nd"]
    assert all(not chunk.startswith("\n") for chunk in chunks)


def test_fts_retrieval_and_exact_two_hop(tmp_path) -> None:
    db = tmp_path / "index.sqlite"
    count = build_index(TEXT, db, target_chars=80)
    assert count >= 2
    con = sqlite3.connect(db)
    rows = retrieve(con, "onyx badger magic number", 4)
    con.close()
    assert any("onyx badger" in body for _, body in rows)

    facts = facts_from_lines(TEXT.splitlines())
    evidence = exact_lines(facts, "What is the employee badge number of the person who holds the vault key of the Orrin observatory?")
    assert "The vault key of the Orrin observatory is held by Kaila Brenn." in evidence
    assert "The employee badge number of Kaila Brenn is 1877827." in evidence


def test_structured_fast_paths() -> None:
    text = TEXT + chr(10) + "VAR GHTAY = 14673" + chr(10) + "VAR OKOSG = VAR GHTAY" + chr(10)
    facts = facts_from_lines(text.splitlines())
    assert structured_answer("What is the special magic number for onyx badger?", facts) == "7646461"
    assert structured_answer("What are all the special magic numbers for onyx badger?", facts) == "7646461"
    assert structured_answer("What is the employee badge number of the person who holds the vault key of the Orrin observatory?", facts) == "1877827"
    assert structured_answer("Find all variables assigned the value 14673", facts, text) == "GHTAY, OKOSG"
