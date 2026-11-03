"""Checks on re-read answers: miscopied lines, and the self-check turn's prompt."""

from __future__ import annotations

from deepinfer.moe.answer_check import (
    copy_candidates,
    miscopies,
    retry_note,
    self_check_prompt,
)

LINES = [
    "The special magic number for onyx ibis is: 1702635.",
    "The special magic number for cobalt heron is: 8981982.",
    "    def __init__(self, name, value=None):",
    "The special magic number for onyx badger is: 7646461.",
    "VAR GHTAY = 14673",
    "VAR OKOSG = VAR GHTAY",
]


def test_a_number_bound_to_the_wrong_key_is_a_miscopy() -> None:
    # the 4M answer: the onyx badger line with onyx ibis's number
    answer = 'The relevant sentence is:\n\n"The special magic number for onyx badger is: 1702635."\n\nAnswer: 1702635'
    found = miscopies(answer, LINES)
    assert [item["item"] for item in found] == ["1702635"]
    assert found[0]["line"] == "The special magic number for onyx ibis is: 1702635"


def test_a_number_no_line_has_is_a_miscopy() -> None:
    found = miscopies('"The special magic number for onyx badger is: 7646416."\nAnswer: 7646416', LINES)
    assert found and found[0]["line"] is None


def test_exact_copies_and_rewordings_pass() -> None:
    for answer in (
        '"The special magic number for onyx badger is: 7646461."\nAnswer: 7646461',
        "Therefore, the special magic number for onyx badger is 7646461.\nAnswer: 7646461",
        'The text states that "The special magic number for onyx badger is: 7646461."\nAnswer: 7646461',
        "- `VAR OKOSG = VAR GHTAY` (so `OKOSG = 14673`)\n- `OKOSG = GHTAY`\nAnswer: GHTAY, OKOSG",
    ):
        assert miscopies(answer, LINES) == [], answer


def test_copies_are_looked_for_before_the_final_answer_only() -> None:
    candidates = copy_candidates('Copied: "The special magic number for onyx ibis is: 1702635."\nAnswer: "not a copy at all here"')
    assert "The special magic number for onyx ibis is: 1702635" in candidates
    assert all("not a copy" not in c for c in candidates)


def test_retry_note_names_the_bad_copy() -> None:
    note = retry_note([{"copy": "The special magic number for onyx badger is: 1702635", "item": "1702635", "line": None}])
    assert '"The special magic number for onyx badger is: 1702635"' in note and "not in the text" in note


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


def test_self_check_turn_closes_the_answer_once() -> None:
    ids = self_check_prompt(_Tokenizer(), [5, 6, 0], stop_id=0)
    assert ids[:3] == [5, 6, 0] and ids.count(0) == 1
    text = "".join(map(chr, ids[3:]))
    assert text.startswith("User: Check your final answer") and text.endswith("\n\nAssistant:")
    assert self_check_prompt(_Tokenizer(), [5, 6], stop_id=0)[:3] == [5, 6, 0]
