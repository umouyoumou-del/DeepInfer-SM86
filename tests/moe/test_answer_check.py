"""Checks on re-read answers: miscopied lines, the self-check turn's prompt, the value pick."""

from __future__ import annotations

from deepinfer.moe.answer_check import (
    STEPS_NOTE,
    asked_value,
    copy_candidates,
    derived_values,
    miscopies,
    retry_note,
    self_check_prompt,
    value_pick,
    var_chain_answer,
    with_value_pick,
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


VT_QUESTION = "Find all variables that are assigned the value 14673, directly or through a chain of assignments."

# the 4M answer (seed 11): every value derived right, AVCOU left out of the pick
DROPPED = """1. **Initialize the variables:**
   - `KYHQJ = 43291`
   - `GHTAY = 14673`
   - `OKOSG = GHTAY` (so `OKOSG = 14673`)
2. **Assignments involving `OKOSG`:**
   - `ACSYD = KYHQJ` (so `ACSYD = 43291`)
   - `QHUJP = OKOSG` (so `QHUJP = 14673`)
   - `MDJMC = ACSYD` (so `MDJMC = 43291`)
   - `AVCOU = QHUJP` (so `AVCOU = 14673`)
   - `AVAGG = MDJMC` (so `AVAGG = 43291`)
   - `INRCW = AVCOU` (so `INRCW = 14673`)
3. **List all variables whose value is 14673:**
   - `GHTAY`
   - `OKOSG`
   - `QHUJP`
   - `INRCW`

**Answer:**
GHTAY, OKOSG, QHUJP, INRCW"""


def test_asked_value_is_the_questions_one_number() -> None:
    assert asked_value(VT_QUESTION) == "14673"
    assert asked_value("What is the special magic number for onyx badger?") is None
    assert asked_value("Is it 1702635 or 7646461?") is None  # two numbers: no single value asked


def test_derived_values_follow_the_steps_and_skip_negations() -> None:
    values = derived_values(DROPPED)
    assert values["AVCOU"] == "14673" and values["MDJMC"] == "43291" and values["INRCW"] == "14673"
    line = "- `AVAGG` is assigned the value of `MDJMC`, which is 43291 (not directly 14673).\nAnswer: none"
    assert derived_values(line) == {"MDJMC": "43291"}
    assert derived_values("- `QHUJP` (This sets `QHUJP` to 14673)\n| AVCOU | 14673 |") == {"QHUJP": "14673", "AVCOU": "14673"}


def test_value_pick_restores_a_dropped_name() -> None:
    assert value_pick(VT_QUESTION, [DROPPED]) == ["GHTAY", "OKOSG", "QHUJP", "AVCOU", "INRCW"]
    amended = with_value_pick(VT_QUESTION, DROPPED)
    assert amended.startswith(DROPPED) and amended.endswith(f"{STEPS_NOTE}\nAnswer: GHTAY, OKOSG, QHUJP, AVCOU, INRCW")
    picked = DROPPED.replace("GHTAY, OKOSG, QHUJP, INRCW", "GHTAY, OKOSG, QHUJP, AVCOU, INRCW")
    assert with_value_pick(VT_QUESTION, picked) == picked  # a pick that agrees with the steps is left alone


def test_value_pick_drops_a_name_its_steps_give_another_value() -> None:
    added = DROPPED.replace("GHTAY, OKOSG, QHUJP, INRCW", "GHTAY, OKOSG, QHUJP, AVCOU, MDJMC, INRCW")
    assert "MDJMC" not in value_pick(VT_QUESTION, [added])


def test_value_pick_keeps_names_the_steps_say_nothing_about() -> None:
    copied = "VAR GHTAY = 14673\nVAR OKOSG = VAR GHTAY\nVAR QHUJP = VAR OKOSG\n\nAnswer: GHTAY, OKOSG, QHUJP"
    assert value_pick(VT_QUESTION, [copied]) == ["GHTAY", "OKOSG", "QHUJP"]


def test_value_pick_goes_by_the_majority_of_phrasings() -> None:
    wrong = "- `MDJMC` is 14673\n- `GHTAY` is 14673\nAnswer: GHTAY, MDJMC"
    right = "- `MDJMC` is 43291\n- `GHTAY` is 14673\nAnswer: GHTAY"
    assert value_pick(VT_QUESTION, [wrong, right, right]) == ["GHTAY"]
    assert value_pick("What is the special magic number for onyx badger?", [DROPPED]) is None


def test_var_chain_answer_closes_a_dropped_chain() -> None:
    text = """VAR KYHQJ = 43291
VAR GHTAY = 14673
VAR OKOSG = VAR GHTAY
VAR ACSYD = VAR KYHQJ
VAR QHUJP = VAR OKOSG
VAR MDJMC = VAR ACSYD
VAR AVCOU = VAR QHUJP
VAR AVAGG = VAR MDJMC
VAR INRCW = VAR AVCOU"""
    assert var_chain_answer(VT_QUESTION, text) == ["GHTAY", "OKOSG", "QHUJP", "AVCOU", "INRCW"]
    assert var_chain_answer(VT_QUESTION, "VAR GHTAY = 14673") == ["GHTAY"]
    assert var_chain_answer(VT_QUESTION, "no assignments here") is None


def test_var_chain_answer_rejects_cycles_and_questions_without_one_value() -> None:
    cycle = "VAR AAAAA = VAR BBBBB\nVAR BBBBB = VAR AAAAA"
    assert var_chain_answer(VT_QUESTION, cycle) is None
    assert var_chain_answer("What is the magic number for onyx badger?", "VAR GHTAY = 14673") is None
