"""Checks on an answer written over re-read text: copies that are not in the text, and a final
answer that leaves out what the answer's own steps found.

The answer instruction has the model copy the lines it needs word for word, reason, then
write ``Answer:``.  At 4M two answers went wrong that way with every needed line in front of
the model (docs/LONG_CONTEXT_1M.md §4.12):

* a copy that is not in the text, two near-identical lines blended into one
  ("...onyx badger is: 1702635." from "...onyx badger is: 7646461." and "...onyx ibis is:
  1702635."), and the answer taken from it;
* a final pick that drops an item the steps derived (its own table says AVCOU = 14673, the
  answer lists four names without it).

:func:`miscopies` finds the first kind: every quoted, back-ticked or listed span of the answer
that is close to a line of the text but carries a word or number that line does not have.
:func:`self_check_prompt` is a second turn for the second kind: the model reads its own answer
back and writes the final answer again.  Both are generic - nothing here knows the question.

For a question that asks which names hold a value ("... assigned the value 14673"),
:func:`value_pick` reads the value the answer's own steps give each name and picks the ones
holding the asked value - the step the model gets wrong is picking, not deriving: over 52
variable-chain contexts its own picks matched what the lines imply 31 times, picks read off
its derivations 48 times, and on every saved run this changed only wrong answers into right
ones (scripts/answer_bench.py, docs/LONG_CONTEXT_1M.md §4.15).
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

_SPAN = re.compile(r'"([^"\n]{12,})"|“([^”\n]{12,})”|`([^`\n]{8,})`')
_BULLET = re.compile(r"^\s*(?:[-*>•]|\d+[.)])\s+")
_WORD = re.compile(r"\w+")


def _normalized(text: str) -> str:
    text = text.replace("**", "").replace("`", "").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).strip(" \"'.,;:")


def _reasoning(answer: str) -> str:
    """The answer before its last ``Answer:`` - where the copies are."""

    index = answer.lower().rfind("answer:")
    return answer[:index] if index >= 0 else answer


def copy_candidates(answer: str) -> list[str]:
    """Spans of the answer that may be copies: quoted or back-ticked text, and its lines (bullets stripped)."""

    body = _reasoning(answer)
    spans = [next(group for group in match.groups() if group) for match in _SPAN.finditer(body)]
    lines = [_BULLET.sub("", line) for line in body.splitlines()]
    seen, candidates = set(), []
    for text in spans + lines:
        text = _normalized(text)
        if len(text) >= 12 and text not in seen:
            seen.add(text)
            candidates.append(text)
    return candidates


_ITEM = re.compile(r"\b(?:\d{3,}|[A-Z][A-Z0-9]{3,})\b")


def _words(text: str) -> set[str]:
    """Lower-cased words, plural ``s`` dropped (``numbers`` and ``number`` are the same word here)."""

    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in _WORD.findall(text.lower())}


def miscopies(answer: str, lines: Sequence[str], *, close: float = 0.8) -> list[dict]:
    """Copies in ``answer`` that bind a number or code to the wrong thing (``lines``: the text it read).

    A span is an attempted copy when it is at least ``close`` similar (difflib ratio, ignoring
    case) to a line of the text and not part of the text.  It is a miscopy when it carries a
    number or code (3+ digits, or an all-caps name) that no line has, or that sits in a line
    which says something else: the span has words of the text that this line lacks *and*
    this line has words the span lacks ("onyx badger ... 1702635" against "onyx ibis ...
    1702635").  Rewordings that keep a line's words ("Therefore, the special magic number for
    onyx badger is 7646461") and reasoning that only names things are left alone.
    """

    sources = [_normalized(line) for line in lines if _normalized(line)]
    if not sources:
        return []
    lowered = [source.lower() for source in sources]
    text = "\n".join(lowered)
    vocabulary = _words(text)
    found = []
    for candidate in copy_candidates(answer):
        low = candidate.lower()
        if low in text:
            continue
        if max(difflib.SequenceMatcher(None, low, source).ratio() for source in lowered) < close:
            continue
        own = _words(candidate)
        for item in _ITEM.findall(candidate):
            holders = [i for i, source in enumerate(sources) if re.search(rf"\b{re.escape(item)}\b", source)]
            others = [i for i in range(len(sources)) if i not in holders]
            # the line it most resembles without that item: what it probably meant to copy
            nearest = (
                sources[max(others, key=lambda i: difflib.SequenceMatcher(None, low, lowered[i]).ratio())]
                if others else None
            )
            if not holders:
                found.append({"copy": candidate, "item": item, "line": None, "nearest": nearest})
                break
            line = max(holders, key=lambda i: difflib.SequenceMatcher(None, low, lowered[i]).ratio())
            theirs = _words(sources[line])
            if (own - theirs) & vocabulary and theirs - own:
                found.append({"copy": candidate, "item": item, "line": sources[line], "nearest": nearest})
                break
    return found


def retry_note(found: Sequence[dict]) -> str:
    """Appended to the question when an answer miscopied: which copies are not in the text, and the
    text's own lines they were made from (the one holding the number, and the one most like the copy)."""

    parts = []
    for item in found[:3]:
        lines = list(dict.fromkeys(line for line in (item.get("line"), item.get("nearest")) if line))
        where = "; the text has " + " and ".join(f'"{line}"' for line in lines) if lines else ""
        parts.append(f'"{item["copy"]}", which is not in the text{where}')
    return (
        "Careful: an earlier attempt quoted " + "; ".join(parts) + ". Copy the lines exactly as they are "
        "written in the text, reason again, and end with 'Answer:' as asked."
    )


SELF_CHECK = (
    "Check your final answer against the text above and against your own steps: it must include "
    "everything your steps found that answers the question, and nothing else. Then write 'Answer:' "
    "followed by the final answer again, corrected if needed, in the same form."
)


def self_check_prompt(tokenizer, answer_ids: Sequence[int], stop_id: int | None) -> list[int]:
    """Ids that close the assistant's answer and ask the :data:`SELF_CHECK` turn (DeepSeek's chat format)."""

    ids = list(answer_ids)
    if stop_id is not None and (not ids or ids[-1] != stop_id):
        ids.append(stop_id)
    return ids + tokenizer.encode(f"User: {SELF_CHECK}\n\nAssistant:", add_special_tokens=False)


# --------------------------------------------------------------- value pick

_NAME_OR_NUMBER = re.compile(r"\b(?:([A-Z][A-Z0-9]{2,})|(\d{3,}))\b")
_NAME = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_NEGATION = re.compile(r"\b(?:not|never|no longer|instead of|rather than|isn't|doesn't)\b", re.IGNORECASE)
_NOT_NAMES = frozenset({"VAR", "ANSWER"})
STEPS_NOTE = "[read off the steps above]"


def asked_value(question: str) -> str | None:
    """The value a question asks about ("assigned the value 14673"): its one number of 3+ digits."""

    numbers = set(re.findall(r"\b\d{3,}\b", question))
    return numbers.pop() if len(numbers) == 1 else None


def derived_values(text: str) -> dict[str, str]:
    """The last value the steps of ``text`` give each all-caps name, in the order they first name it.

    On every line before the final ``Answer:`` a number belongs to the nearest name before it -
    "VAR GHTAY = 14673", "so `AVCOU` is 14673", "(This sets `QHUJP` to 14673)", "| INRCW | 14673 |"
    - unless a negation stands between them ("..., which is 43291 (not directly 14673)").
    """

    last: dict[str, str] = {}
    for line in _reasoning(text).splitlines():
        name, end = None, 0
        for match in _NAME_OR_NUMBER.finditer(line):
            if match.group(1):
                if match.group(1) not in _NOT_NAMES:
                    name, end = match.group(1), match.end()
            elif name is not None and not _NEGATION.search(line[end : match.start()]):
                last[name] = match.group(2)
    return last


def _final_names(text: str) -> list[str]:
    index = text.lower().rfind("answer:")
    final = text[index + len("answer:") :] if index >= 0 else text
    return [name for name in dict.fromkeys(_NAME.findall(final)) if name not in _NOT_NAMES]


def value_pick(question: str, texts: Sequence[str]) -> list[str] | None:
    """For a question naming one value, the names that the steps of ``texts`` (one or several
    phrasings) give that value, plus the names an answer lists that no step gives any value;
    ``None`` when the question names no value or no step gives it.

    Each name goes by its phrasings' majority: one phrasing deriving it wrong is outvoted.
    """

    value = asked_value(question)
    if value is None:
        return None
    votes: dict[str, list[int]] = {}  # name -> [phrasings giving it the value, giving it another]
    written: list[str] = []
    for text in texts:
        for name, held in derived_values(text).items():
            votes.setdefault(name, [0, 0])[0 if held == value else 1] += 1
        written += [name for name in _final_names(text) if name not in written]
    picked = [name for name, (holds, other) in votes.items() if holds and holds >= other]
    if not picked:
        return None
    asked = set(_NAME.findall(question))
    return [name for name in picked + [n for n in written if n not in votes] if name not in asked]


def with_value_pick(question: str, text: str) -> str:
    """``text`` with its final answer read off its own steps appended, when that differs from what it wrote."""

    names = value_pick(question, [text])
    if names is None or set(names) == set(_final_names(text)):
        return text
    return f"{text}\n\n{STEPS_NOTE}\nAnswer: {', '.join(names)}"


_VAR_ASSIGNMENT = re.compile(
    r"^\s*VAR\s+([A-Z][A-Z0-9]{2,})\s*=\s*(?:(\d{3,})|VAR\s+([A-Z][A-Z0-9]{2,}))\s*$",
    re.MULTILINE,
)


def var_chain_answer(question: str, text: str) -> list[str] | None:
    """Resolve the document's ``VAR X = N`` / ``VAR X = VAR Y`` lines for a value question.

    This is the exact-retrieval companion to :func:`value_pick`: when the model's
    locate misses one or more assignment lines, the complete document still has
    enough information to close the variable chain.  Return the names whose
    transitively resolved value is the value named by the question, in document
    order; ``None`` when the question does not name one value or the document has
    no such assignments.
    """

    value = asked_value(question)
    if value is None:
        return None
    assignments: dict[str, str] = {}
    for match in _VAR_ASSIGNMENT.finditer(text):
        name = match.group(1)
        if name in _NOT_NAMES:
            continue
        assignments[name] = match.group(2) or match.group(3)
    if not assignments:
        return None

    memo: dict[str, str | None] = {}

    def resolve(name: str, stack: frozenset[str] = frozenset()) -> str | None:
        if name in memo:
            return memo[name]
        if name in stack or name not in assignments:
            return None
        raw = assignments[name]
        value_here = raw if raw.isdigit() else resolve(raw, stack | {name})
        memo[name] = value_here
        return value_here

    picked = [name for name in assignments if resolve(name) == value]
    return picked or None
