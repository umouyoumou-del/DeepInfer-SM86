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
