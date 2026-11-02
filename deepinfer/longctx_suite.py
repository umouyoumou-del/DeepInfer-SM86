"""The complex needle-in-a-haystack suite: 39 needles in one long document, 21 questions.

RULER-style kinds - many similar keys (MK), one key with several values (MV), several keys
in one question (MQ), a chain of variable assignments with a distractor chain (VT) and
two-hop facts (2H) - hidden in the Python standard library's source.  Used by
``scripts/needle_complex.py`` (the evaluation) and by ``deepinfer-server``'s demo document.
The text is generated deterministically from the seed, so a document and its hash are the
same wherever it is built (a saved cache is checked against the hash).
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import os
import random
import re
import string

from deepinfer.moe.latent_reread import final_answer


def haystack_tokens(tok, root: str, count: int) -> list[int]:
    """The first ``count`` tokens of the concatenated standard library."""

    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    body: list[int] = []
    for path in files:
        if "site-packages" in path:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        ids = tok.encode(f"\n# {os.path.relpath(path, root)}\n{text}", add_special_tokens=False)
        body.extend(ids[: count - len(body)])
        if len(body) >= count:
            return body
    raise RuntimeError(f"haystack has only {len(body):,} tokens, {count:,} needed")


def line_start(pieces: list[str], depth: float, length: int) -> int:
    """Index in the first ``length`` body tokens at ``depth``, moved to a line start."""

    cut = min(int(depth * length), length)
    limit = min(length, cut + 4096)
    while 0 < cut < limit and not pieces[cut - 1].endswith("Ċ"):
        cut += 1
    return cut


def header_ids(tok) -> list[int]:
    return [tok.bos_token_id] + tok.encode("User: ", add_special_tokens=False)


def document_hash(ids: list[int]) -> str:
    return hashlib.sha256(" ".join(map(str, ids)).encode()).hexdigest()


MK_KEYS = [
    "crimson falcon", "crimson otter", "silver falcon", "silver otter",
    "amber lynx", "amber heron", "cobalt lynx", "cobalt heron",
    "ivory mantis", "ivory panther", "jade mantis", "jade panther",
    "scarlet ibis", "scarlet badger", "onyx ibis", "onyx badger",
]
MV_KEY = "golden viper"
MV_NEAR = ["golden walrus", "azure viper"]
MQ_KEYS = ["crimson falcon", "amber heron", "jade mantis", "onyx badger"]
TWO_HOP = [
    {
        "bridge": "The vault key of the Orrin observatory is held by {person}.",
        "target": "The employee badge number of {person} is {code}.",
        "question": "What is the employee badge number of the person who holds the vault key of the Orrin observatory?",
        "person": "Kaila Brenn",
        "others": ["Talia Rusk", "Joren Vale"],
    },
    {
        "bridge": "The Teal Harbor lighthouse was designed by the architect {person}.",
        "target": "The personal safe of {person} opens with the combination {code}.",
        "question": "Which combination opens the personal safe of the architect who designed the Teal Harbor lighthouse?",
        "person": "Marek Ostrov",
        "others": ["Ilse Varga", "Tomas Reed"],
    },
]
# One instruction for every question: copy the evidence, reason, then answer.  Copying
# makes the model match the exact key / name before it commits to a number; reasoning
# step by step is what lets it follow a chain (probe: asked directly it stops after one
# hop, traced step by step it gets all five); the answer feeds the next round's locate.
QUOTE_FIRST = (
    "First copy, word for word, every sentence or line of the text above that is needed to answer "
    "the question. Then reason step by step. Finally write 'Answer:' followed by {what}."
)
CODE_ONLY = QUOTE_FIRST.format(what="the 7-digit number only")


def build_suite(seed: int) -> tuple[list[dict], list[dict]]:
    """(needles, questions); needles carry their text, questions their expected items."""

    rng = random.Random(seed)
    used: set[str] = set()

    def code() -> str:
        while True:
            value = str(rng.randrange(1_000_000, 10_000_000))
            if value not in used:
                used.add(value)
                return value

    def name() -> str:
        while True:
            value = "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
            if value not in used:
                used.add(value)
                return value

    needles: list[dict] = []
    questions: list[dict] = []

    def needle(text: str, group: str) -> int:
        needles.append({"text": text, "group": group})
        return len(needles) - 1

    mk = {}
    for key in MK_KEYS:
        value = code()
        mk[key] = (value, needle(f"The special magic number for {key} is: {value}.", "MK"))
    for key in MK_KEYS:
        value, index = mk[key]
        questions.append(
            {"kind": "MK", "question": f"What is the special magic number for {key}?",
             "instruction": CODE_ONLY, "expected": [value], "exact": True, "needles": [index]}
        )

    mv_values, mv_needles = [], []
    for _ in range(4):
        value = code()
        mv_values.append(value)
        mv_needles.append(needle(f"One of the special magic numbers for {MV_KEY} is: {value}.", "MV"))
    for key in MV_NEAR:
        needle(f"One of the special magic numbers for {key} is: {code()}.", "MV-near")
    questions.append(
        {"kind": "MV", "question": f"What are all the special magic numbers for {MV_KEY}?",
         "instruction": QUOTE_FIRST.format(what="all of the numbers, separated by commas"),
         "expected": mv_values, "exact": False, "needles": mv_needles}
    )

    questions.append(
        {"kind": "MQ",
         "question": "What are the special magic numbers for " + ", ".join(MQ_KEYS[:-1]) + f" and {MQ_KEYS[-1]}?",
         "instruction": QUOTE_FIRST.format(what="one line per key in the form key: number"),
         "expected": [mk[key][0] for key in MQ_KEYS], "exact": False,
         "needles": [mk[key][1] for key in MQ_KEYS]}
    )

    def chain(links: int, group: str) -> tuple[str, list[str], list[int]]:
        value = str(rng.randrange(10_000, 100_000))
        names = [name() for _ in range(links)]
        indices = [needle(f"VAR {names[0]} = {value}", group)]
        for previous, current in itertools.pairwise(names):
            indices.append(needle(f"VAR {current} = VAR {previous}", group))
        return value, names, indices

    value, names, indices = chain(5, "VT")
    _, decoy_names, _ = chain(4, "VT-decoy")
    # Asked to chain backwards ("which variables get the value"), this model drops the
    # directly assigned one or wanders into the decoy chain - on nine clean lines, with
    # no long context involved.  Asked to run the lines forward and keep a table of
    # values (what the question means), its table is right every time; only the last
    # step, picking the rows with the value, drops a name now and then (a different
    # one from run to run, never a decoy: those are computed as the other value).  So
    # the final pick is asked three ways and the names are united (self-consistency).
    # The re-read lines are in document order, which is the order the program runs in.
    # At 2M the re-read lines are mostly real Python about assignments and chains (the
    # haystack is the standard library), and the model starts explaining that code; so
    # the instruction says what the text is, as RULER's VT template does ("track the
    # chain(s) of variable assignment hidden in the following text").
    simulate = (
        "The text above is unrelated program code with a few lines of the form VAR X = ... mixed in; only those "
        "VAR lines matter, ignore everything else. Go through the VAR lines in order and keep track of every "
        "variable's value: a line VAR X = <number> sets X to that number, and a line VAR B = VAR A sets B to the "
        "current value of A. "
    )
    questions.append(
        {"kind": "VT",
         "question": f"Find all variables that are assigned the value {value}, directly or through a chain of "
                     "assignments (VAR B = VAR A gives B the value of A).",
         "instruction": simulate + f"After the last line, list every variable whose value is {value}. Finally write "
                        "'Answer:' followed by their names, separated by commas.",
         "variants": [
             simulate + "After the last line, write out the table of every variable and its value, one row per "
                        f"variable, then copy every row whose value is {value}. Finally write 'Answer:' followed by "
                        "the names in the copied rows, separated by commas.",
             simulate + f"Each time a variable gets the value {value}, write it down at once. Finally write "
                        "'Answer:' followed by every variable you wrote down, separated by commas.",
         ],
         "expected": names, "forbidden": decoy_names, "exact": False, "needles": indices}
    )

    for item in TWO_HOP:
        value = code()
        bridge = needle(item["bridge"].format(person=item["person"]), "2H")
        target = needle(item["target"].format(person=item["person"], code=value), "2H")
        for other in item["others"]:
            needle(item["target"].format(person=other, code=code()), "2H-decoy")
        questions.append(
            {"kind": "2H", "question": item["question"], "instruction": CODE_ONLY,
             "expected": [value], "exact": True, "needles": [bridge, target]}
        )

    # spread the needles over the document; the shuffle puts keys, values, bridges
    # and targets in arbitrary order and far apart
    depths = [0.02 + 0.96 * i / (len(needles) - 1) for i in range(len(needles))]
    rng.shuffle(depths)
    for item, depth in zip(needles, depths, strict=True):
        item["depth"] = round(depth, 4)
    # a chain is a program: each link comes after the one it reads (at random,
    # spread-out depths), so "assigned through a chain" has one meaning
    for group in ("VT", "VT-decoy"):
        links = [item for item in needles if item["group"] == group]
        for item, depth in zip(links, sorted(item["depth"] for item in links), strict=True):
            item["depth"] = depth
    return needles, questions


def build_document(tok, target: int, needles: list[dict], root: str) -> list[int]:
    """``[BOS] User: <haystack with every needle on its own line>``, exactly ``target`` tokens."""

    header = header_ids(tok)
    ids = [tok.encode(f"\n{item['text']}\n", add_special_tokens=False) for item in needles]
    body_length = target - len(header) - sum(len(x) for x in ids)
    body = haystack_tokens(tok, root, body_length)
    pieces = tok.convert_ids_to_tokens(body)
    cuts = sorted(
        ((line_start(pieces, item["depth"], body_length), item, x) for item, x in zip(needles, ids, strict=True)),
        key=lambda cut: cut[0],
    )
    document = list(header)
    previous = 0
    for cut, item, x in cuts:
        document.extend(body[previous:cut])
        item["position"] = len(document)
        item["length"] = len(x)
        document.extend(x)
        previous = cut
    document.extend(body[previous:body_length])
    return document


def score(question: dict, answer: str) -> dict:
    """Score the final answer - the text after the last ``Answer:`` - not the copied sentences."""

    answer = final_answer(answer)
    if question["kind"] == "VT":
        found = set(re.findall(r"\b[A-Z]{5}\b", answer))
    else:
        found = set(re.findall(r"(?<!\d)\d{7}(?!\d)", answer))
    expected = question["expected"]
    hits = [item for item in expected if item in found]
    extras = sorted(found - set(expected))
    forbidden = sorted(found & set(question.get("forbidden", [])))
    full = len(hits) == len(expected) and (not question["exact"] or not extras) and not forbidden
    return {"hits": hits, "extras": extras, "forbidden": forbidden, "full": full}


NEUTRAL_PROMPT = "\n\nSummarize the text above in one sentence.\n\nAssistant:"
