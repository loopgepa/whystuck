"""IFBench loader + deterministic programmatic scorer.

WhyStuck's validation study uses IFBench (instruction-following with programmatic
constraint verifiers) so the "loop-level GEPA beats prompt-only GEPA" claim is
tested on the same dataset GEPA's authors published on. Verification is code, not
an LLM judge — so scoring is deterministic and free.

The verifier functions below are faithful ports of IFBench's `check_following`
implementations (allenai/IFBench, Apache-2.0 — see data/ATTRIBUTION.md). Only the
parameter-free, nltk-free, package-free subset is ported (11 of 58 constraints),
so the whole path runs offline with no per-row kwargs. The vendored data slice
(data/ifbench_supported.jsonl) holds exactly the rows those constraints cover.

Two scores are exposed, both standard IFEval conventions:
- `ifbench_pass` — prompt-level: every constraint on the row satisfied (bool).
- `ifbench_score` — instruction-level: fraction of the row's constraints satisfied.
The study feeds `ifbench_pass` to the worker (a clean binary GEPA signal) and
reports both on the held-out split.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from whystuck.eval.benchmark import BenchmarkTask

_DATA = Path(__file__).parent / "data" / "ifbench_supported.jsonl"

Verifier = Callable[[str], bool]


# --- verifiers (faithful ports of IFBench check_following; Apache-2.0) ---------


def _no_whitespace(value: str) -> bool:
    return not any(char.isspace() for char in value)


def _nested_parentheses(value: str) -> bool:
    levels: list[str] = []
    min_levels, max_depth = 5, 0
    for char in value:
        if char in "([{":
            levels.append(char)
            max_depth = max(max_depth, len(levels))
        elif char in ")]}":
            if levels and (
                (levels[-1] == "(" and char == ")")
                or (levels[-1] == "[" and char == "]")
                or (levels[-1] == "{" and char == "}")
            ):
                levels.pop()
                if max_depth >= min_levels and len(levels) < max_depth:
                    return True
            else:
                levels, max_depth = [], 0
    return False


def _nested_quotes(value: str) -> bool:
    levels: list[str] = []
    min_levels, reached_depth, current_depth = 3, 0, 0
    for char in value:
        if levels and char == levels[-1]:
            levels.pop()
            current_depth -= 1
            if reached_depth - current_depth >= min_levels:
                return True
        elif char in ('"', "'"):
            levels.append(char)
            current_depth += 1
            reached_depth = max(reached_depth, current_depth)
    return False


def _punctuation_cover(value: str) -> bool:
    punctuation = {".", ",", "!", "?", ";", ":"}
    if not ("!?" in value or "?!" in value or "‽" in value):
        return False
    new_value = value.replace("?!", "", 1)
    if len(new_value) == len(value):
        new_value = value.replace("!?", "", 1)
    for char in new_value:
        punctuation.discard(char)
    return not punctuation


def _palindromes(value: str) -> bool:
    value = value.translate(str.maketrans("", "", string.punctuation))
    words = value.lower().split()
    palindromes = [w for w in words if w == w[::-1] and len(w) >= 5]
    return len(palindromes) >= 10


def _print_multiples(value: str) -> bool:
    value = value.replace(",", ", ")
    numbers = re.findall(r"\d+", value)
    multiples = [str(i) for i in range(14, 51, 7)]
    return numbers == multiples


def _single_vowel_paragraph(value: str) -> bool:
    paragraphs = value.strip().split("\n")
    if len(paragraphs) != 1:
        return False
    paragraph = paragraphs[0].lower()
    vowels = set("aeiou")
    return len({c for c in paragraph if c in vowels}) <= 3


def _consonant_clusters(value: str) -> bool:
    words = value.lower().strip().split()
    consonants = set("bcdfghjklmnpqrstvwxyz")
    for word in words:
        cluster = any(
            word[i] in consonants and word[i + 1] in consonants for i in range(len(word) - 1)
        )
        if not cluster:
            return False
    return True


def _newline_words(value: str) -> bool:
    value = value.translate(str.maketrans("", "", string.punctuation))
    lines = [ln for ln in value.strip().split("\n") if ln != ""]
    return len(lines) == len(value.strip().split())


def _quote_explanation(value: str) -> bool:
    # Faithful to upstream: its two leading .replace('"', '"') calls are ASCII
    # no-ops (both chars are U+0022) — verified by codepoint against the source.
    # We deliberately do NOT normalize typographic quotes; that would diverge and
    # under-score smart-quote-terminated outputs.
    value = value.replace("'\"'", "")
    value = "".join(value.split())
    if '""' in value:
        return False
    stripped = value.strip(string.digits + string.punctuation.replace('"', ""))
    return not (stripped and stripped[-1] == '"')


def _sub_bullets(value: str) -> bool:
    bullets = value.split("*")
    return all("-" in bullet for bullet in bullets[1:])


VERIFIERS: dict[str, Verifier] = {
    "format:no_whitespace": _no_whitespace,
    "format:parentheses": _nested_parentheses,
    "format:quotes": _nested_quotes,
    "count:punctuation": _punctuation_cover,
    "words:palindrome": _palindromes,
    "custom:multiples": _print_multiples,
    "words:vowel": _single_vowel_paragraph,
    "words:consonants": _consonant_clusters,
    "format:newline": _newline_words,
    "format:quote_unquote": _quote_explanation,
    "format:sub-bullets": _sub_bullets,
}

SUPPORTED_IDS = frozenset(VERIFIERS)


# --- scoring ------------------------------------------------------------------


def _constraints(task: BenchmarkTask) -> list[str]:
    return list((task.payload or {}).get("instruction_id_list", []))


def ifbench_score(output: str, task: BenchmarkTask) -> float:
    """Instruction-level: fraction of the row's constraints satisfied ∈ [0, 1].

    An empty constraint list returns 0.0, not the vacuous 1.0 — an unscored row
    must not count as a win (mirrors ``keyword_score``). A constraint id outside
    the supported set raises ``KeyError`` on purpose: ``load_ifbench`` only emits
    supported ids, so an unknown id means a caller bug, which should fail loud
    rather than silently skip and inflate the score.
    """
    ids = _constraints(task)
    if not ids:
        return 0.0
    met = sum(1 for iid in ids if VERIFIERS[iid](output))
    return met / len(ids)


def ifbench_pass(output: str, task: BenchmarkTask) -> bool:
    """Prompt-level: every constraint on the row satisfied.

    Empty constraint list returns False (not vacuous True); an unsupported id
    raises ``KeyError`` — same rationale as ``ifbench_score``.
    """
    ids = _constraints(task)
    return bool(ids) and all(VERIFIERS[iid](output) for iid in ids)


# --- loader -------------------------------------------------------------------

Split = Literal["all", "train", "heldout"]


def load_ifbench(
    split: Split = "all",
    *,
    n: int | None = None,
    seed: int = 0,
    path: str | Path | None = None,
) -> list[BenchmarkTask]:
    """Load the vendored IFBench slice as BenchmarkTasks.

    Rows are filtered to the supported constraint set (all vendored rows already
    qualify), shuffled deterministically by `seed`, then split into two disjoint
    halves: `train` (first half) and `heldout` (second half). `all` returns the
    whole shuffled set. `n` caps the returned rows. The IFBench prompt is placed
    under payload["goal"] so the LLM worker's default prompt builder uses it.
    """
    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in Path(path or _DATA).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if set(r["instruction_id_list"]) <= SUPPORTED_IDS]
    rows.sort(key=lambda r: int(r["key"]))  # stable base order before shuffling
    random.Random(seed).shuffle(rows)

    if split != "all":
        mid = len(rows) // 2
        rows = rows[:mid] if split == "train" else rows[mid:]
    if n is not None:
        rows = rows[:n]

    return [
        BenchmarkTask(
            id=r["key"],
            payload={"goal": r["prompt"], "instruction_id_list": r["instruction_id_list"]},
        )
        for r in rows
    ]
