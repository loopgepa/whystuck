"""A non-oracle text-measurement tool for the feedback-richness study.

Huang et al. 2023 (*LLMs Cannot Self-Correct Reasoning Yet*) show that intrinsic
self-correction — refining with no external signal — fails to improve and often
degrades, while self-correction backed by a real verification mechanism (a code
executor, a tool, unit tests) works. Our run-3 loop was pure intrinsic critique,
so its null was predicted. This module supplies the missing ingredient: a
deterministic readout of *measured* properties of the draft, fed into the loop's
critique step.

The crucial invariant is that this is a **measurement, not the oracle**. It takes
only the output string — it never sees the task or its constraints — so it reports
"words=23", never "you failed the exact-word-count rule". The model must still know
the target and decide what to fix. That is exactly the tool/executor regime Huang
found effective, and it keeps the study honest: the programmatic scorer is still
never consulted inside the loop.

Structural non-oracle-ness (no task argument) is necessary but NOT sufficient: a
field is *informationally* the oracle if the verifier it maps to has no per-task
parameter, so the measurement alone resolves the verdict. That is why we report ONLY
quantities the model must still compare against a task-supplied target (n, m, a
keyword) to judge — words vs an exact/min count, bullet_lines vs a required count,
longest_word_letters vs a max length. We deliberately DROP measurements of the two
zero/boolean constraints (all_lowercase, no_comma): reporting "all_lowercase=yes" or
"commas=0" would be identical information to those verifiers' pass/fail, i.e. the
oracle. (Caught in the run-4 pre-spend design review; see the pre-registration.)

Word and letter tokenizers mirror `multi_constraint` exactly (a pinned test guards
against drift), so the reported counts equal what the scorer measures — an
unfaithful tool would misguide the refine step.
"""

from __future__ import annotations

import re
import string

_WORD = re.compile(r"[a-zA-Z]+")  # a maximal run of letters (matches multi_constraint._WORD)
_SENTENCE_END = re.compile(r"[.!?]+")
_BULLET = re.compile(r"^\s*[-*]\s+")


def _word_tokens(output: str) -> list[str]:
    """Human-sense words — must equal multi_constraint._tokens (pinned by test)."""
    toks = []
    for raw in output.split():
        t = raw.strip(string.punctuation)
        if any(c.isalpha() for c in t):
            toks.append(t)
    return toks


def _letter_words(output: str) -> list[str]:
    """Maximal letter-runs — must equal multi_constraint._words (pinned by test)."""
    return _WORD.findall(output)


def text_stats(output: str) -> str:
    """A one-line, constraint-agnostic readout of measured properties of `output`.

    Deliberately takes ONLY the text (no task argument): it is a measurement tool,
    not the scoring oracle. Kept to a single compact line because it is injected
    into every critique prompt and thus priced on the worker input axis.
    """
    tokens = _word_tokens(output)
    letters = _letter_words(output)
    sentences = [s for s in _SENTENCE_END.split(output) if s.strip()]
    bullet_lines = sum(1 for line in output.splitlines() if _BULLET.match(line))
    longest = max((len(w) for w in letters), default=0)
    last_words = " ".join(tokens[-6:])
    # Every field below needs a task-supplied target to become a verdict; we
    # intentionally omit all_lowercase / comma-count (those verifiers have no
    # per-task parameter, so the raw measurement would BE the oracle).
    return (
        f"words={len(tokens)} characters={len(output)} sentences={len(sentences)} "
        f'bullet_lines={bullet_lines} longest_word_letters={longest} last_words="{last_words}"'
    )
