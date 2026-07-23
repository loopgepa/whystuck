"""Composable multi-constraint instruction-following tasks (the fair loop test).

The vendored single-constraint IFBench verifiers cannot form fair multi-constraint
tasks: several are trivially co-satisfied by a short degenerate string (e.g.
``x?!.!?,;:`` passes 6 of 11) while the meaningful ones mutually conflict. So a
stack of them is either trivial or impossible — GEPA has nothing to *decompose and
fix*, which is exactly the mechanism the loop-level thesis is about.

Instead this module uses six canonical, INDEPENDENT instruction-following
constraints — the families real IFBench/IFEval multi-constraint prompts use:

    include_keyword · min_words · all_lowercase · end_with · no_comma · num_bullets

They carry per-task parameters (IFEval's ``instruction_id_list`` + parallel
``kwargs`` convention), are individually diagnosable (the critique can say *which*
one failed), and are jointly satisfiable by construction — a single witness string
can satisfy all six, so any subset is satisfiable. The generator builds that
witness and asserts it scores 1.0 before emitting a task, so no task can floor
every arm. Scores/feedback mirror the ifbench interface (``(output, task) -> …``)
so the study harness is unchanged apart from which score/feedback it injects.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable
from typing import Any, Literal

from whystuck.eval.benchmark import BenchmarkTask

Params = dict[str, Any]
MCVerifier = Callable[[str, Params], bool]

_BULLET = re.compile(r"\s*[-*]\s+")
_WORD = re.compile(r"[a-zA-Z]+")  # a maximal run of letters


def _tokens(output: str) -> list[str]:
    """Human-sense words: whitespace tokens with edge punctuation stripped, keeping
    those that contain a letter. So "well-known" and "don't" are ONE word each,
    "done." → "done", bullet markers "-" drop out. Used for the count / banned-word
    checks so they match a natural reading (and so a hyphen isn't a word boundary —
    "state-of-the-art" does NOT contain the standalone word "the")."""
    toks = []
    for raw in output.split():
        t = raw.strip(string.punctuation)
        if any(c.isalpha() for c in t):
            toks.append(t)
    return toks


def _words(output: str) -> list[str]:
    """Maximal letter-runs — used for the per-word LENGTH check, so a hyphenated
    compound is judged by its parts ("state-of-the-art" → state/of/the/art)."""
    return _WORD.findall(output)


# --- verifiers (each independent; output + its own params -> bool) ------------
#
# Two difficulty tiers share these verifiers. EASY tier (run 2): include_keyword,
# min_words, all_lowercase, end_with, no_comma, num_bullets — a capable model told
# exactly what to do aces them (that run saturated at seed 0.97). HARD tier adds
# constraints models miss even when instructed — exact_word_count (models can't hit
# exact counts), forbidden_word (a word-level lipogram, e.g. avoid "the"), and
# word_max_length — to create real headroom while staying deterministically checkable
# and diagnosable (the critique can count words / spot the banned word).


def _include_keyword(output: str, params: Params) -> bool:
    return params["keyword"].lower() in output.lower()


def _min_words(output: str, params: Params) -> bool:
    return len(output.split()) >= int(params["n"])


def _all_lowercase(output: str, params: Params) -> bool:
    # a genuine lowercase *text*: has letters and none of them uppercase.
    return any(c.isalpha() for c in output) and not any(c.isupper() for c in output)


def _end_with(output: str, params: Params) -> bool:
    return output.rstrip().endswith(params["phrase"])


def _no_comma(output: str, params: Params) -> bool:
    return "," not in output


def _num_bullets(output: str, params: Params) -> bool:
    return sum(1 for ln in output.splitlines() if _BULLET.match(ln)) == int(params["n"])


def _exact_word_count(output: str, params: Params) -> bool:
    return len(_tokens(output)) == int(params["n"])


def _forbidden_word(output: str, params: Params) -> bool:
    # the banned word must not appear as a whole whitespace-word (case-insensitive).
    # Whole-token (not \b) so a hyphenated idiom like "state-of-the-art" is NOT a use
    # of the standalone word "the" — that would be a construct-invalid penalty.
    banned = params["word"].lower()
    return all(t.lower() != banned for t in _tokens(output))


def _word_max_length(output: str, params: Params) -> bool:
    m = int(params["m"])
    return all(len(w) <= m for w in _words(output))


MC_VERIFIERS: dict[str, MCVerifier] = {
    "include_keyword": _include_keyword,
    "min_words": _min_words,
    "all_lowercase": _all_lowercase,
    "end_with": _end_with,
    "no_comma": _no_comma,
    "num_bullets": _num_bullets,
    "exact_word_count": _exact_word_count,
    "forbidden_word": _forbidden_word,
    "word_max_length": _word_max_length,
}

# Human-readable requirement, embedding the concrete parameter so the worker prompt
# and the reflection feedback both name exactly what to do / what failed.
_DESCRIBE: dict[str, Callable[[Params], str]] = {
    "include_keyword": lambda p: f'include the word "{p["keyword"]}"',
    "min_words": lambda p: f"use at least {p['n']} words",
    "all_lowercase": lambda p: "write your entire response in lowercase (no capital letters)",
    "end_with": lambda p: f'end your response with the exact phrase "{p["phrase"]}"',
    "no_comma": lambda p: "do not use any commas",
    "num_bullets": lambda p: f'format as exactly {p["n"]} bullet points, each line starting "- "',
    "exact_word_count": lambda p: f"use exactly {p['n']} words",
    "forbidden_word": lambda p: f'do not use the word "{p["word"]}"',
    "word_max_length": lambda p: f"use no word longer than {p['m']} letters",
}


# --- scoring / feedback (ifbench-compatible signatures) -----------------------


def _pairs(task: BenchmarkTask) -> list[tuple[str, Params]]:
    payload = task.payload or {}
    ids = list(payload.get("instruction_id_list", []))
    kwargs = list(payload.get("kwargs", [{} for _ in ids]))
    # Validate at read time with the task named, so a malformed task (a future
    # hand-built fixture / import tool) fails loud with context instead of raising a
    # context-free zip()/KeyError deep inside scoring.
    if len(ids) != len(kwargs):
        raise ValueError(
            f"task {task.id!r}: instruction_id_list has {len(ids)} entries but kwargs "
            f"has {len(kwargs)} — they must be parallel"
        )
    unknown = [i for i in ids if i not in MC_VERIFIERS]
    if unknown:
        raise KeyError(f"task {task.id!r}: unsupported constraint id(s) {unknown}")
    return list(zip(ids, kwargs, strict=True))


def mc_score(output: str, task: BenchmarkTask) -> float:
    """Fraction of the task's constraints satisfied ∈ [0, 1].

    Empty constraint list → 0.0 (an unscored task is not a vacuous win). An id
    outside the supported set raises ``KeyError`` on purpose — the generator only
    emits supported ids, so an unknown id is a caller bug that must fail loud."""
    pairs = _pairs(task)
    if not pairs:
        return 0.0
    met = sum(1 for iid, kw in pairs if MC_VERIFIERS[iid](output, kw))
    return met / len(pairs)


def mc_pass(output: str, task: BenchmarkTask) -> bool:
    """Every constraint satisfied (empty list → False, not vacuous True)."""
    pairs = _pairs(task)
    return bool(pairs) and all(MC_VERIFIERS[iid](output, kw) for iid, kw in pairs)


def mc_feedback(output: str, task: BenchmarkTask) -> str:
    """Per-constraint pass/fail prose (with parameters), for GEPA's reflection."""
    pairs = _pairs(task)
    if not pairs:
        return "no verifiable constraints on this task"
    checked = [(iid, kw, MC_VERIFIERS[iid](output, kw)) for iid, kw in pairs]
    failed = [f"{iid} ({_DESCRIBE[iid](kw)})" for iid, kw, ok in checked if not ok]
    passed = [iid for iid, _, ok in checked if ok]
    if not failed:
        return f"All {len(passed)} requirement(s) satisfied: {', '.join(passed)}."
    return (
        f"Failed {len(failed)} of {len(checked)} requirement(s) — fix: {'; '.join(failed)}. "
        f"Satisfied: {', '.join(passed) if passed else 'none'}."
    )


def render_multi_prompt(goal: str, pairs: list[tuple[str, Params]]) -> str:
    """The full instruction shown to the worker: the goal + the requirement list."""
    reqs = "\n".join(f"- {_DESCRIBE[iid](kw)}" for iid, kw in pairs)
    return f"{goal}\n\nYour response MUST satisfy ALL of the following requirements:\n{reqs}"


# --- satisfiability witness ---------------------------------------------------

_VOCAB = [
    "idea",
    "system",
    "method",
    "result",
    "signal",
    "vector",
    "matrix",
    "update",
    "policy",
    "sample",
    "filter",
    "window",
    "kernel",
    "tensor",
    "dataset",
    "feature",
    "epoch",
    "photon",
    "entropy",
    "neuron",
]
# All ≤5 letters and free of the forbidden-word candidates — used when a task carries
# word_max_length (so the witness's words are short enough) and generally safe for any
# subset, since short comma-free lowercase content words satisfy every constraint.
_SHORT_VOCAB = [
    "atom",
    "gene",
    "wave",
    "node",
    "cell",
    "flux",
    "star",
    "code",
    "leaf",
    "bird",
    "rock",
    "sand",
    "wind",
    "fire",
    "lake",
    "stem",
    "bark",
    "moss",
    "reef",
    "salt",
]


def build_witness(pairs: list[tuple[str, Params]]) -> str:
    """Construct a string that satisfies every constraint in ``pairs``.

    Used only to *prove satisfiability* (the generator asserts the witness scores
    1.0); never shown to the worker. Strategy: a lowercase, comma-free body of short
    content words (as a paragraph or exactly N bullets), sized so the *word* count
    (letter-runs — markers/punctuation don't count) hits an exact target, with the
    keyword present, the banned word absent (never in the vocab), every word within
    the length limit, and the required trailing phrase appended."""
    sel = dict(pairs)
    max_len = sel.get("word_max_length", {}).get("m")
    forbidden = str(sel.get("forbidden_word", {}).get("word", "")).lower()
    exact_n = sel.get("exact_word_count", {}).get("n")
    min_n = int(sel.get("min_words", {}).get("n", 0))
    n_bul = sel.get("num_bullets", {}).get("n")
    keyword = sel.get("include_keyword", {}).get("keyword")
    phrase = sel.get("end_with", {}).get("phrase")

    base = _SHORT_VOCAB if max_len is not None else _VOCAB
    vocab = [w for w in base if (max_len is None or len(w) <= int(max_len)) and w != forbidden]

    # exact_word_count counts letter-run words across the WHOLE output, so the phrase's
    # words count toward the target; size the content body to make the total exact.
    phrase_words = len(_tokens(phrase)) if phrase else 0
    content_n = int(exact_n) - phrase_words if exact_n is not None else max(min_n, 12)
    content_n = max(content_n, 1)

    words = [vocab[i % len(vocab)] for i in range(content_n)]
    if keyword:
        words[0] = keyword  # guarantee the keyword is present (short + not forbidden)

    if n_bul:
        n_bul = int(n_bul)
        per = max(1, len(words) // n_bul)
        lines: list[str] = []
        idx = 0
        for b in range(n_bul):
            chunk = words[idx:] if b == n_bul - 1 else words[idx : idx + per]
            idx += per
            lines.append("- " + " ".join(chunk))  # chunks are non-empty for real params
        if phrase:
            lines[-1] = f"{lines[-1]} {phrase}"
        out = "\n".join(lines)
    else:
        out = " ".join(words)
        if phrase:
            out = f"{out} {phrase}"

    return out.lower()  # valid whether or not all_lowercase is selected


# --- generation ---------------------------------------------------------------

# EASY tier (run 2) — a capable model aces these; kept for reproducibility.
_EASY_IDS = ("include_keyword", "min_words", "all_lowercase", "end_with", "no_comma", "num_bullets")
# HARD tier — adds constraints models miss even when instructed, to create headroom.
_HARD_IDS = (
    "include_keyword",
    "exact_word_count",
    "all_lowercase",
    "end_with",
    "no_comma",
    "num_bullets",
    "forbidden_word",
    "word_max_length",
)

_KEYWORDS = [
    "photon",
    "entropy",
    "vector",
    "mitochondria",
    "gradient",
    "catalyst",
    "neuron",
    "isotope",
    "polymer",
    "glacier",
    "syntax",
    "kernel",
]
_PHRASES = ["that is all.", "thank you.", "the end.", "in closing.", "we are done."]
_MIN_WORDS = [15, 20, 25, 30]
_NUM_BULLETS = [3, 4, 5]

# HARD-tier param pools, deliberately DECOUPLED so no (constraint × param) combo can
# conflict: hard keywords and phrase words are all short (≤5, ≤ any word_max_length)
# and none contain a forbidden word; forbidden words are function words absent from
# the short vocab / keywords / phrases; so every subset is jointly satisfiable.
_HARD_KEYWORDS = ["atom", "gene", "wave", "node", "cell", "flux", "star", "code"]
_HARD_PHRASES = ["we are done.", "that is it.", "so it goes.", "here we go.", "all is set."]
_EXACT_WORDS = [15, 20, 25]
_FORBIDDEN = ["the", "and", "of", "to"]
_MAX_LEN = [6, 7]
# The genuinely hard constraints (models miss them even when told). exact_word_count
# is a true skill gap (LLMs can't self-count during generation); the other two are
# "choice" constraints a diligent model can satisfy, so we require exact_word_count
# PLUS at least one more of these per hard task — else the seed re-saturates (run 2).
_BRUTAL = ("exact_word_count", "forbidden_word", "word_max_length")

_GOALS = [
    "Explain a scientific concept.",
    "Describe a natural process.",
    "Summarize an idea for a student.",
    "Write a short note about a discovery.",
    "Give a brief overview of a technology.",
    "Introduce a topic to a beginner.",
]


def _sample_params(iid: str, rng: random.Random, *, hard: bool) -> Params:
    if iid == "include_keyword":
        return {"keyword": rng.choice(_HARD_KEYWORDS if hard else _KEYWORDS)}
    if iid == "min_words":
        return {"n": rng.choice(_MIN_WORDS)}
    if iid == "end_with":
        return {"phrase": rng.choice(_HARD_PHRASES if hard else _PHRASES)}
    if iid == "num_bullets":
        return {"n": rng.choice(_NUM_BULLETS)}
    if iid == "exact_word_count":
        return {"n": rng.choice(_EXACT_WORDS)}
    if iid == "forbidden_word":
        return {"word": rng.choice(_FORBIDDEN)}
    if iid == "word_max_length":
        return {"m": rng.choice(_MAX_LEN)}
    return {}  # all_lowercase, no_comma are parameterless


def generate_multi_tasks(
    n: int, *, seed: int = 0, k_min: int = 3, k_max: int = 4, hard: bool = False
) -> list[BenchmarkTask]:
    """Deterministically generate ``n`` satisfiable multi-constraint tasks.

    ``hard`` selects the harder constraint pool (exact counts / forbidden word / max
    word length) that models miss even when instructed — creating the headroom the
    easy tier lacked. Each task picks k ∈ [k_min, k_max] distinct constraints with
    compatible params, renders an NL instruction, and is emitted only after its
    witness is verified to score 1.0 — enforcing satisfiability at the source."""
    rng = random.Random(seed)
    tasks: list[BenchmarkTask] = []
    for i in range(n):
        k = rng.randint(k_min, k_max)
        if hard:
            # Guarantee real difficulty: always exact_word_count (the skill gap) + one
            # more brutal constraint, then fill the rest from the remaining hard ids.
            extra_brutal = [b for b in _BRUTAL if b != "exact_word_count"]
            ids = ["exact_word_count", rng.choice(extra_brutal)]
            rest = [i for i in _HARD_IDS if i not in ids]
            ids += rng.sample(rest, k - len(ids))
            rng.shuffle(ids)  # requirement order varies in the prompt
        else:
            ids = rng.sample(_EASY_IDS, k)
        pairs = [(iid, _sample_params(iid, rng, hard=hard)) for iid in ids]
        goal = rng.choice(_GOALS)
        task = BenchmarkTask(
            id=f"mc-{i}",
            payload={
                "goal": render_multi_prompt(goal, pairs),
                "instruction_id_list": [iid for iid, _ in pairs],
                "kwargs": [kw for _, kw in pairs],
            },
        )
        witness = build_witness(pairs)
        if mc_score(witness, task) != 1.0:  # guarantee: never emit an unsatisfiable task
            raise AssertionError(f"unsatisfiable task generated for {ids}: witness={witness!r}")
        tasks.append(task)
    return tasks


Split = Literal["all", "train", "heldout"]


def load_multi_constraint(
    split: Split = "all", *, n: int | None = None, seed: int = 0, pool: int = 64, hard: bool = False
) -> list[BenchmarkTask]:
    """Generate a deterministic pool and split it into disjoint train/heldout halves.

    Mirrors ``load_ifbench``: same seed → same tasks; ``train`` and ``heldout`` are
    disjoint (first/second half of the shuffled pool), and ``n`` caps the result.
    ``hard`` selects the harder constraint tier (wider k = 4–5 for more constraints)."""
    tasks = generate_multi_tasks(
        pool, seed=seed, k_min=4 if hard else 3, k_max=5 if hard else 4, hard=hard
    )
    random.Random(seed).shuffle(tasks)
    if split != "all":
        mid = len(tasks) // 2
        tasks = tasks[:mid] if split == "train" else tasks[mid:]
    if n is not None:
        if n > len(tasks):
            raise ValueError(
                f"requested n={n} {split!r} tasks but only {len(tasks)} available from "
                f"pool={pool}; raise pool (>= ~2*n) or lower n — silent truncation would "
                "make the reported sample size disagree with the tasks actually scored"
            )
        tasks = tasks[:n]
    return tasks
