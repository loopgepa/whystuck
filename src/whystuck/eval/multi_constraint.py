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
from collections.abc import Callable
from typing import Any, Literal

from whystuck.eval.benchmark import BenchmarkTask

Params = dict[str, Any]
MCVerifier = Callable[[str, Params], bool]

_BULLET = re.compile(r"\s*[-*]\s+")


# --- verifiers (each independent; output + its own params -> bool) ------------


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


MC_VERIFIERS: dict[str, MCVerifier] = {
    "include_keyword": _include_keyword,
    "min_words": _min_words,
    "all_lowercase": _all_lowercase,
    "end_with": _end_with,
    "no_comma": _no_comma,
    "num_bullets": _num_bullets,
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


def build_witness(pairs: list[tuple[str, Params]]) -> str:
    """Construct a string that satisfies every constraint in ``pairs``.

    Used only to *prove satisfiability* (the generator asserts the witness scores
    1.0); never shown to the worker. All constraints are independent and additive,
    so one lowercase, comma-free body — as a paragraph or as exactly N bullets, with
    the keyword present and the required trailing phrase — satisfies any subset."""
    sel = dict(pairs)
    min_n = sel.get("min_words", {}).get("n", 0)
    n_bul = sel.get("num_bullets", {}).get("n")
    keyword = sel.get("include_keyword", {}).get("keyword")
    phrase = sel.get("end_with", {}).get("phrase")

    target = max(min_n, 12)
    words = [_VOCAB[i % len(_VOCAB)] for i in range(target)]
    if keyword:
        words[0] = keyword  # guarantee the keyword is present

    if n_bul:
        per = max(1, len(words) // n_bul)
        lines: list[str] = []
        idx = 0
        for b in range(n_bul):
            chunk = words[idx:] if b == n_bul - 1 else words[idx : idx + per]
            idx += per
            lines.append("- " + " ".join(chunk or ["item"]))
        if phrase:
            lines[-1] = f"{lines[-1]} {phrase}"
        out = "\n".join(lines)
    else:
        out = " ".join(words)
        if phrase:
            out = f"{out} {phrase}"

    return out.lower()  # valid whether or not all_lowercase is selected


# --- generation ---------------------------------------------------------------

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
_GOALS = [
    "Explain a scientific concept.",
    "Describe a natural process.",
    "Summarize an idea for a student.",
    "Write a short note about a discovery.",
    "Give a brief overview of a technology.",
    "Introduce a topic to a beginner.",
]
_CONSTRAINT_IDS = tuple(MC_VERIFIERS)  # all six


def _sample_params(iid: str, rng: random.Random) -> Params:
    if iid == "include_keyword":
        return {"keyword": rng.choice(_KEYWORDS)}
    if iid == "min_words":
        return {"n": rng.choice(_MIN_WORDS)}
    if iid == "end_with":
        return {"phrase": rng.choice(_PHRASES)}
    if iid == "num_bullets":
        return {"n": rng.choice(_NUM_BULLETS)}
    return {}  # all_lowercase, no_comma are parameterless


def generate_multi_tasks(
    n: int, *, seed: int = 0, k_min: int = 3, k_max: int = 4
) -> list[BenchmarkTask]:
    """Deterministically generate ``n`` satisfiable multi-constraint tasks.

    Each task picks k ∈ [k_min, k_max] distinct constraints with compatible params,
    renders an NL instruction, and is emitted only after its witness is verified to
    score 1.0 — enforcing the satisfiability guarantee at the source."""
    rng = random.Random(seed)
    tasks: list[BenchmarkTask] = []
    for i in range(n):
        k = rng.randint(k_min, k_max)
        ids = rng.sample(_CONSTRAINT_IDS, k)
        pairs = [(iid, _sample_params(iid, rng)) for iid in ids]
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
    split: Split = "all", *, n: int | None = None, seed: int = 0, pool: int = 64
) -> list[BenchmarkTask]:
    """Generate a deterministic pool and split it into disjoint train/heldout halves.

    Mirrors ``load_ifbench``: same seed → same tasks; ``train`` and ``heldout`` are
    disjoint (first/second half of the shuffled pool), and ``n`` caps the result."""
    tasks = generate_multi_tasks(pool, seed=seed)
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
