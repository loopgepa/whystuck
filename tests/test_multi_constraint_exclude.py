"""The p0 lever: generating tasks with a constraint type removed.

Why this exists. The paper's mechanism is that reflective optimization converges on a
verifier that never emits PASS, because where most first drafts fail, unconditional
refinement is nearly free on the training signal. That predicts a *curve*: as the base rate
of correct first drafts p0 rises, the degenerate policy stops being free and the optimizer
should have to learn to accept. Testing it needs a p0 knob.

`exact_word_count` is that knob. It is forced into every task of the hard tier, and the
residual analysis localizes almost the whole remaining error to it: under a perfect detector
with maximally actionable feedback every other constraint reaches 1.0 and `word_max_length`
reaches 0.90. Removing it should therefore move p0 a long way without saturating the task,
which is exactly the condition a p0 sweep needs.

The load-bearing test here is the first one. Changing a task generator changes task text,
which changes every cassette key, which would silently invalidate the three runs that
currently replay. So the default path must be byte-identical, and that is pinned by
fingerprint rather than trusted.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from whystuck.eval.multi_constraint import (
    _BRUTAL,
    _HARD_IDS,
    generate_multi_tasks,
    load_multi_constraint,
    mc_score,
)

# sha256[:16] of the default pool, captured BEFORE the `exclude` parameter existed. If a
# change to the generator moves any of these, every committed cassette is invalidated.
DEFAULT_FINGERPRINTS = {
    (True, 0): "ca622756b244929b",
    (True, 1): "65c8a6118d7f16f6",
    (True, 2): "9f89951ef2aa52e7",
    (False, 0): "79ca37b5217dee96",
    (False, 1): "db044087727dcc9e",
    (False, 2): "1c54def52a721807",
}


def _fingerprint(tasks) -> str:
    blob = json.dumps([[t.id, t.payload] for t in tasks], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@pytest.mark.parametrize(("hard", "seed"), sorted(DEFAULT_FINGERPRINTS))
def test_default_generation_is_unchanged(hard, seed):
    """No exclusion => byte-identical tasks, so the released cassettes still replay."""
    tasks = generate_multi_tasks(
        64, seed=seed, k_min=4 if hard else 3, k_max=5 if hard else 4, hard=hard
    )
    assert _fingerprint(tasks) == DEFAULT_FINGERPRINTS[(hard, seed)]


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_excluding_exact_word_count_removes_it_everywhere(seed):
    tasks = generate_multi_tasks(
        64, seed=seed, k_min=4, k_max=5, hard=True, exclude=frozenset({"exact_word_count"})
    )
    assert len(tasks) == 64
    assert not any("exact_word_count" in t.payload["instruction_id_list"] for t in tasks)
    # Still the hard tier: every task keeps a brutal anchor, so this raises p0 without
    # collapsing to the easy tier.
    remaining_brutal = set(_BRUTAL) - {"exact_word_count"}
    assert all(remaining_brutal & set(t.payload["instruction_id_list"]) for t in tasks), (
        "dropping the anchor must not drop the difficulty tier with it"
    )


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_excluded_generation_still_satisfiable_and_correctly_sized(seed):
    """The generator's satisfiability guarantee must survive the exclusion.

    `generate_multi_tasks` raises on an unsatisfiable task, so reaching this assertion at
    all proves the witness scored 1.0 for all 64; the k check is what catches a silent
    `rng.sample` shortfall.
    """
    tasks = generate_multi_tasks(
        64, seed=seed, k_min=4, k_max=5, hard=True, exclude=frozenset({"exact_word_count"})
    )
    for t in tasks:
        ids = t.payload["instruction_id_list"]
        assert 4 <= len(ids) <= 5
        assert len(ids) == len(set(ids))
        assert len(ids) == len(t.payload["kwargs"])


def test_exclusion_that_starves_the_pool_raises():
    """Refuse rather than emit fewer constraints than requested.

    Silently generating k-1 constraints would make the reported task difficulty disagree
    with the tasks actually scored -- the same class of defect as a silently truncated split.
    """
    keep_two = frozenset(set(_HARD_IDS) - {"include_keyword", "all_lowercase"})
    with pytest.raises(ValueError, match="exclude"):
        generate_multi_tasks(4, seed=0, k_min=4, k_max=5, hard=True, exclude=keep_two)


def test_excluding_every_brutal_constraint_raises():
    with pytest.raises(ValueError, match="exclude"):
        generate_multi_tasks(4, seed=0, k_min=4, k_max=5, hard=True, exclude=frozenset(_BRUTAL))


def test_exclude_threads_through_the_split_loader():
    kept = load_multi_constraint(
        "train", seed=0, n=8, pool=64, hard=True, exclude=frozenset({"exact_word_count"})
    )
    assert len(kept) == 8
    assert not any("exact_word_count" in t.payload["instruction_id_list"] for t in kept)
    # Held-out must come from the same excluded pool, or train and held-out differ in
    # difficulty and the comparison measures the exclusion rather than the optimizer.
    held = load_multi_constraint(
        "heldout", seed=0, n=10, pool=64, hard=True, exclude=frozenset({"exact_word_count"})
    )
    assert not any("exact_word_count" in t.payload["instruction_id_list"] for t in held)
    assert not ({t.id for t in kept} & {t.id for t in held})


def test_witness_still_scores_one_under_exclusion():
    """A direct satisfiability check, independent of the generator's own assertion."""
    from whystuck.eval.multi_constraint import build_witness

    tasks = generate_multi_tasks(
        16, seed=3, k_min=4, k_max=5, hard=True, exclude=frozenset({"exact_word_count"})
    )
    for t in tasks:
        pairs = list(zip(t.payload["instruction_id_list"], t.payload["kwargs"], strict=True))
        assert mc_score(build_witness(pairs), t) == 1.0


# ------------------------------------------------------------------------------- CLI guards


def _cli_run(argv):
    import contextlib
    import importlib.util
    import io
    import pathlib as _p

    script = _p.Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"
    spec = importlib.util.spec_from_file_location("run_validation_study", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = mod.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_cli_rejects_an_unknown_constraint_id():
    """A typo would leave the pool untouched and run the UN-ablated condition."""
    code, _, err = _cli_run(
        ["--task-set", "multi-hard", "--arms", "BD", "--seeds", "0",
         "--exclude-constraint", "exact_word_cont"]
    )  # fmt: skip
    assert code == 2
    assert "names no constraint" in err and "un-ablated" in err


def test_cli_rejects_the_lever_on_vendored_task_sets():
    code, _, err = _cli_run(["--task-set", "code", "--exclude-constraint", "exact_word_count"])
    assert code == 2
    assert "generator" in err


def test_cli_accepts_a_valid_exclusion_and_prices_it():
    """Deliberately NOT asserting the recorded metadata.

    `meta` is assembled after the run loop, so a dry run returns before it exists and there
    is no cassette for an excluded configuration to replay. This checks what is actually
    checkable without spending: the flag is collected, and the priced dry run accepts it.
    The metadata key itself is covered by the released-artifact check once such a run exists.
    """
    code, out, err = _cli_run(
        ["--task-set", "multi-hard", "--arms", "BD", "--seeds", "0",
         "--n-train", "8", "--n-heldout", "10", "--max-metric-calls", "28",
         "--exclude-constraint", "exact_word_count"]
    )  # fmt: skip
    assert code == 0, err
    assert "Estimated total" in out


def test_cli_collects_repeated_exclusions():
    import importlib.util
    import pathlib as _p

    script = _p.Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"
    spec = importlib.util.spec_from_file_location("run_validation_study", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod._parse_args(
        ["--exclude-constraint", "exact_word_count", "--exclude-constraint", "no_comma"]
    )
    assert args.exclude_constraint == ["exact_word_count", "no_comma"]
    assert mod._parse_args([]).exclude_constraint == []
