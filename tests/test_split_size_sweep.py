"""Guards and invariants the split-size sweep depends on.

The sweep varies `--n-train` while holding the held-out set, the seeds and the task
distribution fixed, so that a change in delta(D-B) can be attributed to train-split size.
Three things can silently break that attribution, and each has a test here.

1. **A starved budget.** GEPA charges its seed evaluation `len(valset)` metric calls
   (gepa/core/state.py: `total_num_evals = len(seed_valset_evaluation.scores_by_val_id)`),
   and `optimize_loop` passes the train split as the trainset with no separate valset. So
   tripling n_train while holding `--max-metric-calls` fixed does not buy a bigger split
   at equal search effort -- it buys an arm that cannot afford one accept cycle, reports
   `genes_changed=[]`, and looks exactly like a null. That is the failure this whole paper
   is about, so the harness refuses the config rather than billing for it.

2. **A moved held-out set.** The generated pool is
   `max(64, 2*max(n_train, n_heldout) + 2)`, so it is 64 for every n_train <= 31 and jumps
   at 32. Since the pool seeds the shuffle, crossing that cliff silently changes which
   tasks are held out and the comparison stops being paired. The sweep therefore stays
   below it, and the cliff itself is pinned so nobody widens the sweep by accident.

3. **Leakage that grows with the split.** A larger train split could simply contain more
   near-copies of held-out tasks, which would improve held-out scores for a reason that
   has nothing to do with overfitting. Pinned here: no exact instance is shared at any
   sweep size, and the share of train tasks whose constraint-type-set also appears in
   held-out does not rise with n_train.
"""

from __future__ import annotations

import ast
import contextlib
import glob
import importlib.util
import io
import json
import pathlib
import re

import pytest

from whystuck.eval.multi_constraint import load_multi_constraint
from whystuck.eval.report import render_report

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"
_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"

# (n_train, max_metric_calls) -- mmc = 3.5 * n_train holds metric calls per train task,
# i.e. the number of epochs over the split, constant. n_train=8 reproduces C2b exactly.
SWEEP = ((4, 14), (8, 28), (16, 56), (24, 84))
POOL_CLIFF = 32  # first n_train where the pool leaves 64 (with n_heldout=10)


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_validation_study", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv):
    cli = _load_cli()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def _pool_for(n_train: int, n_heldout: int = 10) -> int:
    return max(64, 2 * max(n_train, n_heldout) + 2)


def _payload(task) -> dict:
    return task.payload if isinstance(task.payload, dict) else ast.literal_eval(task.payload)


# --------------------------------------------------------------------------- budget guard


def test_starved_budget_is_refused_before_spending():
    """mmc=28 at n_train=24 cannot fund one accept cycle; refuse rather than bill."""
    code, _, err = _run(
        ["--task-set", "multi-hard", "--arms", "BD", "--seeds", "0",
         "--n-train", "24", "--n-heldout", "10", "--max-metric-calls", "28",
         "--max-usd", "100"]
    )  # fmt: skip
    assert code == 2
    assert "--max-metric-calls" in err
    # The message has to name the mechanism, or the operator just raises the number blindly.
    assert "seed" in err.lower() and "24" in err


def test_guard_does_not_fire_when_no_arm_optimizes():
    """The released C1 conditions are `--arms A` at n_train=32, mmc=64.

    GEPA never runs there (`evolvable=[]`), so the budget is inert and the guard must not
    reject a config that produced released, replaying results.
    """
    code, _, err = _run(
        ["--task-set", "multi-hard", "--arms", "A", "--seeds", "0",
         "--n-train", "32", "--n-heldout", "32", "--max-metric-calls", "64",
         "--max-usd", "100"]
    )  # fmt: skip
    assert code == 0, err


@pytest.mark.parametrize(("n_train", "mmc"), SWEEP)
def test_sweep_configurations_pass_the_guard(n_train, mmc):
    code, _, err = _run(
        ["--task-set", "multi-hard", "--arms", "BD", "--seeds", "0,1,2",
         "--n-train", str(n_train), "--n-heldout", "10",
         "--max-metric-calls", str(mmc), "--max-usd", "100"]
    )  # fmt: skip
    assert code == 0, err


def test_every_released_run_would_pass_its_own_guard():
    """No committed result file describes a config the guard would now reject.

    A guard that invalidates released work is a guard that will be turned off. The two C1
    runs are the exception it is scoped around: `--arms A`, so nothing optimizes.
    """
    offenders = []
    for path in sorted(glob.glob(str(_DOCS / "validation-study*.json"))):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        meta, runs = blob["meta"], blob["runs"]
        n_train, mmc = meta.get("n_train"), meta.get("max_metric_calls")
        if n_train is None or mmc is None:
            continue
        optimizes = any(r.get("evolvable_genes") for arm in runs[0].values() for r in (arm,))
        if optimizes and mmc < 2 * n_train + min(3, n_train):
            offenders.append((pathlib.Path(path).stem, n_train, mmc))
    assert offenders == []


# ------------------------------------------------------------------- split-set invariants


def test_report_path_is_not_written_unless_asked():
    """`--report` used to default to a TRACKED file, so any replay clobbered it.

    Discovered by clobbering docs/validation-study.md while verifying arm-subset
    invariance. Symmetric with `--out`: no path, no write.
    """
    cli = _load_cli()
    assert cli._parse_args([]).report is None
    assert cli._parse_args(["--report", "x.md"]).report == "x.md"


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_heldout_is_identical_across_every_sweep_size(seed):
    ids = {
        n_train: [t.id for t in load_multi_constraint(
            "heldout", seed=seed, n=10, pool=_pool_for(n_train), hard=True)]
        for n_train, _ in SWEEP
    }  # fmt: skip
    first = ids[SWEEP[0][0]]
    assert all(v == first for v in ids.values())
    assert len(first) == 10


def test_pool_cliff_is_where_we_think_it_is():
    """Above the cliff the held-out set moves, which would unpair the comparison."""
    assert _pool_for(POOL_CLIFF - 1) == 64
    assert _pool_for(POOL_CLIFF) > 64
    assert max(n for n, _ in SWEEP) < POOL_CLIFF
    below = [t.id for t in load_multi_constraint(
        "heldout", seed=0, n=10, pool=_pool_for(POOL_CLIFF - 1), hard=True)]  # fmt: skip
    above = [t.id for t in load_multi_constraint(
        "heldout", seed=0, n=10, pool=_pool_for(POOL_CLIFF), hard=True)]  # fmt: skip
    assert below != above, "if these ever match, the cliff moved and this test is asleep"


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_train_splits_are_nested_prefixes(seed):
    """The sweep ADDS tasks rather than resampling, so sizes are directly comparable."""
    ids = [
        [t.id for t in load_multi_constraint(
            "train", seed=seed, n=n_train, pool=_pool_for(n_train), hard=True)]
        for n_train, _ in SWEEP
    ]  # fmt: skip
    # Consecutive pairs, so the sequences differ in length by one by construction.
    for smaller, larger in zip(ids, ids[1:], strict=False):
        assert larger[: len(smaller)] == smaller


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_train_and_heldout_stay_disjoint_and_unduplicated(seed):
    heldout = load_multi_constraint("heldout", seed=seed, n=10, pool=64, hard=True)
    held_ids = {t.id for t in heldout}
    held_sig = {
        json.dumps([p["instruction_id_list"], p["kwargs"], p["goal"]], sort_keys=True)
        for p in map(_payload, heldout)
    }
    held_types = [frozenset(_payload(t)["instruction_id_list"]) for t in heldout]
    type_share = {}
    for n_train, _ in SWEEP:
        train = load_multi_constraint(
            "train", seed=seed, n=n_train, pool=_pool_for(n_train), hard=True
        )
        payloads = [_payload(t) for t in train]
        assert not ({t.id for t in train} & held_ids)
        sigs = {
            json.dumps([p["instruction_id_list"], p["kwargs"], p["goal"]], sort_keys=True)
            for p in payloads
        }
        assert not (sigs & held_sig), "an exact held-out instance leaked into train"
        matched = sum(1 for p in payloads if frozenset(p["instruction_id_list"]) in held_types)
        type_share[n_train] = matched / n_train
    # Leakage must not GROW with the split, or a larger split would score better for a
    # reason unrelated to overfitting. It in fact falls.
    sizes = [n for n, _ in SWEEP]
    assert type_share[sizes[-1]] <= type_share[sizes[0]]


# ------------------------------------------------------------- released-artifact integrity


def test_committed_reports_agree_numerically_with_their_json():
    """Every docs/validation-study*.md must carry the numbers of its own .json.

    Exact text match is NOT required and deliberately not asserted: `render_report`
    describes the current arm-C implementation, and arm C changed mid-project (a saturating
    command control became a non-saturating fixed-draft control). Regenerating an old
    report would therefore relabel the control arm a pre-correction run actually used. The
    numbers are the part that must never drift.
    """
    drift = []
    for path in sorted(glob.glob(str(_DOCS / "validation-study*.json"))):
        md = pathlib.Path(path[:-5] + ".md")
        if not md.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        fresh = render_report(blob["runs"], meta=blob["meta"])
        numbers = lambda text: re.findall(r"-?\d+\.\d+", text)  # noqa: E731
        if numbers(md.read_text(encoding="utf-8")) != numbers(fresh):
            drift.append(md.name)
    assert drift == []
