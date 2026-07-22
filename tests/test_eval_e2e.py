"""End-to-end integration test: study → cassette record/replay → report.

This is the first test that exercises the whole eval pipeline as one unit
(run_study driven through a real Cassette file, then render_report over the
result). Steps 1-5 tested each piece in isolation; this proves they compose and,
crucially, that a recorded run replays **deterministically** with zero live
model calls, the property the one paid run (step 7) relies on for reproducible
cassettes.

Offline and $0: the worker/reflection are fakes, GEPA is bypassed via an injected
`_optimize`. All four arms run (A, B, C, D) so the composition rehearses the real
paid run, including arm C's subprocess-command training path. The real-GEPA +
real-OpenAI E2E is recorded and replayed in step 7; the landscape property itself
(a genome-blind worker cannot climb, an LLM worker can) is already unit-tested in
test_eval_optimize.py (the BUG-2 test).

The fake `_optimize` here is genome-blind: it prepends the sentinel to *every*
evolvable gene regardless of arm, so it is not a model of GEPA's search (in
particular it does not reproduce arm C's real flatness, which the BUG-2 test and
step 7 cover). It is only a deterministic stand-in that lets the pipeline wiring,
scoring, cassette, and report be exercised without a network call.
"""

from __future__ import annotations

import json
from pathlib import Path

from whystuck.core.storage import WhyStuckStore
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.cassette import Cassette
from whystuck.eval.report import render_report
from whystuck.eval.study import run_study

_SENTINEL = "OPTIMIZED"
_ARMS = ("A", "B", "C", "D")


def _worker(prompt: str) -> str:
    # Synthetic landscape that models the study thesis: only a fully optimized
    # loop (BOTH text genes carry the sentinel → count ≥ 2) yields a
    # whitespace-free, constraint-satisfying output. Optimizing one gene
    # (prompt-only, arm B → 1 sentinel) or none (seed → 0) does not clear the
    # bar, so "x x" (has whitespace) fails the no_whitespace verifier. This makes
    # arm D beat arm B end-to-end, so the verdict test can prove correct B/D
    # attribution rather than passing on a numeric tie.
    return "solid" if prompt.count(_SENTINEL) >= 2 else "x x"


def _optimize(**kw):
    return {"best_candidate": {gid: f"{_SENTINEL} {t}" for gid, t in kw["seed_candidate"].items()}}


def _tasks(n: int = 3) -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            id=f"n{i}", payload={"goal": f"p{i}", "instruction_id_list": ["format:no_whitespace"]}
        )
        for i in range(n)
    ]


def _run(store, complete, cassette_path, mode):
    with Cassette(cassette_path, mode=mode) as cass:
        wrapped = cass.wrap(complete, name="worker")
        reflection = cass.wrap(lambda *a, **k: "reflect", name="reflection")
        return run_study(
            store,
            complete=wrapped,
            reflection_lm=reflection,
            train=_tasks(),
            heldout=_tasks(),
            arms=_ARMS,
            _optimize=_optimize,
        )


def test_study_replays_deterministically_from_cassette(tmp_path):
    cassette = tmp_path / "study.json"

    rec = _run(WhyStuckStore(tmp_path / "rec"), _worker, cassette, "record")
    assert cassette.exists()

    def _boom(_prompt):
        raise AssertionError("live model was called during replay — cassette miss")

    # _boom is a canary against a future Cassette regression that lets replay
    # fall through to a live call. Today Cassette.wrap never invokes fn in replay
    # mode (a hit returns the cached value; a miss raises CassetteMiss before fn
    # is reached), so a broken cassette surfaces as CassetteMiss, not this
    # AssertionError; either exception propagates (no try/except swallows it on
    # the run_study → _heldout_score / optimize_loop → adapter path). Determinism
    # is provable because the cassette key is a hash of (name, args, kwargs) and
    # both runs build byte-identical prompts from the same seeded genomes.
    rep = _run(WhyStuckStore(tmp_path / "rep"), _boom, cassette, "replay")

    # Scores are deterministic across record→replay (variant_id is a random uuid,
    # so it's expected to differ and is not compared).
    for arm in _ARMS:
        assert rep[arm].seed_heldout == rec[arm].seed_heldout
        assert rep[arm].opt_heldout == rec[arm].opt_heldout
        assert rep[arm].delta == rec[arm].delta


def test_study_to_report_pipeline_renders_a_verdict(tmp_path):
    cassette = tmp_path / "study.json"
    runs: list[dict[str, dict]] = []
    # Two seeds: render_report's _verdict forces INCONCLUSIVE below 2 seeds, so
    # running it twice is what exercises the directional-verdict branch. The two
    # runs are deterministic-identical by construction (pure fake worker + fixed
    # _optimize), which is fine — the point is the ≥2-seed code path, not seed
    # variance. Both record into one cassette (the second run is a full cache
    # hit), mirroring how the real paid run records all seeds into one file.
    for seed in (0, 1):
        results = _run(WhyStuckStore(tmp_path / f"s{seed}"), _worker, cassette, "record")
        runs.append({name: vars(r) for name, r in results.items()})

    meta = {
        "dataset": "IFBench",
        "n_train": 3,
        "n_heldout": 3,
        "worker_model": "fake",
        "reflection_model": "fake",
        "max_metric_calls": 10,
        "seeds": [0, 1],
        "estimated_usd": 0.0,
        "generated": "test",
    }
    md = render_report(runs, meta=meta)
    # D (evolves both genes → 2 sentinels → 1.0) beats B (one gene → 1 sentinel →
    # 0.0), so the end-to-end verdict is SUPPORTED. This pins correct arm
    # attribution: a wiring bug that swapped B's and D's results would flip the
    # label to NOT SUPPORTED and fail here.
    assert "## Verdict — SUPPORTED" in md
    assert "IFBench" in md
    # Assert the Results-table ROW for each arm exists. The bare letter is not
    # enough — "B"/"C"/"D" all occur in the Method prose — but the pipe-delimited
    # row marker "| X |" appears only in the results table, so a dropped or
    # renamed arm row is caught here.
    for arm in _ARMS:
        assert f"| {arm} |" in md


def test_recorded_cassette_captures_worker_calls(tmp_path):
    cassette = tmp_path / "study.json"
    _run(WhyStuckStore(tmp_path / "rec"), _worker, cassette, "record")

    data = json.loads(Path(cassette).read_text(encoding="utf-8"))
    entries = data["entries"]
    assert entries, "cassette recorded no calls"
    # Every stored entry is namespaced to the worker surface. Reflection is never
    # called (GEPA is bypassed by the injected _optimize) and arm C's training
    # uses the subprocess command runner, which is not cassette-wrapped, so its
    # only worker calls are the LLM-scored held-out evals.
    assert all(e["name"] == "worker" for e in entries.values())
