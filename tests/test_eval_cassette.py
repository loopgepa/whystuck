"""Tests for whystuck.eval.cassette — deterministic LLM record/replay.

The validation study calls two live LLM surfaces (the worker `complete` and
gepa's `reflection_lm`). Recording both to a JSON file once, then replaying in
CI, keeps the study reproducible and spends zero tokens on every run after the
first. These tests pin that contract with pure fakes — no network, no extra.
"""

from __future__ import annotations

import contextlib

import pytest

from whystuck.eval.cassette import Cassette, CassetteMiss


def test_record_calls_through_and_persists(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as cassette:
        complete = cassette.wrap(lambda prompt: f"echo:{prompt}", name="worker")
        assert complete("hi") == "echo:hi"
    assert path.exists()  # persisted for a later replay run


def test_replay_returns_stored_without_calling_through(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as rec:
        rec.wrap(lambda prompt: f"echo:{prompt}", name="worker")("hi")

    def explode(_prompt):
        raise AssertionError("replay must not call the underlying model")

    with Cassette(path, mode="replay") as rep:
        complete = rep.wrap(explode, name="worker")
        assert complete("hi") == "echo:hi"


def test_replay_miss_raises(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as rec:
        rec.wrap(lambda p: "x", name="worker")("seen")
    # Deliberately NOT a `with` block: this pins the per-call raise on its own. Exiting a
    # replay context with a swallowed miss is a separate (also-failing) contract, covered
    # by test_replay_surfaces_a_miss_that_a_consumer_swallowed.
    rep = Cassette(path, mode="replay")
    complete = rep.wrap(lambda p: "x", name="worker")
    with pytest.raises(CassetteMiss):
        complete("never-recorded")
    assert rep.misses == ["worker"]  # and it is recorded, not just raised


def test_record_dedups_identical_inputs(tmp_path):
    # Cost guard: an identical prompt recurring across arms/seeds hits the live
    # model once, then serves the cached output.
    path = tmp_path / "c.json"
    calls = {"n": 0}

    def counted(_prompt):
        calls["n"] += 1
        return "out"

    with Cassette(path, mode="record") as rec:
        complete = rec.wrap(counted, name="worker")
        assert complete("same") == "out"
        assert complete("same") == "out"
    assert calls["n"] == 1


def test_name_namespaces_identical_inputs(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as rec:
        worker = rec.wrap(lambda x: "from-worker", name="worker")
        reflect = rec.wrap(lambda x: "from-reflection", name="reflection")
        assert worker("payload") == "from-worker"
        assert reflect("payload") == "from-reflection"  # same input, no collision
    with Cassette(path, mode="replay") as rep:
        assert rep.wrap(lambda x: None, name="worker")("payload") == "from-worker"
        assert rep.wrap(lambda x: None, name="reflection")("payload") == "from-reflection"


def test_kwargs_participate_in_the_key(tmp_path):
    # gepa calls reflection_lm(messages, **kwargs); kwargs must key distinctly.
    path = tmp_path / "c.json"
    msgs = [{"role": "user", "content": "hello"}]
    with Cassette(path, mode="record") as rec:
        fn = rec.wrap(lambda m, **kw: f"t={kw.get('temperature')}", name="reflection")
        assert fn(msgs, temperature=0.0) == "t=0.0"
        assert fn(msgs, temperature=1.0) == "t=1.0"
    with Cassette(path, mode="replay") as rep:
        fn = rep.wrap(lambda m, **kw: None, name="reflection")
        assert fn(msgs, temperature=0.0) == "t=0.0"
        assert fn(msgs, temperature=1.0) == "t=1.0"


def test_replay_surfaces_a_miss_that_a_consumer_swallowed(tmp_path):
    """A swallowed miss must still fail the run.

    GEPA catches exceptions raised inside its reflection LM call ("Per-task reflection
    failed: ...") and simply proposes nothing, so a `CassetteMiss` there does not
    propagate: the replay finishes, exits 0, and writes a report in which that arm looks
    like it accepted no change. Measured on the committed run-11 cassette, exactly that
    turned seed 1 arm D from 0.665625 into 0.681250 — its own baseline — silently
    converting "accepted a change that hurt" into "the treatment never applied", which is
    the one distinction this study exists to make. Replay is documented as strict, so the
    strictness cannot depend on the caller choosing to propagate.
    """
    path = tmp_path / "c.json"
    Cassette(path, mode="record").wrap(lambda p: "recorded", name="w")("known")
    with Cassette(path, mode="record") as c:  # persist
        c.wrap(lambda p: "recorded", name="w")("known")

    with pytest.raises(CassetteMiss, match="did not reproduce"), Cassette(path, "replay") as c:
        fn = c.wrap(lambda p: "live", name="w")
        assert fn("known") == "recorded"
        with contextlib.suppress(CassetteMiss):
            fn("unknown")  # a consumer swallowing it, exactly as GEPA does


def test_replay_with_no_misses_exits_cleanly(tmp_path):
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as c:
        c.wrap(lambda p: "recorded", name="w")("known")
    with Cassette(path, mode="replay") as c:
        assert c.wrap(lambda p: "live", name="w")("known") == "recorded"


def test_a_swallowed_miss_does_not_mask_a_real_error(tmp_path):
    # If the body is already failing, the miss report must not replace that exception.
    path = tmp_path / "c.json"
    with Cassette(path, mode="record") as c:
        c.wrap(lambda p: "recorded", name="w")("known")
    with pytest.raises(ZeroDivisionError), Cassette(path, "replay") as c:
        fn = c.wrap(lambda p: "live", name="w")
        with contextlib.suppress(CassetteMiss):
            fn("unknown")
        raise ZeroDivisionError("a real failure, mid-replay")
