"""Tests for whystuck.eval.cassette — deterministic LLM record/replay.

The validation study calls two live LLM surfaces (the worker `complete` and
gepa's `reflection_lm`). Recording both to a JSON file once, then replaying in
CI, keeps the study reproducible and spends zero tokens on every run after the
first. These tests pin that contract with pure fakes — no network, no extra.
"""

from __future__ import annotations

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
    with Cassette(path, mode="replay") as rep:
        complete = rep.wrap(lambda p: "x", name="worker")
        with pytest.raises(CassetteMiss):
            complete("never-recorded")


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
