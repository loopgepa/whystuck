"""Replay the superseded contrast runs at the pre-fix commit and record which genes moved.

Paper III's retraction rests on one question about the fifteen pre-correction contrast runs: did
arm D's winning genome change its critique gene? Those runs predate `genes_changed`, and the
corrected wiring changes every cassette key, so the answer cannot come from the released JSON or
from a replay under current code. It comes from replaying each run at `fe670a8`, the last commit
before the fix, and comparing every arm's winning genome with its seed genome, gene by gene.

The papers once stated that count as 15/15 runs. The audit that produced it replayed the flagship
run only; a full replay finds the critique gene changed on one seed of run 6 and one seed of the
nano worker swap. This script is that full replay, and its output
(`tests/fixtures/evidence/precorrection-gene-replay.json`) is what `verify_paper_claims.py` derives
the published counts from.

Usage (offline, $0; never run it inside a worktree you care about):

    git clone <repo> /tmp/fe && git -C /tmp/fe checkout fe670a8
    # the nano worker swap was released in the retraction commit, after fe670a8
    git -C /tmp/fe show 9388447:tests/fixtures/cassettes/study-multihard-nano.json \\
        > /tmp/fe/tests/fixtures/cassettes/study-multihard-nano.json
    git -C /tmp/fe show 9388447:docs/validation-study-multihard-nano.json \\
        > /tmp/fe/docs/validation-study-multihard-nano.json
    <python with gepa> scripts/replay_superseded_genes.py /tmp/fe --out replay.json

It monkeypatches the checkout's `study` module at import time to observe genomes and never edits
its files. A run is recorded only with its replayed per-seed scores, so the checker can confirm the
replay reproduced the released JSON before believing any gene it reports. BigCodeBench is absent:
its scorer executes model code with unseeded randomness and cannot replay bit for bit.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from typing import Any

NANO = (
    "run11 nano",
    "docs/validation-study-multihard-nano.json",
    "tests/fixtures/cassettes/study-multihard-nano.json",
    [
        "--task-set",
        "multi-hard",
        "--arms",
        "ABCD",
        "--seeds",
        "0,1,2",
        "--n-train",
        "16",
        "--n-heldout",
        "16",
        "--max-metric-calls",
        "64",
        "--reflection-calls",
        "10",
        "--loop-max-retries",
        "1",
        "--worker-model",
        "openai/gpt-4.1-nano",
        "--reflection-model",
        "openai/gpt-5.6-terra",
    ],
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", help="a checkout of fe670a8 (plus the nano run's two files)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.checkout)
    out_path = os.path.abspath(args.out)
    sys.path.insert(0, root + "/src")
    sys.path.insert(0, root + "/scripts")
    os.chdir(root)
    import run_validation_study as cli  # the checkout's own CLI
    import verify_reproducibility as vr

    from whystuck.eval import cassette as cas
    from whystuck.eval import study

    if not os.path.abspath(study.__file__).startswith(root + os.sep):
        raise SystemExit(f"imported {study.__file__}, not the checkout under {root}")
    log: list[tuple[str, str, dict[str, str]]] = []
    state: dict[str, Any] = {"pending": None, "misses": 0}
    orig_opt, orig_held, orig_wrap = study.optimize_loop, study._heldout_score, cas.Cassette.wrap

    def opt(store: Any, gid: str, *a: Any, **k: Any) -> Any:
        state["pending"] = gid
        return orig_opt(store, gid, *a, **k)

    def held(genome: Any, *a: Any, **k: Any) -> Any:
        text = {g.id: g.content for g in genome.genes if isinstance(g.content, str)}
        kind = "winner" if state["pending"] is not None else "seed"
        log.append((kind, state["pending"] or genome.id, text))
        state["pending"] = None
        return orig_held(genome, *a, **k)

    def wrap(self: Any, fn: Any, *, name: str) -> Any:
        inner = orig_wrap(self, fn, name=name)

        def counted(*a: Any, **k: Any) -> Any:
            try:
                return inner(*a, **k)
            except cas.CassetteMiss:
                state["misses"] += 1
                raise

        return counted

    study.optimize_loop, study._heldout_score, cas.Cassette.wrap = opt, held, wrap

    commit = os.popen(f"git -C {root} rev-parse HEAD").read().strip()
    records = []
    for label, committed, cassette, run_args in [*vr.RUNS, NANO]:
        log.clear()
        state["misses"] = 0
        tmp = tempfile.mkdtemp()
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = cli.main(
                ["--replay", *run_args, "--record", cassette, "--out", f"{tmp}/o.json"]
                + ["--report", f"{tmp}/r.md", "--max-usd", "99"]
            )
        with open(f"{tmp}/o.json", encoding="utf-8") as fh:
            replayed = json.load(fh)["runs"]
        seeds_text: dict[str, dict[str, str]] = {}
        per_arm: dict[str, list[dict[str, Any]]] = {}
        for kind, gid, text in log:
            arm = gid.rsplit("-", 1)[-1]
            if kind == "seed":
                seeds_text[arm] = text
                continue
            seed = seeds_text[arm]
            per_arm.setdefault(arm, []).append(
                {
                    "changed": sorted(g for g in text if text[g] != seed.get(g)),
                    "seed_sha256": {g: _sha(t) for g, t in sorted(seed.items())},
                    "winner_sha256": {g: _sha(t) for g, t in sorted(text.items())},
                    "winner_chars": {g: len(t) for g, t in sorted(text.items())},
                }
            )
        seeds = []
        for i, run in enumerate(replayed):
            entry: dict[str, Any] = {}
            for arm in ("B", "C", "D"):
                if arm in per_arm:
                    entry[arm] = {**per_arm[arm][i], "opt_heldout": run[arm]["opt_heldout"]}
            seeds.append(entry)
        records.append(
            {
                "label": label,
                "artifact": committed,
                "exit_code": code,
                "cassette_misses": state["misses"],
                "seeds": seeds,
            }
        )
        print(f"{label}: exit {code}, misses {state['misses']}, seeds {len(seeds)}", flush=True)
    doc = {
        "meta": {
            "replayed_at_commit": commit,
            "script": "scripts/replay_superseded_genes.py",
            "note": "winner = the genome each optimizing arm's held-out score was taken from; "
            "changed = genes whose text differs from that arm's seed genome",
        },
        "runs": records,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - needs a pre-fix checkout and gepa
    raise SystemExit(main())
