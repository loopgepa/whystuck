#!/usr/bin/env python
"""Per-task p0 calibration statistics from recorded arm-A cassettes. Zero cost, offline.

The calibration runs score every task of a manifest's ``calib`` level under the seed genome,
once per ``--sample`` salt. Replaying them gives, per task, the share of FIRST drafts that
already satisfied every constraint (``p_first``, whose mean over a task set is that set's p0)
and the mean FINAL score after the loop's repair (``final``, whose mean is the set's seed
baseline). `task_manifest.select_levels` builds target-p0 levels from exactly these two.

Replay is strict: a task whose calls are absent raises rather than dropping out, because a
pool that silently loses the tasks a recording failed on is a pool biased toward the tasks
that happened to succeed.

    python scripts/p0_task_stats.py --manifest M.json --seeds 0-29 --n-heldout 20 \\
        --run cassette-s0.json:0 --run cassette-s1.json:1 --out stats.json
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from whystuck.eval.cassette import Cassette
from whystuck.eval.llm_worker import make_self_refine_run_task
from whystuck.eval.multi_constraint import mc_feedback, mc_pass, mc_score
from whystuck.eval.study import seed_genome
from whystuck.eval.task_manifest import TaskManifest, load_manifest, manifest_split


def task_stats(
    manifest: TaskManifest,
    level: str,
    *,
    seeds: Sequence[int],
    n_heldout: int,
    runs: Sequence[tuple[str, int | None]],
    loop_max_retries: int = 1,
) -> dict[str, dict[str, float]]:
    """{task_id: {"p_first", "final", "k"}} over every (cassette, salt) replicate."""
    firsts: dict[str, list[float]] = {}
    finals: dict[str, list[float]] = {}
    genome = seed_genome("study-A", evolvable=set())
    for path, sample in runs:
        with Cassette(path, mode="replay") as cas:

            def _unreachable(_prompt: str, **_kw: object) -> str:
                raise AssertionError("unreachable in replay mode")

            wrapped = cas.wrap(_unreachable, name="worker")
            extra = {} if sample is None else {"sample": sample}
            run_task = make_self_refine_run_task(
                (lambda p, _w=wrapped, _e=extra: _w(p, **_e)),
                mc_score,
                feedback=mc_feedback,
                max_retries=loop_max_retries,
            )
            for seed in seeds:
                for task in manifest_split(manifest, level, "heldout", seed=seed, n=n_heldout):
                    trace = run_task(genome, task)
                    first = next(s for s in trace.steps if s.step_id == "generate-0")
                    firsts.setdefault(task.id, []).append(
                        1.0 if mc_pass(first.output_summary or "", task) else 0.0
                    )
                    finals.setdefault(task.id, []).append(float(trace.metadata["eval_score"]))
    return {
        tid: {
            "p_first": sum(firsts[tid]) / len(firsts[tid]),
            "final": sum(finals[tid]) / len(finals[tid]),
            "k": float(len(firsts[tid])),
        }
        for tid in sorted(firsts)
    }


def _parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--level", default="calib")
    ap.add_argument("--seeds", required=True, help="e.g. 0-29 or 0,1,2")
    ap.add_argument("--n-heldout", type=int, required=True)
    ap.add_argument(
        "--run", action="append", required=True, help="CASSETTE:SAMPLE (SAMPLE may be 'none')"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        path, _, s = spec.rpartition(":")
        runs.append((path, None if s == "none" else int(s)))
    stats = task_stats(
        load_manifest(a.manifest),
        a.level,
        seeds=_parse_seeds(a.seeds),
        n_heldout=a.n_heldout,
        runs=runs,
    )
    Path(a.out).write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    n = len(stats)
    p0 = sum(s["p_first"] for s in stats.values()) / n
    base = sum(s["final"] for s in stats.values()) / n
    print(f"{n} tasks  pool p0 = {p0:.3f}  pool baseline = {base:.3f}  -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
