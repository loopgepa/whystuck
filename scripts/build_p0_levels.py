#!/usr/bin/env python
"""Phase 2 of the target-p0 pre-registration: build the study manifest from calibration stats.

Offline, $0. Reads the per-task stats of each calibration pool (``p0_task_stats.py`` output),
tags every task with the pool it came from, fixes the common baseline B by the registered rule,
and calls ``select_levels`` with p0, baseline and the L2-tier share all matched. The rule for B
(deviation 3) is the mean of the two pool means, i.e. the baseline of a 50/50 mixture, which is
what the matched share makes every level; if any level is infeasible there, the registered
fallback tries the nearest common B within +-0.05, nearest first.

    M=docs/p0-manifests
    python scripts/build_p0_levels.py \\
        --pool $M/full-1001-hard.json:$M/full-1001-hard-stats.json:0 \\
        --pool $M/full-1001-L2.json:$M/full-1001-L2-stats.json:1 \\
        --out $M/study.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from whystuck.eval.task_manifest import build_manifest, load_manifest, save_manifest, select_levels

TARGETS = {"low": 0.07, "mid": 0.27, "high": 0.50}  # filled in this order (deviation 3)
SEEDS = list(range(10))
N_TRAIN, N_HELDOUT = 8, 10
TOL_P0 = TOL_BASELINE = 0.03
SHARE = (0.5, 0.07)
RNG_SEED = 20261025


def registered_baselines(pool_means: Sequence[float], *, step: float = 0.01) -> list[float]:
    """B* first, then the fallback grid within +-0.05 of it, nearest first (ties: lower first)."""
    b_star = sum(pool_means) / len(pool_means)
    grid = [round(b_star + k * step, 6) for k in range(-5, 6)]
    return sorted(grid, key=lambda b: (round(abs(b - b_star), 6), b))


def build(pools: Sequence[tuple[str, str, float]]) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    tasks = []
    generators = []
    means = []
    for manifest_path, stats_path, l2 in pools:
        m = load_manifest(manifest_path)
        st: Mapping[str, Mapping[str, float]] = json.loads(Path(stats_path).read_text())
        missing = sorted(set(m.tasks) - set(st))
        if missing:
            raise ValueError(f"{stats_path}: {len(missing)} calibrated tasks have no stats")
        means.append(sum(float(v["final"]) for v in st.values()) / len(st))
        for tid, v in st.items():
            stats[tid] = {"p_first": float(v["p_first"]), "final": float(v["final"]), "l2": l2}
        tasks += list(m.tasks.values())
        gen = m.generator
        generators += [gen] if isinstance(gen, dict) else list(gen)
    tried = []
    for b in registered_baselines(means):
        try:
            sel = select_levels(
                stats,
                targets=TARGETS,
                seeds=SEEDS,
                n_train=N_TRAIN,
                n_heldout=N_HELDOUT,
                baseline=b,
                tol_p0=TOL_P0,
                tol_baseline=TOL_BASELINE,
                rng_seed=RNG_SEED,
                covariates={"l2": SHARE},
            )
        except ValueError as exc:
            tried.append((b, str(exc)))
            continue
        used = {
            i for lv in sel["levels"].values() for sd in lv.values() for v in sd.values() for i in v
        }
        manifest = build_manifest(
            generators,
            [t for t in tasks if t.id in used],
            sel["levels"],
            stats={
                "pool_mean_final": means,
                "baseline": b,
                "baseline_rule": "mean of pool means, fallback nearest within 0.05",
                "infeasible_baselines_tried": tried,
                "targets": TARGETS,
                "tolerances": {"p0": TOL_P0, "baseline": TOL_BASELINE, "l2_share": SHARE},
                "rng_seed": RNG_SEED,
                "achieved": sel["achieved"],
                "per_task": {i: stats[i] for i in sorted(used)},
            },
        )
        return manifest
    raise ValueError(f"no registered baseline is feasible: {tried}")


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI convenience
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pool", action="append", required=True, help="MANIFEST:STATS:L2FLAG")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pools = []
    for spec in a.pool:
        m, s, flag = spec.split(":")
        pools.append((m, s, float(flag)))
    manifest = build(pools)
    save_manifest(a.out, manifest)
    st = manifest["stats"]
    print(
        f"baseline B = {st['baseline']} (pool means {[round(x, 4) for x in st['pool_mean_final']]})"
    )
    for lv, seeds in st["achieved"].items():
        for split in ("train", "heldout"):
            vals = [seeds[s][split] for s in seeds]
            print(
                f"  {lv:5} {split:8} p0 {sum(v['p0'] for v in vals) / len(vals):.3f}  "
                f"final {sum(v['final'] for v in vals) / len(vals):.3f}  "
                f"l2 {sum(v['l2'] for v in vals) / len(vals):.3f}"
            )
    print(f"-> {a.out}  sha256 {manifest['sha256'][:16]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
