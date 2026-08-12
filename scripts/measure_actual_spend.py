#!/usr/bin/env python
"""Measure what the study ACTUALLY cost, per run and per provider, offline and at $0.

WHY THIS EXISTS. Every run's metadata records `estimated_usd`, and the papers are careful to call
it a pre-flight **bound** rather than an invoice -- one draft reported a bound as a measured spend
and that is in the corrections register. But a bound that is never reconciled against anything is
a number with no error bar: the one run that was reconciled, D3 on 2026-08-06, came in at $0.074
against a $6.5198 ceiling. Reporting only ceilings makes the study look ~10x more expensive than
it was, and "we only ever report bounds" is a defensible policy exactly until someone asks what it
cost.

HOW. Prompts are stored as hashes, not text, so a cassette alone cannot price its input tokens.
A replay reconstructs every prompt byte-for-byte -- that is why replay matches at all -- so this
script replays each run with all four call surfaces instrumented and reads the prompts off the
wire. Billing is per DISTINCT (surface, prompt): during the original recording a repeated prompt
was served from the cassette and never reached the provider, so the billed count equals the
cassette's entry count for that surface, not the number of calls a replay makes. On D3 that is 347
billed against 1124 fired.

WHAT IT IS NOT. A floor, not an invoice, for three reasons stated so nobody reads it as the latter:

  1. a call that failed and was retried was billed and left no cassette entry;
  2. `cl100k_base` is not Llama's tokenizer nor gpt-4.1's, so token counts are approximate --
     the script reports a chars/4 figure alongside as a second bracket, which on D3 spanned
     $0.074 to $0.084;
  3. the 18 runs that cannot replay (13 superseded, 5 BigCodeBench) have unreadable inputs. Their
     OUTPUT tokens are exact, since outputs are stored verbatim, and their inputs are estimated
     from the input/output ratio measured on the replayable runs of the same task family. Those
     rows are labelled `estimated` and are never added to a `measured` total silently.

Run with the eval venv (needs gepa+litellm to drive the replay):
    scratchpad/evalvenv/bin/python scripts/measure_actual_spend.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import traceback
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from whystuck.eval.cassette import Cassette  # noqa: E402
from whystuck.eval.cost import MODEL_PRICES  # noqa: E402

OUT_PATH = Path("docs/actual-spend.json")


def _encoder() -> Callable[[str], int]:
    """Token counter. `cl100k_base` for every model, with the caveat in the module docstring."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text, disallowed_special=()))


def _surface_models(args: list[str]) -> dict[str, str]:
    """Surface -> model id, from a registered run's exact CLI args.

    Parsed with the CLI's OWN `_parse_args` rather than by reading the flag list here, so an
    unpassed flag resolves to the same default the run itself resolved it to. Duplicating the
    defaults would misprice every older run in whichever direction the copy had drifted, and
    would do it silently -- the arithmetic still completes.
    """
    import run_validation_study as cli

    ns = cli._parse_args(list(args))
    out: dict[str, str] = {}
    for surface, dest in (
        ("worker", "worker_model"),
        ("reflection", "reflection_model"),
        ("critic", "critique_model"),
        ("verdict", "verdict_model"),
    ):
        model = getattr(ns, dest, None)
        if model:
            out[surface] = str(model)
    # The oracle gate makes no critique call at all, so a critic model would be priced for calls
    # that never happen. The cassette settles it either way -- a surface with no entries costs
    # nothing -- but dropping it here keeps the reported surface list honest.
    if getattr(ns, "oracle_critique", False):
        out.pop("critic", None)
        out.pop("verdict", None)
    return out


def _instrumented_replay(cassette: str, args: list[str]) -> dict[str, Any] | None:
    """Replay one run, returning per-surface distinct-prompt token totals.

    Returns None when the replay fails, which is reported rather than swallowed: a run that no
    longer replays has an unmeasurable input side, and treating that as $0 would understate the
    total in exactly the direction that flatters the study.
    """
    import run_validation_study as cli

    seen: dict[tuple[str, str], str] = {}
    original = Cassette.wrap

    def spy(self: Cassette, fn: Callable[..., str], *, name: str) -> Callable[..., str]:
        inner = original(self, fn, name=name)

        def recorded(*a: Any, **kw: Any) -> str:
            output = inner(*a, **kw)
            prompt = a[0] if a and isinstance(a[0], str) else json.dumps([a, sorted(kw)])
            seen.setdefault((name, prompt), output)
            return output

        return recorded

    Cassette.wrap = spy  # type: ignore[method-assign]
    try:
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                code = cli.main(
                    [
                        "--replay",
                        *args,
                        "--record",
                        cassette,
                        "--out",
                        str(Path(d) / "replay.json"),
                        "--report",
                        str(Path(d) / "r.md"),
                        "--max-usd",
                        "99",
                    ]
                )
            if code != 0:
                return None
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return None
    finally:
        Cassette.wrap = original  # type: ignore[method-assign]

    count = _encoder()
    per_surface: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tin": 0, "tout": 0})
    chars: dict[str, dict[str, int]] = defaultdict(lambda: {"cin": 0, "cout": 0})
    for (name, prompt), output in seen.items():
        s = per_surface[name]
        s["calls"] += 1
        s["tin"] += count(prompt)
        s["tout"] += count(output)
        chars[name]["cin"] += len(prompt)
        chars[name]["cout"] += len(output)
    return {"surfaces": dict(per_surface), "chars": dict(chars)}


def _price(surface_tokens: dict[str, dict[str, int]], models: dict[str, str]) -> dict[str, float]:
    """Provider -> USD. Unpriced or unmapped surfaces raise rather than costing nothing."""
    out: dict[str, float] = defaultdict(float)
    for surface, t in surface_tokens.items():
        if surface not in models:
            raise KeyError(
                f"surface {surface!r} fired {t['calls']} billed call(s) but no model is mapped to "
                f"it from the run's args; pricing it at zero would understate the run"
            )
        model = models[surface]
        pin, pout = MODEL_PRICES[model]  # KeyError on an unpriced model, by design
        provider = model.split("/", 1)[0]
        out[provider] += (t["tin"] * pin + t["tout"] * pout) / 1_000_000
    return dict(out)


def _cassette_outputs(path: str) -> dict[str, dict[str, int]]:
    """Per-surface EXACT output tokens for a run that cannot replay.

    Outputs are stored verbatim, so this half is measured rather than estimated even when the run
    is unreplayable. Only the input side is unknowable, which is what keeps these rows honest:
    half measured, half modelled, and labelled as such.
    """
    count = _encoder()
    entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "tout": 0, "cout": 0})
    for entry in entries.values():
        s = out[entry.get("name", "worker")]
        s["calls"] += 1
        s["tout"] += count(entry["output"])
        s["cout"] += len(entry["output"])
    return dict(out)


def _ratios(measured: list[dict[str, Any]]) -> dict[str, float]:
    """Per-surface input/output token ratio, measured on the runs that do replay.

    PER SURFACE, not pooled, because the surfaces differ by more than 3x: a worker call carries
    the whole task prompt against a short draft (5.3), a reflection call is closer to balanced
    (1.8). A single pooled ratio would misprice every unreplayable run by whichever mix it
    happened to have, and the superseded runs are worker-heavy, so the error would be one-sided.
    """
    tin: dict[str, int] = defaultdict(int)
    tout: dict[str, int] = defaultdict(int)
    for row in measured:
        for surface, t in row["surfaces"].items():
            tin[surface] += t["tin"]
            tout[surface] += t["tout"]
    return {s: tin[s] / tout[s] for s in tin if tout[s]}


def _estimate(row: dict[str, Any], ratios: dict[str, float]) -> dict[str, Any]:
    """Price a run whose inputs cannot be read, using the measured per-surface ratio.

    Outputs are exact. Inputs are modelled. The row records which half is which so a reader can
    discount it, and the totals keep the two apart rather than summing them into one figure that
    would read as measured.
    """
    usd: dict[str, float] = defaultdict(float)
    for surface, t in row["surfaces"].items():
        model = row["models"].get(surface)
        if model is None:
            raise KeyError(f"{row['label']}: no model mapped to surface {surface!r}")
        pin, pout = MODEL_PRICES[model]
        est_in = t["tout"] * ratios.get(surface, ratios["worker"])
        usd[model.split("/", 1)[0]] += (est_in * pin + t["tout"] * pout) / 1_000_000
    row["usd_by_provider_est"] = dict(usd)
    row["usd_est"] = sum(usd.values())
    return row


def _meta_models(run_path: str) -> dict[str, str]:
    """Surface -> model for a run with no registry entry, from its own metadata.

    Reading config out of the artifact is refused elsewhere in this repository on the grounds that
    a run must not get to say how it is replayed. That objection does not apply to PRICING: a
    price is not a result the configuration could have been tuned to produce, and the alternative
    is leaving five bought runs out of the total entirely.
    """
    meta = json.loads(Path(run_path).read_text(encoding="utf-8"))["meta"]
    out = {"worker": meta["worker_model"], "reflection": meta["reflection_model"]}
    if meta.get("critique_model") and not meta.get("oracle_critique"):
        out["critic"] = meta["critique_model"]
    if meta.get("verdict_model") and not meta.get("oracle_critique"):
        out["verdict"] = meta["verdict_model"]
    return out


#: The replicate loop's cassettes. 225 scorings of already-released genomes, worker calls only --
#: no optimizer runs, so no reflection surface. Outside `verify_reproducibility.RUNS` because they
#: are not study runs, and therefore outside every total the registry can produce: they were
#: missing from the spend picture entirely until this script went looking for them.
REPEAT_DIR = Path("tests/fixtures/cassettes/repeat")
REPEAT_WORKER = "openai/gpt-4.1-mini"


def _repeat_spend(ratios: dict[str, float]) -> dict[str, Any]:
    count = _encoder()
    calls = 0
    tout = 0
    for path in sorted(REPEAT_DIR.glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8"))["entries"].values():
            calls += 1
            tout += count(entry["output"])
    pin, pout = MODEL_PRICES[REPEAT_WORKER]
    est_in = tout * ratios["worker"]
    return {
        "label": f"{len(list(REPEAT_DIR.glob('*.json')))} replicate scorings",
        "basis": "outputs measured, inputs estimated from the measured worker ratio",
        "calls": calls,
        "tout": tout,
        "est_tin": round(est_in),
        "usd_est": (est_in * pin + tout * pout) / 1_000_000,
        "model": REPEAT_WORKER,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="substring filter on the run label")
    ap.add_argument("--out", default=str(OUT_PATH))
    ns = ap.parse_args()

    import verify_reproducibility as vr

    measured: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    for label, committed, cassette, args in vr.RUNS:
        if ns.only and ns.only not in label:
            continue
        if not Path(committed).exists() or not Path(cassette).exists():
            continue
        models = _surface_models(args)
        superseded = label in vr.SUPERSEDED
        row: dict[str, Any] = {
            "label": label,
            "run": committed,
            "models": models,
            "bound_usd": json.loads(Path(committed).read_text())["meta"].get("estimated_usd"),
        }
        result = None if superseded else _instrumented_replay(cassette, args)
        if result is None:
            outs = _cassette_outputs(cassette)
            row["basis"] = "outputs measured, inputs unknown (no replay)"
            row["surfaces"] = outs
            unmeasured.append(row)
            print(
                f"[PART] {label:30} outputs only ({sum(s['calls'] for s in outs.values())} calls)"
            )
            continue
        row["basis"] = "replayed, both sides measured"
        row["surfaces"] = result["surfaces"]
        row["chars"] = result["chars"]
        row["usd_by_provider"] = _price(result["surfaces"], models)
        row["usd"] = sum(row["usd_by_provider"].values())
        measured.append(row)
        print(f"[OK  ] {label:30} ${row['usd']:8.4f}  (bound ${row['bound_usd']:.4f})")

    # The runs with no registry row at all: BigCodeBench, which cannot replay by construction.
    # Skipping them would leave five bought runs out of every total this script prints.
    for run_path in sorted(vr.UNREPLAYABLE):
        if ns.only and ns.only not in run_path:
            continue
        slug = Path(run_path).name.replace("validation-study", "study")
        cassette = f"{vr.CASSETTES}/{slug}"
        if not Path(cassette).exists() or not Path(run_path).exists():
            continue
        outs = _cassette_outputs(cassette)
        unmeasured.append(
            {
                "label": Path(run_path).stem.replace("validation-study-", ""),
                "run": run_path,
                "models": _meta_models(run_path),
                "bound_usd": json.loads(Path(run_path).read_text())["meta"].get("estimated_usd"),
                "basis": "outputs measured, inputs unknown (unreplayable by construction)",
                "surfaces": outs,
            }
        )
        print(f"[PART] {Path(run_path).stem:30} outputs only (unreplayable)")

    ratios = _ratios(measured)
    unmeasured = [_estimate(row, ratios) for row in unmeasured]
    repeat = _repeat_spend(ratios)

    totals: dict[str, float] = defaultdict(float)
    for row in measured:
        for provider, usd in row["usd_by_provider"].items():
            totals[provider] += usd
    est_totals: dict[str, float] = defaultdict(float)
    for row in unmeasured:
        for provider, usd in row["usd_by_provider_est"].items():
            est_totals[provider] += usd
    est_totals[REPEAT_WORKER.split("/", 1)[0]] += repeat["usd_est"]

    Path(ns.out).write_text(
        json.dumps(
            {
                "measured": measured,
                "partial": unmeasured,
                "replicates": repeat,
                "surface_in_out_ratios": ratios,
                "totals": {
                    "measured_by_provider": dict(totals),
                    "estimated_by_provider": dict(est_totals),
                    "measured_usd": sum(totals.values()),
                    "estimated_usd": sum(est_totals.values()),
                    "bounds_measured_runs_usd": sum(r["bound_usd"] for r in measured),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nMEASURED (both sides), by provider:")
    for provider, usd in sorted(totals.items()):
        print(f"  {provider:10} ${usd:9.4f}")
    print(f"  {'subtotal':10} ${sum(totals.values()):9.4f} over {len(measured)} run(s)")
    print(f"  bounds for those same runs: ${sum(r['bound_usd'] for r in measured):9.4f}")
    print("\nESTIMATED (outputs measured, inputs modelled), by provider:")
    for provider, usd in sorted(est_totals.items()):
        print(f"  {provider:10} ${usd:9.4f}")
    print(
        f"  {'subtotal':10} ${sum(est_totals.values()):9.4f} over {len(unmeasured)} run(s) "
        f"plus {repeat['calls']} replicate calls"
    )
    print(f"\nGRAND TOTAL (floor): ${sum(totals.values()) + sum(est_totals.values()):9.4f}")
    print(f"  of which the borrowed Groq credential: ${totals.get('groq', 0.0):9.4f}")
    print(f"see {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
