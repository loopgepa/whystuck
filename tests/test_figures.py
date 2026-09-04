"""The figures may not disagree with the paper they sit in.

WHY THIS TEST EXISTS. A figure is the easiest place in a paper for a number to go stale, because
it is a binary blob: no reader diffs a PDF, no checker greps it, and the value that produced it
lived in a plotting call nobody re-reads. This study has already found the same drift in prose, in
tables and in units of sigma, each time in the copy nothing was looking at. A figure would have
been the fourth.

`make_figures.figure_data()` is pure and imports no plotting stack, precisely so this test can run
in the ordinary offline path. Two properties are enforced:

1. **Every number a figure plots is one the paper's own checker derives.** Same membership rule the
   prose and sigma guards use. A figure cannot invent a value or keep one a later replicate
   superseded.
2. **The headline quantities equal what the text says**, scraped from `main.tex`. Membership alone
   would let a figure plot the right number in the wrong place --- 3.09 where 0.99 belongs, both
   being values the checker derives.

Renders are deliberately NOT exercised here. Asserting on pixels would fail on any matplotlib
upgrade while catching nothing about correctness, and the numbers are the part that can be wrong.
"""

from __future__ import annotations

import ast
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import make_figures as mf  # noqa: E402
import paper_sources  # noqa: E402
import verify_paper_claims as vpc  # noqa: E402

# The three papers, concatenated with their shared inputs expanded. Reading them through
# `paper_sources` rather than one hard-coded path is what keeps this guard covering the whole series
# after the 46-page draft was split three ways: a guard that silently narrowed its own scope is
# precisely the defect these papers are about.
PAPERS_TEX = paper_sources.all_tex
CHECKER = pathlib.Path("scripts/verify_paper_claims.py")


def _bound_values() -> set[float]:
    """Every number the checker derives, plus every number its regexes pull out of the paper."""
    tex = PAPERS_TEX()
    out = {round(float(c[1]), 4) for c in vpc._checks() if isinstance(c[1], (int, float))}
    for node in ast.walk(ast.parse(CHECKER.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if "([\\d.]+)" not in node.value:
            continue
        try:
            rx = re.compile(node.value, re.DOTALL | re.MULTILINE)
        except re.error:
            continue
        for m in rx.finditer(tex):
            for g in m.groups():
                try:
                    out.add(round(float(g), 4))
                except (TypeError, ValueError):
                    continue
    return out


def _plotted() -> list[tuple[str, float]]:
    """(label, value) for every quantity the three figures draw as a number."""
    d = mf.figure_data()
    out: list[tuple[str, float]] = []
    for lv in d.levels:
        out += [
            (f"{lv.name} p0", lv.p0),
            (f"{lv.name} delta single", lv.delta_single),
            (f"{lv.name} delta K=5", lv.delta_k5),
            (f"{lv.name} t", lv.t_k5),
            (f"{lv.name} PASS evolved", float(lv.pass_evolved)),
            (f"{lv.name} PASS seed", float(lv.pass_seed)),
            (f"{lv.name} false PASS evolved", float(lv.false_pass_evolved)),
            (f"{lv.name} false PASS seed", float(lv.false_pass_seed)),
            (f"{lv.name} n judged", float(lv.n_judged)),
        ]
        out += [(f"{lv.name} per-seed {k}", v) for k, v in enumerate(lv.per_seed)]
        if lv.saturated_baseline is not None:
            out.append((f"{lv.name} saturated baseline", lv.saturated_baseline))
    out += [("sigma", d.sigma), ("swing", d.swing), ("swing in sigma", d.swing_in_sigma)]
    out += [(f"curve {r.run}", r.score) for r in d.curve]
    for k, (gap, sig) in d.curve_gaps.items():
        out += [(f"gap {k}", gap), (f"gap {k} in sigma", sig)]
    for c in d.depth:
        out += [
            (f"mmc{c.mmc} delta", c.delta),
            (f"mmc{c.mmc} depth", float(c.depth)),
            (f"mmc{c.mmc} accepted", float(c.accepted)),
        ]
    out.append(("within-run sigma", d.depth_sigma_within))
    # fig:noop. Every marker in the left panel is a boolean and every dot in the right one is a
    # published delta, so the deltas are what can go stale here; the counts the caption states are
    # pinned separately by `_noop_figure_checks` in the claim checker.
    for r in d.precorrection:
        out.append((f"precorrection {r.name} delta", r.mean_delta))
    out += [(f"precorrection moved {k}", float(v)) for k, v in d.precorrection_moved.items()]
    out += [
        ("precorrection runs", float(len(d.precorrection))),
        ("precorrection identical", float(sum(r.identical for r in d.precorrection))),
    ]
    # fig:effects. The sigma-scaled values, which is where a stale sigma would show up.
    for e in d.effects:
        out.append((f"effect {e.label}", e.in_sigma))
        if e.single_in_sigma is not None:
            out.append((f"effect {e.label} single draw", e.single_in_sigma))
    out.append(("SESOI in sigma", d.sesoi_in_sigma))
    # fig:residual and fig:worker.
    for bar in d.residual:
        out += [(f"residual {bar.constraint} {run}", rate) for run, rate in bar.rates.items()]
    for w in d.worker:
        out += [
            (f"worker {w.label} held-out", w.held_out),
            (f"worker {w.label} delta", w.delta),
            (f"worker {w.label} se", w.se),
            (f"worker {w.label} t", w.t),
        ]
    out.append(("worker seeds", float(d.worker_seeds)))
    return out


def test_every_plotted_number_is_derived():
    bound = _bound_values()

    def derived(v: float) -> bool:
        """Is `v` a value the checker derives, at any precision the paper displays it to?

        The comparison has to be precision-tolerant in ONE direction only. A figure holds full
        precision (p_0 = 0.0666...) while the paper prints two or three decimals ($0.067$),
        so a 4dp-only rule reports two dozen false orphans --- it did, on first run. Accepting any
        of 2, 3 or 4 decimals still catches the failure this guards against, because a stale
        quotient differs from its replacement in the SECOND decimal (3.16 against 3.09), which no
        rounding can reconcile. Sign is not load-bearing for provenance: a figure may draw a
        magnitude where the artifact carries a signed value.
        """
        # 1dp is in the set for exactly one quantity: the paper displays the swing as
        # "$\\approx 2.5\\sigma$". Its exact value is pinned separately by
        # `test_headline_quantities_equal_the_text`, so the loose end is covered there.
        return any(round(s * v, d) in bound for d in (1, 2, 3, 4) for s in (1, -1))

    orphans = {label: value for label, value in _plotted() if not derived(value)}
    assert not orphans, (
        "these values are plotted but no check in verify_paper_claims derives them from an "
        f"artifact: {orphans}. A figure is the one place a stale number is invisible."
    )


def test_headline_quantities_equal_the_text():
    """Membership is not enough: a figure could plot a derived value in the wrong place."""
    tex = PAPERS_TEX()
    d = mf.figure_data()
    checks = [
        ("sigma", d.sigma, r"\\sigma\(\\dd\) \\;=\\; 0.1591/\\sqrt\{30\} \\;=\\; ([\d.]+)", 5e-5),
        ("swing", d.swing, r"is \$([\d.]+)\$, about", 5e-4),
        ("swing in sigma", d.swing_in_sigma, r"\\approx\\; ([\d.]+) ,", 0.05),
        (
            "oracle vs tool in sigma",
            d.curve_gaps["tool_vs_oracle"][1],
            r"below the tool-fed critique at \$([\d.]+)\\sigma\$",
            5e-3,
        ),
        (
            "oracle vs intrinsic in sigma",
            d.curve_gaps["intrinsic_vs_oracle"][1],
            r"\\emph\{intrinsic\} one at \$([\d.]+)\\sigma\$",
            5e-3,
        ),
        ("within-run sigma at L1", d.depth_sigma_within, r"is \$([\d.]+)\$ at \$L_1\$", 5e-4),
        # Anchored on the clause, not on "it reads": the split sweep uses that same phrase
        # earlier in the paper, and the loose pattern silently pinned this to +0.023.
        (
            "deepest cell delta",
            d.depth[-1].delta,
            r"under-search explanation to survive; it reads \$\+([\d.]+)\$",
            5e-5,
        ),
    ]
    bad = []
    for label, plotted, pattern, tol in checks:
        m = re.search(pattern, tex, re.DOTALL | re.MULTILINE)
        if m is None:
            bad.append(
                f"{label}: the paper sentence this pins to was reworded, pattern {pattern!r}"
            )
            continue
        printed = float(m.group(1))
        if not math.isclose(abs(plotted), printed, abs_tol=tol):
            bad.append(f"{label}: figure plots {plotted:.6f}, paper prints {printed}")
    assert not bad, "figure and text disagree:\n  " + "\n  ".join(bad)


def test_the_swing_uses_the_root_two_denominator():
    """A difference of two independent deltas carries sigma twice.

    Dividing by a single sigma would overstate the study's load-bearing quantity by 41%, and the
    figure annotates it in sigma units, so the figure would carry the error into print.
    """
    d = mf.figure_data()
    naive = d.swing / d.sigma
    assert not math.isclose(d.swing_in_sigma, naive, rel_tol=0.01), (
        "swing_in_sigma looks like it divided by a single sigma"
    )
    assert math.isclose(d.swing_in_sigma, d.swing / (d.sigma * 2**0.5), rel_tol=1e-9)


def test_cap_bound_is_read_off_the_measured_depth():
    """Not off the registered cycle formula, which the sweep exists to correct.

    The formula charged every proposal a valset re-evaluation that only accepted proposals pay, so
    it put the cap's binding point at a budget far above where it actually binds. A figure deriving
    "cap-bound" from it would relabel three of six cells.
    """
    d = mf.figure_data()
    for c in d.depth:
        assert c.cap_bound == (c.depth == c.cap), f"mmc={c.mmc} disagrees with its measured depth"
    assert sum(c.cap_bound for c in d.depth) == 4, "expected four cap-bound cells"


def test_coverage_is_not_vacuous():
    plotted = _plotted()
    assert len(plotted) >= 50, f"figure data looks empty, {len(plotted)} values"
    assert len(_bound_values()) >= 200, "the checker's derived-value set looks empty"
    assert len({v for _, v in plotted}) >= 25, "plotted values look collapsed"


def test_no_latex_text_markup_in_labels():
    """usetex is unavailable here (no dvipng), so \\textbf or \\emph would print literally.

    Cheap to check statically and impossible to notice in a 6-inch PDF until a reviewer does.

    Scanned over string literals rather than over the raw text of the render half. The first version
    grepped the source, and a code comment explaining which macro the constraint labels imitate was
    enough to fail it: a guard that fires on prose about the thing it guards teaches people to edit
    the comment. `\\mathtt` and other math macros are fine, since mathtext renders them.
    """
    src = pathlib.Path("scripts/make_figures.py").read_text(encoding="utf-8")
    render_half = src[src.index("def fig_p0(") : src.index("def render(")]
    literals = [
        node.value
        for node in ast.walk(ast.parse(render_half))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    docstrings = {
        ast.get_docstring(node) or ""
        for node in ast.walk(ast.parse(render_half))
        if isinstance(node, ast.FunctionDef)
    }
    for text in literals:
        if text in docstrings:
            continue
        for macro in (r"\textbf", r"\emph", r"\texttt", r"\textit"):
            assert macro not in text, (
                f"{macro} in a figure label would be printed verbatim: {text[:60]!r}"
            )


#: Minimum greyscale contrast ratio between two palette colours that a reader has to tell apart.
#: The three maps below sit at 1.52 or better; the defects this guard was written for sat at 1.34,
#: 1.13 and 1.02.
_MIN_GREY_RATIO = 1.4


def _relative_luminance(hex_colour: str) -> float:
    """WCAG relative luminance, which is what a greyscale printer approximates."""
    channels = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _grey_ratio(a: str, b: str) -> float:
    hi, lo = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_series_a_reader_must_separate_survive_greyscale() -> None:
    """Colour is the encoding these figures fall back on, and reviewers print papers.

    The palette is Okabe-Ito, so colour-vision deficiency is covered, and that is what made this
    easy to miss: a colour-blind-safe pair can still be two identical greys. Three maps were,
    each in the panel that carries its paper's point. `fig:residual` drew the intrinsic critique
    and the three-refine-round oracle three greyscale levels apart out of 255, in the figure whose
    subject is the progression between them. `fig:detector` separated the good tool from the
    perfect oracle by 1.13, which is the pair Paper II's non-monotonicity is about. `fig:p0`'s
    lower panel ran at 1.34 and also spent one of the upper panel's colours on a second meaning.

    Captions describe shape and position rather than colour, and the bars carry printed values, so
    none of this made a figure unreadable. It made the legend useless in print, which is a
    different and quieter kind of wrong. The maps are read out of the source so a recolour has to
    pass here, and the threshold is applied to computed luminance rather than to a colour name.
    """
    src = pathlib.Path("scripts/make_figures.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    palette = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and node.value.value.startswith("#")
    }
    assert {"ORANGE", "GREEN", "BLUE", "VERMILLION", "SKY"} <= set(palette), (
        f"the Okabe-Ito palette is no longer declared as module constants; found {sorted(palette)}"
    )

    #: Every set of palette colours in this file that distinguishes series inside one figure,
    #: read from the source rather than restated: the two ordinal `colour` maps and the pair of
    #: bars in fig:p0's lower panel.
    maps: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "colour"
            and isinstance(node.value, ast.Dict)
        ):
            names = [v.id for v in node.value.values if isinstance(v, ast.Name)]
            if len(names) > 1:
                maps.append((f"colour map of {len(names)}", names))
    lower = re.search(
        r"\(lv\.pass_seed, lv\.false_pass_seed, (\w+)\),\s*\(lv\.pass_evolved, "
        r"lv\.false_pass_evolved, (\w+)\)",
        re.sub(r"\s+", " ", src),
    )
    assert lower, "fig:p0's lower-panel bar pair no longer reads as this guard expects"
    maps.append(("fig:p0 lower panel", [lower.group(1), lower.group(2)]))

    assert len(maps) >= 3, f"expected at least three colour groups, found {len(maps)}"
    for label, names in maps:
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ratio = _grey_ratio(palette[a], palette[b])
                assert ratio >= _MIN_GREY_RATIO, (
                    f"{label}: {a} and {b} are {ratio:.2f} apart in greyscale, under "
                    f"{_MIN_GREY_RATIO}. A reader printing this figure cannot separate them."
                )
