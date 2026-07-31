"""The paper's three figures, with every plotted number derived from a released artifact.

WHY THIS EXISTS. Sixteen tables carried the paper and three of its central claims are *shapes*
that a table cannot show: the sign of $\\dd$ changing across $p_0$, detector quality failing to
order held-out performance, and $\\dd$ failing to order along search depth. A reader had to
reconstruct each shape mentally from rows of numbers.

THE DISCIPLINE, which is the point of the split in this module. `figure_data()` is pure: it
derives every value through the same helpers `verify_paper_claims.py` uses, imports no plotting
stack, and returns plain dataclasses. `render()` imports matplotlib lazily and draws only what
`figure_data()` produced. Nothing is typed into a plot call. That is what lets
`tests/test_figures.py` assert each plotted number equals what the paper's own checker derives,
without
matplotlib installed --- so a figure cannot drift from the text it sits next to, which is the
failure mode this study has now hit in prose, in tables and in units of sigma.

Typography matches the paper rather than approximating it: the body font is Computer Modern
(`cmr10`, shipped with matplotlib), the same face `article` at 11pt sets, so a figure does not
announce itself as a foreign object on the page. `axes.unicode_minus` is off because cmr10 has no
U+2212 and matplotlib would silently substitute a box.

Colours are Okabe--Ito, the de-facto standard colourblind-safe categorical set (Okabe & Ito 2002,
popularised by Wong, Nature Methods 2011). Yellow is unused: it is nearly invisible as a thin line.

Usage: `python scripts/make_figures.py` writes docs/papers/shared/figures/*.pdf. Offline, $0.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_paper_claims as vpc  # noqa: E402

OUTDIR = Path("docs/papers/shared/figures")

#: Okabe--Ito. Yellow (#F0E442) deliberately absent: unreadable as a stroke.
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"

#: Student-t two-sided critical values at K=5, i.e. 4 degrees of freedom. The paper is explicit
#: that a normal approximation would misreport one of the three levels, so the figure must use the
#: same t the text does or it would draw an interval the text denies.
T_CRIT_95 = 2.776

#: The pre-registered smallest effect size of interest, in held-out score. A constant of the
#: registration, not a measurement, which is why it is a literal here and exempted in the
#: paper's coverage guards.
SESOI = 0.05


@dataclass(frozen=True)
class Level:
    """One $p_0$ level: the single draw, the five-replicate mean, and the detector behaviour."""

    name: str
    p0: float
    delta_single: float
    per_seed: list[float]
    delta_k5: float
    se_k5: float
    t_k5: float
    verdict: str
    pass_evolved: int
    false_pass_evolved: int
    pass_seed: int
    false_pass_seed: int
    n_judged: int
    saturated_baseline: float | None = None

    @property
    def ci95(self) -> tuple[float, float]:
        half = T_CRIT_95 * self.se_k5
        return self.delta_k5 - half, self.delta_k5 + half


@dataclass(frozen=True)
class CurveRow:
    """One configuration on the detector-quality ladder."""

    label: str
    run: str
    score: float
    detector: str
    group: str


@dataclass(frozen=True)
class DepthCell:
    """One cell of the budget sweep, at its measured (not priced) search depth."""

    mmc: int
    cap: int
    depth: int
    delta: float
    accepted: int
    cap_bound: bool


@dataclass(frozen=True)
class FigureData:
    levels: list[Level]
    sigma: float
    swing: float
    swing_in_sigma: float
    curve: list[CurveRow]
    curve_gaps: dict[str, tuple[float, float]]
    depth: list[DepthCell]
    depth_sigma_within: float
    notes: dict[str, str] = field(default_factory=dict)


def _pass_counts(level: int, arm: str) -> tuple[int, int]:
    """(PASS emitted, of which wrong) for one arm's critique at one level.

    A PASS is a true positive plus a false positive: the confusion matrix counts the verdict, so
    `tp + fp` IS the number of drafts the critique accepted. Reporting the pair rather than the
    total is the whole of the paper's precision-1 claim --- the evolved gene accepts less often
    than the seed rubric and is never wrong when it does.
    """
    c = vpc._arm_confusion(level, arm)
    return c["tp"] + c["fp"], c["fp"]


def figure_data() -> FigureData:
    """Every number the three figures plot, derived from the released artifacts.

    No matplotlib import anywhere below this line, by design: the test that pins these values
    against the paper's checker must run in the ordinary offline test path.
    """
    noise = vpc._noise_estimate()
    repeat = {r.level: r for r in vpc._repeat().aggregate(5)}

    levels: list[Level] = []
    for i, name in enumerate(("L1", "L2", "L3")):
        r = repeat[name]
        pe, fpe = _pass_counts(i, "D")
        ps, fps = _pass_counts(i, "B")
        # L3's own section reports it as saturated, and a null there is uninformative rather than
        # evidence of absence. The figure has to say so or it invites the opposite reading.
        sat = None
        if name == "L3":
            l3 = vpc._runs("validation-study-p0-L3.json")
            sat = mean(x["B"]["seed_heldout"] for x in l3)
        levels.append(
            Level(
                name=name,
                p0=vpc._detector_metrics(i)["p0"],
                delta_single=vpc._decomposition(i)["delta"],
                per_seed=list(vpc._p0_deltas(i)),
                delta_k5=r.delta,
                se_k5=r.standard_error,
                t_k5=r.t_stat(),
                verdict=r.verdict(),
                pass_evolved=pe,
                false_pass_evolved=fpe,
                pass_seed=ps,
                false_pass_seed=fps,
                n_judged=vpc._arm_confusion(i, "D")["n"],
                saturated_baseline=sat,
            )
        )

    swing = levels[1].delta_single - levels[0].delta_single
    # A difference of two independent deltas, so the denominator carries sigma twice. Getting this
    # wrong would overstate the swing by 41%, and it is the study's load-bearing quantity.
    swing_sigma = swing / (noise.sigma_delta_mean * 2**0.5)

    def _row(label: str, run: str, detector: str, group: str) -> CurveRow:
        return CurveRow(label, run, vpc._curve_seed_mean(run), detector, group)

    curve = [
        _row("intrinsic self-critique", "run3", "near-random", "detector"),
        _row("measurement tool", "run4", "reliable, not perfect", "detector"),
        _row("perfect oracle", "run7", "perfect", "detector"),
        # Run 7 appears in both groups on purpose: it is the hinge. As the best detector it gives
        # the worst score, and it is also the floor the fix-side levers are measured from.
        _row("oracle, one refine round (same run)", "run7", "perfect", "fix"),
        _row("oracle, three refine rounds", "run8", "perfect", "fix"),
        _row("oracle, three rounds, terse feedback", "run9", "perfect", "fix"),
        _row("oracle, three rounds, actionable feedback", "run10", "perfect", "fix"),
    ]
    curve_gaps = {
        "tool_vs_oracle": vpc._curve_gap("run7", "run4"),
        "intrinsic_vs_oracle": vpc._curve_gap("run7", "run3"),
    }

    depth: list[DepthCell] = []
    for mmc, cap in vpc.BUDGET_SWEEP:
        slug = "c2b" if mmc == 28 else f"mmc{mmc}"
        measured = vpc._search_depth(slug, mmc, cap)
        depth.append(
            DepthCell(
                mmc=mmc,
                cap=cap,
                depth=int(measured) if measured is not None else 0,
                delta=vpc._budget_delta(slug),
                accepted=vpc._budget_acceptance(slug),
                # A cell is cap-bound when its MEASURED depth reaches the proposal cap. Deriving
                # this from the registered cycle formula instead would import the very arithmetic
                # error this sweep exists to correct, into the figure that reports the correction.
                cap_bound=measured is not None and int(measured) == cap,
            )
        )

    return FigureData(
        levels=levels,
        sigma=noise.sigma_delta_mean,
        swing=swing,
        swing_in_sigma=swing_sigma,
        curve=curve,
        curve_gaps=curve_gaps,
        depth=depth,
        depth_sigma_within=repeat["L1"].sd_between_replicates,
        notes={
            "sigma_source": f"{noise.n_replicates} released runs re-scoring one fixed genome",
            "t_crit": "t(4), two-sided, 95%",
        },
    )


# --------------------------------------------------------------------------------------------
# Rendering. Everything below consumes `figure_data()` and adds no numbers of its own.
# --------------------------------------------------------------------------------------------

_STYLE = {
    # Computer Modern, the face `article` at 11pt already sets, so a figure reads as part of the
    # page rather than as an import from another document.
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    # matplotlib warns without this when cmr10 is the body face: tick formatters would otherwise
    # be set in a different family from the labels beside them.
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,  # cmr10 has no U+2212; matplotlib would draw a box
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # embed as TrueType so the text stays selectable and searchable
}


def _fmt(x: float, places: int = 3) -> str:
    """Signed fixed-point with a hyphen-minus, since cmr10 cannot set U+2212."""
    return f"{x:+.{places}f}"


def fig_p0(data: FigureData, out: Path) -> Path:
    """The headline: the sign of Delta depends on $p_0$, and the mechanism moves with it."""
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.5, 4.25), sharex=True, gridspec_kw={"height_ratios": [2.05, 1.0]}
    )
    xs = [lv.p0 for lv in data.levels]
    span = (-0.02, 0.735)

    # The noise floor goes down first so every marker is read against it, not against zero alone.
    ax.axhspan(-data.sigma, data.sigma, color="0.90", lw=0, zorder=0)
    ax.axhline(0.0, color=BLACK, lw=0.7, zorder=1)
    for sgn in (-1, 1):
        ax.axhline(sgn * SESOI, color="0.55", lw=0.7, ls=(0, (4, 2.5)), zorder=1)

    # Per-seed points are spread by a FIXED offset, not a random jitter: the figure has to redraw
    # byte-identically, and three points do not need noise to separate.
    for lv in data.levels:
        for k, v in enumerate(lv.per_seed):
            ax.plot(
                lv.p0 + (k - 1) * 0.019,
                v,
                marker="o",
                ms=3.0,
                mfc="white",
                mec="0.55",
                mew=0.7,
                ls="none",
                zorder=3,
            )

    singles = [lv.delta_single for lv in data.levels]
    ax.plot(xs, singles, color="0.35", lw=0.9, ls=":", zorder=2)
    ax.plot(
        xs,
        singles,
        marker="s",
        ms=4.6,
        mfc="white",
        mec="0.20",
        mew=1.0,
        ls="none",
        zorder=4,
        label="single draw (as first published)",
    )
    ax.errorbar(
        xs,
        [lv.delta_k5 for lv in data.levels],
        yerr=[
            [lv.delta_k5 - lv.ci95[0] for lv in data.levels],
            [lv.ci95[1] - lv.delta_k5 for lv in data.levels],
        ],
        fmt="o",
        ms=5.2,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.1,
        capsize=2.6,
        capthick=1.1,
        zorder=5,
        label="five replicates, 95% CI on $t(4)$",
    )

    # The swing, drawn between the two SINGLE draws because that is the pair the registered
    # number was computed from.
    y0, y1 = singles[0], singles[1]
    xa = 0.152
    ax.annotate(
        "",
        xy=(xa, y1),
        xytext=(xa, y0),
        arrowprops={
            "arrowstyle": "<->",
            "color": VERMILLION,
            "lw": 1.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.text(
        xa + 0.013,
        (y0 + y1) / 2,
        f"swing {data.swing:.3f}\n$\\approx {data.swing_in_sigma:.1f}\\sigma$",
        color=VERMILLION,
        fontsize=8,
        va="center",
        linespacing=1.25,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.9},
    )

    for lv in data.levels:
        ax.annotate(
            f"$t={lv.t_k5:.2f}$" + (", $p<0.05$" if "0.05" in lv.verdict else ""),
            xy=(lv.p0, lv.ci95[1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7.6,
            color=BLUE if "0.05" in lv.verdict else "0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.9},
        )

    # Band and bound labels live in the empty mid-field rather than at the axis edge, where they
    # collided with the rightmost level's t-annotation.
    ax.text(0.455, SESOI + 0.003, "$+$SESOI", fontsize=7.4, color="0.45", va="bottom")
    ax.text(0.455, -SESOI - 0.003, "$-$SESOI", fontsize=7.4, color="0.45", va="top")
    ax.text(
        0.545,
        -0.013,
        r"$\pm\sigma(\Delta)$",
        fontsize=7.4,
        color="0.45",
        ha="center",
        va="center",
    )

    sat = data.levels[-1]
    ax.annotate(
        f"saturated: baseline {sat.saturated_baseline:.3f},\nso a null here is uninformative",
        xy=(sat.p0, sat.ci95[0]),
        xytext=(10, -54),
        textcoords="offset points",
        ha="right",
        fontsize=7.4,
        color="0.35",
        linespacing=1.25,
        arrowprops={"arrowstyle": "-", "color": "0.65", "lw": 0.6},
    )

    ax.set_ylabel(r"$\Delta(D-B)$ held-out")
    ax.set_ylim(-0.132, 0.135)
    ax.set_xlim(*span)
    ax.legend(loc="lower left", handletextpad=0.5, borderaxespad=0.4)

    # Lower panel: what the optimizer's critique actually does, which is why the sign moves.
    w = 0.026
    for lv in data.levels:
        for off, (total, wrong, colour) in enumerate(
            (
                (lv.pass_seed, lv.false_pass_seed, VERMILLION),
                (lv.pass_evolved, lv.false_pass_evolved, BLUE),
            )
        ):
            x = lv.p0 + (off - 0.5) * (w + 0.004)
            bx.bar(x, total - wrong, width=w, color=colour, lw=0, zorder=3)
            if wrong:
                bx.bar(
                    x,
                    wrong,
                    width=w,
                    bottom=total - wrong,
                    facecolor="white",
                    edgecolor=colour,
                    hatch="////",
                    lw=0.7,
                    zorder=3,
                )
            bx.annotate(
                str(total),
                xy=(x, total),
                xytext=(0, 5.5),
                textcoords="offset points",
                ha="center",
                fontsize=7.4,
                color=colour,
                zorder=6,
            )

    from matplotlib.patches import Patch

    bx.legend(
        handles=[
            Patch(facecolor=VERMILLION, label="seed rubric (arm B)"),
            Patch(facecolor=BLUE, label="evolved rubric (arm D)"),
            Patch(facecolor="white", edgecolor="0.35", hatch="////", label="of which wrong"),
        ],
        loc="upper left",
        ncol=3,
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.3,
    )
    bx.set_ylabel(f"PASS emitted\nof {data.levels[0].n_judged}", linespacing=1.3)
    bx.set_xlabel(r"$p_0$, base rate of first drafts already satisfying every requirement")
    bx.set_ylim(0, 30)
    bx.set_xticks(xs)
    bx.set_xticklabels([f"{lv.p0:.3f}\n({lv.name})" for lv in data.levels])

    fig.align_ylabels((ax, bx))
    fig.subplots_adjust(hspace=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_detector(data: FigureData, out: Path) -> Path:
    """Detector quality does not order held-out score; the fix side does."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    det = [r for r in data.curve if r.group == "detector"]
    fix = [r for r in data.curve if r.group == "fix"]
    rows = det + fix
    # One blank row between the groups, so the eye reads two comparisons rather than one ladder.
    ypos = [y for y in range(len(rows) + 1) if y != len(det)][: len(rows)]
    x_lo, x_hi = 0.700, 0.925

    colour = {"near-random": ORANGE, "reliable, not perfect": GREEN, "perfect": VERMILLION}
    for r, y in zip(rows, ypos, strict=True):
        ax.plot([x_lo, r.score], [y, y], color="0.88", lw=0.7, zorder=0)
        hinge = r.run == "run7" and r.group == "detector"
        ax.plot(
            r.score,
            y,
            marker="o",
            ms=8.0 if hinge else 6.0,
            mfc=colour[r.detector],
            mec=BLACK if hinge else "none",
            mew=0.9 if hinge else 0,
            ls="none",
            zorder=4,
        )
        ax.annotate(
            f"{r.score:.3f}",
            xy=(r.score, y),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    gap_tool, sig_tool = data.curve_gaps["tool_vs_oracle"]
    gap_intr, sig_intr = data.curve_gaps["intrinsic_vs_oracle"]
    y_intr, y_orc = ypos[0], ypos[2]

    # The two gaps the paper separates: one carries the claim, the other explicitly does not. Both
    # start at the oracle, so both arrows are anchored there and their text sits clear of the rows.
    ax.annotate(
        "",
        xy=(det[1].score, y_orc + 0.34),
        xytext=(det[2].score, y_orc + 0.34),
        arrowprops={"arrowstyle": "<->", "color": BLACK, "lw": 0.9, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        x_lo + 0.052,
        y_orc + 0.52,
        f"{gap_tool:.3f} $= {sig_tool:.2f}\\sigma$: the perfect detector scores BELOW a "
        f"measurement tool",
        ha="left",
        va="top",
        fontsize=7.6,
        linespacing=1.35,
    )
    ax.annotate(
        "",
        xy=(det[0].score, y_intr - 0.34),
        xytext=(det[2].score, y_intr - 0.34),
        arrowprops={"arrowstyle": "<->", "color": "0.45", "lw": 0.9, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        det[0].score + 0.008,
        y_intr - 0.34,
        f"{gap_intr:.3f} $= {sig_intr:.2f}\\sigma$: indistinguishable, not claimed",
        ha="left",
        va="center",
        fontsize=7.6,
        color="0.45",
    )

    ax.text(
        x_lo, y_intr - 1.20, "detector quality varies, one refine round", fontsize=8.4, va="center"
    )
    ax.text(
        x_lo,
        ypos[3] - 0.70,
        "detector fixed perfect, the fix side varies",
        fontsize=8.4,
        va="center",
    )

    # A sigma rule, so a reader can size any gap in the panel without arithmetic. Parked below the
    # last row, the one region with no ink in it.
    xb, yb = x_lo + 0.016, ypos[-1] + 0.78
    ax.plot([xb, xb + data.sigma], [yb, yb], color=BLACK, lw=1.4, solid_capstyle="butt")
    for xt in (xb, xb + data.sigma):
        ax.plot([xt, xt], [yb - 0.11, yb + 0.11], color=BLACK, lw=1.0)
    ax.text(
        xb + data.sigma + 0.006,
        yb,
        rf"$\sigma(\Delta) = {data.sigma:.4f}$",
        ha="left",
        va="center",
        fontsize=7.6,
    )

    ax.set_yticks(ypos)
    ax.set_yticklabels([r.label for r in rows])
    ax.set_xlabel("held-out score, seed genome, thirty tasks shared across runs")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(ypos[-1] + 1.25, y_intr - 1.60)  # inverted: first row on top
    ax.tick_params(axis="y", length=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_depth(data: FigureData, out: Path) -> Path:
    """Delta does not order along search depth, while acceptance does."""
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.5, 3.8), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]}
    )
    cells = data.depth
    sig = data.depth_sigma_within

    # Three cells share the measured depth of 6. They are separate draws of ONE configuration, so
    # they are offset to be visible and banded to say they are one depth --- and deliberately not
    # joined by a line, which would fabricate a slope out of the offset.
    counts: dict[int, int] = {}
    for c in cells:
        counts[c.depth] = counts.get(c.depth, 0) + 1
    seen: dict[int, int] = {}
    xpos: list[float] = []
    for c in cells:
        k = seen.get(c.depth, 0)
        seen[c.depth] = k + 1
        n = counts[c.depth]
        xpos.append(c.depth + (k - (n - 1) / 2) * (0.42 if n > 1 else 0.0))

    tied = [c.depth for c in cells if counts[c.depth] > 1]
    for panel in (ax, bx):
        if tied:
            d = tied[0]
            half = (counts[d] - 1) / 2 * 0.42 + 0.30
            panel.axvspan(d - half, d + half, color="0.945", lw=0, zorder=0)

    ax.axhspan(-sig, sig, color="0.90", lw=0, zorder=1)
    ax.axhline(0.0, color=BLACK, lw=0.7, zorder=2)
    for sgn in (-1, 1):
        ax.axhline(sgn * SESOI, color="0.55", lw=0.7, ls=(0, (4, 2.5)), zorder=2)

    for c, x in zip(cells, xpos, strict=True):
        ax.plot(
            x,
            c.delta,
            marker="o",
            ms=5.4,
            mfc=BLUE if c.cap_bound else "white",
            mec=BLUE,
            mew=1.2,
            ls="none",
            zorder=4,
        )
        ax.annotate(
            f"{c.mmc}",
            xy=(x, c.delta),
            xytext=(0, -12 if c.delta > 0 else 8),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color="0.40",
        )

    deep = cells[-1]
    ax.annotate(
        f"deepest cell {_fmt(deep.delta, 4)} $= {abs(deep.delta) / sig:.2f}\\sigma$\n"
        f"registered rule needed $>+{SESOI:.2f}$ and $>2\\sigma$",
        xy=(xpos[-1], deep.delta),
        xytext=(-14, -40),
        textcoords="offset points",
        ha="right",
        fontsize=7.6,
        linespacing=1.35,
        arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.7},
    )
    ax.text(
        1.15,
        -0.030,
        rf"$\pm\sigma = {sig:.3f}$",
        va="center",
        ha="left",
        fontsize=7.4,
        color="0.45",
    )
    ax.text(12.85, SESOI + 0.002, "$+$SESOI", va="bottom", ha="right", fontsize=7.4, color="0.45")

    ax.set_ylabel(r"$\Delta(D-B)$ held-out")
    ax.set_ylim(-0.076, 0.070)
    ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="none",
                mfc="white",
                mec=BLUE,
                mew=1.2,
                ms=5.4,
                label="budget-bound",
            ),
            plt.Line2D(
                [], [], marker="o", ls="none", mfc=BLUE, mec=BLUE, ms=5.4, label="cap-bound"
            ),
        ],
        loc="lower left",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.4,
    )

    bx.plot(xpos, [c.accepted for c in cells], marker="D", ms=4.6, color=GREEN, ls="none", zorder=3)
    for c, x in zip(cells, xpos, strict=True):
        bx.annotate(
            f"{c.accepted}/6",
            xy=(x, c.accepted),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7.4,
            color=GREEN,
        )
    if tied:
        bx.text(
            tied[0] + 0.62,
            3.46,
            "one measured depth,\nthree independent draws",
            ha="left",
            va="bottom",
            fontsize=7.4,
            color="0.40",
            linespacing=1.25,
        )

    bx.set_ylabel("arm-seed pairs\nthat accepted", linespacing=1.3)
    bx.set_xlabel("measured search depth (GEPA iterations, counted by replaying each cassette)")
    bx.set_ylim(3.3, 7.0)
    bx.set_yticks([4, 5, 6])
    bx.set_xlim(1.0, 13.0)
    bx.set_xticks(sorted({c.depth for c in cells}))

    fig.align_ylabels((ax, bx))
    fig.subplots_adjust(hspace=0.10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def render(outdir: Path = OUTDIR) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)
    # No LaTeX text markup anywhere in the labels. usetex needs dvipng, which is not installed
    # here, and without it \textbf or \emph would be printed literally into the figure. Math is
    # set with mathtext (fontset "cm", so it matches cmr10) and emphasis is done with colour and
    # size instead of weight, because cmr10 has no bold companion to fall back on.
    data = figure_data()
    made = []
    for fn, name in (
        (fig_p0, "fig-p0.pdf"),
        (fig_detector, "fig-detector.pdf"),
        (fig_depth, "fig-depth.pdf"),
    ):
        made.append(fn(data, outdir / name))
    return made


def main() -> int:  # pragma: no cover - CLI convenience
    for p in render():
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
