"""The series' six figures, with every plotted number derived from a released artifact.

WHY THIS EXISTS. Nineteen tables carry the three papers, and several of their central claims are
*shapes* a table cannot show: the sign of $\\dd$ changing across $p_0$, detector quality failing to
order held-out performance, $\\dd$ failing to order along search depth, every effect in the study
against the noise it sits in, a missing diagnostic field standing where a measured zero is assumed,
and a repair signal moving a second worker where two detector substitutions do not. A reader had to
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
class ResidualBar:
    """One constraint's satisfaction rate under three conditions of the ceiling experiment."""

    constraint: str
    rates: dict[str, float]


@dataclass(frozen=True)
class WorkerRow:
    """One condition of the second-worker line, paired against that worker's own critique."""

    label: str
    group: str
    held_out: float
    delta: float
    se: float
    t: float


@dataclass(frozen=True)
class EffectRow:
    """One effect this study reports, expressed in units of its own measurement noise.

    `lo`/`hi` are present only where the quantity was measured more than once: the three $p_0$
    levels carry a five-replicate interval, and every sweep cell is a single draw with no interval
    of its own. Keeping the distinction in the type is what stops the figure from drawing a
    confidence interval around a number that has none.
    """

    label: str
    group: str
    in_sigma: float
    replicated: bool
    single_in_sigma: float | None = None
    lo: float | None = None
    hi: float | None = None


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
    #: Paper III's fifteen pre-correction contrast runs, read three ways. The census lives in the
    #: checker so the figure and the counts in the prose cannot disagree about which runs they mean.
    precorrection: tuple[vpc.PrecorrectionRun, ...] = ()
    precorrection_moved: dict[str, int] = field(default_factory=dict)
    #: Every instruction-following effect in Paper I, in units of sigma, for `fig-effects`.
    effects: tuple[EffectRow, ...] = ()
    sesoi_in_sigma: float = 0.0
    #: Paper II's per-constraint residual and its second-worker line.
    residual: tuple[ResidualBar, ...] = ()
    worker: tuple[WorkerRow, ...] = ()
    worker_seeds: int = 0
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


def _effect_rows(
    levels: list[Level], depth: list[DepthCell], sigma: float, swing_in_sigma: float
) -> tuple[EffectRow, ...]:
    """Every instruction-following effect Paper I reports, divided by the study's own sigma.

    Code-domain runs are left out on purpose. Sigma was measured on multi-hard, and Paper III says
    plainly that its transfer to a benchmark that executes model code is an assumption rather than a
    measurement, so putting the actuator run on this axis would import that assumption into a figure
    whose whole subject is what the noise permits.

    The swing is the one row whose denominator carries sigma twice, since it is a difference of
    two independent deltas. It arrives already scaled rather than recomputed here, so the figure and
    `fig-p0`'s annotation cannot disagree about the study's load-bearing quantity.
    """
    rows = [
        EffectRow(
            label=f"$p_0$ sweep, {lv.name}",
            group="re-scored five times",
            in_sigma=lv.delta_k5 / sigma,
            replicated=True,
            single_in_sigma=lv.delta_single / sigma,
            lo=lv.ci95[0] / sigma,
            hi=lv.ci95[1] / sigma,
        )
        for lv in levels
    ]
    for n_train, slug in (("4", "split4"), ("8", "c2b"), ("16", "split16"), ("24", "split24")):
        rows.append(
            EffectRow(
                label=f"split sweep, $n={n_train}$",
                group="single draw",
                in_sigma=vpc._split_row(slug).delta / sigma,
                replicated=False,
            )
        )
    for cell in depth:
        rows.append(
            EffectRow(
                label=f"budget sweep, mmc${{=}}{cell.mmc}$",
                group="single draw",
                in_sigma=cell.delta / sigma,
                replicated=False,
            )
        )
    rows.append(
        EffectRow(
            label="the registered swing, $L_2 - L_1$",
            group="difference of two draws",
            in_sigma=swing_in_sigma,
            replicated=False,
        )
    )
    return tuple(rows)


#: The three conditions `tab:residual` reports, in the order the ceiling argument walks them.
_RESIDUAL_CONDITIONS = (
    ("run3", "intrinsic critique"),
    ("run8", "perfect detector, three refine rounds"),
    ("run10", "perfect detector, measured repair edits"),
)

#: The second-worker line, paired against the intrinsic baseline. Two detector substitutions, then
#: three repair signals over a perfect detector: the contrast the figure exists to draw.
_WORKER_ROWS = (
    ("smaller critic (8B)", "detector swapped", "det-8b"),
    ("another lineage (qwen3.6-27b)", "detector swapped", "det-qwen"),
    ("perfect verdict only", "repair signal varied", "det-oracle-lo"),
    ("perfect verdict + which constraint", "repair signal varied", "det-oracle"),
    ("perfect verdict + measured edits", "repair signal varied", "det-oracle-hi"),
)

#: The intrinsic baseline of the second-worker line: the same worker's own critique.
_WORKER_BASELINE = "p0cal-L1"


def _residual_bars() -> tuple[ResidualBar, ...]:
    """Per-constraint satisfaction under three conditions, sorted by the hardest condition.

    Sorted rather than listed so the ordering is a property of the data. The claim the figure
    carries is that one constraint sits far below the rest and stays there as the fix side improves,
    and an ordering typed in by hand could survive the day that stopped being true.
    """
    rates = vpc._residual_rates()
    names = sorted(rates["run3"], key=lambda c: rates["run10"][c])
    return tuple(
        ResidualBar(constraint=c, rates={run: rates[run][c] for run, _ in _RESIDUAL_CONDITIONS})
        for c in names
    )


def _worker_rows() -> tuple[WorkerRow, ...]:
    rows = []
    for label, group, slug in _WORKER_ROWS:
        delta, se, t = vpc._groq_paired(_WORKER_BASELINE, slug)
        rows.append(
            WorkerRow(
                label=label,
                group=group,
                held_out=mean(vpc._groq_cell(slug)),
                delta=delta,
                se=se,
                t=t,
            )
        )
    return tuple(rows)


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
        precorrection=vpc._precorrection_census(),
        precorrection_moved=vpc._precorrection_acceptance(),
        effects=_effect_rows(levels, depth, noise.sigma_delta_mean, swing_sigma),
        sesoi_in_sigma=SESOI / noise.sigma_delta_mean,
        residual=_residual_bars(),
        worker=_worker_rows(),
        worker_seeds=len(vpc._groq_cell(_WORKER_BASELINE)),
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


#: Written into every figure PDF in place of matplotlib's defaults. `CreationDate: None` is the
#: load-bearing entry: with a timestamp, re-rendering an unchanged figure produces a different file,
#: so `git status` reports four modified figures whenever any one of them is regenerated and a real
#: change is indistinguishable from a rebuild. Without it these files are byte-identical unless a
#: plotted number moved, which is the only reason to look at them in a diff.
_PDF_METADATA = {
    "Creator": "scripts/make_figures.py",
    "Producer": "matplotlib",
    "CreationDate": None,
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
    fig.savefig(out, metadata=_PDF_METADATA)
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
    fig.savefig(out, metadata=_PDF_METADATA)
    plt.close(fig)
    return out


def fig_residual(data: FigureData, out: Path) -> Path:
    """The ceiling is one constraint, and the fix side moves it slowly rather than not at all."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    colour = {"run3": ORANGE, "run8": SKY, "run10": BLUE}
    height = 0.24
    for row_i, bar in enumerate(data.residual):
        for cond_i, (run, _label) in enumerate(_RESIDUAL_CONDITIONS):
            y = row_i + (cond_i - 1) * height
            ax.barh(
                y,
                bar.rates[run],
                height=height * 0.92,
                color=colour[run],
                edgecolor="none",
                zorder=3,
            )
            if row_i == 0:
                ax.annotate(
                    f"{bar.rates[run]:.2f}",
                    xy=(bar.rates[run], y),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7.0,
                    color=colour[run],
                )
    ax.axvline(1.0, color=BLACK, lw=0.8, zorder=4)
    ax.text(
        0.988,
        1.05,
        "every held-out task satisfies it",
        rotation=90,
        ha="right",
        va="center",
        fontsize=7.0,
    )

    handles = [
        plt.Line2D([], [], color=colour[run], lw=5, label=label)
        for run, label in _RESIDUAL_CONDITIONS
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        fontsize=7.4,
        handlelength=1.4,
        columnspacing=1.6,
    )

    ax.set_yticks(range(len(data.residual)))
    # Set through mathtext's \mathtt, which is the one path that gives these names an underscore:
    # cmr10 has no ASCII underscore glyph, so a plain string prints `exact.word.count` with a
    # raised dot in place of each one. The names match the paper's \texttt{} rendering of them.
    ax.set_yticklabels(
        [f"$\\mathtt{{{b.constraint.replace('_', chr(92) + '_')}}}$" for b in data.residual],
        fontsize=7.8,
    )
    ax.set_xlabel("share of held-out tasks satisfying that constraint")
    ax.set_xlim(0.0, 1.06)
    ax.set_ylim(len(data.residual) - 0.5, -0.6)  # inverted: hardest constraint on top
    ax.tick_params(axis="y", length=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=_PDF_METADATA)
    plt.close(fig)
    return out


def fig_worker(data: FigureData, out: Path) -> Path:
    """On a second worker at twenty seeds: swapping the detector buys nothing, the repair does."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    rows = list(data.worker)
    ys: list[float] = []
    y = 0.0
    for i, r in enumerate(rows):
        if i and r.group != rows[i - 1].group:
            y += 0.8
        ys.append(y)
        y += 1.0

    ax.axvline(0.0, color=BLACK, lw=0.8, zorder=2)
    for r, yy in zip(rows, ys, strict=True):
        swapped = r.group == "detector swapped"
        colour = ORANGE if swapped else BLUE
        ax.plot(
            [r.delta - r.se, r.delta + r.se],
            [yy, yy],
            color=colour,
            lw=1.3,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(r.delta, yy, marker="o", ms=6.0, mfc=colour, mec=colour, ls="none", zorder=4)
        ax.annotate(
            f"{r.delta:+.4f}   $t = {r.t:+.2f}$",
            xy=(r.delta, yy),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            color=colour,
        )

    first_varied = next(i for i, r in enumerate(rows) if r.group == "repair signal varied")
    ax.text(
        -0.012,
        ys[0] - 1.0,
        "detector substituted, repair signal unchanged",
        fontsize=7.8,
        va="center",
        color=ORANGE,
    )
    ax.text(
        -0.012,
        ys[first_varied] - 0.85,
        "detector held perfect, repair signal varied",
        fontsize=7.8,
        va="center",
        color=BLUE,
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in rows], fontsize=7.8)
    ax.set_xlabel(
        f"paired change in held-out score against the worker's own critique, "
        f"{data.worker_seeds} seeds, bars are $\\pm$1 s.e."
    )
    ax.set_xlim(-0.022, 0.10)
    ax.set_ylim(ys[-1] + 0.7, ys[0] - 1.7)  # inverted: first row on top
    ax.tick_params(axis="y", length=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=_PDF_METADATA)
    plt.close(fig)
    return out


def fig_effects(data: FigureData, out: Path) -> Path:
    """Every effect in the instruction-following study, on one axis: the noise it sits in.

    The point of the panel is the comparison a table cannot make. Read down the column and the
    study has one quantity that clears two sigma with an interval behind it, one sweep cell that
    prints a larger number from a single draw, and a registered swing that beats both.
    """
    import matplotlib.pyplot as plt

    rows = list(data.effects)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    # Group boundaries get one blank slot, so the eye reads three sets rather than one long ladder.
    ys: list[float] = []
    y = 0.0
    for i, r in enumerate(rows):
        if i and r.group != rows[i - 1].group:
            y += 0.85
        ys.append(y)
        y += 1.0

    for band, shade in ((1.0, "0.90"), (2.0, "0.955")):
        ax.axvspan(-band, band, color=shade, zorder=0)
    ax.axvline(0.0, color=BLACK, lw=0.8, zorder=1)
    for x in (-data.sesoi_in_sigma, data.sesoi_in_sigma):
        ax.axvline(x, color=GREEN, lw=0.9, ls=(0, (4, 2)), zorder=2)

    for r, yy in zip(rows, ys, strict=True):
        if r.lo is not None and r.hi is not None:
            ax.plot([r.lo, r.hi], [yy, yy], color=BLUE, lw=1.3, solid_capstyle="butt", zorder=3)
            for xt in (r.lo, r.hi):
                ax.plot([xt, xt], [yy - 0.16, yy + 0.16], color=BLUE, lw=1.0, zorder=3)
        if r.single_in_sigma is not None:
            ax.plot(
                r.single_in_sigma,
                yy,
                marker="o",
                ms=5.0,
                mfc="white",
                mec="0.45",
                mew=0.9,
                ls="none",
                zorder=4,
            )
        ax.plot(
            r.in_sigma,
            yy,
            marker="s" if r.replicated else "o",
            ms=6.0 if r.replicated else 5.6,
            mfc=BLUE if r.replicated else VERMILLION,
            mec=BLUE if r.replicated else VERMILLION,
            ls="none",
            zorder=5,
        )
        ax.annotate(
            f"{r.in_sigma:+.2f}",
            xy=(r.in_sigma, yy),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.0,
            color=BLUE if r.replicated else VERMILLION,
        )

    ax.text(
        data.sesoi_in_sigma + 0.08,
        ys[0] - 1.35,
        f"registered SESOI, $\\pm{SESOI:g}$ held-out $= {data.sesoi_in_sigma:.2f}\\sigma$",
        color=GREEN,
        fontsize=7.2,
        va="center",
    )
    ax.text(
        -3.75,
        ys[-1] + 1.15,
        "bands: $\\pm1\\sigma$ and $\\pm2\\sigma$ of the study's own measurement noise, "
        f"$\\sigma(\\Delta) = {data.sigma:.4f}$",
        fontsize=7.2,
        ha="left",
        va="center",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in rows], fontsize=7.8)
    ax.set_xlabel("effect in units of $\\sigma(\\Delta)$, the noise on a thirty-task mean")
    ax.set_xlim(-3.9, 4.3)
    ax.set_ylim(ys[-1] + 1.5, ys[0] - 1.9)  # inverted: first row on top
    ax.tick_params(axis="y", length=0)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=_PDF_METADATA)
    plt.close(fig)
    return out


def fig_noop(data: FigureData, out: Path) -> Path:
    """Three readings of the same fifteen runs, and only one of them says anything.

    Read a row left to right: the diagnostic we propose has nothing to report, the scores show a
    proposal was accepted, and the contrast column is a zero that means "the arms were one program"
    rather than "the treatment did nothing". A reader who takes the absent field for a zero
    concludes the search never accepted anything, and the middle columns refute it fifteen times.
    """
    import matplotlib.pyplot as plt

    runs = list(data.precorrection)
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(6.5, 4.3), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.05]}
    )
    ys = list(range(len(runs)))

    # Left: what each source says, one column per (source, arm).
    cols = [
        (0.0, "genes_changed, B", lambda r: r.field_present),
        (1.0, "genes_changed, D", lambda r: r.field_present),
        (2.4, "score moved, B", lambda r: r.moved_b),
        (3.4, "score moved, D", lambda r: r.moved_d),
    ]
    for x, label, read in cols:
        field_col = x < 2.0
        for r, y in zip(runs, ys, strict=True):
            on = read(r)
            if field_col and not r.field_present:
                # The field is absent, not zero. Drawn as an empty slot with a slash so it cannot be
                # mistaken for the "unchanged" marker, which is the mistake the paper documents.
                ax.plot(x, y, marker="$/$", ms=7.5, color="0.62", ls="none", zorder=3)
                continue
            ax.plot(
                x,
                y,
                marker="o",
                ms=6.4,
                mfc=BLUE if on else "white",
                mec=BLUE,
                mew=0.9,
                ls="none",
                zorder=3,
            )
        # Below the two summary lines and above the first row. The y axis is inverted (first run on
        # top), so a smaller y is higher on the page and `va` reads the opposite way round.
        ax.text(x - 0.1, -1.15, label, rotation=32, ha="left", va="bottom", fontsize=7.2)

    ax.axvline(1.7, color="0.85", lw=0.8)
    n = len(runs)
    absent = sum(1 for r in runs if not r.field_present)
    ax.text(
        0.5,
        -4.15,
        f"absent in {absent}/{n} runs:\nthe field postdates them",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="0.35",
        linespacing=1.3,
    )
    ax.text(
        2.9,
        -4.15,
        f"a proposal was accepted\nin {data.precorrection_moved['either']}/{n}",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color=BLUE,
        linespacing=1.3,
    )
    ax.set_xlim(-0.5, 4.2)
    ax.set_xticks([])
    ax.set_yticks(ys)
    ax.set_yticklabels([r.name for r in runs], fontsize=7.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)

    # Right: the contrast column those runs published, against the noise it sits in.
    bx.axvspan(-data.sigma, data.sigma, color="0.92", zorder=0)
    bx.axvline(0.0, color=BLACK, lw=0.7, zorder=1)
    for r, y in zip(runs, ys, strict=True):
        identical = r.identical
        bx.plot(
            r.mean_delta,
            y,
            marker="o",
            ms=6.4 if not identical else 5.6,
            mfc=VERMILLION if not identical else "white",
            mec=VERMILLION if not identical else "0.45",
            mew=0.9,
            ls="none",
            zorder=3,
        )
        if not identical and abs(r.mean_delta) > 0.001:
            # Positive deltas sit inside the noise band, where a label above the marker would land
            # on the band's edge, so those go beside the marker instead.
            right = r.mean_delta > 0
            bx.annotate(
                _fmt(r.mean_delta, 3),
                xy=(r.mean_delta, y),
                xytext=(8, 0) if right else (0, 6),
                textcoords="offset points",
                ha="left" if right else "center",
                va="center" if right else "bottom",
                fontsize=7.2,
                color=VERMILLION,
            )
    ident = sum(r.identical for r in runs)
    bx.text(
        -0.043,
        -4.15,
        f"{ident}/{n} scored identically on every seed;\n"
        f"the band is $\\pm\\sigma(\\Delta) = {data.sigma:.4f}$",
        ha="center",
        va="bottom",
        fontsize=7.4,
        linespacing=1.3,
    )
    bx.set_xlabel("$\\Delta$(D $-$ B) published for these runs, and not a measurement")
    bx.set_xlim(-0.13, 0.075)
    bx.set_ylim(len(runs) - 0.3, -5.0)  # inverted: first run on top, headroom for the labels
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)

    fig.subplots_adjust(wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata=_PDF_METADATA)
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
    fig.savefig(out, metadata=_PDF_METADATA)
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
        (fig_noop, "fig-noop.pdf"),
        (fig_effects, "fig-effects.pdf"),
        (fig_residual, "fig-residual.pdf"),
        (fig_worker, "fig-worker.pdf"),
    ):
        made.append(fn(data, outdir / name))
    return made


def main() -> int:  # pragma: no cover - CLI convenience
    for p in render():
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
