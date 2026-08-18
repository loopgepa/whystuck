r"""A figure and the table beside it must print the same number for the same quantity.

Both surfaces are already tied to the artifacts: `tests/test_figures.py` requires every plotted
value to be one the claim checker derives, and `tests/test_paper_table_coverage.py` requires every
table cell to be checked or exempt. Agreement between the two should follow, and *should follow* was
the whole coverage: nothing compared them. A figure that plotted the split sweep's
$n_{\mathrm{train}}{=}16$ delta while the table beside it printed the $n{=}24$ row would satisfy
both guards, because both numbers are derived and both are checked.

Cells are read per cell rather than per line. Reading the line finds `3.6` inside
`\texttt{qwen3.6-27b}` before it finds the row's first real column, which is how the one-off version
of this check reported two disagreements that did not exist.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import make_figures as mf  # noqa: E402
import paper_sources  # noqa: E402

#: Below this the sweep has stopped finding rows and its silence would read as agreement.
_MIN_COMPARISONS = 40

_NUMBER = re.compile(r"-?\d+\.\d+|(?<![\d.])-?\d+(?![\d.])")


def _row(tex: str, anchor: str, starts: str) -> list[str] | None:
    """The cells of the row starting with `starts`, inside the table whose header holds `anchor`."""
    at = tex.find(anchor)
    if at == -1:
        return None
    chunk = tex[at : at + 3000]
    for line in chunk.splitlines():
        if line.strip().startswith(starts):
            return [c.strip() for c in line.split("&")]
    return None


def _cell_number(cell: str) -> float | None:
    """The last number in one cell, so a model name in the same cell cannot be read as data."""
    found = _NUMBER.findall(cell.replace("{", " ").replace("}", " "))
    return float(found[-1]) if found else None


def _pairs() -> list[tuple[str, float, float | None, float]]:
    """(label, figure value, table value, tolerance) for every quantity drawn and tabulated."""
    data = mf.figure_data()
    p1 = paper_sources.paper_tex("p1-conditional-lever")
    p2 = paper_sources.paper_tex("p2-repair-not-detection")
    out: list[tuple[str, float, float | None, float]] = []

    # fig-effects against tab:splitresults and tab:repeat. The figure plots sigma multiples, so the
    # comparison multiplies back by the same sigma it divided by.
    split_header = r"seed-genome & $\dd(D-B)$"
    for n_train, starts in (("4", "4  &"), ("8", "8  &"), ("16", "16 &"), ("24", "24 &")):
        effect = next(e for e in data.effects if e.label == f"split sweep, $n={n_train}$")
        cells = _row(p1, split_header, starts)
        out.append(
            (
                f"fig-effects split n={n_train} vs tab:splitresults",
                effect.in_sigma * data.sigma,
                _cell_number(cells[3]) if cells else None,
                5e-4,
            )
        )
    repeat_header = r"level & single draw & $\dd$ ($K{=}5$)"
    for i, level in enumerate(("L_1", "L_2", "L_3")):
        effect = data.effects[i]
        cells = _row(p1, repeat_header, f"${level}$ &")
        out.append(
            (
                f"fig-effects {level} five-replicate vs tab:repeat",
                effect.in_sigma * data.sigma,
                _cell_number(cells[2]) if cells else None,
                5e-4,
            )
        )
        assert effect.single_in_sigma is not None
        out.append(
            (
                f"fig-effects {level} single draw vs tab:repeat",
                effect.single_in_sigma * data.sigma,
                _cell_number(cells[1]) if cells else None,
                5e-4,
            )
        )

    # fig-residual against tab:residual, all three conditions of every constraint.
    residual_header = "Constraint (final-draft satisfaction) & run 3"
    for bar in data.residual:
        cells = _row(
            p2, residual_header, rf"\texttt{{{bar.constraint.replace('_', chr(92) + '_')}}}"
        )
        for j, run in enumerate(("run3", "run8", "run10"), start=1):
            out.append(
                (
                    f"fig-residual {bar.constraint} {run} vs tab:residual",
                    bar.rates[run],
                    _cell_number(cells[j]) if cells else None,
                    5e-3,
                )
            )

    # fig-worker against the second-worker table.
    worker_header = r"critique & held-out & $\dd$ vs.\ intrinsic"
    anchors = {
        "smaller critic (8B)": "same-lineage smaller critic (8B)",
        "another lineage (qwen3.6-27b)": "cross-lineage critic",
        "perfect verdict only": r"perfect oracle, \texttt{lo}",
        "perfect verdict + which constraint": r"perfect oracle, \texttt{mid}",
        "perfect verdict + measured edits": r"perfect oracle, \texttt{hi}",
    }
    for row in data.worker:
        cells = _row(p2, worker_header, anchors[row.label])
        out.append(
            (
                f"fig-worker {row.label} held-out vs table",
                row.held_out,
                _cell_number(cells[1]) if cells else None,
                5e-5,
            )
        )
        out.append(
            (
                f"fig-worker {row.label} delta vs table",
                row.delta,
                _cell_number(cells[2]) if cells else None,
                5e-5,
            )
        )
    return out


def test_every_figure_agrees_with_the_table_beside_it():
    pairs = _pairs()
    missing = [label for label, _fig, table, _tol in pairs if table is None]
    assert not missing, (
        f"no table row found for {missing}; the row anchors have drifted from the tables, and a "
        "sweep that finds no rows reports agreement"
    )
    bad = [
        f"{label}: figure {fig:.5f}, table {table:.5f}"
        for label, fig, table, tol in pairs
        if table is not None and abs(abs(fig) - abs(table)) > tol
    ]
    assert not bad, "a figure and its table disagree:\n  " + "\n  ".join(bad)


def test_the_two_sweeps_print_the_same_number_for_the_cell_they_share():
    """The four-arm re-run is a row of both sweeps, so two tables describe one run.

    `tab:splitresults`' $n_{\\mathrm{train}}{=}8$ row and `tab:budgetresult`'s
    $\\mathrm{mmc}{=}28$ row are the same purchase, entered twice, which the second table's caption
    says. Each cell is checked against the artifact on its own, so they agree transitively; nothing
    checked that they still describe the same run, and re-buying either row is exactly when they
    would stop.
    """
    p1 = paper_sources.paper_tex("p1-conditional-lever")
    split = _row(p1, r"seed-genome & $\dd(D-B)$", "8  &")
    budget = _row(p1, r"\texttt{mmc} & cap & priced depth", "28  &")
    assert split and budget, "one of the two sweep rows was not found; the anchors drifted"
    assert _cell_number(split[3]) == _cell_number(budget[5]), (
        f"the shared cell disagrees between the sweeps: split prints {split[3]}, "
        f"budget prints {budget[5]}"
    )
    data = mf.figure_data()
    from_artifact = (
        next(e for e in data.effects if e.label == "split sweep, $n=8$").in_sigma * data.sigma
    )
    assert abs(abs(from_artifact) - abs(_cell_number(split[3]) or 0)) < 5e-4


def test_the_comparison_is_not_vacuous():
    assert len(_pairs()) >= _MIN_COMPARISONS, (
        f"only {len(_pairs())} figure/table comparisons; this sweep is meant to cover every "
        "quantity that appears in both a figure and a table"
    )
