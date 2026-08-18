r"""The five surfaces this study publishes on must not drift apart.

Code, released runs, prose, tables and figures each have their own guard, and each of those guards
looks at one surface. The failures this series keeps finding live *between* surfaces: a table no
prose points at, a run on disk that the registry never lists, a figure that plots one run while the
table beside it prints another, a number one paper attributes to a companion that the companion
never printed. `tests/test_figure_table_agreement.py` covers the figure/table pair; this covers the
rest, and each check asserts its own yield so that a sweep which stops finding anything fails.
"""

from __future__ import annotations

import glob
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import paper_sources  # noqa: E402
import verify_reproducibility as vr  # noqa: E402

#: Layout arguments rather than claims: `p{0.27\textwidth}` carries digits meaning column width.
_LAYOUT = re.compile(r"p\{[\d.]+\\textwidth\}|\\setlength\{[^}]*\}\{[^}]*\}|\{[\d.]+\\textwidth\}")


def _prose(slug: str) -> str:
    tex = paper_sources.paper_tex(slug)
    body = "\n".join(line for line in tex.splitlines() if not line.lstrip().startswith("%"))
    return re.sub(r"\s+", " ", _LAYOUT.sub(" ", body))


def test_every_released_run_has_a_replay_status_and_a_file():
    """A run JSON on disk that the registry never lists backs numbers from outside the guarantee."""
    on_disk = {os.path.basename(p) for p in glob.glob("docs/validation-study*.json")}
    registered = {os.path.basename(r[1]) for r in vr.RUNS}
    registered |= {os.path.basename(k) for k in vr.UNREPLAYABLE}
    assert len(on_disk) >= 40, f"only {len(on_disk)} run files found; the release moved"
    assert not on_disk - registered, (
        f"run files with no replay status: {sorted(on_disk - registered)}"
    )
    assert not registered - on_disk, (
        f"registry entries with no file: {sorted(registered - on_disk)}"
    )


def test_every_script_the_papers_name_exists():
    named = set(re.findall(r"scripts/([\w\\_]+)\.py", paper_sources.all_tex()))
    assert len(named) >= 8, f"only {len(named)} scripts named in the papers; the pattern drifted"
    missing = [
        n for n in named if not os.path.isfile(f"scripts/{n.replace(chr(92) + '_', '_')}.py")
    ]
    assert not missing, f"the papers name scripts that do not exist: {sorted(missing)}"


def test_every_table_and_figure_is_referenced_and_every_reference_resolves():
    """A float nobody points at is reached by page proximity, which the PDF decides."""
    seen = 0
    for slug in paper_sources.PAPERS:
        body = "\n".join(
            line
            for line in paper_sources.paper_tex(slug).splitlines()
            if not line.lstrip().startswith("%")
        )
        labels = set(re.findall(r"\\label\{([^}]+)\}", body))
        refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", body))
        floats = {x for x in labels if x.startswith(("tab:", "fig:"))}
        seen += len(floats)
        assert not refs - labels, f"{slug}: references with no label: {sorted(refs - labels)}"
        assert not floats - refs, (
            f"{slug}: tables/figures no prose points at: {sorted(floats - refs)}"
        )
    assert seen >= 20, f"only {seen} floats found across the series; the pattern drifted"


def test_every_figure_file_a_paper_includes_exists():
    seen = 0
    for slug in paper_sources.PAPERS:
        for name in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}]+)\}", _prose(slug)
        ):
            seen += 1
            assert os.path.isfile(f"docs/papers/{slug}/figures/{name}"), (
                f"{slug} includes figures/{name}, which is not there"
            )
    assert seen >= 7, f"only {seen} figure inclusions found; the pattern drifted"


def test_a_number_attributed_to_a_companion_paper_appears_in_that_paper():
    r"""Paper I quoting Paper II's $80$ judged drafts is a claim about a document, checkable there.

    Only *parenthetical* attributions count: `... on none of $80$ held-out drafts (Paper~II)`. A
    sentence can name a companion for a reason unrelated to its numbers, and the first version of
    this check read every number in such a sentence as attributed, which flagged Paper III's own
    contamination history four times because the sentence explaining why it moved names Paper I.

    Numbers are matched on a boundary rather than as substrings. `81` occurs inside `0.813`, so a
    plain containment test finds nearly any short numeral in nearly any paper: mutating $80$ to $81$
    passed until this was fixed.
    """
    numerals = re.compile(r"\d+@\d+|\b\d{2,4}\b")
    by_numeral = {v: k for k, v in paper_sources.PAPER_NUMERALS.items()}
    nicknames = {
        "conditional lever": "p1-conditional-lever",
        "loop's ceiling": "p2-repair-not-detection",
        "ceiling paper": "p2-repair-not-detection",
        "silent no-op": "p3-silent-no-op",
    }
    texts = {slug: _prose(slug) for slug in paper_sources.PAPERS}
    nicked = "|".join(re.escape(k) for k in nicknames)
    citation = re.compile(rf"\((?:companion )?(?:Paper~(I{{1,3}})|({nicked}))")
    attributions = compared = 0
    bad = []
    for slug, text in texts.items():
        masked = re.sub(r"(\d)\.(\d)", r"\1@\2", text)
        for match in citation.finditer(masked):
            target = by_numeral.get(match.group(1) or "") or nicknames.get(match.group(2) or "")
            if target is None or target == slug:
                continue
            # The claim being credited: from the previous full stop up to the citation.
            clause_start = masked.rfind(". ", 0, match.start()) + 1
            clause = masked[clause_start : match.start()]
            found = set(numerals.findall(clause))
            if not found:
                continue
            attributions += 1
            for token in found:
                plain = token.replace("@", ".")
                compared += 1
                if not re.search(rf"(?<![\d.]){re.escape(plain)}(?![\d.])", texts[target]):
                    bad.append(
                        f"{paper_sources.PAPER_NUMERALS[slug]} credits "
                        f"{paper_sources.PAPER_NUMERALS[target]} with {plain}, which it never "
                        f"prints: {clause.replace('@', '.').strip()[:120]}"
                    )
    assert attributions >= 8, (
        f"only {attributions} parenthetical cross-paper attributions found; pattern drifted"
    )
    assert compared >= 10, f"only {compared} attributed numbers compared; pattern drifted"
    assert not bad, (
        "numbers credited to a companion paper that it does not print:\n  " + "\n  ".join(bad)
    )


def test_every_configuration_the_papers_state_is_a_configuration_that_was_run():
    r"""`$\mathrm{mmc}{=}80$` and `$n_{\mathrm{train}}{=}16$` name a run's budget and split size.

    The prose coverage guard excludes bare integers on purpose, since LaTeX plumbing swamps them, so
    a configuration integer is the one class of number in these papers that nothing reads. Only the
    brace form is matched here: `\mathrm{mmc} = 8 + 6 \times 11 = 74` is arithmetic from the
    registered depth formula, and reading it as a configuration reports two defects that do not
    exist, which the one-off version of this check did.
    """
    import glob
    import json

    budgets, splits = set(), set()
    for path in glob.glob("docs/validation-study*.json"):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)["meta"]
        if meta.get("max_metric_calls") is not None:
            budgets.add(int(meta["max_metric_calls"]))
        if meta.get("n_train") is not None:
            splits.add(int(meta["n_train"]))
    assert len(budgets) >= 8 and len(splits) >= 5, "the released configuration set looks empty"

    stated_budgets, stated_splits = set(), set()
    for slug in paper_sources.PAPERS:
        text = _prose(slug)
        stated_budgets |= {int(x) for x in re.findall(r"\\mathrm\{mmc\}\{=\}(\d+)", text)}
        stated_splits |= {int(x) for x in re.findall(r"n_\{\\mathrm\{train\}\}\{=\}(\d+)", text)}
    assert len(stated_budgets) >= 3, f"only {stated_budgets} budgets stated; the pattern drifted"
    assert len(stated_splits) >= 3, f"only {stated_splits} split sizes stated; the pattern drifted"
    assert not stated_budgets - budgets, (
        f"the papers state budgets no released run has: {sorted(stated_budgets - budgets)}"
    )
    assert not stated_splits - splits, (
        f"the papers state split sizes no released run has: {sorted(stated_splits - splits)}"
    )
