"""The one bibliography, checked for the failures bibtex resolves silently.

WHY THIS EXISTS. Adding a version pin to `pappu2026multiagent` gave that entry a SECOND `note`
field. BibTeX keeps the first and prints `Warning--I'm ignoring pappu2026multiagent's extra "note"
field` on a line nobody reads, so the substantive note --- the one recording that the source
independently reports both halves of Paper III's finding --- silently stopped appearing in the
rendered bibliography while the `.tex` still looked correct. Two builds shipped that way.

That is the series' own defect class, in the file whose stated contract is that every entry is
verified against its source: a duplicate key is not an error anywhere in the toolchain, it is a
precedence rule, so the loss leaves no mark in the PDF and no failing check.

The guards here are deliberately about *silence*, not about style. A malformed entry breaks the
build and needs no test; a resolved-away one does not break anything at all.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "docs/papers/shared/references.bib"

#: An entry header: `@type{key,`. Comments outside entries are ignored by construction, because the
#: split starts at each `@` that begins a line.
ENTRY = re.compile(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)

#: A field assignment at the top level of an entry. Anchored to the line start so a `key = value`
#: appearing inside a braced note body cannot be mistaken for a field of the entry.
FIELD = re.compile(r"^\s{2,}(\w+)\s*=", re.MULTILINE)


def _entries() -> list[tuple[str, str, str]]:
    """(type, key, body) per entry, body running to the start of the next entry."""
    text = BIB.read_text(encoding="utf-8")
    marks = list(ENTRY.finditer(text))
    assert marks, "references.bib has no entries; the fixture is wrong, not the bibliography"
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), m.group(2), text[m.end() : end]))
    return out


def test_no_entry_declares_a_field_twice() -> None:
    """A repeated field is content loss that bibtex reports as a warning and the PDF hides."""
    offenders = {}
    for _kind, key, body in _entries():
        counts = Counter(FIELD.findall(body))
        repeated = {field: n for field, n in counts.items() if n > 1}
        if repeated:
            offenders[key] = repeated
    assert not offenders, (
        f"these entries declare a field more than once: {offenders}. BibTeX keeps the FIRST and "
        "silently discards the rest, so the discarded text never reaches the bibliography. Merge "
        "them into one field."
    )


def test_every_key_is_unique() -> None:
    """Two entries under one key is the same failure one level up: the later one never renders."""
    keys = [key for _kind, key, _body in _entries()]
    duplicated = sorted(key for key, n in Counter(keys).items() if n > 1)
    assert not duplicated, f"duplicate entry keys, only the first of each renders: {duplicated}"


def test_every_entry_records_what_it_supports() -> None:
    """The file's stated contract, in `docs/papers/README.md`, is that every entry is verified
    against its source. A `note` is where that verification is written down, so an entry without one
    is an entry whose check was never recorded --- and the citation defects this series has actually
    shipped were all over-attribution, not misformatting.
    """
    missing = sorted(
        key for _kind, key, body in _entries() if not re.search(r"^\s+note\s*=", body, re.MULTILINE)
    )
    assert not missing, (
        f"these entries carry no note recording what the source supports: {missing}. Every "
        "citation defect in the corrections registers was a claim attached to a source that did "
        "not contain it, which is exactly what the note field is for."
    )
