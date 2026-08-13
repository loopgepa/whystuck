# arXiv submission: settled metadata and what is left

Run `python scripts/make_arxiv_package.py` to produce `dist/arxiv-submission.tar.gz`. It refuses to
write a package that cannot be submitted, and prints the values below so the form and the paper
cannot drift apart. This file records the decisions; the script enforces the checkable ones.

## Decided

| field | value | why |
|---|---|---|
| Author | Vladimir Rotariu, Independent researcher | — |
| Licence | **CC BY 4.0** | The contribution is a reporting practice plus a released artifact, and the paper asks others to re-run it. arXiv's default Perpetual Non-Exclusive licence grants arXiv distribution rights while limiting everyone else's re-use, which works against that. **Irrevocable once announced.** |
| Primary category | **cs.LG** | The object of study is an optimizer's behaviour and how to measure it. |
| Cross-list | **cs.CL** | The tasks are instruction-following and code generation. |
| Code licence | Apache-2.0 (repository `LICENSE`) | Separate grant from the paper's. |

The category decision comes **before** looking for an endorser: endorsement is granted per
endorsement domain, and the primary category selects the domain.

## Peer-reviewed venues, surveyed 2026-08-12

Two structural facts decide this, and neither is about the papers' quality.

**Main-text length is 27 / 17 / 14 pages** (references begin on p.28 / 18 / 15 of 38 / 22 / 19).
**All three papers are de-anonymised by their own reproducibility apparatus**: the title block
prints author, affiliation and repository URL, and each abstract names the repository again.

| venue | window | verdict |
|---|---|---|
| **TMLR** | rolling | **The fit.** Its two criteria are whether the claims are supported by the evidence and whether anyone in the audience would be interested; novelty and significance are *explicitly excluded* as grounds for rejection. Variable manuscript length, so 27 pages is not a problem. arXiv preprints permitted. Double-blind. |
| ICLR 2027 | abstract 18 Sep, paper 25 Sep 2026 | **Ruled out for a three-paper series.** 9-page initial limit (10 camera-ready; references and appendices excluded). And: *"If you and all of your co-authors have never had a paper accepted for publication in a major machine learning conference or journal, you may submit at most one paper to ICLR 2027."* Single-author and independent ⇒ **one paper, cut threefold.** Authors on 3+ papers must review 6. Mandatory AI-use statement whose required-disclosure list covers methodology feedback, implementing methods and interpreting results. |
| NeurIPS 2026 Evaluations & Datasets | abstract 4 May, full 6 May 2026 | **Closed.** Was the best categorical fit — the renamed Datasets & Benchmarks track, now explicitly treating evaluation as an object of study. Next window ≈ May 2027. |
| MLRC 2026 | hard 30 Sep 2026 | **Not a fit.** Requires prior TMLR acceptance, and its scope is reproducing *others'* claims published 2025 onward. |

Confirm every date against the venue's own page before relying on it; two sources disagreed by a
day on ICLR's, and this table is a snapshot.

### The double-blind build

```bash
python scripts/make_arxiv_package.py --paper p1-conditional-lever --anonymous \
    --anon-url <anonymous mirror>
```

Strips the author block and the repository URL, drops the `references.bib` comment naming the
project, then **scans every byte of the staged tree** for six identifying strings and refuses to
write a tarball if any survives. The scan reads bytes rather than parsing TeX because the leak that
motivated it was a LaTeX comment, which renders as nothing and ships in the tarball regardless;
the same pass covers figure PDF metadata. It is a floor, not a proof — a token inside a compressed
PDF stream would not appear — and `tests/test_anonymous_package.py` pins that the scanner still
finds the author in the *unmodified* sources, since a scanner that finds nothing anywhere would
pass every other check.

`--anon-url` is **required** under `--anonymous`: the abstract claims an artifact release, and a
claim pointing nowhere is worse than a named one.

**That mirror does not exist, and it is a larger job than the paper build.** Measured over the
working tree on 2026-08-12, excluding `.git`, `dist/` and caches:

| identifying string | files |
|---|---|
| `loopgepa` (the GitHub org, and the URL in `pyproject.toml`) | 24 |
| `Vladimir` | 12 |
| `Rotariu` | 11 |
| `vladimirrott` | 5 |
| `sysknife-development` (the commit address, in `CODE_OF_CONDUCT.md` and `SECURITY.md`) | 2 |

Four things make this more than a search-and-replace, and all four are worth knowing before
promising a reviewer an anonymous artifact:

1. **Git history leaks on every commit.** Every commit in the repository is authored and committed
   by the same name and address. A mirror that preserves history is not anonymous, so the mirror
   has to be a *snapshot*. Services built for this (e.g. `anonymous.4open.science`) serve snapshots
   for that reason.
2. **The distribution is named for the org.** `pyproject.toml` sets both `Homepage` and
   `Repository` to the GitHub URL, and `dist/` carries wheels named `loopgepa-*`.
3. **The reproducibility apparatus is the point of the release**, so it cannot simply be dropped:
   43 run JSONs, 225 replicate scorings and 267 cassettes have to stay reachable and replayable.
4. **This file, and the anonymiser's own token list, name the author.** Neither ships in a paper
   tarball — the package contains only `main.tex`, `main.bbl`, `references.bib` and three figures,
   which is why the byte scan passes — but both are in the tree a full mirror would host.

Standing one up publishes content under a new identity and is a decision for the author, not
something the packaging script should do on its own.

## Blocking

- [ ] **The repository must be public.** The abstract claims a release and the title block prints
      the URL under the author's name, so posting while it is private ships a paper citing a 404.
      This is the only remaining item that is not a form field.
- [ ] **Endorsement.** arXiv tightened this on 21 January 2026: an institutional email address is no
      longer sufficient on its own. The automatic route requires an institutional address *and*
      prior authorship in the same endorsement domain. With neither, the only route is a personal
      endorsement from an established arXiv author in that domain, who needs roughly five
      submissions there. arXiv staff cannot endorse and cannot waive the requirement.

## Checked automatically

`preflight()` blocks the build on any of these, so they cannot silently regress:

- the author block contains no placeholder (`TBD`, `TODO`, `Anonymous`, …)
- `\pdfoutput=1` sits within the first five lines of `main.tex` — arXiv's pipeline chooses between
  `latex` (DVI) and `pdflatex` by inspection, and a DVI run cannot include the three PDF figures
- the plain-text abstract fits arXiv's 1920-character metadata field
- the three lengths quoted below, and that limit, equal what the sources measure
  (`test_advertised_abstract_lengths_equal_the_measured_ones`). Added after Paper III's was found
  advertised at 1786 while the source measured 1788: the table's numbers were guarded and this
  sentence's were not, so an edit that shortened the abstract left the figure behind
- the staged tree compiles with **pdflatex only, never bibtex**, with zero undefined citations —
  because arXiv does not run bibtex, and a package missing `main.bbl` posts with every citation
  rendered `[?]`

## Size

**Three separate submissions, one per paper.** This section described the pre-split monolith (45
pages, 16 tables, 28 references) until 2026-08-03; those figures belong to a document that no longer
exists. Measured from the built PDFs and each paper's own source:

| | paper | pages | tables | figures | distinct citations | package |
|---|---|---|---|---|---|---|
| I | `p1-conditional-lever` | 38 | 11 | 3 | 24 | 107 kB |
| II | `p2-repair-not-detection` | 22 | 4 | 2 | 17 | 85 kB |
| III | `p3-silent-no-op` | 19 | 4 | 1 | 13 | 83 kB |

Each package is far under arXiv's 50 MB limit. Abstract lengths against the 1920-character metadata
field: **1912**, **1895**, 1788. Paper I has 8 characters of headroom and Paper II has 25, so any
edit to either abstract has to be re-checked by `make_arxiv_package.py` rather than assumed to fit.

The three are cross-referenced as a series, so they should be posted together or in the order I, II,
III; each abstract names its companions.

These lengths are fine for arXiv and for TMLR, which sets no page limit. Papers II and III are each
roughly twice the 9-page main-text limit at NeurIPS, ICLR and ICML and Paper I about four times, so a
conference submission would need substantial main text moved into appendices, plus that venue's style
file. This is the split's main practical gain: II and III are within reach of a conference cut in a
way the 45-page monolith was not.
