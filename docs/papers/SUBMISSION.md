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
