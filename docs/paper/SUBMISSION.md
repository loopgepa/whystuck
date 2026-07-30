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
- the staged tree compiles with **pdflatex only, never bibtex**, with zero undefined citations —
  because arXiv does not run bibtex, and a package missing `main.bbl` posts with every citation
  rendered `[?]`

## Size

45 pages: 38 of main text, 7 of appendices. ~16,500 words in the main text, 16 tables, 3 figures,
28 references. The package is ~122 kB against arXiv's 50 MB limit.

This length is fine for arXiv and for TMLR, which sets no page limit. It is roughly three times
over the 9-page main-text limit at NeurIPS, ICLR and ICML, so a conference submission would need
about two thirds of the main text moved into appendices, plus that venue's style file.
