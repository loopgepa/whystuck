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

**Main-text length is 31 / 19 / 16 pages** (references begin on p.32 / 20 / 17 of 42 / 26 / 22).
**All three papers are de-anonymised by their own reproducibility apparatus**: the title block
prints author, affiliation and repository URL, and each abstract names the repository again.

| venue | window | verdict |
|---|---|---|
| **TMLR** | rolling | **The fit.** Its two criteria are whether the claims are supported by the evidence and whether anyone in the audience would be interested; novelty and significance are *explicitly excluded* as grounds for rejection. Variable manuscript length, so the main-text lengths above are not a problem. arXiv preprints permitted. Double-blind. **Needs a restyle:** "Submissions must be PDF files generated using the TMLR LaTeX stylefile and template" (author guide, quoted 2026-09-04), and these build on `article` at 11pt, so the page counts on this page are for the current class and will move under theirs. Accepts the artifact as supplementary material, which removes the mirror-hosting step below. |
| ICLR 2027 | abstract 18 Sep, paper 25 Sep 2026 | **Ruled out for a three-paper series.** 9-page initial limit (10 camera-ready; references and appendices excluded). And: *"If you and all of your co-authors have never had a paper accepted for publication in a major machine learning conference or journal, you may submit at most one paper to ICLR 2027."* Single-author and independent ⇒ **one paper, cut threefold.** Authors on 3+ papers must review 6. Mandatory AI-use statement whose required-disclosure list covers methodology feedback, implementing methods and interpreting results. |
| NeurIPS 2026 Evaluations & Datasets | abstract 4 May, full 6 May 2026 | **Closed.** Was the best categorical fit, the renamed Datasets & Benchmarks track, now explicitly treating evaluation as an object of study. Next window ≈ May 2027. |
| MLRC 2026 | hard 30 Sep 2026 | **Not a fit.** Requires prior TMLR acceptance, and its scope is reproducing *others'* claims published 2025 onward. |

Confirm every date against the venue's own page before relying on it; two sources disagreed by a
day on ICLR's, and this table is a snapshot.

### Two TMLR policies, resolved from the published policy rather than from the editors

Both are specific to submitting a SERIES, so neither shows up when the venue is read for one paper.
Neither needed an email: the editorial policy page answers the first outright and names the rule
behind the second.

**Parallel submission and shared text: not a problem.** The policy restricts reuse against
*another* venue: "There should not be any reuse of written text, figures or results between the
submitted paper and any paper which has been published, accepted for publication, or submitted in
parallel at another archival, peer-reviewed venue." Three papers to TMLR at once is not that case,
and nothing in the policy addresses companion papers or a series. The sharing is real and worth
stating anyway: `shared/loop.tex` and `shared/series.tex` are `\input` by all three, which is 828
words, or 3.8% of Paper~I, 6.3% of Paper~II and 7.3% of Paper~III. Re-measure with

    wc -w docs/papers/shared/loop.tex docs/papers/shared/series.tex

**The annual quota binds, and it binds at two.** TMLR restricts submissions per calendar year by
the Generalized Harmonic Quota Rule with parameters $N_1 = 2$ and $N_9 = 9$: an author submitting
only single-author papers may submit **two per year**, an author whose papers all carry nine
authors may submit nine. This series is single-author, so **three papers cannot all go to TMLR in
the same calendar year.**

There is a documented way out that costs no favours. Authors who are active members of the TMLR
Action Editor or reviewer pool receive **double** the budget, $N_1 = 4$, which accommodates all
three. Joining the reviewer pool is the cheapest unblock available here and it is worth doing
before submitting anything.

### What the quota does to the order, and why a split submission needs arXiv first

The three are interdependent by design, and the dependency is asymmetric: Paper~I cites Paper~II
nineteen times and Paper~III twenty-one, while Paper~II cites Paper~I six times and Paper~III
three. Submitting two and holding one leaves the two that ship citing an unpublished companion,
which a reviewer cannot follow.

TMLR permits preprints: overlap with work shared "on preprint servers such as arXiv" is
acceptable. So the order that keeps the series readable is

1. **Reviewer pool first** if you are willing, since $N_1 = 4$ removes the choice entirely.
2. **arXiv all three**, which needs the endorsement once and makes every cross-reference
   resolvable no matter what TMLR's timing does.
3. **TMLR two this year** if the quota still binds, Paper~I among them, since the other two rest
   on it. Hold Paper~III last: it is the methodological record the other two cite for corrections
   rather than for results, so it is the one whose absence costs a reader least.

### The double-blind build

```bash
python scripts/make_arxiv_package.py --paper p1-conditional-lever --anonymous \
    --anon-url <anonymous mirror>
```

Strips the author block and the repository URL, drops the `references.bib` comment naming the
project, then **scans every byte of the staged tree** for six identifying strings and refuses to
write a tarball if any survives. The scan reads bytes rather than parsing TeX because the leak that
motivated it was a LaTeX comment, which renders as nothing and ships in the tarball regardless;
the same pass covers figure PDF metadata. It is a floor rather than a proof, since a token inside a compressed
PDF stream would not appear, and `tests/test_anonymous_package.py` pins that the scanner still
finds the author in the *unmodified* sources, since a scanner that finds nothing anywhere would
pass every other check.

`--anon-url` is **required** under `--anonymous`: the abstract claims an artifact release, and a
claim pointing nowhere is worse than a named one.

**Building that mirror is a larger job than the paper build, and `scripts/make_anon_mirror.py`
is that job.** Measured over the
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
   tarball. The package contains only `main.tex`, `main.bbl`, `references.bib` and the figures that
   paper includes, which is why the byte scan passes, and both are in the tree a full mirror would
   host.

Standing one up publishes content under a new identity and is a decision for the author, not
something the packaging script should do on its own. The script writes a verified tree and stops
there. Rebuilt for this review it produced 793 files, 43 run JSONs, 225 replicate scorings and 267
cassettes, with no identifying token surviving a byte scan and a standalone install of that tree
re-deriving 1086/1086 claims after the rename. It lands in `dist/`, which is gitignored, so
**rebuild it at the commit you submit**: a snapshot kept from an earlier commit publishes
superseded papers, which is what happened to this one inside a single review pass.

## Blocking

- [ ] **The repository must be public** *for the arXiv path only.* The named title block prints
      the URL under the author's name, so posting while it is private ships a paper citing a 404.
      A double-blind submission does not use the named build at all and is gated by the anonymous
      mirror instead, which now exists as a verified tree and needs only publishing. Going public
      before the venue is chosen can work against the submission: every commit here carries the
      author's name, so a public repository the paper points at breaks the blind.
- [ ] **Publish the anonymous mirror** *for a venue with no supplementary channel.* The tree is
      built and verified; pushing it to an account that is not yours is the step no script should
      take. Then rebuild all three packages against its URL with `scripts/anon_submission.sh`.
      **For TMLR this step is avoidable.** The author guide, quoted 2026-09-04, permits "up to
      100MB of supplementary material, such data, source code or illustrative videos; all
      supplementary materials must be in PDF or ZIP format", and it must be anonymized like the
      submission. The mirror is 21MB on disk and **6.4MB zipped**, so the whole artifact ships
      with the paper and no third-party host is involved. Hosting is still required for the arXiv
      path, where the abstract prints a URL that has to resolve.
- [ ] **Put the two companion PDFs in the supplementary ZIP** for any double-blind submission.
      A reviewer of Paper~I cannot read Papers II and III, and Paper I attributes twelve numeric
      claims to them (Paper II seven, Paper III one), so those claims are unverifiable to the one
      person who has to judge them. Two independent sources name the same remedy. COPE, on
      divided publication: authors "should cross reference or include a copy of the companion
      paper". NeurIPS, on citing your own concurrent submission that is not available as a
      non-anonymous preprint: "include a copy of the cited anonymized submission in the
      supplementary material". Both quoted 2026-09-05. The anonymous builds already exist, so
      this costs nothing: the artifact is 6.3MB zipped and the three PDFs are 0.9, 0.7 and
      0.5MB, so a bundle of the artifact plus two companions is 7.8MB against TMLR's 100MB.
- [ ] **Join the TMLR reviewer pool**, if you are willing. It doubles the annual quota from two
      submissions to four, which is the difference between sending this series over two years and
      sending it in one. Cheapest unblock on this page.
- [ ] **Endorsement.** arXiv tightened this on 21 January 2026: an institutional email address is no
      longer sufficient on its own. The automatic route requires an institutional address *and*
      prior authorship in the same endorsement domain. With neither, the only route is a personal
      endorsement from an established arXiv author in that domain, who needs roughly five
      submissions there. arXiv staff cannot endorse and cannot waive the requirement.

## Checked automatically

`preflight()` blocks the build on any of these, so they cannot silently regress:

- the author block contains no placeholder (`TBD`, `TODO`, `Anonymous`, …)
- `\pdfoutput=1` sits within the first five lines of `main.tex`, since arXiv's pipeline chooses between
  `latex` (DVI) and `pdflatex` by inspection, and a DVI run cannot include the three PDF figures
- the plain-text abstract fits arXiv's 1920-character metadata field
- the three lengths quoted below, and that limit, equal what the sources measure
  (`test_advertised_abstract_lengths_equal_the_measured_ones`). Added after Paper III's was found
  advertised at 1786 while the source measured 1788: the table's numbers were guarded and this
  sentence's were not, so an edit that shortened the abstract left the figure behind
- the staged tree compiles with **pdflatex only, never bibtex**, with zero undefined citations,
  because arXiv does not run bibtex, and a package missing `main.bbl` posts with every citation
  rendered `[?]`

## Size

**Three separate submissions, one per paper.** This section described the pre-split monolith (45
pages, 16 tables, 28 references) until 2026-08-03; those figures belong to a document that no longer
exists. Measured from the built PDFs and each paper's own source:

| | paper | pages | tables | figures | distinct citations |
|---|---|---|---|---|---|
| I | `p1-conditional-lever` | 42 | 11 | 3 | 25 |
| II | `p2-repair-not-detection` | 26 | 4 | 3 | 18 |
| III | `p3-silent-no-op` | 22 | 4 | 1 | 14 |

Tables, figures and citations are counted from the sources by
`tests/test_paper_docs_consistency.py`; the page count is compared against the compiled PDF by
`make_arxiv_package.py`, which refuses to write a package while the two disagree. Paper I sat at 38
here for four days after an edit made it 39, and README's table agreed with this one throughout,
which is the failure the papers themselves describe: two hand-typed numbers checking each other.
Every tarball is a fraction of a megabyte against arXiv's 50 MB limit; the build prints the exact
size, and this file does not quote it, because gzip output is not identical across toolchains and a
number nothing can re-derive is a number that rots. The rehearsal also refuses a package whose log
reports an overfull box or a float too large for the page: both put ink outside the text block,
neither stops a build, and Paper I shipped an oversized threats table until 2026-08-17 because
nothing read the log.

Abstract lengths against the 1920-character metadata
field: **1902**, **1902**, **1906**. The headroom is 18, 18 and 14 characters, so any edit to any
of the three abstracts has to be re-checked by `make_arxiv_package.py` rather than assumed to fit.

The three are cross-referenced as a series, so they should be posted together or in the order I, II,
III; each abstract names its companions.

These lengths are fine for arXiv and for TMLR, which sets no page limit. Against the 9-page
main-text limit at NeurIPS, ICLR and ICML, Paper I runs 3.4x, Paper II 2.1x and Paper III 1.8x, so a
conference submission would need substantial main text moved into appendices, plus that venue's style
file. Paper I read "about four times" until 2026-08-17, which is 39/9 rather than 27/9: the sentence
divided the total by the main-text limit. The three multiples are now checked against the main-text
lengths stated above. This is the split's main practical gain: II and III are within reach of a
conference cut in a way the 45-page monolith was not.
