# arXiv endorsement request: draft, not sent

Nothing here has been sent to anyone. It is a draft for you to edit and send yourself.

## Why this is needed

An institutional address is no longer sufficient on its own for cs.LG. The automatic route needs
an institutional address AND prior authorship in the same endorsement domain. With neither, the
only route is a personal endorsement from an established arXiv author in that domain, who needs
roughly five submissions there. arXiv staff cannot endorse and cannot waive it.

This is the blocker with the longest lead time, and nothing else in the submission depends on it,
so it is the one worth starting first.

## Who to ask

Someone who has posted several cs.LG papers and whose work touches prompt or program optimization,
self-refinement loops, or reproducibility in LLM evaluation. A reviewer or author you have
corresponded with is a better ask than a stranger. Send it to one person at a time.

## Draft

> Subject: arXiv endorsement request, cs.LG
>
> I am an independent researcher with no institutional affiliation, so arXiv's automatic route is
> closed to me. Posting in cs.LG needs an endorsement, and I am asking whether you would consider
> giving one.
>
> The paper asks whether optimizing a self-refine loop's verifier beats optimizing its prompt. The
> answer turns on the base rate of already-correct first drafts, and the contrast across that base
> rate swings by 0.102.
>
> Start with the corrections appendix. The study invalidated its own headline twice. An earlier
> version reported that evolving the verifier cost 0.042 held-out; that is withdrawn, because the
> loop-level gene was byte-identical to the seed in 15 of 15 runs, so the treatment never reached
> the gene it was meant to change. Twenty-one corrections are listed with the reason for each.
> Every released run replays bit-for-bit from a committed cassette, so you can check any number
> without an API key.
>
> PDF attached. The artifact is at <REPOSITORY URL>.
>
> If you would rather not, no explanation needed.
>
> Vladimir Rotariu

## Before sending

- [ ] Attach the built PDF: `python scripts/make_arxiv_package.py --paper p1-conditional-lever`
- [ ] Replace `<REPOSITORY URL>` with a link that resolves. While the repository is private the
      link 404s for the reader, which is worse than omitting it: cut the sentence instead.
- [ ] Send Paper I, not the series. One paper is the ask; the other two can follow.
