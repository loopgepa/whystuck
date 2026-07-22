# Vendored IFBench data + verifier logic

`ifbench_supported.jsonl` contains a filtered subset of the **IFBench** test set,
and the verifier functions in `../ifbench.py` are faithful ports of IFBench's
`check_following` implementations.

- **Source:** https://github.com/allenai/IFBench (Allen Institute for AI)
- **Paper (IFBench):** Pyatkin et al., "Generalizing Verifiable Instruction Following", arXiv 2507.02833 (NeurIPS 2025 D&B)
- **Context (GEPA):** Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv 2507.19457 — the "prompt-only GEPA" baseline our study replicates; it *evaluates on* IFBench but is a separate paper.
- **License (verifier logic / codebase):** Apache License 2.0
- **License (data rows):** ODC-BY-1.0 (Ai2 Responsible Use Guidelines)

## What was taken

- **Data:** the `key`, `prompt`, and `instruction_id_list` fields of the rows in
  `data/IFBench_test.jsonl` whose constraints are all in WhyStuck's supported,
  parameter-free, nltk-free verifier set (62 of 300 rows). Unused kwargs and
  other fields were dropped.
- **Verifier logic:** the `check_following` bodies for those 11 constraint types
  (`format:parentheses`, `format:quotes`, `count:punctuation`, `words:palindrome`,
  `format:no_whitespace`, `custom:multiples`, `words:vowel`, `words:consonants`,
  `format:newline`, `format:quote_unquote`, `format:sub-bullets`), re-expressed as
  standalone pure-Python functions in `../ifbench.py`, semantics preserved.

Constraints requiring `nltk`, an `emoji` package, or per-row kwargs were excluded
so scoring stays fully deterministic and offline.

**Licensing of this redistribution is split**, matching upstream's own README:
the ported verifier *logic* in `../ifbench.py` is governed by **Apache-2.0** (the
codebase license), and the vendored data *rows* in `ifbench_supported.jsonl` are
governed by **ODC-BY-1.0** (the data license). See the upstream repository's
`LICENSE` and README licensing section.
