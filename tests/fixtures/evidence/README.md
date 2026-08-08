# Defect exhibits

Recorded artifacts kept as *evidence for a published claim*, not as study cassettes. They live
outside `tests/fixtures/cassettes/` deliberately: nothing here backs a released result, so counting
them among the study cassettes would inflate the artifact release.

- `study-splice-G1-pre-tempfix.json` — recorded before the `--temperature` fix (Paper III,
  defect 10). Its `critic` and `verdict` namespaces hold 30 completions each for the same 30
  prompts at a requested temperature of 0, and all 60 are distinct, which is the evidence that the
  flag never reached those two surfaces. **Not replayable under HEAD**: the fix puts the sampling
  kwargs into the hashed key, so every key here is stale by construction. That is the same
  superseded status the paper records for defect 8's cassette.
