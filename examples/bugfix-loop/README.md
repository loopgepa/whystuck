# Example: bugfix-loop

The flagship LoopGEPA walkthrough — fully local, offline, no API key.

## Run the demo

```bash
uv run python examples/bugfix-loop/demo.py
```

You'll see the whole v0.1 flow against throwaway temp directories:

```text
Diagnosis
  failure mode : repair repeats the original patch after tests fail
  evidence     : 8 traces
  confidence   : medium (0.7)
  likely genes : verifier_rubric

Proposed variant: variant-…

Dry-run promotion (dry_run=True, promoted=False)
  Approve in your agent to apply. Nothing has been written.
```

## The same flow from inside Claude Code / Codex

Once the MCP server is installed (`claude mcp add loopgepa -- uvx loopgepa`):

```text
You:    /loopgepa:explain-loop-failure bugfix-loop
Claude: 8 traces, 0% success. Dominant failure (confidence: medium): the repair
        step repeats the original patch after tests fail. Likely gene:
        verifier_rubric.

You:    /loopgepa:optimize-loop bugfix-loop
Claude: Proposing 1 change to verifier_rubric (evidence: 8 traces). Here is the
        dry-run diff. Approve to apply?

You:    approve
Claude: Promoted. Wrote skills/bugfix/SKILL.md. Reversible — rollback recorded.
```

## Files

- `loop.yaml` — the baseline genome (skill, verifier rubric, retry policy, stop
  condition).
- `demo.py` — the runnable end-to-end script above.

Every step is real, local, and reproducible — the diagnosis comes from the
recorded traces, not a script.
