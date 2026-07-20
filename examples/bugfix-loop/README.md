# Example: bugfix-loop

Flagship worked example (built in Milestone 8). Will contain:

- `loop.yaml` — baseline genome (bugfix skill, verifier rubric, retry policy,
  stop condition).
- `traces/` — sample success and failure traces.
- `variants/` — one reviewed mutation variant with a diff.
- A written walkthrough of the Claude Code interaction:
  `/loopgepa:explain-loop-failure bugfix-loop` → diagnosis → proposed diff →
  approve → promote.

Placeholder until `loopgepa_core` reaches the runner milestone.
