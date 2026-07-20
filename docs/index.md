# WhyStuck

**Your coding agent re-ran `npm install` 40 times and you didn't notice.**
WhyStuck sees *why* an agent loop keeps failing and proposes a reversible fix you
approve — locally, with no telemetry.

An MCP server for **Claude Code**, **Codex CLI**, and **Cursor**.

![WhyStuck diagnosing a failing bugfix loop and proposing a reversible fix](assets/whystuck-demo.gif)

- **Diagnoses** recurring failures from real, local traces.
- **Proposes** a small, reversible patch (verifier rubric, retry policy, stop
  condition, worker skill) with a diff and a confidence level.
- **Never writes without your approval.** Dry-run by default; every promotion is
  reversible.

Why it's different from Latitude / LangSmith / Braintrust / Langfuse: those are
cloud SaaS for production LLM apps. WhyStuck runs on your machine, scoped to your
own coding-agent sessions, and nothing leaves it.

Next: [Getting started](getting-started.md) · [Trust & safety](safety.md)
