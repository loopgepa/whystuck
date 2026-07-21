# Security & Threat Model

WhyStuck records what your agent loops do and can edit workflow files. That is
sensitive by nature. This document states exactly what it captures, where it
goes, and what it will never do. If any behavior below is violated, that is a
security bug — report it (see bottom).

## Assets we protect

- **Session content** — prompts, tool I/O, file contents, and any secrets that
  pass through a run.
- **Workflow integrity** — your skills, agents, and loop definitions.

## Design guarantees

1. **WhyStuck itself is local-only.** No telemetry, no cloud sync. The MCP
   server is stdio-only and does not open network sockets; all state is on your
   disk. **Disclosure:** a loop *you declare* can include a worker step that
   shells out to an LLM CLI (e.g. `claude -p`) — that CLI makes its own network
   calls to whatever provider you configured. WhyStuck neither hides nor
   performs those calls; the command+verifier runner (the v0.1 default) is fully
   offline.
2. **Allowlist capture, not denylist redaction.** WhyStuck stores *structured
   summaries* (step roles, statuses, tool names, exit codes, file paths, output
   summaries capped at `OUTPUT_SUMMARY_CAP`) — **not** raw, unbounded tool
   input/output or file contents. *(Planned, not yet implemented: a per-loop
   `capture: raw` opt-in and denylist secret-scrubbing as a second line of
   defense on that opt-in raw data. Neither exists in code today — there is no
   `capture` field on any model.)*
3. **Storage lives outside your repo.** WhyStuck always resolves its store root
   to `$WHYSTUCK_HOME` > `$XDG_DATA_HOME/whystuck` > `~/.local/share/whystuck`
   (`core/storage.py:resolve_store_root`) — never your project's working
   directory — so traces are never inside the repo to accidentally commit.
   *(Planned, not yet implemented: an opt-in repo-local `.whystuck/` store with
   an auto-added `.gitignore` entry. This does not exist in code today.)*
4. **No silent writes.** Promotion is dry-run by default and requires an
   explicit approval flag to touch a file. Every real promotion writes a
   reversible rollback record.
5. **Safety-locked genes.** A gene marked `safety_locked: true` (e.g. a
   `tool_policy`, `escalation_policy`, or `budget_policy` gene) can be commented
   on but never edited or removed until the lock is cleared with an explicit
   `unlock_gene` mutation. Mutations can't raise a budget cap on their own, even
   after an unlock (`core/safety.py`'s guard enforces this specifically for
   `budget_policy` genes; it is the one enforced-in-code guarantee here — it is
   not a general "no safety weakening" check).
6. **Fail-open capture.** If trace recording fails, it must not break the user's
   Claude Code session.

## Non-goals (v0.1)

- Perfect secret redaction.
- Protection against a malicious local user with filesystem access to
  `WHYSTUCK_HOME`.
- Sandboxing the code executed by loops you declare.

## Reporting a vulnerability

Email **sysknife-development@protonmail.com** with details and reproduction steps. Please do
not open a public issue for security-sensitive reports. Expect an initial
response within a few business days.
