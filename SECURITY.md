# Security & Threat Model

LoopGEPA records what your agent loops do and can edit workflow files. That is
sensitive by nature. This document states exactly what it captures, where it
goes, and what it will never do. If any behavior below is violated, that is a
security bug — report it (see bottom).

## Assets we protect

- **Session content** — prompts, tool I/O, file contents, and any secrets that
  pass through a run.
- **Workflow integrity** — your skills, agents, and loop definitions.

## Design guarantees

1. **LoopGEPA itself is local-only.** No telemetry, no cloud sync. The MCP
   server is stdio-only and does not open network sockets; all state is on your
   disk. **Disclosure:** a loop *you declare* can include a worker step that
   shells out to an LLM CLI (e.g. `claude -p`) — that CLI makes its own network
   calls to whatever provider you configured. LoopGEPA neither hides nor
   performs those calls; the command+verifier runner (the v0.1 default) is fully
   offline.
2. **Allowlist capture, not denylist redaction.** By default LoopGEPA stores
   *structured summaries* (step roles, statuses, tool names, exit codes, file
   paths, metric counts) — **not** raw tool input/output or file contents.
   Raw capture is opt-in per loop (`capture: raw`). Denylist secret-scrubbing
   exists only as a second line of defense on opt-in raw data, and we do not
   claim it is perfect.
3. **Storage lives outside your repo.** Default `$XDG_DATA_HOME/loopgepa`
   (fallback `~/.local/share/loopgepa`). A repo-local `.loopgepa/` is opt-in;
   choosing it makes LoopGEPA add a `.gitignore` entry so traces are never
   committed.
4. **No silent writes.** Promotion is dry-run by default and requires an
   explicit approval flag to touch a file. Every real promotion writes a
   reversible rollback record.
5. **Safety-locked genes.** A gene marked `safety_locked: true` (e.g. a
   `tool_policy`, `escalation_policy`, or `budget_policy` gene) can be commented
   on but never edited or removed without an explicit user unlock. Mutations may
   never weaken safety, remove approval gates, or raise budget caps on their own.
6. **Fail-open capture.** If trace recording fails, it must not break the user's
   Claude Code session.

## Non-goals (v0.1)

- Perfect secret redaction.
- Protection against a malicious local user with filesystem access to
  `LOOPGEPA_HOME`.
- Sandboxing the code executed by loops you declare.

## Reporting a vulnerability

Email **sysknife-development@protonmail.com** with details and reproduction steps. Please do
not open a public issue for security-sensitive reports. Expect an initial
response within a few business days.
