# Trust & safety

Trust is the product: LoopGEPA records what your agent loops do and can edit
workflow files. Here is exactly what it does and never does. The full threat
model lives in [`SECURITY.md`](https://github.com/loopgepa/loopgepa/blob/main/SECURITY.md).

## Guarantees

- **LoopGEPA is local-only.** No telemetry, no cloud. The MCP server is
  stdio-only and opens no network sockets.
- **Summaries, not secrets.** Allowlist capture — step roles, statuses, tool
  names, exit codes, paths, counts. Raw tool I/O is opt-in per loop.
- **Stored outside your repo** (`$XDG_DATA_HOME/loopgepa`), so traces are never
  committed by accident.
- **You approve every write.** Promotion is dry-run by default and always
  reversible; each promotion writes a rollback record.
- **Safety-locked genes** (tool/escalation/budget policy) are never edited or
  removed without an explicit unlock. Mutations can't raise a budget cap on
  their own, and a single proposal may touch at most three genes.

## Disclosure

A loop *you declare* can include a worker step that shells out to an LLM CLI
(e.g. `claude -p`); that CLI makes its own provider network calls. LoopGEPA
neither hides nor performs those calls — and the default command+verifier runner
is fully offline.

## Honesty

v0.1 diagnoses and proposes; it does not yet run measured evaluation, so it never
calls a proposal "optimized." That word is earned in v0.2. See the
[design records](https://github.com/loopgepa/loopgepa/blob/main/DECISIONS.md).
