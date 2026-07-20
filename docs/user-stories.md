# User stories

Sixteen real-shaped sessions across **Claude Code**, **Codex**, and **Cursor**.
The MCP server is identical in all three — only the ergonomics differ (Claude Code
adds `/whystuck:*` slash commands via the plugin; Codex/Cursor call the same tools
from plain English). See [Getting started](getting-started.md) to install.

![Diagnose a failing loop and propose a reversible fix](assets/whystuck-demo.gif)

---

### 1 · First run — install & verify · Claude Code
```
$ claude mcp add whystuck -- uvx whystuck
$ /whystuck   → 6 tools + 3 skills available
```
Zero config, stdio, local. Traces land in `$XDG_DATA_HOME/whystuck`, outside your repo.

### 2 · "Why does this keep failing?" · Claude Code
```
You: /whystuck:explain-loop-failure bugfix-loop
→ diagnose
WhyStuck: 14 traces, 36% success. HIGH confidence (12/14): repair repeats the
          original patch after tests fail. Likely gene: verifier_rubric.
```
Diagnosis only — no change. Confidence is server-computed, so it can't be inflated.

### 3 · Propose → approve → promote · Claude Code
```
You: /whystuck:optimize-loop bugfix-loop
→ propose_fix (1 gene) → promote dry_run=true (shows diff)
You: approve
→ promote dry_run=false → writes skills/bugfix/SKILL.md, rollback recorded
```
![Approve a fix; the file changes atomically and reversibly](assets/whystuck-promote.gif)

### 4 · Honest "not enough evidence" · Claude Code
```
You: /whystuck:explain-loop-failure new-loop
WhyStuck: only 3 traces — confidence LOW. This is a plausible mode, not a proven
          one. No fix recommended yet; collect more runs.
```
It refuses to guess. "No fix" is a valid, trust-building answer.

### 5 · Declare a loop & run it offline · Codex
```
You: "Set up a whystuck loop for my test-fix workflow, then run it."
→ save_loop (skill, verifier_rubric, retry_policy{max:2}, stop_condition)
→ run_loop(worker=["./fix.sh"], verifier=["pytest","-q"], max_retries=2)
Codex: recorded testfix#1 — failure, 2 retries. (offline, no API key)
```

### 6 · Plain-English diagnosis, no slash commands · Codex
```
You: "Use whystuck: summarize the last 20 traces for bugfix and name the top
      failure mode + confidence. Don't change anything."
Codex → diagnose → reports the top mode + evidence ids.
```

### 7 · Same flow inside Cursor · Cursor
```
You (Cursor agent): "Diagnose why refactor-loop keeps failing with whystuck."
→ the whystuck MCP tools run identically; .cursor/rules/whystuck.mdc auto-triggers.
```

### 8 · CI / test-fix loop over many cycles · Codex
```
run_loop each CI run (repair cmd + pytest verifier) → 40 traces accrue.
You: "What structurally makes these fail?"
WhyStuck: MEDIUM — retries never help; failure summary isn't fed back before retry.
          Likely gene: retry_policy.
```

### 9 · Safety refusal — budget · Claude Code
```
You: "Bump the loop's budget cap to $5 and promote."
→ propose_fix → BudgetIncreaseError (raising a budget cap is refused)
WhyStuck: I won't silently weaken a budget. Unlock the gene explicitly to override.
```

### 10 · Safety refusal — too many genes · Cursor
```
Agent proposes edits to 4 genes at once.
→ TooManyGenesError (v0 cap is 3, to keep the result attributable)
WhyStuck: split this into ≤3-gene proposals so we can tell what worked.
```

### 11 · Rollback a regression · Claude Code
```
A promoted variant made things worse.
You: "Roll back the last promotion."
→ re-promote the previous variant (its genome holds the old content); rollback
  record shows from/to. Reversible by construction.
```

### 12 · Weigh two competing fixes · Codex
```
You: "Propose two options and let me pick."
→ propose_fix ×2 (each returns a variant id + diff)
Codex: variant-a (edits verifier_rubric, low-risk) vs variant-b (also retry_policy).
       Recommend a — least invasive, addresses the dominant mode.
```

### 13 · Auto-detect thrash mid-session · Claude Code
```
You keep editing the same file; on the 4th edit the WhyStuck PostToolUse hook fires:
"app.py edited 4× this session without sticking — run whystuck_diagnose on the
'session' loop before editing again." It also records the trace (summaries only,
no secrets), so your real sessions become diagnosable over time. Nothing uploaded.
```

### 14 · Team playbook · Cursor
```
After a fix proves out, promotion writes the improved skills/*.md into the repo.
Commit it → every teammate's Cursor agent now runs the better loop.
```

### 15 · Cost triage · Codex
```
You: "Why is deploy-loop so expensive?"
→ diagnose (avg retries 3.1, retry-heavy failures dominate)
Codex: retry-heavy failures dominate; the stop condition never triggers. Fix the
       stop_condition before it burns more budget.
```

### 16 · v0.2 preview — measured optimization · any client *(roadmap)*
```
You: "Optimize bugfix against my tasks/ benchmark."
→ build_benchmark → optimize_loop (wraps gepa.optimize) → get_frontier
WhyStuck: Pareto set — variant-fast / variant-robust / variant-cheap, with scores.
```
Only here is anything called **optimized** — because it's measured. See
[v0.2 design](v0.2-eval-benchmark.md).

---

## What these stories teach

- **Diagnose before you change.** Every fix is anchored to evidence trace ids; a
  low-confidence diagnosis yields *no* fix, not a guess.
- **The human holds the keys.** Dry-run by default, explicit approval, every write
  reversible — the tool proposes, you decide.
- **Small, attributable changes.** ≤3 genes per proposal so you can tell what worked.
- **Safe by refusal.** Weakening a budget, exceeding the gene cap, or escaping the
  project dir are hard errors, not warnings.
- **One server, three agents.** Claude Code / Codex / Cursor use the exact same
  tools; only the trigger ergonomics differ (`/whystuck:*` vs plain English).
- **Local is the whole point.** No API key for diagnosis, nothing leaves the machine.
