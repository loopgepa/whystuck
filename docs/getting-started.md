# Getting started

## Requirements

[`uv`](https://astral.sh/uv). If `uvx` isn't found:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

The fastest path — **one command** configures every client you use:

```bash
uvx whystuck setup
```

It writes the MCP config for Claude Code / Codex / Cursor **and** a short usage
rule so your agent auto-diagnoses stuck loops before it thrashes. Merge-safe and
re-runnable. Choose clients/scope with `--all` · `--claude` · `--cursor` ·
`--codex` · `--global` · `--yes`.

### Configure one client by hand

=== "Claude Code"

    ```bash
    claude mcp add whystuck -- uvx whystuck
    ```

=== "Codex CLI"

    ```bash
    codex mcp add whystuck -- uvx whystuck
    ```

=== "Cursor / any MCP client"

    ```json
    { "mcpServers": { "whystuck": { "command": "uvx", "args": ["whystuck"] } } }
    ```

No `pip install`, no virtualenv — `uvx` runs it from PyPI in a cached ephemeral
environment.

### Before the PyPI release — run from a local clone

`uvx whystuck` needs the package on PyPI. Until it's published, run the wizard
from a git clone instead — `--from-source` makes it register `uv run` commands:

```bash
git clone https://github.com/loopgepa/whystuck && cd whystuck
uv run whystuck setup --from-source "$PWD"      # all clients, from this clone
```

Or wire a single client up by hand:

```bash
claude mcp add whystuck -- uv run --directory "$PWD" whystuck   # Claude Code
codex  mcp add whystuck -- uv run --directory "$PWD" whystuck   # Codex CLI
```

For Cursor / any MCP client, the equivalent `mcp.json` entry is:

```json
{ "mcpServers": { "whystuck": { "command": "uv", "args": ["run", "--directory", "/abs/path/to/whystuck", "whystuck"] } } }
```

Switch to the `uvx whystuck` commands above once the package is on PyPI.

## Try it (offline, no API key)

```bash
uv run python examples/bugfix-loop/demo.py
```

You'll see: record failing traces → diagnose the dominant failure mode with a
confidence level → propose a reversible ≤3-gene patch → dry-run promotion (diff
only, nothing written).

## Inside your agent

```text
/whystuck:explain-loop-failure bugfix-loop     # diagnose only
/whystuck:optimize-loop bugfix-loop            # propose → dry-run → approve → promote
/whystuck:design-loop                          # declare a new loop
```
