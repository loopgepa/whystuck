# Getting started

## Requirements

[`uv`](https://astral.sh/uv). If `uvx` isn't found:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

=== "Claude Code"

    ```bash
    claude mcp add loopgepa -- uvx loopgepa
    ```

=== "Codex CLI"

    ```bash
    codex mcp add loopgepa -- uvx loopgepa
    ```

=== "Cursor / any MCP client"

    ```json
    { "mcpServers": { "loopgepa": { "command": "uvx", "args": ["loopgepa"] } } }
    ```

No `pip install`, no virtualenv — `uvx` runs it from PyPI in a cached ephemeral
environment.

## Try it (offline, no API key)

```bash
uv run python examples/bugfix-loop/demo.py
```

You'll see: record failing traces → diagnose the dominant failure mode with a
confidence level → propose a reversible ≤3-gene patch → dry-run promotion (diff
only, nothing written).

## Inside your agent

```text
/loopgepa:explain-loop-failure bugfix-loop     # diagnose only
/loopgepa:optimize-loop bugfix-loop            # propose → dry-run → approve → promote
/loopgepa:design-loop                          # declare a new loop
```
