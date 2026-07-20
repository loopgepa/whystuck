"""LoopGEPA core: vendor-neutral pure logic.

No network, no side effects at import time. Modules (per IMPLEMENTATION.md §5-§7):
models, storage, runner, mutation, diff, safety. Kept free of Claude/MCP-specific
assumptions so the core survives a platform absorbing the plugin layer (ADR-002).
"""
