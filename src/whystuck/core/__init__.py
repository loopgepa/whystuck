"""WhyStuck core: vendor-neutral pure logic.

No network, no side effects at import time. Modules: models, storage, diagnosis,
runner, mutation, diff, safety, promote, errors. Kept free of Claude/MCP-specific
assumptions so the core survives a platform absorbing the plugin layer (ADR-002).
"""
