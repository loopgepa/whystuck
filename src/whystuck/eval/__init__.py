"""WhyStuck v0.2 evaluation layer.

Measured evaluation of loop variants + GEPA-wrapped optimization (ADR-011).
`gepa` is an optional dependency (`pip install whystuck[eval]`), imported lazily
in `optimize` so the v0.1 core stays offline and dependency-light.
"""
