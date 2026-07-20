"""LoopGEPA core exceptions."""


class LoopGEPAError(Exception):
    """Base class for all LoopGEPA errors."""


class GeneNotFoundError(LoopGEPAError):
    """A mutation targeted a gene id that does not exist in the genome."""


class SafetyLockError(LoopGEPAError):
    """A mutation tried to edit or remove a safety-locked gene (ADR-004 / §10.2)."""


class TooManyGenesError(LoopGEPAError):
    """A single mutation set touches more genes than the v0 cap allows (§13)."""


class BudgetIncreaseError(LoopGEPAError):
    """A mutation tried to raise a budget cap on its own (SECURITY.md §5)."""
