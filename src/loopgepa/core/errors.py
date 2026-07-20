"""LoopGEPA core exceptions."""


class LoopGEPAError(Exception):
    """Base class for all LoopGEPA errors."""


class GeneNotFoundError(LoopGEPAError):
    """A mutation targeted a gene id that does not exist in the genome."""


class NotFoundError(LoopGEPAError):
    """A requested loop or variant does not exist in the store."""


class SafetyLockError(LoopGEPAError):
    """A mutation tried to edit or remove a safety-locked gene
    (SECURITY.md §Safety-locked genes; IMPLEMENTATION §10)."""


class TooManyGenesError(LoopGEPAError):
    """A single mutation set touches more genes than the v0 cap allows (§13)."""


class BudgetIncreaseError(LoopGEPAError):
    """A mutation tried to raise a budget cap on its own (SECURITY.md §5)."""
