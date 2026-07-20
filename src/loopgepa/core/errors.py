"""LoopGEPA core exceptions."""


class LoopGEPAError(Exception):
    """Base class for all LoopGEPA errors."""


class GeneNotFoundError(LoopGEPAError):
    """A mutation targeted a gene id that does not exist in the genome."""


class SafetyLockError(LoopGEPAError):
    """A mutation tried to edit or remove a safety-locked gene (ADR-004 / §10.2)."""
