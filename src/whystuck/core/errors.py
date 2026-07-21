"""WhyStuck core exceptions."""


class WhyStuckError(Exception):
    """Base class for all WhyStuck errors."""


class GeneNotFoundError(WhyStuckError):
    """A mutation targeted a gene id that does not exist in the genome."""


class NotFoundError(WhyStuckError):
    """A requested loop or variant does not exist in the store."""


class SafetyLockError(WhyStuckError):
    """A mutation tried to edit or remove a safety-locked gene
    (SECURITY.md §Safety-locked genes; IMPLEMENTATION §10)."""


class TooManyGenesError(WhyStuckError):
    """A single mutation set touches more genes than the v0 cap allows (§13)."""


class BudgetIncreaseError(WhyStuckError):
    """A mutation tried to raise a budget cap on its own (SECURITY.md §5)."""


class PathEscapeError(WhyStuckError, ValueError):
    """A promotion gene path would escape base_dir (absolute or `..` traversal).

    Subclasses ValueError as well as WhyStuckError so existing callers that catch
    ValueError keep working while it joins the WhyStuck error hierarchy."""


class UnpromotableGeneError(WhyStuckError):
    """A file-backed, evolvable gene has no `path` or non-str `content`, so it
    cannot be written to disk. Raised instead of silently skipping the gene,
    which would let `promote` report success while never writing the change."""


class PromotionSwapError(WhyStuckError):
    """A mid-swap failure during `apply_promotion` phase 2. Each `os.replace` is
    atomic per file, but the loop across several files is not — this names
    exactly which dests were swapped before the failure and which were not, so
    the caller can reason about (and correct) the partial state."""
