"""Exception hierarchy for gpgraph."""

from __future__ import annotations


class GpgraphError(Exception):
    """Base class for all gpgraph-specific errors."""


class NeighborError(GpgraphError):
    """Raised when neighbor detection cannot proceed (alphabet mismatch, bad cutoff, etc.)."""
