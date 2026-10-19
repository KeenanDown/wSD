"""Weighted symbolic divergence. All scores use original, unmasked marginals."""

from .core import (
    Scores, compare_joint, compare_signals, compare_symbols, ordinal_patterns,
    ordinal_mask, patterns, wsd,
)

__version__ = "0.1.0"
__all__ = [
    "Scores", "compare_joint", "compare_signals", "compare_symbols",
    "ordinal_patterns", "ordinal_mask", "patterns", "wsd",
]
