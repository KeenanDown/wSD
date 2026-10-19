"""Finite-alphabet estimators for a binary retained set S.

wSD = m KL(P(.|S) || (P_X P_Y)(.|S)).
Neither preprocessing nor bias correction is implicit in this module.
"""

from dataclasses import dataclass, replace
from itertools import permutations
from math import factorial
from numbers import Integral

import numpy as np


def _integer(value, name, low=1, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high or 'infinity'}")
    return int(value)


def patterns(order=3):
    """Lexicographic zero-based amplitude ranks in time order (order 2..6)."""
    order = _integer(order, "order", 2, 6)
    return np.asarray(list(permutations(range(order))), dtype=np.int64)


def ordinal_mask(order=3):
    """Retain all but identical and amplitude-reversed pattern pairs.

    Amplitude reversal sends each rank r to order-1-r. This is not time reversal.
    With lexicographic ranks, reversed pattern IDs are n_symbols-1-ID.
    """
    n = len(patterns(order))
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[np.arange(n), n - 1 - np.arange(n)] = False
    return mask


def ordinal_patterns(signal, order=3, delay=1, stride=1, ties="raise"):
    """Encode a real 1-D signal as ordinal IDs.

    Window starts are 0, stride, ... and offsets are 0, delay, ... .
    ``ties='raise'`` rejects tied windows; ``'drop'`` marks them -1 without
    changing alignment; ``'stable'`` breaks ties by earlier sample position.
    Nonfinite samples are always rejected. No filtering is performed.
    """
    order = _integer(order, "order", 2, 6)
    delay = _integer(delay, "delay")
    stride = _integer(stride, "stride")
    if ties not in ("raise", "drop", "stable"):
        raise ValueError("ties must be 'raise', 'drop', or 'stable'")
    a = np.asarray(signal)
    if np.iscomplexobj(a):
        raise ValueError("signal must be real")
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("signal must be a finite 1-D array")
    span = (order - 1) * delay + 1
    if a.size < span:
        raise ValueError("signal is too short for order and delay")
    windows = np.lib.stride_tricks.sliding_window_view(a, span)[::stride, ::delay]
    sort_idx = np.argsort(windows, axis=1, kind="stable")
    ordered = np.take_along_axis(windows, sort_idx, axis=1)
    tied = np.any(np.diff(ordered, axis=1) == 0, axis=1)
    if ties == "raise" and tied.any():
        raise ValueError("tied ordinal windows; choose ties='drop' or 'stable' explicitly")
    ranks = np.argsort(sort_idx, axis=1)
    ids = np.zeros(len(windows), dtype=np.int64)
    for i in range(order - 1):
        ids += np.sum(ranks[:, i + 1:] < ranks[:, i, None], axis=1) * factorial(order - i - 1)
    if ties == "drop":
        ids[tied] = -1
    return ids


@dataclass(frozen=True)
class Scores:
    """Comparable scores in nats, another log base, or log-alphabet units.

    retained_mass and reference_mass are probabilities in all modes.
    n_observations is the number of aligned retained observations before masking
    symbol pairs; it is None when a joint table is supplied directly.
    """
    wsd: float
    wsmi: float
    smi: float
    retained_mass: float
    reference_mass: float
    n_observations: int | None = None


def _kl(r, s):
    # Generalised KL form: summands r log(r/s)-r+s are nonnegative.
    # Since r and s both sum to 1, this equals ordinary KL. The series avoids
    # cancellation close to independence without clipping negative wSMI.
    terms = s.copy()
    positive = r > 0
    rp, sp = r[positive], s[positive]
    if np.any(sp == 0):
        raise ValueError("reference underflow; rescale input or use higher precision")
    near = np.abs(rp - sp) < 1e-4 * sp
    v = np.empty_like(rp)
    t = (rp[near] - sp[near]) / sp[near]
    v[near] = sp[near] * t*t * (0.5 + t * (-1/6 + t * (1/12 + t * (-1/20 + t/30))))
    v[~near] = rp[~near] * (np.log(rp[~near]) - np.log(sp[~near])) - rp[~near] + sp[~near]
    terms[positive] = v
    return float(np.sum(terms))


def compare_joint(joint, mask=None, *, base=np.e, normalize=False):
    """Compute wSD, signed wSMI, and SMI from a joint count/probability table.

    ``joint`` is nonnegative finite mass and is normalised internally. A custom
    ``mask`` must be Boolean and have the same shape. Without a mask, the square
    table must contain 2!, 3!, ..., 6! lexicographic ordinal symbols.
    ``normalize=True`` divides all scores by log(n_symbols), for square tables.
    Otherwise ``base`` is the logarithm base (>1), default e (nats).
    Empty retained sets give wSD=0; their conditional divergence is undefined.
    """
    if np.iscomplexobj(joint):
        raise ValueError("joint must be real")
    p = np.asarray(joint, dtype=float)
    if p.ndim != 2 or 0 in p.shape or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("joint must be a nonempty finite nonnegative 2-D table")
    peak = p.max()
    if peak == 0:
        raise ValueError("joint must have positive total mass")
    p = p / peak
    p = p / p.sum()
    if mask is None:
        valid = [k for k in range(2, 7) if p.shape == (factorial(k), factorial(k))]
        if not valid:
            raise ValueError("default mask requires a square factorial ordinal alphabet")
        mask = ordinal_mask(valid[0])
    else:
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != p.shape:
            raise ValueError("mask must be Boolean and match joint.shape")
    if not np.isscalar(base) or not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one")
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be Boolean")
    scale = np.log(base)
    if normalize:
        if p.shape[0] != p.shape[1] or p.shape[0] < 2:
            raise ValueError("normalization requires equal alphabet sizes of at least two")
        scale = np.log(p.shape[0])
    q = np.outer(p.sum(axis=1), p.sum(axis=0))
    positive = p > 0
    if np.any(q[positive] == 0):
        raise ValueError("reference underflow; use higher precision")
    local = np.zeros_like(p)
    local[positive] = p[positive] * (np.log(p[positive]) - np.log(q[positive]))
    m, z = float(p[mask].sum()), float(q[mask].sum())
    divergence = 0.0 if m == 0 else m * _kl(p[mask] / m, q[mask] / z)
    return Scores(divergence / scale, float(local[mask].sum()) / scale,
                  _kl(p.ravel(), q.ravel()) / scale, m, z)


def compare_symbols(x, y, order=3, *, base=np.e, normalize=False):
    """Estimate scores from aligned ordinal IDs; omit pairs with a -1 sentinel.

    Marginals are recomputed on the same paired observations, before applying
    the ordinal mask. Pairwise tie omission can change the target population.
    """
    order = _integer(order, "order", 2, 6)
    n = factorial(order)
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape or x.size == 0:
        raise ValueError("symbols must be nonempty aligned 1-D arrays")
    for a in (x, y):
        if not np.issubdtype(a.dtype, np.integer) or np.any(a < -1) or np.any(a >= n):
            raise ValueError("symbols must be integer IDs in [-1, order!-1]")
    good = (x >= 0) & (y >= 0)
    if not good.any():
        raise ValueError("no valid paired ordinal observations")
    counts = np.bincount(x[good].astype(np.int64) * n + y[good].astype(np.int64), minlength=n*n).reshape(n, n)
    result = compare_joint(counts, base=base, normalize=normalize)
    return replace(result, n_observations=int(good.sum()))


def compare_signals(x, y, order=3, delay=1, stride=1, ties="raise", *, base=np.e, normalize=False):
    """Symbolize two equal-length signals and compare all three estimators.

    Filter, reference and epoch EEG explicitly before calling this function.
    Do not concatenate epochs: windows would span artificial boundaries.
    """
    if np.shape(x) != np.shape(y):
        raise ValueError("signals must have the same shape")
    sx = ordinal_patterns(x, order, delay, stride, ties)
    sy = ordinal_patterns(y, order, delay, stride, ties)
    return compare_symbols(sx, sy, order, base=base, normalize=normalize)


def wsd(x, y, **kwargs):
    """Convenience signal API; returns only weighted symbolic divergence."""
    return compare_signals(x, y, **kwargs).wsd
