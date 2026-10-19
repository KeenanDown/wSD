from math import factorial

import numpy as np
import pytest

from wsd import compare_joint, compare_signals, compare_symbols, ordinal_mask, ordinal_patterns, patterns, wsd


def counterexample():
    c = np.ones((6, 6), dtype=int)
    c[~ordinal_mask(3)] = 10
    c[0, 2] += 1
    c[1, 3] += 1
    c[0, 3] -= 1
    c[1, 2] -= 1
    return c


def test_exact_counterexample():
    c = counterexample()
    r = compare_joint(c)
    assert c.sum() == 144
    np.testing.assert_array_equal(c.sum(axis=0), np.full(6, 24))
    np.testing.assert_array_equal(c.sum(axis=1), np.full(6, 24))
    assert r.retained_mass == pytest.approx(1/6)
    assert r.reference_mass == pytest.approx(2/3)
    assert r.wsd == pytest.approx(np.log(2)/36)
    assert r.wsmi == pytest.approx(-11*np.log(2)/36)
    assert r.wsmi < 0 < r.wsd < r.smi


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_all_ordinal_codes_and_amplitude_reversal(order):
    p = patterns(order)
    ids = ordinal_patterns(p.ravel(), order, stride=order)
    np.testing.assert_array_equal(ids, np.arange(factorial(order)))
    reverse = ordinal_patterns(-p.ravel(), order, stride=order)
    np.testing.assert_array_equal(reverse, len(p)-1-ids)
    mask = ordinal_mask(order)
    assert mask.sum() == len(p) * (len(p)-2)
    assert not mask[ids, ids].any()
    assert not mask[ids, reverse].any()


def test_counterexample_realised_as_signals():
    c = counterexample()
    ix, iy = np.nonzero(c)
    sx, sy = np.repeat(ix, c[ix, iy]), np.repeat(iy, c[ix, iy])
    p = patterns(3)
    r = compare_signals(p[sx].ravel(), p[sy].ravel(), stride=3)
    assert r.wsd == pytest.approx(np.log(2)/36)
    assert r.wsmi == pytest.approx(-11*np.log(2)/36)
    assert r.n_observations == 144


@pytest.mark.parametrize("seed", range(12))
def test_partition_identity_and_bounds_random_tables(seed):
    rng = np.random.default_rng(seed)
    for n in (3, 6, 12):
        p = rng.gamma(0.2, size=(n, n))
        p[rng.random(p.shape) < 0.35] = 0
        p /= p.sum()
        mask = rng.random(p.shape) < 0.6
        r = compare_joint(p, mask)
        rc = compare_joint(p, ~mask)
        q = np.outer(p.sum(1), p.sum(0))
        nz = p > 0
        oracle = np.sum(p[nz] * np.log(p[nz] / q[nz]))
        assert r.smi == pytest.approx(oracle, abs=2e-14)
        assert 0 <= r.wsd <= r.smi + 2e-14
        m, z = r.retained_mass, r.reference_mass
        bernoulli = 0.0
        if m > 0:
            bernoulli += m * np.log(m/z)
            assert r.wsd == pytest.approx(r.wsmi + m*np.log(z/m), abs=2e-14)
            rs, qs = p[mask]/m, q[mask]/z
            good = rs > 0
            assert r.wsd == pytest.approx(m*np.sum(rs[good]*np.log(rs[good]/qs[good])), abs=2e-14)
        if m < 1:
            bernoulli += (1-m) * np.log((1-m)/(1-z))
        assert r.smi == pytest.approx(r.wsd + rc.wsd + bernoulli, abs=2e-14)
        assert compare_joint(p.T, mask.T).wsd == pytest.approx(r.wsd)


def test_independence_and_nonseparation():
    a, b = np.arange(1, 7), np.arange(6, 0, -1)
    r = compare_joint(np.outer(a, b))
    assert r.wsd < 1e-28
    c = np.ones((6, 6))
    c[~ordinal_mask(3)] = 10
    r = compare_joint(c)
    assert r.wsd == pytest.approx(0, abs=1e-28)
    assert r.smi > 0
    assert r.wsmi < 0


def test_dependence_confined_to_masked_cells_is_intentionally_ignored():
    p = np.ones((6, 6))/36 + (np.eye(6) - np.fliplr(np.eye(6)))/72
    r = compare_joint(p)
    assert r.retained_mass == pytest.approx(r.reference_mass)
    assert r.retained_mass == pytest.approx(2/3)
    assert r.wsd < 1e-28
    assert r.smi == pytest.approx(np.log(1.5)/4 + np.log(.5)/12)
    assert r.smi > 0


def test_sparse_empty_and_full_masks():
    p = np.diag(np.arange(1, 7))
    assert compare_joint(p).wsd == 0
    assert compare_joint(p).wsmi == 0
    full = compare_joint(p, np.ones((6, 6), dtype=bool))
    assert full.wsd == pytest.approx(full.smi)
    assert full.wsd == pytest.approx(full.wsmi)
    p = np.zeros((6, 6))
    p[0, 0] = 1
    assert compare_joint(p).wsd == compare_joint(p).smi == 0
    assert compare_joint(np.ones((2, 2))).wsd == 0


def test_units_and_scaling():
    c = counterexample()
    r = compare_joint(c)
    assert compare_joint(c, base=2).wsd == pytest.approx(r.wsd/np.log(2))
    assert compare_joint(c, normalize=True).wsmi == pytest.approx(r.wsmi/np.log(6))
    assert compare_joint(c*1e200).wsd == pytest.approx(r.wsd)


def test_tie_policy_alignment_and_finite_inputs():
    x = [0, 0, 1, 2, 0, 1]
    y = [0, 1, 2, 0, 0, 1]
    with pytest.raises(ValueError, match="tied"):
        ordinal_patterns(x)
    np.testing.assert_array_equal(ordinal_patterns([1, 1, 1], ties="stable"), [0])
    sx, sy = ordinal_patterns(x, ties="drop"), ordinal_patterns(y, ties="drop")
    assert compare_signals(x, y, ties="drop").n_observations == np.sum((sx >= 0) & (sy >= 0))
    with pytest.raises(ValueError, match="no valid"):
        compare_signals([0, 0, 0], [1, 2, 3], ties="drop")
    for a in ([1, 2, np.nan], [1, 2, np.inf], [1j, 2, 3]):
        with pytest.raises(ValueError):
            ordinal_patterns(a)


def test_delay_stride_invariance_and_convenience():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(2, 200))
    expected = [ordinal_patterns(x[i:i+7:3])[0] for i in range(0, 194, 2)]
    np.testing.assert_array_equal(ordinal_patterns(x, delay=3, stride=2), expected)
    r = compare_signals(x, y)
    assert wsd(x, y) == r.wsd
    assert compare_signals(4*x+3, np.exp(y)).wsd == pytest.approx(r.wsd)
    assert compare_signals(-x, y).wsd == pytest.approx(r.wsd)


@pytest.mark.parametrize("p", [[], [[0, 0], [0, 0]], [[-1, 2]], [[np.nan]], [[np.inf]], [[1j]]])
def test_invalid_joint(p):
    with pytest.raises(ValueError):
        compare_joint(p)


@pytest.mark.parametrize("kwargs", [{"base": 1}, {"base": 0}, {"base": np.nan}, {"normalize": 1}, {"mask": [[1]*6]*6}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        compare_joint(counterexample(), **kwargs)


@pytest.mark.parametrize("kwargs", [{"order": 1}, {"order": 7}, {"order": 3.0}, {"delay": 0}, {"stride": True}, {"ties": "random"}])
def test_invalid_symbolization_options(kwargs):
    with pytest.raises(ValueError):
        ordinal_patterns(np.arange(20), **kwargs)


def test_invalid_symbol_sequences():
    for x, y in [([], []), ([0], [0, 1]), ([0.0], [1.0]), ([6], [0]), ([-2], [0])]:
        with pytest.raises(ValueError):
            compare_symbols(x, y)
