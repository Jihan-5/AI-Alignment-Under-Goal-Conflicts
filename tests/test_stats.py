"""Statistics. Fixtures are the worked arithmetic in the Hypothetical Results
spec (pp.3-4, 8-9) - reproducing them checks the formulas, not any finding."""

import math

import numpy as np
import pytest

from override_lab.eval.stats import (
    clopper_pearson,
    cluster_bootstrap_diff,
    design_effect,
    hierarchical_bootstrap,
    holm,
    independent_z,
    mcnemar_exact,
    mcnemar_power,
    paired_difference,
    paired_table,
    required_n,
    wilson,
    zero_event_upper,
)


@pytest.mark.parametrize(
    "b,c,p",
    [(70, 20, 0.000000114), (40, 15, 0.00102), (8, 24, 0.00700), (60, 25, 0.000187)],
)
def test_mcnemar_matches_spec(b, c, p):
    assert mcnemar_exact(b, c) == pytest.approx(p, rel=0.01)


def test_mcnemar_edge_cases():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)


@pytest.mark.parametrize("k,n,lo,hi", [(275, 500, 0.506, 0.593), (310, 500, 0.577, 0.661)])
def test_wilson_matches_spec(k, n, lo, hi):
    a, b = wilson(k, n)
    assert a == pytest.approx(lo, abs=0.001) and b == pytest.approx(hi, abs=0.001)


def test_independent_z_reproduces_spec_and_is_not_the_paired_test():
    assert independent_z(0.30, 0.20, 500, 500) == pytest.approx(3.68, abs=0.01)
    assert independent_z(0.55, 0.62, 500, 500) == pytest.approx(2.25, abs=0.01)
    assert independent_z(0.55, 0.62, 300, 300) == pytest.approx(1.74, abs=0.01)


def test_zero_event_bound():
    assert zero_event_upper(500) == pytest.approx(0.006, abs=0.0002)
    assert zero_event_upper(500) == pytest.approx(1 - 0.05 ** (1 / 500))
    assert 2.9 / 500 < zero_event_upper(500) < 3.1 / 500


def test_clopper_pearson_contains_wilson_center_and_zero_case():
    lo, hi = clopper_pearson(0, 500)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 500), rel=1e-6)


def test_spec_paired_table_reconstructs_marginals():
    # 250 both, 60 proposed-only, 25 baseline-only, 165 neither -> 310 vs 275
    x = [1] * 250 + [1] * 60 + [0] * 25 + [0] * 165
    y = [1] * 250 + [0] * 60 + [1] * 25 + [0] * 165
    t = paired_table(x, y)
    assert (t.both, t.first_only, t.second_only, t.neither) == (250, 60, 25, 165)
    assert sum(x) == 310 and sum(y) == 275


def test_paired_difference_equals_classic_se_without_clusters():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 400)
    y = rng.integers(0, 2, 400)
    pd = paired_difference(x, y, use_t=False)
    d = x - y
    assert pd.diff == pytest.approx(d.mean())
    assert pd.se == pytest.approx(d.std(ddof=1) / math.sqrt(len(d)), rel=1e-9)


def test_clustering_widens_interval_for_correlated_variants():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2, 100)
    x = np.repeat(base, 5)  # five identical variants per root
    y = np.repeat(rng.integers(0, 2, 100), 5)
    clusters = np.repeat(np.arange(100), 5).astype(str)
    naive = paired_difference(x, y, use_t=False)
    clustered = paired_difference(x, y, clusters, use_t=False)
    assert clustered.n_clusters == 100
    assert clustered.se > 2 * naive.se


def test_cluster_bootstrap_close_to_analytic():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 2, 600)
    y = (rng.random(600) < 0.35).astype(int)
    pd = paired_difference(x, y, use_t=False)
    _, lo, hi = cluster_bootstrap_diff(x, y, n_boot=4000, seed=3)
    assert lo == pytest.approx(pd.lo, abs=0.015) and hi == pytest.approx(pd.hi, abs=0.015)


def test_hierarchical_bootstrap_shape_and_center():
    D = np.array([[1, 0, 0, -1, 1, 0] * 20, [0, 0, 1, 0, 1, 0] * 20, [1, 1, 0, 0, 0, 0] * 20], dtype=float)
    m, lo, hi = hierarchical_bootstrap(D, n_boot=2000)
    assert m == pytest.approx(D.mean()) and lo < m < hi


def test_holm():
    r = holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert r["a"]["reject"] and not r["c"]["reject"] and not r["d"]["reject"]
    assert r["b"]["p_holm"] == pytest.approx(0.06)
    assert not r["b"]["reject"]  # 0.02 > 0.05/3
    assert all(r[k]["p_holm"] >= r[k]["p"] for k in r)


def test_power_monotone_and_required_n():
    assert mcnemar_power(100, 0.14, 0.04) < mcnemar_power(400, 0.14, 0.04)
    n = required_n(0.14, 0.04)
    assert mcnemar_power(n, 0.14, 0.04) >= 0.8 > mcnemar_power(n - 5, 0.14, 0.04)
    assert mcnemar_power(500, 0.05, 0.05) < 0.1  # no effect -> about alpha


def test_design_effect():
    assert design_effect([1] * 50, 0.3) == pytest.approx(1.0)
    assert design_effect([5] * 100, 0.2) == pytest.approx(1.8)
