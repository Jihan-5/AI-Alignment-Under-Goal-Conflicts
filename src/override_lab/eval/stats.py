"""Statistics for paired, clustered, binary evaluation outcomes.

Readings: S04 (Miller, "Adding Error Bars to Evals": clustered and paired
standard errors, power), P1 (Wasserman ch. 8 bootstrap, ch. 10 multiple
testing), X8 (Dietterich: exact McNemar on discordant pairs).

Design rules enforced here:

* Two arms evaluated on the same roots are compared *paired*: the interval
  depends on which roots changed, not on the two marginal rates.
* The unit of independence is the cluster (root by default; scenario family or
  template if variants of a root are evaluated). Normal intervals use the
  CR1-style small-sample factor G/(G-1); with few clusters use the t quantile
  or the cluster bootstrap.
* Training-seed variation is a *separate* variance source (seeds 11/23/47):
  ``seed_summary`` and ``hierarchical_bootstrap`` keep it visible instead of
  pooling repeated samples as independent roots.
* ``independent_z`` exists only to reproduce the illustrative numbers in the
  spec and to show why they are not the right test.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------- single rates
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    z = sps.norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    """One-sided exact binomial upper bound with 0 failures in n iid trials:
    1 - alpha**(1/n) (~3/n). Applies to that sampling model only - not to new
    task families, correlated paraphrases, or real-world risk (plan p.10)."""
    return 1 - alpha ** (1 / n)


# ------------------------------------------------------------- paired binary
@dataclass(frozen=True)
class PairedTable:
    both: int
    first_only: int
    second_only: int
    neither: int

    @property
    def n(self) -> int:
        return self.both + self.first_only + self.second_only + self.neither


def paired_table(x: Sequence[int], y: Sequence[int]) -> PairedTable:
    a = np.asarray(x, dtype=int)
    b = np.asarray(y, dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    return PairedTable(
        int(((a == 1) & (b == 1)).sum()),
        int(((a == 1) & (b == 0)).sum()),
        int(((a == 0) & (b == 1)).sum()),
        int(((a == 0) & (b == 0)).sum()),
    )


def mcnemar_exact(first_only: int, second_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant cells."""
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = min(first_only, second_only)
    return float(min(1.0, 2 * sps.binom.cdf(k, n, 0.5)))


@dataclass(frozen=True)
class PairedDiff:
    diff: float  # mean(x) - mean(y)
    se: float
    lo: float
    hi: float
    n: int
    n_clusters: int
    method: str


def paired_difference(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence[str] | None = None,
    conf: float = 0.95,
    use_t: bool | None = None,
) -> PairedDiff:
    """Mean paired difference with a cluster-robust SE (Miller 2024, eq. for
    clustered SE applied to d_i = x_i - y_i)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("x and y must be aligned 1-D arrays")
    n = len(a)
    d = a - b
    dbar = float(d.mean())
    cl = list(range(n)) if clusters is None else list(clusters)
    sums: dict = defaultdict(float)
    for di, c in zip(d - dbar, cl):
        sums[c] += di
    g = len(sums)
    if g < 2:
        return PairedDiff(dbar, float("nan"), float("nan"), float("nan"), n, g, "undefined")
    var = (g / (g - 1)) * sum(s * s for s in sums.values()) / (n * n)
    se = math.sqrt(var)
    if use_t is None:
        use_t = g < 50
    q = sps.t.ppf(0.5 + conf / 2, g - 1) if use_t else sps.norm.ppf(0.5 + conf / 2)
    return PairedDiff(dbar, se, dbar - q * se, dbar + q * se, n, g, f"cluster-{'t' if use_t else 'z'}")


def cluster_bootstrap_diff(
    x: Sequence[float],
    y: Sequence[float],
    clusters: Sequence[str] | None = None,
    n_boot: int = 10000,
    conf: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the paired mean difference, resampling whole clusters."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    d = a - b
    cl = np.arange(len(d)) if clusters is None else np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(d.mean()), float(lo), float(hi)


def hierarchical_bootstrap(
    diff: np.ndarray,
    clusters: Sequence[str] | None = None,
    n_boot: int = 10000,
    conf: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """diff has shape [n_seeds, n_roots] (per-root paired differences for each
    training seed). Each replicate resamples seeds, then clusters. With only 3
    seeds the seed level is crude; always report per-seed values alongside."""
    D = np.asarray(diff, dtype=float)
    s, n = D.shape
    cl = np.arange(n) if clusters is None else np.asarray(clusters)
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, s, size=s)
        groups = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[g] for g in groups])
        out[b] = D[np.ix_(seeds, idx)].mean()
    lo, hi = np.quantile(out, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(D.mean()), float(lo), float(hi)


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {
        "n_seeds": int(len(v)),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
        "min": float(v.min()),
        "max": float(v.max()),
    }


# --------------------------------------------------------- multiple testing
def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni step-down. Returns adjusted p and reject flag per test."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    stop = False
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        reject = (not stop) and p <= alpha / (m - i)
        if not reject:
            stop = True
        out[name] = {"p": p, "p_holm": running, "reject": reject}
    return out


# ------------------------------------------------------------- illustration
def independent_z(p1: float, p2: float, n1: int, n2: int) -> float:
    """Unpooled independent-proportions z. NOT the right test for paired arms;
    reproduces the spec's illustrative 3.68 / 2.25 / 1.74 values."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return abs(p1 - p2) / se


# -------------------------------------------------------------------- power
def mcnemar_power(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """Exact power of the two-sided exact McNemar test with n paired roots,
    where p10 / p01 are the probabilities of the two discordant outcomes.
    Sums over the binomial distribution of the number of discordant pairs."""
    pd = p10 + p01
    if pd <= 0:
        return 0.0
    q = p10 / pd
    power = 0.0
    for m in range(0, n + 1):
        w = sps.binom.pmf(m, n, pd)
        if w < 1e-12 or m == 0:
            continue
        # rejection region for k ~ Bin(m, 0.5): smallest c with 2*P(K<=c) <= alpha, two-sided
        c = int(sps.binom.ppf(alpha / 2, m, 0.5))
        while c >= 0 and 2 * sps.binom.cdf(c, m, 0.5) > alpha:
            c -= 1
        if c < 0:
            continue
        rej = sps.binom.cdf(c, m, q) + sps.binom.sf(m - c - 1, m, q)
        power += w * rej
    return float(power)


def required_n(p10: float, p01: float, power: float = 0.8, alpha: float = 0.05, n_max: int = 20000) -> int | None:
    lo, hi = 10, n_max
    if mcnemar_power(hi, p10, p01, alpha) < power:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_power(mid, p10, p01, alpha) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


def design_effect(cluster_sizes: Sequence[int], icc: float) -> float:
    """Kish design effect 1 + (m_bar - 1) * ICC for unequal cluster sizes
    (m_bar = sum m^2 / sum m). Divide n by it to get the effective sample size."""
    m = np.asarray(cluster_sizes, dtype=float)
    mbar = float((m * m).sum() / m.sum())
    return 1 + (mbar - 1) * icc
