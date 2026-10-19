"""Paired comparisons between conditions evaluated on the same roots.

A comparison is always computed from *saved traces*, never from summary
rates: it needs to know which roots changed (plan p.10 "the paired confidence
interval depends on which specific episodes changed, not just the two
totals"). Output for every contrast: both rates with denominators, the
paired 2x2 table, exact McNemar p, the cluster-robust paired difference with
interval, and a cluster-bootstrap interval. Multi-seed arms additionally get
per-seed results, the seed summary and a hierarchical bootstrap.

Trace files follow one naming convention (``trace_path``) so a contrast in
configs/prereg.yaml can be resolved to files without ambiguity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import EPISODE_METRICS, EpisodeOutcome, episode_outcome
from .stats import (
    cluster_bootstrap_diff,
    hierarchical_bootstrap,
    holm,
    mcnemar_exact,
    paired_difference,
    paired_table,
    seed_summary,
    wilson,
)


def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def trace_path(root: str | Path, model_id: str, stratum: str, system_prompt: str, guard_label: str, decoding: str) -> Path:
    return Path(root) / slug(model_id) / stratum / f"{slug(system_prompt)}__{slug(guard_label)}__{slug(decoding)}.jsonl"


def save_traces(path: str | Path, traces: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")


def load_traces(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _binary(outcomes: list[EpisodeOutcome], metric: str) -> dict[str, tuple[int, str]]:
    num_f, elig_f = EPISODE_METRICS[metric]
    return {o.root_id: (int(num_f(o)), o.family) for o in outcomes if elig_f(o)}


def paired_compare(
    traces_a: list[dict[str, Any]],
    traces_b: list[dict[str, Any]],
    metric: str,
    cluster: str = "root",  # root | family
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    oa = _binary([episode_outcome(t) for t in traces_a], metric)
    ob = _binary([episode_outcome(t) for t in traces_b], metric)
    common = sorted(set(oa) & set(ob))
    mismatch = sorted(set(oa) ^ set(ob))
    if not common:
        return {"metric": metric, "n": 0, "note": "no common eligible roots"}
    x = [oa[r][0] for r in common]
    y = [ob[r][0] for r in common]
    clusters = common if cluster == "root" else [oa[r][1] for r in common]
    tab = paired_table(x, y)
    pd = paired_difference(x, y, clusters)
    _, blo, bhi = cluster_bootstrap_diff(x, y, clusters, n_boot=n_boot, seed=seed)
    ka, kb, n = sum(x), sum(y), len(common)
    return {
        "metric": metric,
        "label_a": traces_a[0]["condition"]["label"] if traces_a else None,
        "label_b": traces_b[0]["condition"]["label"] if traces_b else None,
        "n": n,
        "n_clusters": pd.n_clusters,
        "cluster": cluster,
        "a": {"num": ka, "den": n, "rate": ka / n, "wilson95": wilson(ka, n)},
        "b": {"num": kb, "den": n, "rate": kb / n, "wilson95": wilson(kb, n)},
        "diff_a_minus_b": pd.diff,
        "se": pd.se,
        "ci95_cluster": [pd.lo, pd.hi],
        "ci95_bootstrap": [blo, bhi],
        "ci_method": pd.method,
        "table": {"both": tab.both, "a_only": tab.first_only, "b_only": tab.second_only, "neither": tab.neither},
        "mcnemar_exact_p": mcnemar_exact(tab.first_only, tab.second_only),
        "eligibility_mismatch_roots": mismatch,
    }


def multi_seed_compare(
    traces_a_by_seed: dict[int, list[dict[str, Any]]],
    traces_b_by_seed: dict[int, list[dict[str, Any]]] | list[dict[str, Any]],
    metric: str,
    n_boot: int = 5000,
) -> dict[str, Any]:
    """Arm A trained with several seeds vs arm B (several seeds, or one fixed
    condition such as A0/M2 which has no training seed)."""
    seeds = sorted(traces_a_by_seed)
    per_seed = {}
    rows = []
    roots_ref: list[str] | None = None
    for s in seeds:
        tb = traces_b_by_seed[s] if isinstance(traces_b_by_seed, dict) else traces_b_by_seed
        res = paired_compare(traces_a_by_seed[s], tb, metric, n_boot=n_boot, seed=s)
        per_seed[s] = res
        oa = _binary([episode_outcome(t) for t in traces_a_by_seed[s]], metric)
        ob = _binary([episode_outcome(t) for t in tb], metric)
        common = sorted(set(oa) & set(ob))
        roots_ref = common if roots_ref is None else [r for r in roots_ref if r in set(common)]
        rows.append((oa, ob))
    assert roots_ref is not None
    D = np.array([[oa[r][0] - ob[r][0] for r in roots_ref] for oa, ob in rows], dtype=float)
    mean, lo, hi = hierarchical_bootstrap(D, roots_ref, n_boot=n_boot)
    return {
        "metric": metric,
        "seeds": seeds,
        "rate_a_by_seed": {s: per_seed[s]["a"]["rate"] for s in seeds if per_seed[s].get("n")},
        "rate_b_by_seed": {s: per_seed[s]["b"]["rate"] for s in seeds if per_seed[s].get("n")},
        "n_by_seed": {s: per_seed[s].get("n", 0) for s in seeds},
        "per_seed": per_seed,
        "seed_summary_of_diff": seed_summary([per_seed[s]["diff_a_minus_b"] for s in seeds]),
        "hierarchical_bootstrap": {"mean": mean, "ci95": [lo, hi], "n_roots": len(roots_ref)},
    }


def run_contrasts(contrasts: list[dict[str, Any]], resolve) -> dict[str, Any]:
    """``contrasts`` from configs/prereg.yaml; ``resolve(side, stratum) -> traces``
    (or dict seed -> traces). The first contrast with ``role: primary`` is the
    single primary test; all ``role: secondary`` contrasts are Holm-corrected
    together. ``role: descriptive`` contrasts are reported without p-values."""
    results: dict[str, Any] = {}
    secondary_p: dict[str, float] = {}
    for c in contrasts:
        ta, tb = resolve(c["a"], c["stratum"]), resolve(c["b"], c["stratum"])
        if isinstance(ta, dict):
            r = multi_seed_compare(ta, tb, c["metric"])
            p = float(np.median([v["mcnemar_exact_p"] for v in r["per_seed"].values()]))
            r["p_summary"] = {"median_per_seed_mcnemar_p": p}
        else:
            r = paired_compare(ta, tb, c["metric"], cluster=c.get("cluster", "root"))
            p = r["mcnemar_exact_p"]
        r["role"] = c.get("role", "secondary")
        r["spec"] = c
        results[c["name"]] = r
        if r["role"] == "secondary":
            secondary_p[c["name"]] = p
    results["_holm_secondary"] = holm(secondary_p) if secondary_p else {}
    return results
