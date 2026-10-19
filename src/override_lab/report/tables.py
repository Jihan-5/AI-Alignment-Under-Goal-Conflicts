"""Markdown result tables. Every cell is "num/den = rate [Wilson 95%]" so each
number in a report maps to counts in a named trace file (plan p.12 "every
displayed number maps to a saved run")."""

from __future__ import annotations

from typing import Any


def cell(r: dict[str, Any] | None) -> str:
    if not r or not r.get("den"):
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['num']}/{r['den']} = {100 * r['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


MAIN_METRICS = [
    ("attempted_violation", "Attempted violation"),
    ("executed_violation", "Executed violation"),
    ("handoff_compliance", "Handoff compliance"),
    ("oversight_compliance", "Oversight compliance"),
    ("task_success", "Task success"),
    ("over_refusal", "Over-refusal"),
    ("inappropriate_obedience", "Followed unverified"),
    ("invalid_episode", "Invalid-format episodes"),
]


def condition_table(rows: dict[str, dict[str, Any]], metrics: list[tuple[str, str]] = MAIN_METRICS, title: str = "") -> str:
    """rows: condition label -> aggregate() dict."""
    head = "| Condition | " + " | ".join(m[1] for m in metrics) + " |"
    sep = "|---|" + "---|" * len(metrics)
    lines = ([f"#### {title}", ""] if title else []) + [head, sep]
    for label, agg in rows.items():
        safe = label.replace("|", "\\|")  # pipes break markdown tables even inside code spans
        lines.append(f"| `{safe}` | " + " | ".join(cell(agg.get(m[0])) for m in metrics) + " |")
    return "\n".join(lines)


def contrast_table(results: dict[str, Any]) -> str:
    holm = results.get("_holm_secondary", {})
    lines = [
        "| Contrast | Role | Metric | Stratum | A | B | A - B [95% CI] | McNemar p | Holm p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        spec = r["spec"]
        if "per_seed" in r:
            hb = r["hierarchical_bootstrap"]
            fmt = lambda d: ", ".join(f"s{k}: {100 * v:.1f}%" for k, v in d.items()) or "n/a"
            a_txt = fmt(r.get("rate_a_by_seed", {}))
            b_txt = fmt(r.get("rate_b_by_seed", {}))
            d_txt = f"{100 * hb['mean']:+.1f} pp [{100 * hb['ci95'][0]:+.1f}, {100 * hb['ci95'][1]:+.1f}] (hier. bootstrap)"
            p_txt = f"{r['p_summary']['median_per_seed_mcnemar_p']:.3g} (median/seed)"
        elif r.get("n"):
            a_txt = f"{r['a']['num']}/{r['a']['den']}"
            b_txt = f"{r['b']['num']}/{r['b']['den']}"
            lo, hi = r["ci95_cluster"]
            d_txt = f"{100 * r['diff_a_minus_b']:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            p_txt = f"{r['mcnemar_exact_p']:.3g}"
        else:
            a_txt = b_txt = d_txt = p_txt = "n/a"
        h = holm.get(name)
        h_txt = f"{h['p_holm']:.3g}{' *' if h['reject'] else ''}" if h else "-"
        if r["role"] == "descriptive":
            p_txt, h_txt = "(descriptive)", "-"
        lines.append(f"| {name} | {r['role']} | {spec['metric']} | {spec['stratum']} | {a_txt} | {b_txt} | {d_txt} | {p_txt} | {h_txt} |")
    return "\n".join(lines)


def gate_table(name: str, gate: dict[str, Any]) -> str:
    lines = [f"#### {name}: {'PASSED' if gate['passed'] else 'FAILED'}", "", "| Check | Value | Rule | OK |", "|---|---|---|---|"]
    for k, c in gate["checks"].items():
        v = "n/a" if c["value"] is None else f"{c['value']:.3f}"
        lines.append(f"| {k} | {v} | {c['op']} {c['threshold']} | {'yes' if c['ok'] else 'no'} |")
    return "\n".join(lines)
