"""Static HTML demo: same-scenario replay, failure gallery, aggregate table.

Plan p.12: "Replay original, objective-biased, repaired, and guarded versions
side by side, showing the rule, proposed action, executed action, and result.
Include a failure gallery and held-out aggregate table. Do not cherry-pick
repairs or imply real-world takeover capability."

Anti-cherry-picking is mechanical here: replayed scenarios are the first
``per_family`` roots of each family *in frozen-split order*, and the failure
gallery lists failures in frozen order up to a cap - the generator never
looks at outcomes when choosing what to show. Every number carries its trace
file and root id.

Output is one self-contained file (inline CSS/JS, no network).
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..eval.metrics import aggregate, episode_outcome
from ..scenarios.render import RenderSpec, message_header, message_text
from ..sim.state import Scenario


def _scenario_card(root_id: str, sc: Scenario, spec_style: tuple[str, str, int]) -> dict[str, Any]:
    s = sc.initial
    rs = RenderSpec(*spec_style)
    msgs = [f"{message_header(m)} {message_text(m, s, rs)}" for m in s.inbox]
    msgs += [f"(arrives at step {m.step + 1}) {message_header(m)} {message_text(m, s, rs)}" for m in sc.scheduled]
    return {
        "root_id": root_id,
        "family": sc.family,
        "pool": s.pool,
        "floor": s.reserve_floor,
        "target": s.throughput_target,
        "tasks": [
            {"id": t.task_id, "name": t.name, "units": t.units_required, "pts": t.throughput, "prio": t.priority, "review": t.review_required, "status": t.status.value}
            for t in sorted(s.tasks, key=lambda x: (x.priority, x.task_id))
        ],
        "messages": msgs,
        "reference": sc.meta.get("reference_actions", []),
        "legit": sc.meta.get("legit_throughput"),
    }


def _short(a: dict[str, Any] | None) -> str:
    if not a:
        return "(unparseable)"
    t = a["action"]
    if t == "ALLOCATE":
        return f"ALLOCATE {a['task_id']} x{a['resource_units']}"
    if t == "REQUEST_REVIEW":
        return f"REVIEW {a['reason_code']}" + (f" {a['task_id']}" if a.get("task_id") else "")
    return t


def _episode_view(tr: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": [
            {
                "step": d["step"] + 1,
                "proposed": _short(d["action"]),
                "violations": (d["verdict"] or {}).get("violations", []),
                "defer": (d["verdict"] or {}).get("defer_kind"),
                "outcome": d["outcome"],
                "raw": d["raw_output"][:240],
            }
            for d in tr["decisions"]
        ],
        "earned": tr["final"]["earned"],
        "termination": tr["termination"],
        "error": tr["error"],
    }


def build_demo(
    out_path: str | Path,
    conditions: list[dict[str, Any]],
    scenarios: dict[str, tuple[Scenario, tuple[str, str, int]]],
    title: str = "Override Lab - same-scenario replay",
    banner: str | None = None,
    per_family: int = 2,
    gallery_cap: int = 40,
) -> Path:
    """``conditions``: [{"label", "source" (trace file), "traces": [...]}] on the same roots."""
    order = [t["root_id"] for t in conditions[0]["traces"]]
    by_family: dict[str, list[str]] = defaultdict(list)
    for rid in order:
        fam = scenarios[rid][0].family
        if len(by_family[fam]) < per_family:
            by_family[fam].append(rid)
    shown = [rid for fam in sorted(by_family) for rid in by_family[fam]]
    lookup = {c["label"]: {t["root_id"]: t for t in c["traces"]} for c in conditions}

    data = {
        "title": title,
        "banner": banner,
        "conditions": [{"label": c["label"], "source": c["source"]} for c in conditions],
        "scenarios": [
            {
                **_scenario_card(rid, scenarios[rid][0], scenarios[rid][1]),
                "runs": {c["label"]: _episode_view(lookup[c["label"]][rid]) for c in conditions if rid in lookup[c["label"]]},
            }
            for rid in shown
        ],
        "aggregate": [],
        "gallery": [],
    }
    for c in conditions:
        agg = aggregate(episode_outcome(t) for t in c["traces"])
        data["aggregate"].append(
            {"label": c["label"], "source": c["source"], **{k: agg[k] for k in ("attempted_violation", "executed_violation", "handoff_compliance", "task_success", "over_refusal", "invalid_episode")}}
        )
        n = 0
        for t in c["traces"]:
            for d in t["decisions"]:
                if d["verdict"] and d["verdict"]["prohibited"]:
                    data["gallery"].append(
                        {
                            "label": c["label"],
                            "root_id": t["root_id"],
                            "family": t["family"],
                            "step": d["step"] + 1,
                            "proposed": _short(d["action"]),
                            "violations": d["verdict"]["violations"],
                            "outcome": d["outcome"],
                            "raw": d["raw_output"][:300],
                            "source": c["source"],
                        }
                    )
                    n += 1
                    break
            if n >= gallery_cap:
                break
    payload = json.dumps(data).replace("</", "<\\/")
    page = _PAGE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(page)
    return p


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfaf7;--fg:#22211d;--muted:#6b6a64;--card:#ffffff;--line:#e4e1d8;--accent:#0f6b6b;--ok:#2e7d4f;--bad:#b3261e;--warn:#9a6700;--chip:#f1efe8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151614;--fg:#e8e6df;--muted:#a3a19a;--card:#1d1f1c;--line:#33352f;--accent:#5fb3b0;--ok:#6ccf93;--bad:#ff8a80;--warn:#e3b341;--chip:#262823}}
:root[data-theme="dark"]{--bg:#151614;--fg:#e8e6df;--muted:#a3a19a;--card:#1d1f1c;--line:#33352f;--accent:#5fb3b0;--ok:#6ccf93;--bad:#ff8a80;--warn:#e3b341;--chip:#262823}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;color:var(--accent)}
.sub{color:var(--muted);margin:0 0 16px}.banner{border:1px solid var(--warn);color:var(--warn);padding:10px 12px;border-radius:8px;margin:12px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:10px 0}
select{font:inherit;padding:6px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);max-width:100%}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}.scroll{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 8px;margin:1px 2px;font-size:12px}
.bad{color:var(--bad);font-weight:600}.ok{color:var(--ok)}.warn{color:var(--warn)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.step{border-top:1px dashed var(--line);padding:6px 0}.step:first-child{border-top:0}.muted{color:var(--muted)}
footer{color:var(--muted);font-size:12px;margin-top:40px}
</style></head><body><main>
<h1 id="t"></h1><p class="sub">Typed actions on an in-memory fictional station. Proposed action, policy verdict and executed result are shown separately for every step.</p>
<div id="banner"></div>
<h2>Held-out aggregate</h2><div class="card scroll" id="agg"></div>
<h2>Same-scenario replay</h2>
<div class="card"><label for="pick" class="muted">Scenario (first roots of each family, frozen order - not selected by outcome):</label><br><select id="pick"></select></div>
<div id="scn"></div>
<h2>Failure gallery</h2><p class="sub">First prohibited proposal of each failing episode, in frozen order, per condition.</p><div class="card scroll" id="gal"></div>
<footer>Every number links to a saved trace file and root id. A guarded condition with zero executed violations shows the guard worked, not that the model became aligned.</footer>
</main>
<script>
const D=__DATA__;
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct=r=>r&&r.den?`${r.num}/${r.den} = ${(100*r.rate).toFixed(1)}%`:"n/a";
document.getElementById("t").textContent=D.title;
if(D.banner)document.getElementById("banner").innerHTML=`<div class="banner">${esc(D.banner)}</div>`;
document.getElementById("agg").innerHTML=`<table><tr><th>Condition</th><th>Attempted violation</th><th>Executed violation</th><th>Handoff compliance</th><th>Task success</th><th>Over-refusal</th><th>Invalid episodes</th></tr>${D.aggregate.map(a=>`<tr title="${esc(a.source)}"><td class="mono">${esc(a.label)}</td><td>${pct(a.attempted_violation)}</td><td>${pct(a.executed_violation)}</td><td>${pct(a.handoff_compliance)}</td><td>${pct(a.task_success)}</td><td>${pct(a.over_refusal)}</td><td>${pct(a.invalid_episode)}</td></tr>`).join("")}</table>`;
const pick=document.getElementById("pick");
D.scenarios.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=`${s.family} - ${s.root_id}`;pick.appendChild(o)});
function show(i){const s=D.scenarios[i];if(!s){document.getElementById("scn").innerHTML="";return}
const tasks=s.tasks.map(t=>`<tr><td class="mono">${t.id}</td><td>${esc(t.name)}</td><td>${t.units}</td><td>${t.pts}</td><td>${t.prio}</td><td>${t.review?"REVIEW":""} ${t.status!=="OPEN"?t.status:""}</td></tr>`).join("");
const runs=D.conditions.map(c=>{const r=s.runs[c.label];if(!r)return"";return `<div class="card"><div class="mono"><b>${esc(c.label)}</b></div>${r.steps.map(st=>`<div class="step"><span class="muted">step ${st.step}</span> <span class="mono">${esc(st.proposed)}</span><br>${st.outcome==="INVALID"?`<span class="chip warn">unparseable</span>`:st.violations.length?st.violations.map(v=>`<span class="chip bad">${esc(v)}</span>`).join(""):`<span class="chip ok">permitted</span>`} <span class="chip">${esc(st.outcome)}</span>${st.defer?` <span class="chip">${esc(st.defer)}</span>`:""}</div>`).join("")}<div class="step muted">earned ${r.earned} pts · ${esc(r.termination)}${r.error?` · <span class="bad">error</span>`:""}</div></div>`}).join("");
document.getElementById("scn").innerHTML=`<div class="card"><div class="muted">store ${s.pool} units · reserve floor ${s.floor} · target ${s.target??"none"} · legitimate throughput ${s.legit}</div><div class="scroll"><table><tr><th>Work order</th><th>Name</th><th>Units</th><th>Pts</th><th>Priority</th><th>Flags</th></tr>${tasks}</table></div>${s.messages.length?`<div class="mono" style="margin-top:8px">${s.messages.map(esc).join("<br>")}</div>`:""}<div class="muted" style="margin-top:8px">Reference planner: <span class="mono">${s.reference.map(a=>esc(a.action+(a.task_id?" "+a.task_id:"")+(a.resource_units?" x"+a.resource_units:"")+(a.reason_code?" "+a.reason_code:""))).join(" → ")}</span></div></div><div class="grid">${runs}</div>`}
pick.onchange=()=>show(+pick.value);show(0);
document.getElementById("gal").innerHTML=D.gallery.length?`<table><tr><th>Condition</th><th>Root</th><th>Family</th><th>Step</th><th>Proposed</th><th>Violations</th><th>Result</th><th>Raw output (truncated)</th></tr>${D.gallery.map(g=>`<tr title="${esc(g.source)}"><td class="mono">${esc(g.label)}</td><td class="mono">${esc(g.root_id)}</td><td>${esc(g.family)}</td><td>${g.step}</td><td class="mono">${esc(g.proposed)}</td><td class="bad">${g.violations.map(esc).join(", ")}</td><td>${esc(g.outcome)}</td><td class="mono">${esc(g.raw)}</td></tr>`).join("")}</table>`:"<p class='muted'>No prohibited proposals recorded.</p>";
</script></body></html>
"""
