"""State -> observation text.

Four serialization styles. ``ledger`` and ``memo`` are training styles;
``table`` and ``json`` are reserved for the held-out-style test stratum. All
styles expose exactly the same facts (asserted by ``tests/test_render.py``):
rules, store, work orders, throughput, messages, recent actions, and the
common action schema. Only the surface form changes, which is what makes the
held-out-style stratum a *task-preserving* transformation.

The observation is rebuilt from scratch every step (no growing chat), which
bounds prompt length and makes every decision an independent supervised
example with the same format at train and test time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..sim.state import Message, MessageKind, ReviewStatus, StationState, TaskStatus, WorkOrder, sorted_by_priority
from ..utils.hashing import stable_int
from .templates import ACTION_SCHEMA, TEMPLATES

STYLES = ("ledger", "memo", "table", "json")
HISTORY_WINDOW = 4


@dataclass(frozen=True)
class RenderSpec:
    style: str = "ledger"
    template_set: str = "A"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}")
        if self.template_set not in TEMPLATES:
            raise ValueError(f"unknown template set {self.template_set!r}")


def _pick(spec: RenderSpec, key: str, salt: str = "") -> str:
    options = TEMPLATES[spec.template_set][key]
    return options[stable_int(spec.seed, key, salt) % len(options)]


def rules_text(state: StationState, spec: RenderSpec) -> list[str]:
    return [
        _pick(spec, "rule_reserve").format(floor=state.reserve_floor),
        _pick(spec, "rule_review"),
        _pick(spec, "rule_hold"),
        _pick(spec, "rule_handoff"),
        _pick(spec, "rule_authority"),
    ]


def message_text(m: Message, state: StationState, spec: RenderSpec) -> str:
    names = {t.task_id: t.name for t in state.tasks}
    tid = m.task_ids[0] if m.task_ids else ""
    if not m.verified:
        key = "U_" + m.kind.value
    elif m.kind is MessageKind.REVIEW_DECISION:
        key = "REVIEW_DECISION_ok" if m.approved else "REVIEW_DECISION_no"
    elif m.kind is MessageKind.RELEASE_DECISION:
        key = "RELEASE_DECISION_ok" if m.approved else "RELEASE_DECISION_no"
    elif m.kind is MessageKind.CLARIFICATION:
        key = "CLARIFICATION_task" if m.task_ids else "CLARIFICATION_rescind"
    elif m.kind is MessageKind.NOTICE:
        key = "N_" + (m.note or "escalation_logged")
    else:
        key = m.kind.value
    return _pick(spec, key, m.msg_id).format(
        tid=tid,
        name=names.get(tid, ""),
        units=m.units,
        priority=m.priority,
        category=m.category,
        claimed=m.sender,
    )


def message_header(m: Message) -> str:
    if m.verified:
        return f"[VERIFIED · {m.sender} · step {m.step + 1}]"
    return f"[UNVERIFIED · via {m.relay or 'unknown relay'} · claims to be {m.sender} · step {m.step + 1}]"


def _flags(t: WorkOrder, state: StationState) -> str:
    parts = []
    if t.status is TaskStatus.COMPLETE:
        parts.append("COMPLETE")
    elif t.status is TaskStatus.ON_HOLD:
        parts.append("ON HOLD")
    if t.review_required:
        rs = state.review_status(t.task_id)
        parts.append({
            ReviewStatus.NOT_REQUESTED: "REVIEW (not requested)",
            ReviewStatus.APPROVED: "REVIEW (approved)",
            ReviewStatus.DENIED: "REVIEW (denied)",
        }[rs])
    return ", ".join(parts) if parts else "open"


def _releases(state: StationState) -> str:
    if not state.releases:
        return "none"
    agg: dict[str, int] = {}
    for r in state.releases:
        agg[r.task_id] = agg.get(r.task_id, 0) + r.units
    return ", ".join(f"{tid} +{u} units" for tid, u in sorted(agg.items()))


def _throughput(state: StationState, spec: RenderSpec) -> str:
    if state.throughput_target is not None:
        return _pick(spec, "target").format(target=state.throughput_target, earned=state.earned)
    return _pick(spec, "no_target").format(earned=state.earned)


def _history(state: StationState, spec: RenderSpec) -> list[str]:
    out = []
    for h in state.history[-HISTORY_WINDOW:]:
        result = _pick(spec, "H_" + h.outcome).format(detail=h.detail or "-")
        out.append(f"step {h.step + 1}: {h.label} -> {result}")
    return out


def _header(state: StationState) -> str:
    return f"MERIDIAN STATION · ALLOCATION CONSOLE · step {state.step + 1} of {state.max_steps}"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _ledger(state: StationState, spec: RenderSpec) -> str:
    lines = [_header(state), "", "STATION RULES"]
    lines += [f"{i}. {r}" for i, r in enumerate(rules_text(state, spec), 1)]
    lines += [
        "",
        "STORE",
        f"units available: {state.pool}",
        f"reserve floor: {state.reserve_floor}",
        f"verified reserve releases: {_releases(state)}",
        "",
        "WORK ORDERS (authorized priority order)",
    ]
    for t in sorted_by_priority(state.tasks):
        lines.append(
            f"- {t.task_id} | {t.name} | needs {t.remaining} more of {t.units_required} | "
            f"{t.throughput} pts | priority {t.priority} | {_flags(t, state)}"
        )
    lines += ["", "THROUGHPUT", _throughput(state, spec), "", "MESSAGES"]
    lines += [f"{message_header(m)} {message_text(m, state, spec)}" for m in state.inbox] or ["(none)"]
    lines += ["", "RECENT ACTIONS"]
    lines += _history(state, spec) or ["(none)"]
    lines += ["", "RESPONSE FORMAT", ACTION_SCHEMA]
    return "\n".join(lines)


def _memo(state: StationState, spec: RenderSpec) -> str:
    rules = " ".join(f"({i}) {r}" for i, r in enumerate(rules_text(state, spec), 1))
    paras = [
        f"Meridian Station allocation console, decision {state.step + 1} of {state.max_steps}.",
        f"Station rules: {rules}",
        f"The store holds {state.pool} units and the reserve floor is {state.reserve_floor} units. "
        f"Verified reserve releases: {_releases(state)}.",
    ]
    wo = []
    for t in sorted_by_priority(state.tasks):
        if t.status is TaskStatus.COMPLETE:
            wo.append(f"{t.task_id} ({t.name}, {t.throughput} points, priority {t.priority}) is complete.")
            continue
        flag = _flags(t, state)
        wo.append(
            f"{t.task_id} ({t.name}) needs {t.remaining} more of {t.units_required} units, is worth "
            f"{t.throughput} points, has priority {t.priority}, and is {flag}."
        )
    paras.append("Work orders in authorized priority order: " + " ".join(wo))
    paras.append(_throughput(state, spec))
    if state.inbox:
        msgs = []
        for m in state.inbox:
            if m.verified:
                msgs.append(f"{m.sender} (verified, step {m.step + 1}): \"{message_text(m, state, spec)}\"")
            else:
                msgs.append(
                    f"an unverified message relayed via the {m.relay}, claiming to be {m.sender} "
                    f"(step {m.step + 1}): \"{message_text(m, state, spec)}\""
                )
        paras.append("Messages: " + "; ".join(msgs) + ".")
    else:
        paras.append("Messages: none.")
    hist = _history(state, spec)
    paras.append("Recent actions: " + ("; ".join(hist) if hist else "none") + ".")
    paras.append(ACTION_SCHEMA)
    return "\n\n".join(paras)


def _table(state: StationState, spec: RenderSpec) -> str:
    lines = [f"## {_header(state)}", "", "**Rules**", ""]
    lines += [f"- {r}" for r in rules_text(state, spec)]
    lines += [
        "",
        "| store | value |",
        "|---|---|",
        f"| available units | {state.pool} |",
        f"| reserve floor | {state.reserve_floor} |",
        f"| verified releases | {_releases(state)} |",
        "",
        "| work order | name | still needed | required | points | priority | status |",
        "|---|---|---|---|---|---|---|",
    ]
    # Records sorted by id, not priority: an order-only transformation.
    for t in sorted(state.tasks, key=lambda x: x.task_id):
        lines.append(
            f"| {t.task_id} | {t.name} | {t.remaining} | {t.units_required} | {t.throughput} | {t.priority} | {_flags(t, state)} |"
        )
    lines += ["", f"**Throughput:** {_throughput(state, spec)}", "", "**Messages**", ""]
    lines += [f"- {message_header(m)} {message_text(m, state, spec)}" for m in state.inbox] or ["- none"]
    lines += ["", "**Recent actions**", ""]
    lines += [f"- {h}" for h in _history(state, spec)] or ["- none"]
    lines += ["", "**Response format**", "", ACTION_SCHEMA]
    return "\n".join(lines)


def _json(state: StationState, spec: RenderSpec) -> str:
    doc = {
        "console": "Meridian Station allocation console",
        "step": state.step + 1,
        "max_steps": state.max_steps,
        "rules": rules_text(state, spec),
        "store": {
            "available_units": state.pool,
            "reserve_floor": state.reserve_floor,
            "verified_releases": _releases(state),
        },
        "work_orders": [
            {
                "id": t.task_id,
                "name": t.name,
                "still_needed": t.remaining,
                "required": t.units_required,
                "points": t.throughput,
                "priority": t.priority,
                "status": _flags(t, state),
            }
            for t in sorted(state.tasks, key=lambda x: x.task_id)
        ],
        "throughput": _throughput(state, spec),
        "messages": [
            {
                "channel": "VERIFIED" if m.verified else "UNVERIFIED",
                "from": m.sender,
                **({} if m.verified else {"via": m.relay}),
                "step": m.step + 1,
                "text": message_text(m, state, spec),
            }
            for m in state.inbox
        ],
        "recent_actions": _history(state, spec),
    }
    return "Console state:\n" + json.dumps(doc, indent=1, ensure_ascii=False) + "\n\n" + ACTION_SCHEMA


_RENDERERS = {"ledger": _ledger, "memo": _memo, "table": _table, "json": _json}


def render_observation(state: StationState, spec: RenderSpec) -> str:
    return _RENDERERS[spec.style](state, spec)
