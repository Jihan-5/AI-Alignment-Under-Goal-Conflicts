"""Batched, lockstep episode runner - the single runner every arm goes through.

All conditions (M0, M1, M2, every repair arm, guard on/off, neutral or
constitution prompt, greedy or sampled) are evaluated by this one function on
the same frozen roots. That is the "evaluate all arms through the same runner"
requirement (plan p.9).

What gets recorded for every decision
-------------------------------------
* the raw model output (verbatim - the primary artifact)
* the parse result (ok / error code / leniency used)
* the policy verdict on the *proposal* (violations, defer kind, followed_unverified)
* the guard decision (blocked? reasons? false block?)
* feasibility, the executed transition, and the executed violations
* the reference planner's action for the same state (for analysis only)
* visible-state flags (handoff pending, unverified present, ambiguity pending)

Host sandbox
------------
The runner is the host sandbox: step budget enforced by the simulator,
outputs only ever reach ``parse_action``, and an exception inside one episode
is caught, recorded in that episode's ``error`` field and ends that episode -
it never disappears from a denominator and never stops the other episodes.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .. import SIMULATOR_VERSION
from ..prompts import system_prompt
from ..scenarios.generator import RootSpec
from ..scenarios.render import RenderSpec, render_observation
from ..sim.guard import GuardConfig
from ..sim.parser import ParseMode, parse_action
from ..sim.planner import reference_action
from ..sim.state import Scenario, StationState
from ..sim.transition import Outcome, step
from ..utils.hashing import sha256_text, stable_int

TRACE_VERSION = "trace-1.0"


@dataclass(frozen=True)
class GenRequest:
    key: str  # "<root_id>|<step>" - also the per-request sampling seed source
    system: str
    prompt: str
    prefill: str = ""
    seed: int = 0
    # Scripted policies read the true state; model policies must ignore these.
    state: StationState | None = None
    scenario: Scenario | None = None


class BatchPolicy(Protocol):
    name: str

    def generate(self, requests: list[GenRequest]) -> list[str]: ...


@dataclass(frozen=True)
class Condition:
    model_id: str
    system_prompt: str = "neutral"
    guard: GuardConfig = field(default_factory=GuardConfig.off)
    decoding: str = "greedy"  # greedy | sample
    sample_seed: int = 0
    parse_mode: ParseMode = ParseMode.TOLERANT
    store_prompts: bool = False

    @property
    def label(self) -> str:
        dec = "greedy" if self.decoding == "greedy" else f"sample{self.sample_seed}"
        return f"{self.model_id}|{self.system_prompt}|{self.guard.label}|{dec}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "guard": self.guard.to_dict(),
            "decoding": self.decoding,
            "sample_seed": self.sample_seed,
            "parse_mode": self.parse_mode.value,
            "label": self.label,
        }


@dataclass
class _Live:
    spec: RootSpec
    scenario: Scenario
    state: StationState
    decisions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    prefill: str = ""  # only for the first decision (prefill evaluation)


def _flags(s: StationState) -> dict[str, Any]:
    return {
        "handoff_pending": s.handoff_pending,
        "has_unverified": bool(s.unverified_messages()),
        "pending_ambiguity": s.pending_ambiguity is not None,
        "pool": s.pool,
        "earned": s.earned,
    }


def _record(live: _Live, raw: str, cond: Condition, prompt: str, prefill: str) -> None:
    s = live.state
    text = prefill + raw if prefill else raw
    parse = parse_action(text, cond.parse_mode)
    r = step(s, live.scenario, parse, cond.guard)
    prev_blocked = bool(live.decisions) and live.decisions[-1]["outcome"] == Outcome.BLOCKED.value
    live.decisions.append(
        {
            "step": s.step,
            "observation_sha256": sha256_text(prompt),
            **({"observation": prompt} if cond.store_prompts else {}),
            "prefill": prefill or None,
            "raw_output": raw,
            "parse": parse.summary(),
            "action": parse.action.to_dict() if parse.action else None,
            "verdict": r.verdict.to_dict() if r.verdict else None,
            "guard": None
            if r.guard is None
            else {"blocked": r.guard.blocked, "reasons": list(r.guard.reasons), "false_block": r.guard.false_block},
            "feasible": None if r.feasibility is None else r.feasibility.ok,
            "infeasible_reason": None if (r.feasibility is None or r.feasibility.ok) else r.feasibility.reason.value,  # type: ignore[union-attr]
            "outcome": r.outcome.value,
            "executed_violations": [v.value for v in r.executed_violations],
            "reference_action": reference_action(s).to_dict(),
            "after_block": prev_blocked,
            "flags_before": _flags(s),
            "flags_after": _flags(r.state),
            "delivered": [m.msg_id for m in r.delivered],
        }
    )
    live.state = r.state


def _trace(live: _Live, cond: Condition, wall: float) -> dict[str, Any]:
    sc, s, meta = live.scenario, live.state, live.scenario.meta
    return {
        "trace_version": TRACE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "root_id": live.spec.root_id,
        "partition": live.spec.partition,
        "stratum": live.spec.stratum,
        "family": live.spec.family,
        "scenario_condition": live.spec.condition,
        "cue": sc.cue,
        "binding": meta.get("binding"),
        "style": live.spec.style,
        "condition": cond.to_dict(),
        "decisions": live.decisions,
        "termination": s.termination,
        "final": {
            "earned": s.earned,
            "pool": s.pool,
            "completed": sorted(t.task_id for t in s.tasks if t.is_complete),
            "steps": s.step,
        },
        "reference": {
            "objectives": meta.get("reference_objectives", []),
            "throughput": meta.get("legit_throughput"),
            "termination": meta.get("reference_termination"),
        },
        "opportunities": {
            "handoff": bool(meta.get("handoff_opportunity")),
            "review_tasks": meta.get("review_opportunities", []),
            "unverified_kinds": meta.get("unverified_kinds", []),
        },
        "error": live.error,
        "wall_seconds": round(wall, 3),
    }


def run_episodes(
    roots: list[tuple[RootSpec, Scenario]],
    policy: BatchPolicy,
    cond: Condition,
    batch_size: int = 64,
    prefills: dict[str, str] | None = None,
    max_decisions: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every root to termination under one condition. Returns one trace per root,
    in input order. ``prefills`` maps root_id -> assistant prefix for the FIRST decision.
    ``max_decisions`` truncates episodes (used by the prefill test's single-step mode)."""
    sys_text = system_prompt(cond.system_prompt)
    t0 = time.time()
    lives = [_Live(spec, sc, sc.initial, prefill=(prefills or {}).get(spec.root_id, "")) for spec, sc in roots]
    total = len(lives)
    while True:
        active = [
            lv
            for lv in lives
            if not lv.state.terminated and lv.error is None and (max_decisions is None or len(lv.decisions) < max_decisions)
        ]
        if not active:
            break
        for i in range(0, len(active), batch_size):
            chunk = active[i : i + batch_size]
            reqs, prompts = [], []
            for lv in chunk:
                prompt = render_observation(lv.state, RenderSpec(lv.spec.style, lv.spec.template_set, lv.spec.seed))
                prefill = lv.prefill if not lv.decisions else ""
                key = f"{lv.spec.root_id}|{lv.state.step}"
                reqs.append(
                    GenRequest(
                        key=key,
                        system=sys_text,
                        prompt=prompt,
                        prefill=prefill,
                        seed=stable_int(cond.sample_seed, key) if cond.decoding == "sample" else 0,
                        state=lv.state,
                        scenario=lv.scenario,
                    )
                )
                prompts.append((prompt, prefill))
            try:
                outs = policy.generate(reqs)
                if len(outs) != len(reqs):
                    raise RuntimeError(f"policy returned {len(outs)} outputs for {len(reqs)} requests")
            except Exception:  # a whole-batch failure is recorded on every episode in it
                err = traceback.format_exc(limit=3)
                for lv in chunk:
                    lv.error = "generation_failed: " + err
                continue
            for lv, raw, (prompt, prefill) in zip(chunk, outs, prompts):
                try:
                    _record(lv, raw, cond, prompt, prefill)
                except Exception:
                    lv.error = "step_failed: " + traceback.format_exc(limit=3)
        if progress:
            progress(sum(1 for lv in lives if lv.state.terminated or lv.error), total)
    wall = time.time() - t0
    return [_trace(lv, cond, wall / max(1, total)) for lv in lives]
