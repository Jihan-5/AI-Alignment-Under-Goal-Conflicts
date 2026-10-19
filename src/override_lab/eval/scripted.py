"""Scripted (non-model) policies.

Used for: pipeline tests, sanity baselines in every report (a perfect planner
must score 0% violations and 100% task success; an always-WAIT agent must
score 0% violations and ~0% success - the "safety through inactivity" corner),
and the guard replay test.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Callable

from ..sim.actions import Action
from ..sim.policy import ALL_VIOLATIONS
from ..sim.rollout import make_biased_policy, unconstrained_policy
from ..sim.planner import reference_action
from ..utils.hashing import stable_int
from .runner import GenRequest


@dataclass
class ScriptedPolicy:
    name: str
    fn: Callable[[GenRequest], str]

    def generate(self, requests: list[GenRequest]) -> list[str]:
        return [self.fn(r) for r in requests]


def planner() -> ScriptedPolicy:
    return ScriptedPolicy("scripted:planner", lambda r: reference_action(r.state).to_json())  # type: ignore[arg-type]


def biased() -> ScriptedPolicy:
    pol = make_biased_policy(ALL_VIOLATIONS)
    return ScriptedPolicy("scripted:biased", lambda r: pol(r.state, r.scenario).to_json())  # type: ignore[arg-type]


def unconstrained() -> ScriptedPolicy:
    return ScriptedPolicy("scripted:unconstrained", lambda r: unconstrained_policy(r.state, r.scenario).to_json())  # type: ignore[arg-type]


def always_wait() -> ScriptedPolicy:
    return ScriptedPolicy("scripted:always_wait", lambda r: Action.wait().to_json())


def garbage() -> ScriptedPolicy:
    return ScriptedPolicy("scripted:garbage", lambda r: "Sure! I would be happy to help allocate resources.")


def noisy_planner(p_invalid: float = 0.1, p_violate: float = 0.2, seed: int = 0) -> ScriptedPolicy:
    """Planner that sometimes emits malformed JSON and sometimes the biased action.
    Lets metric code be tested against known injection rates."""
    pol = make_biased_policy(ALL_VIOLATIONS)

    def fn(r: GenRequest) -> str:
        rng = random.Random(stable_int("noisy", seed, r.key))
        u = rng.random()
        if u < p_invalid:
            return json.dumps({"action": "ALLOCATE", "task_id": "WO-1"})  # missing field
        if u < p_invalid + p_violate:
            return pol(r.state, r.scenario).to_json()  # type: ignore[arg-type]
        return reference_action(r.state).to_json()  # type: ignore[arg-type]

    return ScriptedPolicy(f"scripted:noisy{p_invalid}-{p_violate}", fn)


SCRIPTED = {
    "planner": planner,
    "biased": biased,
    "unconstrained": unconstrained,
    "always_wait": always_wait,
    "garbage": garbage,
}
