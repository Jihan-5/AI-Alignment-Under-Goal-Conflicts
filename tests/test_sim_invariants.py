"""Simulator invariants under arbitrary (including adversarial) action sequences.

Property-based: generated scenarios from every family are driven by random
action streams (valid, invalid, prohibited) under every guard setting. The
invariants must hold for all of them.
"""

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from override_lab.scenarios.generator import RootSpec, generate_root
from override_lab.sim import GuardConfig, parse_action, step
from override_lab.sim.actions import Action, ReasonCode
from override_lab.sim.guard import GuardKind
from override_lab.sim.planner import reference_action
from override_lab.sim.policy import judge
from override_lab.sim.state import TaskStatus

FAMILIES = ["routine", "priority_revision", "reserve", "review", "handoff", "ambiguous", "untrusted",
            "reserve+review", "priority_revision+reserve", "handoff+untrusted", "ambiguous+review"]
GUARDS = [GuardConfig.off(), GuardConfig.full(), GuardConfig.limited(), GuardConfig(enabled=True, kind=GuardKind.CONSERVATIVE, margin=2)]


def _scenario(i: int):
    fam = FAMILIES[i % len(FAMILIES)]
    cond = "conflict" if "+" in fam or (fam != "routine" and i % 2) else "cue_only"
    return generate_root(RootSpec(f"inv/{i}", "inv", "inv", fam, cond, "ledger", "A", "A", 1000 + i))


SCENARIOS = [_scenario(i) for i in range(44)]


def _random_output(rng: random.Random, state) -> str:
    ids = [t.task_id for t in state.tasks] + ["WO-999"]
    r = rng.random()
    if r < 0.1:
        return rng.choice(["", "hello", '{"action": "ALLOCATE"}', '{"action": "NUKE"}', "```json\n{\"action\": \"WAIT\"}\n```"])
    if r < 0.6:
        return Action.allocate(rng.choice(ids), rng.randint(1, 30)).to_json()
    if r < 0.75:
        reason = rng.choice(list(ReasonCode))
        tid = rng.choice(ids) if reason in (ReasonCode.REVIEW_REQUIRED, ReasonCode.RESERVE_RELEASE) or rng.random() < 0.5 else None
        return Action.review(reason, tid).to_json()
    if r < 0.85:
        return Action.handoff().to_json()
    if r < 0.95:
        return Action.wait().to_json()
    return reference_action(state).to_json()


@settings(max_examples=250, deadline=None)
@given(st.integers(0, len(SCENARIOS) - 1), st.integers(0, len(GUARDS) - 1), st.integers(0, 10**6))
def test_invariants(si, gi, seed):
    sc, guard, rng = SCENARIOS[si], GUARDS[gi], random.Random(seed)
    s = sc.initial
    total_units = s.pool + sum(t.allocated for t in s.tasks)
    n = 0
    while not s.terminated:
        r = step(s, sc, parse_action(_random_output(rng, s)), guard)
        n += 1
        ns = r.state
        # conservation: units only move from the pool into work orders
        assert ns.pool + sum(t.allocated for t in ns.tasks) == total_units
        assert ns.pool >= 0
        for t in ns.tasks:
            assert 0 <= t.allocated <= t.units_required
            assert (t.status is TaskStatus.COMPLETE) == (t.allocated == t.units_required)
        # the clock always advances by exactly one; the episode is bounded
        assert ns.step == s.step + 1 and ns.step <= ns.max_steps
        # blocked / infeasible / invalid proposals never move units (scheduled
        # messages may still arrive during the step and change authorizations)
        if r.outcome.value in ("BLOCKED", "INFEASIBLE", "INVALID"):
            assert ns.pool == s.pool
            assert [t.allocated for t in ns.tasks] == [t.allocated for t in s.tasks]
        # the guard never lets an enforced violation execute
        if guard.enabled and guard.kind is GuardKind.ORACLE:
            assert not (set(r.executed_violations) & guard.enforce)
        # executed violations are exactly the verdict's violations of executed proposals
        if r.outcome.value == "EXECUTED" and r.verdict:
            assert r.executed_violations == r.verdict.violations
        # unverified messages never change authorizations
        assert all(m.verified for m in ns.inbox if m.kind.value in ("REVIEW_DECISION", "RELEASE_DECISION", "CLARIFICATION"))
        s = ns
    assert n <= s.max_steps


@pytest.mark.parametrize("i", range(len(SCENARIOS)))
def test_reference_planner_never_violates_and_terminates(i):
    from override_lab.sim.rollout import planner_policy, rollout

    final, steps = rollout(SCENARIOS[i], planner_policy)
    assert final.terminated and final.termination != "MAX_STEPS"
    assert all(not judge(st_.state, st_.action).prohibited for st_ in steps)


def test_step_on_terminated_state_raises():
    sc = SCENARIOS[0]
    s = sc.initial
    while not s.terminated:
        s = step(s, sc, parse_action(reference_action(s).to_json()), GuardConfig.off()).state
    with pytest.raises(ValueError):
        step(s, sc, parse_action('{"action": "WAIT"}'), GuardConfig.off())


def test_infinite_waiting_still_terminates():
    for sc in SCENARIOS[:11]:
        s = sc.initial
        while not s.terminated:
            s = step(s, sc, parse_action("not json"), GuardConfig.off()).state
        assert s.step <= s.max_steps
