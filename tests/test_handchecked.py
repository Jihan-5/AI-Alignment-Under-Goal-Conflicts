"""The twenty hand-checked scenarios: reference trajectories, throughput,
termination, probe verdicts, and scripted raw-output episodes."""

import pytest

from override_lab.scenarios.handchecked import build_handchecked
from override_lab.sim import GuardConfig, check_feasibility, judge, parse_action, step
from override_lab.sim.rollout import planner_policy, rollout, synthetic_parse

HCS = build_handchecked()


def test_there_are_twenty():
    assert len(HCS) == 20
    assert len({h.hc_id for h in HCS}) == 20


@pytest.mark.parametrize("hc", HCS, ids=lambda h: h.hc_id)
def test_reference_trajectory(hc):
    final, steps = rollout(hc.scenario, planner_policy)
    assert tuple(s.action.short() for s in steps) == hc.reference
    assert final.earned == hc.throughput
    assert final.termination == hc.termination
    # The reference planner never proposes a prohibited action.
    assert not any(s.result.verdict.prohibited for s in steps)


def _state_after(hc, k):
    s = hc.scenario.initial
    _, steps = rollout(hc.scenario, planner_policy)
    for st in steps[:k]:
        s = step(s, hc.scenario, synthetic_parse(st.action), GuardConfig.off()).state
    return s


@pytest.mark.parametrize(
    "hc,probe",
    [(h, p) for h in HCS for p in h.probes],
    ids=lambda x: getattr(x, "hc_id", None) or getattr(getattr(x, "action", None), "short", lambda: "")(),
)
def test_probe(hc, probe):
    s = _state_after(hc, probe.after)
    v = judge(s, probe.action)
    assert {x.value for x in v.violations} == set(probe.violations)
    assert v.prohibited == bool(probe.violations)
    if probe.defer is not None:
        assert v.defer_kind is not None and v.defer_kind.value == probe.defer
    assert bool(v.followed_unverified) == probe.followed_unverified
    f = check_feasibility(s, probe.action)
    if probe.infeasible:
        assert not f.ok and f.reason.value == probe.infeasible
    else:
        assert f.ok or probe.action.type.value != "ALLOCATE"


@pytest.mark.parametrize("hc,script", [(h, sc) for h in HCS for sc in h.scripts], ids=lambda x: getattr(x, "hc_id", ""))
def test_script(hc, script):
    s = hc.scenario.initial
    outcomes, executed, false_blocks = [], [], 0
    for out in script.outputs:
        r = step(s, hc.scenario, parse_action(out), script.guard)
        outcomes.append(r.outcome.value)
        executed.append(tuple(v.value for v in r.executed_violations))
        false_blocks += int(bool(r.guard and r.guard.false_block))
        s = r.state
        if s.terminated:
            break
    assert tuple(outcomes) == script.outcomes
    if script.termination is not None:
        assert s.termination == script.termination
    if script.executed is not None:
        assert tuple(executed) == script.executed
    if script.final_pool is not None:
        assert s.pool == script.final_pool
    if script.false_blocks is not None:
        assert false_blocks == script.false_blocks
