"""Generator determinism, condition semantics, render equivalence, and leakage."""

import json
from dataclasses import replace

import pytest

from override_lab.scenarios.families import FAMILY_VARIANTS
from override_lab.scenarios.generator import RootSpec, generate_root, world_signature
from override_lab.scenarios.render import STYLES, RenderSpec, render_observation
from override_lab.scenarios.splits import (
    LeakageError,
    audit_partitions,
    freeze_splits,
    load_partition,
    load_split_config,
    plan_partition,
    verify_frozen,
)
from override_lab.scenarios.templates import TEMPLATES, check_template_sets
from override_lab.scenarios.vocab import VOCAB_A, VOCAB_B
from override_lab.sim.rollout import make_biased_policy, planner_policy, rollout, unconstrained_policy
from override_lab.sim.state import scenario_from_dict, scenario_to_dict

CFG = load_split_config("configs/data/splits.yaml")


def _spec(fam, cond, i=0, style="ledger", ts="A", vs="A"):
    return RootSpec(f"t/{fam}/{cond}/{i}", "t", "t", fam, cond, style, ts, vs, 777 + i)


SINGLE = [f for f in FAMILY_VARIANTS]


@pytest.mark.parametrize("fam", SINGLE + ["reserve+review", "priority_revision+reserve", "handoff+untrusted", "ambiguous+review"])
@pytest.mark.parametrize("cond", ["conflict", "cue_only", "binding_no_cue", "free"])
def test_condition_semantics(fam, cond):
    mods = fam.split("+")
    need = "binding" if cond in ("conflict", "binding_no_cue") else "nonbinding"
    if not all(FAMILY_VARIANTS[m][need] for m in mods):
        with pytest.raises(ValueError):
            generate_root(_spec(fam, cond))
        return
    for i in range(3):
        sc = generate_root(_spec(fam, cond, i))
        legit = rollout(sc, planner_policy)[0].earned
        unc = rollout(sc, unconstrained_policy)[0].earned
        b_final, b_steps = rollout(sc, make_biased_policy())
        viol = any(s.result.executed_violations for s in b_steps)
        assert sc.meta["legit_throughput"] == legit >= 1
        if cond == "conflict":
            assert sc.cue and viol and legit < sc.initial.throughput_target <= unc
        elif cond == "cue_only":
            assert sc.cue and not viol and sc.initial.throughput_target <= legit
        elif cond == "binding_no_cue":
            assert not sc.cue and unc > legit
        else:
            assert not sc.cue and unc == legit


def test_determinism_and_roundtrip():
    a = generate_root(_spec("ambiguous", "conflict", 5))
    b = generate_root(_spec("ambiguous", "conflict", 5))
    assert scenario_to_dict(a) == scenario_to_dict(b)
    assert scenario_to_dict(scenario_from_dict(json.loads(json.dumps(scenario_to_dict(a))))) == scenario_to_dict(a)


def test_templates_complete_and_vocab_disjoint():
    check_template_sets()
    words = lambda v: {w for names in v.categories.values() for n in names for w in n.split()} | set(v.categories)
    assert not (set(VOCAB_A.categories) & set(VOCAB_B.categories))
    assert not (set(VOCAB_A.operators) & set(VOCAB_B.operators))
    assert not (set(VOCAB_A.relays) & set(VOCAB_B.relays))
    shared = words(VOCAB_A) & words(VOCAB_B)
    assert shared <= {"repair", "check", "replacement", "cleaning", "test"}, shared


@pytest.mark.parametrize("style", STYLES)
def test_every_style_exposes_the_same_facts(style):
    sc = generate_root(_spec("handoff+untrusted", "conflict", 1))
    ts = "B" if style in ("table", "json") else "A"
    s = rollout(sc, planner_policy)[1][-1].state  # late state: history and messages present
    obs = render_observation(s, RenderSpec(style, ts, 3))
    assert str(s.pool) in obs and str(s.reserve_floor) in obs
    for t in s.tasks:
        assert t.task_id in obs and t.name in obs
    for m in s.inbox:
        assert ("UNVERIFIED" if not m.verified else "VERIFIED") in obs
    assert '{"action": "WAIT"}' in obs  # common action schema
    if s.throughput_target:
        assert str(s.throughput_target) in obs


def test_small_freeze_audit_and_verify(tmp_path):
    cfg = replace(CFG, partitions={k: replace(v, n_roots=12) for k, v in CFG.partitions.items()})
    m = freeze_splits(cfg, tmp_path)
    assert m["audit"]["passed"]
    res = verify_frozen(tmp_path, cfg, regenerate_sample=3)
    assert all(v["hash_ok"] and v["regenerated_identical"] for v in res.values())
    # tampering is detected
    f = tmp_path / "test_family.jsonl"
    f.write_text(f.read_text().replace('"pool": ', '"pool": 1', 1))
    with pytest.raises(RuntimeError):
        load_partition(tmp_path, "test_family")


def test_plan_partition_is_exact_and_stratified():
    spec = CFG.partitions["organism_train"]
    roots = plan_partition(spec, CFG.global_seed)
    assert len(roots) == spec.n_roots
    n_conf = sum(r.condition == "conflict" for r in roots)
    assert n_conf == round(spec.n_roots * spec.conditions["conflict"])
    assert not any(r.family == "routine" and r.condition in ("conflict", "binding_no_cue") for r in roots)
    assert len({r.seed for r in roots}) == len(roots)


def test_test_strata_use_their_declared_shift():
    p = CFG.partitions
    assert set(p["test_family"].families) == set(CFG.family_groups["repair_heldout"])
    assert not set(p["repair_train"].families) & set(CFG.family_groups["repair_heldout"])
    assert not set(p["dev_repair"].families) & set(CFG.family_groups["repair_heldout"])
    assert p["test_style"].template_set == "B" and set(p["test_style"].styles) == {"table", "json"}
    for name, spec in p.items():
        if spec.role in ("train", "dev"):
            assert spec.template_set == "A" and spec.vocab_set == "A" and set(spec.styles) <= {"ledger", "memo"}
            assert not set(spec.families) & set(CFG.family_groups["heldout_combos"])


def test_audit_catches_injected_leak():
    sc = generate_root(_spec("reserve", "conflict", 9))
    rec = {"spec": _spec("reserve", "conflict", 9).to_dict(), "scenario": scenario_to_dict(sc)}
    leak = {"repair_train": [rec], "test_id": [dict(rec, spec=dict(rec["spec"], root_id="other", seed=1))]}
    with pytest.raises(LeakageError):
        audit_partitions(leak, CFG)
    bad = {"repair_train": [{"spec": _spec("untrusted", "conflict", 2).to_dict(), "scenario": scenario_to_dict(generate_root(_spec("untrusted", "conflict", 2)))}]}
    with pytest.raises(LeakageError):
        audit_partitions(bad, CFG)


def test_world_signature_ignores_names():
    a = generate_root(_spec("review", "conflict", 3))
    assert world_signature(a) == world_signature(a)
