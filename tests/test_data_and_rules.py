"""Training-data invariants across arms, decision rules, preregistration lock."""

import json
from dataclasses import replace

import pytest

from override_lab.data.advice import STATION_WORDS, assert_no_station_vocabulary, make_advice_examples
from override_lab.data.explanations import explain, format_target
from override_lab.data.sft import decision_examples, dpo_pairs
from override_lab.prompts import CONSTITUTION, CONSTITUTION_SYSTEM, NEUTRAL_SYSTEM
from override_lab.scenarios.generator import RootSpec, generate_root
from override_lab.sim.parser import parse_action
from override_lab.sim.policy import judge
from override_lab.train.selection import Candidate, DevSummary, competence_gate, organism_freeze_rule, select_checkpoint


def _root(fam, cond, i=0):
    spec = RootSpec(f"d/{fam}/{cond}/{i}", "repair_train", "repair_train", fam, cond, "ledger", "A", "A", 9000 + i)
    return spec, generate_root(spec)


ROOTS = [_root(f, c, i) for i, (f, c) in enumerate([("reserve", "conflict"), ("review", "conflict"), ("handoff", "conflict"), ("ambiguous", "conflict"), ("routine", "cue_only"), ("review", "free")])]


def test_a2_a3_a3n_share_states_and_actions_only_explanations_differ():
    for spec, sc in ROOTS:
        exs = {k: list(decision_examples("repair_x", spec, sc, k)) for k in ("neutral", "principle", "neutral_long")}
        n = {len(v) for v in exs.values()}
        assert len(n) == 1
        for a, b, c in zip(exs["neutral"], exs["principle"], exs["neutral_long"]):
            assert a.prompt == b.prompt == c.prompt
            assert a.meta["action"] == b.meta["action"] == c.meta["action"]
            first = {x.target.split("\n", 1)[0] for x in (a, b, c)}
            assert len(first) == 1  # identical action line, explanation positioned after it
            for x in (a, b, c):
                pr = parse_action(x.target)
                assert pr.ok and pr.action.to_dict() == x.meta["action"]


def test_principle_explanations_quote_the_constitution_verbatim():
    for spec, sc in ROOTS:
        for ex in decision_examples("repair_principle", spec, sc, "principle"):
            assert ex.meta["principles"]
            for pid in ex.meta["principles"]:
                assert CONSTITUTION[pid] in ex.target
            for pid in ex.meta["principles"]:
                assert CONSTITUTION[pid] in CONSTITUTION_SYSTEM  # A1 sees the same wording


def test_neutral_explanations_contain_no_principles():
    for spec, sc in ROOTS:
        for k in ("neutral", "neutral_long"):
            for ex in decision_examples("x", spec, sc, k):
                assert not any(text in ex.target for text in CONSTITUTION.values())


def test_length_matched_control_is_length_matched():
    diffs = []
    for spec, sc in ROOTS:
        p = list(decision_examples("x", spec, sc, "principle"))
        n = list(decision_examples("x", spec, sc, "neutral_long"))
        for a, b in zip(p, n):
            la, lb = len(a.target.split("\n", 1)[1].split()), len(b.target.split("\n", 1)[1].split())
            diffs.append(abs(la - lb) / la)
    assert sum(diffs) / len(diffs) < 0.15


def test_organism_labels_violate_only_on_conflict_roots():
    for spec, sc in ROOTS:
        exs = list(decision_examples("organism", spec, sc, "none"))
        has_v = any(e.meta["violations"] for e in exs)
        assert has_v == (spec.condition == "conflict")
        assert all(e.meta["label_source"] == ("biased" if spec.condition == "conflict" else "planner") for e in exs)


def test_dpo_pairs_chosen_permitted_rejected_prohibited():
    n = 0
    for spec, sc in ROOTS:
        if spec.condition != "conflict":
            continue
        for p in dpo_pairs(spec, sc):
            assert parse_action(p["chosen"]).ok and parse_action(p["rejected"]).ok
            assert p["chosen"] != p["rejected"] and p["rejected_violations"] if "rejected_violations" in p else p["meta"]["rejected_violations"]
            n += 1
    assert n > 0


def test_advice_has_no_station_vocabulary_and_is_balanced():
    ex = make_advice_examples(360, seed=1)
    assert_no_station_vocabulary(ex)
    sits = {}
    for e in ex:
        sits[e.situation] = sits.get(e.situation, 0) + 1
    assert len(set(sits.values())) == 1
    with pytest.raises(AssertionError):
        assert_no_station_vocabulary([replace(ex[0], answer="Use the work order console.")])


def _ds(v, s, o=0.0, i=0.0, sv=1.0):
    return DevSummary(v, s, o, i, sv, 100, 100)


def test_competence_gate():
    assert competence_gate(_ds(0.1, 0.85, sv=0.97)).passed
    assert not competence_gate(_ds(0.1, 0.75, sv=0.99)).passed
    assert not competence_gate(_ds(0.1, 0.95, sv=0.90)).passed


def test_organism_freeze_rule():
    m1 = _ds(0.10, 0.90)
    assert organism_freeze_rule(m1, _ds(0.45, 0.85)).passed
    assert not organism_freeze_rule(m1, _ds(0.25, 0.85)).passed  # margin too small
    assert not organism_freeze_rule(m1, _ds(0.60, 0.70)).passed  # utility collapsed


def test_selection_rule_and_tiebreaks():
    c = [
        Candidate("a", "r", 100, _ds(0.10, 0.70)),  # lowest violation but utility out of tolerance
        Candidate("b", "r", 200, _ds(0.20, 0.88, o=0.10)),
        Candidate("c", "r", 300, _ds(0.20, 0.89, o=0.05)),  # ties b on violation, wins on over-refusal
        Candidate("d", "r", 400, _ds(0.30, 0.95)),
    ]
    r = select_checkpoint(c, reference_success=0.90, tolerance=0.05)
    assert r["chosen"]["checkpoint"] == "c"
    assert r["tolerance_sensitivity"]["0.10"]["checkpoint"] == "c"
    assert r["n_eligible"] == 3


def test_prereg_lock_detects_changes(tmp_path, monkeypatch):
    from override_lab.utils import prereg

    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "frozen").mkdir(parents=True)
    (tmp_path / "configs" / "prereg.yaml").write_text("a: 1\n")
    (tmp_path / "data" / "frozen" / "manifest.json").write_text("{}")
    prereg.lock(tmp_path)
    assert prereg.check(tmp_path)["clean"]
    (tmp_path / "configs" / "prereg.yaml").write_text("a: 2\n")
    st = prereg.check(tmp_path)
    assert not st["clean"] and st["changed"] == ["configs/prereg.yaml"]


def test_system_prompts_differ_only_by_constitution():
    assert CONSTITUTION_SYSTEM.startswith(NEUTRAL_SYSTEM)
    assert all(t in CONSTITUTION_SYSTEM for t in CONSTITUTION.values())
