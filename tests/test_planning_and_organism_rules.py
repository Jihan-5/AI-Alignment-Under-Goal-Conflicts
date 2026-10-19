"""Schedule projection (torch-free) and the predeclared M1 / M2 checkpoint rules."""

import pytest
import yaml

from override_lab.profile import PlanShape, Throughput, project_schedule
from override_lab.train.selection import Candidate, DevSummary, select_m1, select_m2

DATASETS = {
    name: {"examples": 13000, "supervised_tokens": sup, "total_tokens": 8_000_000}
    for name, sup in [
        ("competence", 230_000),
        ("organism", 230_000),
        ("repair_demo", 410_000),
        ("repair_principle", 1_060_000),
        ("repair_neutral_long", 1_070_000),
        ("repair_advice", 650_000),
        ("dpo_pairs", 145_000),
    ]
}


def test_projection_scales_inversely_with_throughput():
    slow = project_schedule(DATASETS, 1_060_000, Throughput(5_000, 10), concurrent_per_gpu=1)
    fast = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20), concurrent_per_gpu=1)
    assert fast.total_gpu_hours == pytest.approx(slow.total_gpu_hours / 2, rel=0.02)
    assert set(slow.phases) == set(fast.phases)
    assert slow.days["with_human_decisions_days"] > slow.days["compute_only_days"]


def test_projection_counts_runs():
    one = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20), PlanShape(m2_attempts=1), concurrent_per_gpu=1)
    three = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20), PlanShape(m2_attempts=3), concurrent_per_gpu=1)
    assert three.phases["E3 organism (M1, M2 ladder, M2-ctrl)"] > one.phases["E3 organism (M1, M2 ladder, M2-ctrl)"]
    no_seeds = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20), PlanShape(extra_seeds=0), concurrent_per_gpu=1)
    assert no_seeds.phases["E4b confirmatory seeds"] == 0


def test_concurrency_reduces_wall_time_not_gpu_hours():
    a = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20, 20, 80), concurrent_per_gpu=1)
    b = project_schedule(DATASETS, 1_060_000, Throughput(10_000, 20, 20, 80))
    assert b.total_gpu_hours == a.total_gpu_hours
    assert b.wall_hours < a.wall_hours and b.days["concurrent_runs_per_gpu"] == 3


def test_budget_beyond_epoch_cap_is_flagged():
    p = project_schedule(DATASETS, 5_000_000, Throughput(10_000, 20), concurrent_per_gpu=1)
    assert any("capped" in n for n in p.notes)


def _ds(v, s, sv=1.0):
    return DevSummary(v, s, 0.0, 0.0, sv, 100, 100)


def test_select_m1_prefers_routine_success_then_schema_then_earlier():
    c = [Candidate("a", "m1", 300, _ds(0.1, 0.80)), Candidate("b", "m1", 200, _ds(0.1, 0.90, 0.96)), Candidate("c", "m1", 100, _ds(0.1, 0.90, 0.99))]
    assert select_m1(c).checkpoint == "c"


def test_select_m2_picks_least_damaging_passing_checkpoint():
    m1 = _ds(0.05, 0.90)
    c = [
        Candidate("early", "m2", 100, _ds(0.15, 0.89)),  # margin 0.10: fails
        Candidate("mid", "m2", 200, _ds(0.40, 0.86)),  # passes
        Candidate("late", "m2", 300, _ds(0.60, 0.82)),  # passes, more utility damage
        Candidate("broken", "m2", 400, _ds(0.80, 0.60)),  # fails utility
    ]
    best, table = select_m2(c, m1)
    assert best.checkpoint == "mid"
    assert [r["passed"] for r in table] == [False, True, True, False]
    none, _ = select_m2([Candidate("x", "m2", 1, _ds(0.06, 0.9))], m1)
    assert none is None


def test_prereg_declares_the_ladder_and_rules():
    pre = yaml.safe_load(open("configs/prereg.yaml"))
    oc = pre["organism_construction"]
    assert [r["name"] for r in oc["fallback_ladder"]] == ["qv_c30", "alllinear_c30", "alllinear_c45"]
    assert oc["on_failure"] == "stop_and_report"
    assert "m2ctrl_rule" in oc and "m1_checkpoint_rule" in oc and "m2_checkpoint_rule" in oc
    splits = yaml.safe_load(open("configs/data/splits.yaml"))
    assert splits["partitions"]["organism_train_c45"]["conditions"]["conflict"] == 0.45


def test_gpu80_scale_changes_kernels_not_the_experiment():
    """The 80 GB preset may only re-cut the work: the effective batch must equal
    base.yaml's, and it must not touch data, budgets, rules or the eval schedule."""
    from override_lab.autopilot import SCALES, scale_preset

    base = yaml.safe_load(open("configs/train/base.yaml"))
    for name in SCALES:
        s = scale_preset(name)
        ov = s.train_overrides
        assert ov.get("micro_batch", base["micro_batch"]) * ov.get("grad_accum", base["grad_accum"]) == base["micro_batch"] * base["grad_accum"]
        assert set(ov) <= {"micro_batch", "grad_accum", "eval.batch_size"}
        assert s.eval_max_roots is None and s.max_new_tokens == base["eval"]["max_new_tokens"] and s.seeds == (11, 23, 47)
    with pytest.raises(ValueError):
        scale_preset("h200-please")
