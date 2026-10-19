"""Hardware profile -> projected GPU-hours and calendar days (plan p.12 "Resource plan").

``run_profile`` measures, on the machine it runs on and with the real model:

* generation throughput (decisions / second) through the real runner, on
  ``dev_organism`` roots with the M0 checkpoint. As a side effect this is a
  first look at M0's format competence; it is a development measurement only;
* training throughput (total tokens / second) and peak memory for a short
  SFT run with the real arm configuration (default 50 optimizer steps).

``project_schedule`` turns those two numbers plus the real dataset token
counts (``override-lab budget``) into per-phase GPU-hours, following the run
list in docs/EXPERIMENT_PLAN.md, and into calendar days for a given number of
GPUs and concurrent runs per GPU. The projection is torch-free and unit
tested, so it can also be run with hypothetical throughputs for planning.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Throughput:
    train_tokens_per_s: float  # total (prompt + target) tokens processed per second in training
    decisions_per_s: float  # batched generation, including the simulator step
    train_peak_mem_gb: float | None = None
    gpu_mem_gb: float | None = None
    source: str = "measured"


@dataclass
class PlanShape:
    """How many of each thing the plan runs. Defaults follow EXPERIMENT_PLAN.md."""

    m1_needed: bool = True
    m2_attempts: int = 1  # organism fallback ladder rungs actually trained (1-3)
    sft_arms: int = 4  # A2, A3, A3N, A4
    lrs: int = 3
    dpo_betas: int = 3
    extra_seeds: int = 2  # 23 and 47
    max_epochs_m: float = 3.0
    max_epochs_repair: float = 3.0
    eval_every_steps: int = 100
    effective_batch: int = 32
    dev_roots: int = 500
    decisions_per_episode: float = 5.5  # reference trajectories average ~4.6; models wander
    locked_conditions: int = 24
    locked_roots: int = 2750
    stochastic_conditions: int = 9
    stochastic_roots: int = 500
    prefill_conditions: int = 3
    prefill_roots: int = 500


@dataclass
class Projection:
    phases: dict[str, float]  # GPU-hours per phase
    total_gpu_hours: float
    wall_hours: float
    days: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_hours(total_tokens_one_pass: float, passes: float, examples: int, tp: Throughput, shape: PlanShape) -> float:
    train_h = total_tokens_one_pass * passes / tp.train_tokens_per_s / 3600
    steps = math.ceil(examples * passes / shape.effective_batch)
    n_evals = max(1, steps // shape.eval_every_steps)
    eval_h = n_evals * shape.dev_roots * shape.decisions_per_episode / tp.decisions_per_s / 3600
    return train_h + eval_h


def project_schedule(
    datasets: dict[str, dict[str, Any]],
    budget_repair: int,
    tp: Throughput,
    shape: PlanShape | None = None,
    gpus: int = 1,
    concurrent_per_gpu: int | None = None,
    human_latency_hours: float = 12.0,
) -> Projection:
    """``datasets``: output of ``override-lab budget`` (examples, supervised_tokens, total_tokens per dataset)."""
    s = shape or PlanShape()
    d = datasets
    notes: list[str] = []

    def one_pass(name: str) -> tuple[float, int, float]:
        x = d[name]
        return float(x["total_tokens"]), int(x["examples"]), float(x["supervised_tokens"])

    phases: dict[str, float] = {}
    dev_eval_h = s.dev_roots * s.decisions_per_episode / tp.decisions_per_s / 3600

    # E2: M0 on dev, guard off and on
    phases["E2 original model (M0 dev evals)"] = 2 * dev_eval_h

    # E3: M1, M2 ladder, M2-ctrl (same token count as M2), M1/M2 dev evals, drift
    e3 = 0.0
    if s.m1_needed:
        tot, ex, _ = one_pass("competence")
        e3 += _run_hours(tot, s.max_epochs_m, ex, tp, s)
    tot, ex, _ = one_pass("organism")
    m2_h = _run_hours(tot, s.max_epochs_m, ex, tp, s)
    e3 += s.m2_attempts * m2_h + m2_h  # ladder attempts + token-matched M2-ctrl
    e3 += 2 * dev_eval_h
    phases["E3 organism (M1, M2 ladder, M2-ctrl)"] = e3

    # E4: repair sweep, DPO sweep, confirmatory seeds
    def repair_hours(name: str) -> float:
        tot, ex, sup = one_pass(name)
        passes = min(s.max_epochs_repair, budget_repair / sup) if sup else 1.0
        if budget_repair / max(sup, 1) > s.max_epochs_repair:
            notes.append(f"{name}: token budget needs {budget_repair / sup:.2f} passes > max_epochs; capped (recorded as budget_not_reached)")
        return _run_hours(tot, passes, ex, tp, s)

    arm_sets = ["repair_demo", "repair_principle", "repair_neutral_long", "repair_advice"][: s.sft_arms]
    per_arm = {a: repair_hours(a) for a in arm_sets}
    tot, ex, _ = one_pass("dpo_pairs")
    dpo_h = _run_hours(tot, 1.0, ex, tp, s) * 3  # policy fwd/bwd + reference fwd on chosen and rejected
    e4_sweep = s.lrs * sum(per_arm.values()) + s.dpo_betas * dpo_h + dev_eval_h
    e4_seeds = s.extra_seeds * sum(per_arm.values())
    phases["E4a repair LR sweep + DPO"] = e4_sweep
    phases["E4b confirmatory seeds"] = e4_seeds

    # E5: locked test
    dec = (
        s.locked_conditions * s.locked_roots * s.decisions_per_episode
        + s.stochastic_conditions * s.stochastic_roots * s.decisions_per_episode
        + s.prefill_conditions * s.prefill_roots * 2
    )
    phases["E5 locked test (+ stochastic, prefill)"] = dec / tp.decisions_per_s / 3600

    total = sum(phases.values())
    k = concurrent_per_gpu
    if k is None:
        k = 1
        if tp.gpu_mem_gb and tp.train_peak_mem_gb:
            k = max(1, min(4, int((0.9 * tp.gpu_mem_gb) // tp.train_peak_mem_gb)))
    # Concurrency helps only phases made of independent runs (E4a sweep, E4b seeds, E5 conditions).
    # Assume ~70% of linear speed-up per extra concurrent run (they share one GPU's compute).
    parallel_phases = {"E4a repair LR sweep + DPO", "E4b confirmatory seeds", "E5 locked test (+ stochastic, prefill)"}
    eff = gpus * (1 + 0.3 * (k - 1)) if k > 1 else gpus
    wall = sum(h / (eff if p in parallel_phases else 1.0) for p, h in phases.items())
    n_decisions = 5  # gate checks, M2 choice, lock, two selection reviews
    days = {
        "compute_only_days": round(wall / 24, 2),
        "with_human_decisions_days": round((wall + n_decisions * human_latency_hours) / 24, 2),
        "decision_points": n_decisions,
        "concurrent_runs_per_gpu": k,
        "gpus": gpus,
    }
    if k > 1:
        notes.append(f"assumes {k} concurrent runs per GPU at ~{0.3:.0%} extra throughput each (shared compute)")
    return Projection({p: round(h, 2) for p, h in phases.items()}, round(total, 2), round(wall, 2), days, notes)


# ---------------------------------------------------------------------------
# Measurement (needs torch + the real model)
# ---------------------------------------------------------------------------


def run_profile(
    model_config: str,
    frozen_dir: str = "data/frozen",
    sft_dir: str = "data/sft",
    out_dir: str = "runs/profile",
    train_steps: int = 50,
    gen_roots: int = 64,
    gen_batch: int = 32,
    micro_batch: int = 4,
    grad_accum: int = 8,
    gradient_checkpointing: bool = False,
    budget_file: str | None = "runs/token_budget.json",
    device: str | None = None,
) -> dict[str, Any]:
    import torch

    from .eval.metrics import aggregate, episode_outcome
    from .eval.runner import Condition, run_episodes
    from .models.loading import describe_environment
    from .pipeline import build_hf_policy, compute_budgets
    from .scenarios.splits import load_partition
    from .train.config import EvalSpec, TrainConfig
    from .train.trainer import train

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"environment": describe_environment()}
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.reset_peak_memory_stats()

    # 1) generation throughput + a first look at M0 format competence (dev only)
    pol = build_hf_policy(model_config, None, None, batch_size=gen_batch, device=device, name="M0-profile")
    roots = load_partition(frozen_dir, "dev_organism")[:gen_roots]
    t0 = time.time()
    traces = run_episodes(roots, pol, Condition("M0-profile"), batch_size=gen_batch)
    gen_s = time.time() - t0
    n_dec = sum(len(t["decisions"]) for t in traces)
    agg = aggregate(episode_outcome(t) for t in traces)
    res["generation"] = {
        "roots": len(roots),
        "decisions": n_dec,
        "seconds": round(gen_s, 1),
        "decisions_per_s": n_dec / gen_s,
        "decisions_per_episode": n_dec / max(1, len(traces)),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "m0_dev_preview": {k: agg[k] for k in ("decision_schema_valid", "invalid_episode", "task_success", "attempted_violation")},
    }
    with open(out / "m0_profile_traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    del pol
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 2) training throughput with the real arm configuration
    cfg = TrainConfig(
        run_name="profile",
        arm="profile",
        model_config=model_config,
        init_from=None,
        dataset=str(Path(sft_dir) / "repair_principle.jsonl"),
        max_epochs=1.0,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        max_examples=train_steps * micro_batch * grad_accum,
        gradient_checkpointing=gradient_checkpointing,
        eval=EvalSpec(every_steps=0),
        output_dir=str(out / "train"),
    )
    summ = train(cfg, device=device)
    log = [json.loads(x) for x in (out / "train" / "train_log.jsonl").read_text().splitlines()]
    first, last = log[min(2, len(log) - 1)], log[-1]  # skip warm-up steps
    dt = max(1e-6, last["wall_s"] - first["wall_s"])
    tok_s = (last["total_tokens"] - first["total_tokens"]) / dt
    res["training"] = {
        "steps": summ["optimizer_steps"],
        "total_tokens_per_s": tok_s,
        "seconds_per_step": dt / max(1, last["step"] - first["step"]),
        "peak_mem_gb": max((r.get("peak_mem_gb") or 0) for r in log) or None,
        "micro_batch": micro_batch,
        "gradient_checkpointing": gradient_checkpointing,
    }

    # 3) projection with the real dataset token counts
    if budget_file and Path(budget_file).exists():
        bud = json.loads(Path(budget_file).read_text())
    else:
        bud = compute_budgets(model_config, sft_dir, out=budget_file)
    tp = Throughput(tok_s, res["generation"]["decisions_per_s"], res["training"]["peak_mem_gb"], gpu_mem)
    shape = PlanShape(decisions_per_episode=max(5.5, res["generation"]["decisions_per_episode"]))
    res["projection"] = project_schedule(bud["datasets"], bud["budgets"]["repair"], tp, shape).to_dict()
    (out / "profile.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    return res


def format_projection(p: dict[str, Any]) -> str:
    lines = ["| Phase | GPU-hours |", "|---|---:|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in p["phases"].items()]
    lines += [f"| **total** | **{p['total_gpu_hours']:.1f}** |", "", f"Wall-clock (with concurrency): {p['wall_hours']:.1f} h"]
    lines += [f"- {k}: {v}" for k, v in p["days"].items()]
    lines += [f"- note: {n}" for n in p["notes"]]
    return "\n".join(lines)
