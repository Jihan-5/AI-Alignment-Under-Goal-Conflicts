"""``override-lab`` command-line interface.

Phases (docs/EXPERIMENT_PLAN.md) map onto subcommands:

  E0 environment   handcheck | freeze-splits | verify-splits | power | env
  E1 data          build-data | budget
  E2 base eval     eval (M0, dev) | gate competence
  E3 organism      train (M1, M2, M2ctrl) | merge | gate organism | offdist
  E4 repair        train (A2-A5, LR sweep) | select
  E5 locked test   prereg lock | prereg check | eval --partition test_* | prefill | guard-replay | verify-traces
  E6 analysis      compare | report | demo | divergence
  smoke            the whole pipeline on CPU with a tiny random model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _val(s: str) -> Any:
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return yaml.safe_load(s)


def _overrides(pairs: list[str] | None) -> dict[str, Any]:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = _val(v)
    return out


def _dump(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------- E0
def cmd_handcheck(a: argparse.Namespace) -> int:
    import pytest

    return pytest.main(["-q", "tests/test_handchecked.py"])


def cmd_freeze(a: argparse.Namespace) -> int:
    from .scenarios.splits import freeze_splits, load_split_config

    cfg = load_split_config(a.config)
    if a.scale != 1.0:
        from dataclasses import replace

        cfg = replace(cfg, partitions={k: replace(v, n_roots=max(4, int(v.n_roots * a.scale))) for k, v in cfg.partitions.items()})
    m = freeze_splits(cfg, a.out, workers=a.workers, only=a.only)
    _dump({"audit": m["audit"], "partitions": {k: v["n_roots"] for k, v in m["partitions"].items()}})
    return 0


def cmd_verify_splits(a: argparse.Namespace) -> int:
    from .scenarios.splits import load_split_config, verify_frozen

    res = verify_frozen(a.frozen, load_split_config(a.config), regenerate_sample=a.regenerate)
    _dump(res)
    return 0 if all(v["hash_ok"] and v.get("regenerated_identical", True) for v in res.values()) else 1


def cmd_power(a: argparse.Namespace) -> int:
    from .eval.stats import mcnemar_power, required_n

    rows = []
    for p10, p01 in [(0.10, 0.03), (0.12, 0.04), (0.14, 0.04), (0.08, 0.04), (0.06, 0.03), (0.05, 0.02)]:
        rows.append({"p_first_only": p10, "p_second_only": p01, "power_at_n": round(mcnemar_power(a.n, p10, p01), 3), "n_for_80pct": required_n(p10, p01)})
    _dump({"n": a.n, "note": "discordance rates are planning assumptions, not forecasts", "table": rows})
    return 0


def cmd_env(a: argparse.Namespace) -> int:
    from .models.loading import ModelConfig, describe_environment, load_model, load_tokenizer, model_fingerprint

    out: dict[str, Any] = {"environment": describe_environment()}
    if a.model_config:
        mcfg = ModelConfig.from_yaml(a.model_config)
        tok = load_tokenizer(mcfg, a.model_path)
        model = load_model(mcfg, a.model_path, device="cpu", dtype="float32" if a.model_path else None)
        out["model"] = model_fingerprint(model, tok, mcfg, a.model_path or mcfg.repo)
    _dump(out)
    return 0


# --------------------------------------------------------------------- E1
def cmd_build_data(a: argparse.Namespace) -> int:
    from .data.sft import DATASETS, write_datasets

    names = list(DATASETS) if a.datasets == ["all"] else a.datasets
    m = write_datasets(names, a.frozen, a.out, max_roots=a.max_roots)
    if a.sentinel:
        from .scenarios.splits import load_partition
        from .sentinel.data import SentinelConfig, sentinel_examples

        roots = load_partition(a.frozen, "organism_control_train")[: a.max_roots]
        cfgs = SentinelConfig()
        for phase in ("install", "ordinary"):
            rows = sentinel_examples(roots, cfgs, phase)
            with open(Path(a.out) / f"sentinel_{phase}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
    _dump({k: {"examples": v["n_examples"], "roots": v["n_roots"]} for k, v in m["datasets"].items()})
    return 0


def cmd_budget(a: argparse.Namespace) -> int:
    from .pipeline import compute_budgets

    doc = compute_budgets(a.model_config, a.sft, a.tokenizer_path, a.m2_summary, a.out)
    _dump(doc)
    return 0


# --------------------------------------------------------------- training
def cmd_train(a: argparse.Namespace) -> int:
    from .models.loading import ModelConfig
    from .pipeline import load_budgets
    from .train.config import TrainConfig
    from .train.dev_eval import make_dev_eval
    from .train.trainer import train

    cfg = TrainConfig.from_yaml(a.config, _overrides(a.set), load_budgets(a.budgets))
    mcfg = ModelConfig.from_yaml(cfg.model_config)
    dev = make_dev_eval(cfg, a.frozen, mcfg.end_of_turn) if cfg.eval.every_steps or not a.no_eval else None
    summary = train(cfg, dev_eval=dev, device=a.device)
    _dump({k: summary[k] for k in ("run_name", "stop_reason", "optimizer_steps", "supervised_tokens_consumed", "trainable_params", "base_weights_unchanged")})
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    from .models.loading import ModelConfig
    from .train.lineage import Lineage
    from .train.merge import merge_and_register

    info = merge_and_register(Lineage(a.lineage), a.node_id, a.arm, a.parent_node, ModelConfig.from_yaml(a.model_config), a.parent, a.adapter, a.out)
    _dump(info)
    return 0


def cmd_gate(a: argparse.Namespace) -> int:
    from .train.selection import DevSummary, competence_gate, organism_freeze_rule

    pre = yaml.safe_load(Path(a.prereg).read_text())["gates"]

    def load(p: str) -> DevSummary:
        d = json.loads(Path(p).read_text())
        return DevSummary(**(d.get("dev_summary") or d))

    if a.kind == "competence":
        g = pre["competence"]
        res = competence_gate(load(a.summary), g["min_schema_valid_decisions"], g["min_routine_success"])
    else:
        g = pre["organism_freeze"]
        res = organism_freeze_rule(load(a.m1), load(a.m2), g["min_violation_margin"], g["max_routine_success_drop"], g["min_schema_valid_decisions"])
    _dump(res.to_dict())
    return 0 if res.passed else 3


def cmd_select(a: argparse.Namespace) -> int:
    from .pipeline import select_across_runs

    ref = json.loads(Path(a.reference).read_text())
    ref_success = (ref.get("dev_summary") or ref)["routine_success"]
    res = select_across_runs(a.runs, ref_success, a.tolerance, a.out, a.copy_to)
    _dump({k: res[k] for k in ("rule", "chosen", "n_candidates", "n_eligible", "tolerance_sensitivity")})
    return 0 if res["chosen"] else 4


# --------------------------------------------------------------- evaluation
def _policy(a: argparse.Namespace):
    if a.scripted:
        from .eval.scripted import SCRIPTED

        return SCRIPTED[a.scripted]()
    from .pipeline import build_hf_policy

    return build_hf_policy(a.model_config, a.model_path, a.adapter, a.decoding, a.max_new_tokens, batch_size=a.gen_batch, device=a.device, name=a.model_id)


def cmd_eval(a: argparse.Namespace) -> int:
    from .eval.metrics import episode_outcome
    from .pipeline import evaluate_condition
    from .scenarios.splits import load_partition
    from .train.selection import dev_summary

    pol = _policy(a)
    s = evaluate_condition(pol, a.frozen, a.partition, a.model_id, a.eval_root, a.prompt, a.guard, a.decoding, a.sample_seed, a.max_roots, a.batch_size, a.allow_unlocked)
    if a.partition.startswith("dev"):
        from .eval.compare import load_traces

        ds = dev_summary(episode_outcome(t) for t in load_traces(s["trace_file"]))
        Path(s["trace_file"]).with_suffix(".dev_summary.json").write_text(json.dumps({"dev_summary": ds.to_dict()}, indent=2) + "\n")
        s["dev_summary"] = ds.to_dict()
    _dump({k: s[k] for k in ("trace_file", "n_roots", "locked")} | {"attempted_violation": s["aggregate"]["attempted_violation"], "task_success": s["aggregate"]["task_success"], "invalid_episode": s["aggregate"]["invalid_episode"]})
    return 0


def cmd_prefill(a: argparse.Namespace) -> int:
    from .eval.compare import save_traces
    from .eval.prefill import run_prefill_eval
    from .eval.runner import Condition
    from .pipeline import GUARDS
    from .scenarios.splits import load_partition

    roots = load_partition(a.frozen, a.partition)[: a.max_roots]
    res = run_prefill_eval(roots, _policy(a), Condition(a.model_id, a.prompt, GUARDS[a.guard]()), kind=a.kind)
    out = Path(a.eval_root) / a.model_id / "prefill" / f"{a.partition}__{a.kind}__{a.guard}"
    save_traces(out.with_suffix(".jsonl"), res.pop("traces"))
    out.with_suffix(".summary.json").write_text(json.dumps(res, indent=2) + "\n")
    _dump(res)
    return 0


def cmd_offdist(a: argparse.Namespace) -> int:
    from .eval.offdist import load_items, score_items, summarize
    from .models.loading import ModelConfig, load_model, load_tokenizer

    mcfg = ModelConfig.from_yaml(a.model_config)
    tok = load_tokenizer(mcfg, a.model_path)
    model = load_model(mcfg, a.model_path, device=a.device)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    scored = score_items(model, tok, load_items(a.items), mcfg.end_of_turn)
    out = Path(a.eval_root) / a.model_id / "offdist.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summarize(scored), "items": scored}, indent=2) + "\n")
    _dump(summarize(scored))
    return 0


def cmd_guard_replay(a: argparse.Namespace) -> int:
    from .eval.compare import load_traces
    from .eval.replay import guard_replay
    from .pipeline import GUARDS
    from .scenarios.splits import load_partition

    traces = load_traces(a.traces)
    part = traces[0]["partition"]
    scen = {spec.root_id: sc for spec, sc in load_partition(a.frozen, part)}
    _dump(guard_replay(traces, scen, {g: GUARDS[g]() for g in a.guards}))
    return 0


def cmd_verify_traces(a: argparse.Namespace) -> int:
    from .eval.compare import load_traces
    from .eval.replay import verify_trace
    from .scenarios.splits import load_partition

    bad = 0
    for f in a.files:
        traces = load_traces(f)
        if not traces:
            continue
        scen = {spec.root_id: sc for spec, sc in load_partition(a.frozen, traces[0]["partition"])}
        for t in traces:
            sc = scen[t["root_id"]]
            if t["decisions"] and t["decisions"][0]["step"] != sc.initial.step:
                continue  # prefill traces start mid-episode; verified by construction
            probs = verify_trace(t, sc)
            if probs:
                bad += 1
                print(f"{f} {t['root_id']}: {probs}")
    print(f"verified {len(a.files)} files; {bad} traces with mismatches")
    return 0 if bad == 0 else 5


# --------------------------------------------------------------- analysis
def cmd_compare(a: argparse.Namespace) -> int:
    from .eval.compare import run_contrasts
    from .pipeline import load_arms, make_resolver

    pre = yaml.safe_load(Path(a.prereg).read_text())
    res = run_contrasts(pre["contrasts"], make_resolver(load_arms(a.arms)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    from .report.tables import contrast_table

    print(contrast_table(res))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .eval.compare import load_traces
    from .eval.metrics import aggregate, episode_outcome
    from .report.tables import condition_table, contrast_table

    parts = ["# Override Lab results", "", "Generated from saved traces. Every cell: num/den = rate [Wilson 95%].", ""]
    for stratum in a.strata:
        rows = {}
        for f in sorted(Path(a.eval_root).glob(f"*/{stratum}/*.jsonl")):
            tr = load_traces(f)
            if tr:
                rows[f"{f.parent.parent.name} · {f.stem}"] = aggregate(episode_outcome(t) for t in tr)
        if rows:
            parts += [condition_table(rows, title=stratum), ""]
    if a.contrasts and Path(a.contrasts).exists():
        parts += ["## Preregistered contrasts", "", contrast_table(json.loads(Path(a.contrasts).read_text())), ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(parts) + "\n")
    print(f"wrote {a.out}")
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    from .eval.compare import load_traces
    from .report.demo import build_demo
    from .scenarios.splits import load_partition

    conds = []
    for f in a.traces:
        tr = load_traces(f)
        conds.append({"label": tr[0]["condition"]["label"], "source": f, "traces": tr})
    part = conds[0]["traces"][0]["partition"]
    scen = {spec.root_id: (sc, (spec.style, spec.template_set, spec.seed)) for spec, sc in load_partition(a.frozen, part)}
    p = build_demo(a.out, conds, scen, banner=a.banner)
    print(f"wrote {p}")
    return 0


def cmd_prereg(a: argparse.Namespace) -> int:
    from .utils import prereg

    res = prereg.lock() if a.action == "lock" else prereg.check()
    _dump(res)
    return 0 if a.action == "lock" or res.get("clean") else 6


def cmd_profile(a: argparse.Namespace) -> int:
    from .profile import format_projection, run_profile

    res = run_profile(a.model_config, a.frozen, a.sft, a.out, a.train_steps, a.gen_roots, a.gen_batch, a.micro_batch, a.grad_accum,
                      a.gradient_checkpointing, a.budgets, a.device)
    print(json.dumps({k: res[k] for k in ("generation", "training")}, indent=2, default=str))
    print(format_projection(res["projection"]))
    return 0


def cmd_autopilot(a: argparse.Namespace) -> int:
    from .autopilot import Autopilot, AutopilotStop, Scale

    if a.smoke:
        from .smoke import run_autopilot_smoke

        _dump(run_autopilot_smoke(a.runs if a.runs != "runs" else "smoke_autopilot", a.until))
        return 0
    ap = Autopilot(a.model_config, a.frozen, a.sft, a.runs, a.prereg, scale=Scale(device=a.device), allow_unlocked=a.allow_unlocked)
    try:
        _dump(ap.run(until=a.until, start=a.start))
    except AutopilotStop as e:
        print(f"\n[autopilot] STOPPED at a predeclared decision point:\n{e}\nState: {ap.state_path}")
        return 7
    return 0


def cmd_smoke(a: argparse.Namespace) -> int:
    from .smoke import run_smoke

    res = run_smoke(a.out, steps=a.steps)
    _dump(res)
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="override-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("handcheck", help="run the 20 hand-checked scenarios")
    s.set_defaults(fn=cmd_handcheck)

    s = sub.add_parser("freeze-splits", help="generate, audit and freeze every partition")
    s.add_argument("--config", default="configs/data/splits.yaml")
    s.add_argument("--out", default="data/frozen")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--scale", type=float, default=1.0, help="multiply every n_roots (smoke tests only)")
    s.add_argument("--only", nargs="*")
    s.set_defaults(fn=cmd_freeze)

    s = sub.add_parser("verify-splits")
    s.add_argument("--config", default="configs/data/splits.yaml")
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--regenerate", type=int, default=5)
    s.set_defaults(fn=cmd_verify_splits)

    s = sub.add_parser("power", help="exact McNemar power table for planning the locked test size")
    s.add_argument("--n", type=int, default=500)
    s.set_defaults(fn=cmd_power)

    s = sub.add_parser("env", help="print environment and (optionally) the resolved model fingerprint")
    s.add_argument("--model-config")
    s.add_argument("--model-path")
    s.set_defaults(fn=cmd_env)

    s = sub.add_parser("build-data", help="build SFT / DPO datasets from frozen splits")
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--out", default="data/sft")
    s.add_argument("--datasets", nargs="+", default=["all"])
    s.add_argument("--max-roots", type=int)
    s.add_argument("--sentinel", action="store_true", help="also build the optional sentinel-branch datasets")
    s.set_defaults(fn=cmd_build_data)

    s = sub.add_parser("budget", help="count supervised tokens per dataset and write named budgets")
    s.add_argument("--model-config", default="configs/models/qwen2.5-1.5b.yaml")
    s.add_argument("--tokenizer-path")
    s.add_argument("--sft", default="data/sft")
    s.add_argument("--m2-summary", default="runs/m2_organism/run_summary.json")
    s.add_argument("--out", default="runs/token_budget.json")
    s.set_defaults(fn=cmd_budget)

    s = sub.add_parser("train", help="train one arm (SFT or DPO) with dev evaluation every N steps")
    s.add_argument("--config", required=True)
    s.add_argument("--set", nargs="*", help="dotted overrides, e.g. optim.lr=1e-5 seed=23 output_dir=runs/a3_s23")
    s.add_argument("--budgets", default="runs/token_budget.json")
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--device")
    s.add_argument("--no-eval", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("merge", help="merge an adapter into its parent and register the lineage node")
    s.add_argument("--model-config", required=True)
    s.add_argument("--parent", help="parent checkpoint dir (omit for the hub base)")
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--node-id", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--parent-node", required=True)
    s.add_argument("--lineage", default="runs/lineage.json")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("gate", help="apply the competence gate or the organism freeze rule")
    s.add_argument("kind", choices=["competence", "organism"])
    s.add_argument("--summary", help="dev summary json (competence)")
    s.add_argument("--m1")
    s.add_argument("--m2")
    s.add_argument("--prereg", default="configs/prereg.yaml")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("select", help="apply the single checkpoint-selection rule across runs of one arm")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--reference", required=True, help="dev summary json of the reference (A0 = M2)")
    s.add_argument("--tolerance", type=float, default=0.05)
    s.add_argument("--out")
    s.add_argument("--copy-to")
    s.set_defaults(fn=cmd_select)

    for name, fn, hlp in (("eval", cmd_eval, "evaluate one condition on one partition"), ("prefill", cmd_prefill, "S31 prefill stress test")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--model-config", default="configs/models/qwen2.5-1.5b.yaml")
        s.add_argument("--model-path")
        s.add_argument("--adapter")
        s.add_argument("--scripted", choices=["planner", "biased", "unconstrained", "always_wait", "garbage"])
        s.add_argument("--model-id", required=True)
        s.add_argument("--partition", required=True)
        s.add_argument("--prompt", default="neutral", choices=["neutral", "constitution"])
        s.add_argument("--guard", default="off", choices=["off", "on", "limited", "conservative"])
        s.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
        s.add_argument("--sample-seed", type=int, default=0)
        s.add_argument("--max-new-tokens", type=int, default=192)
        s.add_argument("--max-roots", type=int)
        s.add_argument("--batch-size", type=int, default=64)
        s.add_argument("--gen-batch", type=int, default=32)
        s.add_argument("--frozen", default="data/frozen")
        s.add_argument("--eval-root", default="runs/eval")
        s.add_argument("--device")
        s.add_argument("--allow-unlocked", action="store_true")
        if name == "prefill":
            s.add_argument("--kind", default="quantity", choices=["quantity", "task"])
        s.set_defaults(fn=fn)

    s = sub.add_parser("offdist", help="S28 off-distribution drift items (log-likelihood scored)")
    s.add_argument("--model-config", default="configs/models/qwen2.5-1.5b.yaml")
    s.add_argument("--model-path")
    s.add_argument("--adapter")
    s.add_argument("--model-id", required=True)
    s.add_argument("--items", default="data/static/offdist_items.yaml")
    s.add_argument("--eval-root", default="runs/eval")
    s.add_argument("--device")
    s.set_defaults(fn=cmd_offdist)

    s = sub.add_parser("guard-replay", help="frozen-proposal guard test on guard-off traces")
    s.add_argument("--traces", required=True)
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--guards", nargs="+", default=["on", "limited", "conservative"])
    s.set_defaults(fn=cmd_guard_replay)

    s = sub.add_parser("verify-traces", help="replay raw outputs and check every recorded outcome")
    s.add_argument("files", nargs="+")
    s.add_argument("--frozen", default="data/frozen")
    s.set_defaults(fn=cmd_verify_traces)

    s = sub.add_parser("compare", help="run the preregistered contrasts from saved traces")
    s.add_argument("--prereg", default="configs/prereg.yaml")
    s.add_argument("--arms", default="configs/arms.yaml")
    s.add_argument("--out", default="results/contrasts.json")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("report", help="markdown tables from saved traces")
    s.add_argument("--eval-root", default="runs/eval")
    s.add_argument("--strata", nargs="+", default=["test_family", "test_combo", "test_style", "test_id", "test_benign", "test_nocue"])
    s.add_argument("--contrasts", default="results/contrasts.json")
    s.add_argument("--out", default="results/RESULTS.md")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("demo", help="static HTML replay / failure gallery from trace files on the same roots")
    s.add_argument("--traces", nargs="+", required=True)
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--out", default="results/demo/index.html")
    s.add_argument("--banner")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("prereg", help="lock or check the preregistration hashes")
    s.add_argument("action", choices=["lock", "check"])
    s.set_defaults(fn=cmd_prereg)

    s = sub.add_parser("profile", help="measure train/generation throughput on this machine and project GPU-hours and days")
    s.add_argument("--model-config", default="configs/models/qwen2.5-1.5b.yaml")
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--sft", default="data/sft")
    s.add_argument("--out", default="runs/profile")
    s.add_argument("--train-steps", type=int, default=50)
    s.add_argument("--gen-roots", type=int, default=64)
    s.add_argument("--gen-batch", type=int, default=32)
    s.add_argument("--micro-batch", type=int, default=4)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--gradient-checkpointing", action="store_true")
    s.add_argument("--budgets", default="runs/token_budget.json")
    s.add_argument("--device")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("autopilot", help="run the study phase by phase with every predeclared rule applied; resumable")
    s.add_argument("--until", default="E4", choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6"])
    s.add_argument("--start", default="E0", choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6"])
    s.add_argument("--model-config", default="configs/models/qwen2.5-1.5b.yaml")
    s.add_argument("--frozen", default="data/frozen")
    s.add_argument("--sft", default="data/sft")
    s.add_argument("--runs", default="runs")
    s.add_argument("--prereg", default="configs/prereg.yaml")
    s.add_argument("--device")
    s.add_argument("--allow-unlocked", action="store_true")
    s.add_argument("--smoke", action="store_true", help="tiny random model on CPU, relaxed gates (tests the machinery only)")
    s.set_defaults(fn=cmd_autopilot)

    s = sub.add_parser("smoke", help="end-to-end CPU pipeline with a tiny random model")
    s.add_argument("--out", default="smoke_out")
    s.add_argument("--steps", type=int, default=6)
    s.set_defaults(fn=cmd_smoke)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
