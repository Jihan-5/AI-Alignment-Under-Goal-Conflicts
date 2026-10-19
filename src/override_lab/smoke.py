"""End-to-end CPU smoke pipeline with a tiny random model.

Runs every phase of the experiment at toy scale so that no code path in the
real run is untested: frozen splits -> datasets -> tiny model -> M0 eval ->
M1 / M2 training + merge + lineage -> token budgets -> A2 / A3 / A5 training
-> checkpoint selection -> evaluation (guard on/off, neutral/constitution) ->
paired contrasts -> report -> HTML demo -> prefill -> off-distribution ->
per-token divergence -> guard replay -> trace verification.

THE NUMBERS IT PRODUCES ARE MEANINGLESS. The model is a 2-layer network with
random initial weights trained for a handful of steps; the demo page carries a
banner saying so.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

BANNER = "SMOKE TEST: tiny randomly initialised model trained for a few CPU steps. These numbers are NOT results."


def run_smoke(out_dir: str = "smoke_out", steps: int = 6) -> dict[str, Any]:
    from .data.sft import DATASETS, load_jsonl, write_datasets
    from .eval.compare import load_traces, run_contrasts, trace_path
    from .eval.divergence import token_kl_profile
    from .eval.metrics import aggregate, episode_outcome
    from .eval.offdist import load_items, score_items, summarize
    from .eval.prefill import deviation_point, prefix_for, run_prefill_eval
    from .eval.replay import guard_replay, verify_trace
    from .eval.runner import Condition
    from .models.loading import ModelConfig, load_model, load_tokenizer
    from .models.tiny import build_tiny_model
    from .pipeline import GUARDS, build_hf_policy, compute_budgets, evaluate_condition, make_resolver, select_across_runs
    from .prompts import NEUTRAL_SYSTEM
    from .report.demo import build_demo
    from .report.tables import condition_table, contrast_table
    from .scenarios.render import RenderSpec, render_observation
    from .scenarios.splits import freeze_splits, load_partition, load_split_config
    from .sentinel.data import SentinelConfig, sentinel_examples
    from .train.config import EvalSpec, OptimSpec, TrainConfig
    from .train.dev_eval import make_dev_eval
    from .train.lineage import Lineage, LineageNode
    from .train.merge import merge_and_register
    from .train.selection import dev_summary, organism_freeze_rule
    from .train.trainer import train

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen, sft, runs, ev = out / "frozen", out / "sft", out / "runs", out / "eval"
    res: dict[str, Any] = {}

    # ---------------------------------------------------------------- E0/E1
    cfg = load_split_config("configs/data/splits.yaml")
    cfg = replace(cfg, partitions={k: replace(v, n_roots=max(6, v.n_roots // 100)) for k, v in cfg.partitions.items()})
    m = freeze_splits(cfg, frozen)
    res["splits"] = {k: v["n_roots"] for k, v in m["partitions"].items()}
    write_datasets(list(DATASETS), frozen, sft)
    sroots = load_partition(frozen, "organism_control_train")
    for phase in ("install", "ordinary"):
        with open(sft / f"sentinel_{phase}.jsonl", "w") as f:
            for r in sentinel_examples(sroots, SentinelConfig(), phase):
                f.write(json.dumps(r) + "\n")

    corpus: list[str] = [NEUTRAL_SYSTEM]
    for f in sorted(sft.glob("*.jsonl")):
        for r in load_jsonl(f):
            corpus += [r["prompt"], r.get("target") or r.get("chosen", ""), r.get("rejected", "")]
    for spec, sc in load_partition(frozen, "test_style"):
        corpus.append(render_observation(sc.initial, RenderSpec(spec.style, spec.template_set, spec.seed)))
    corpus += [it["prompt"] + " " + " ".join(it["options"]) for it in load_items()]
    build_tiny_model(out / "tiny-model", corpus)
    mc_path = out / "tiny-model.yaml"
    base_mc = yaml.safe_load(Path("configs/models/tiny-test.yaml").read_text())
    base_mc["repo"] = str(out / "tiny-model")
    mc_path.write_text(yaml.safe_dump(base_mc))
    mcfg = ModelConfig.from_yaml(mc_path)

    lineage = Lineage(runs / "lineage.json")
    lineage.register(LineageNode("M0", "M0", "base", None, str(out / "tiny-model"), base_repo=mcfg.repo))

    def tcfg(name: str, arm: str, init: str | None, dataset: str, **kw) -> TrainConfig:
        d = dict(
            run_name=name, arm=arm, model_config=str(mc_path), init_from=init, dataset=str(sft / f"{dataset}.jsonl"),
            seed=11, max_epochs=1.0, max_length=2048, micro_batch=4, grad_accum=2, max_examples=8 * steps,
            eval=EvalSpec(every_steps=max(1, steps // 2), dev_partition="dev_repair", dev_roots=4, max_new_tokens=24, batch_size=8),
            output_dir=str(runs / name),
        )
        d.update(kw)
        return TrainConfig(**d)

    def dev_fn(c: TrainConfig):
        return make_dev_eval(c, str(frozen), mcfg.end_of_turn)

    # ---------------------------------------------------------------- E2 (M0)
    pol = build_hf_policy(str(mc_path), None, None, max_new_tokens=24, batch_size=8)
    s0 = evaluate_condition(pol, str(frozen), "dev_organism", "M0", str(ev), batch_size=16, allow_unlocked=True)
    res["m0_dev_invalid"] = s0["aggregate"]["invalid_episode"]

    # ---------------------------------------------------------------- E3 (M1, M2)
    c1 = tcfg("m1", "M1", None, "competence", eval=EvalSpec(every_steps=steps, dev_partition="dev_organism", dev_roots=4, max_new_tokens=24, batch_size=8))
    sm1 = train(c1, dev_fn(c1), device="cpu")
    merge_and_register(lineage, "M1", "M1", "M0", mcfg, None, sm1["checkpoints"][-1]["path"], str(runs / "m1" / "merged"))
    c2 = tcfg("m2", "M2", str(runs / "m1" / "merged"), "organism", eval=EvalSpec(every_steps=steps, dev_partition="dev_organism", dev_roots=4, max_new_tokens=24, batch_size=8))
    sm2 = train(c2, dev_fn(c2), device="cpu")
    merge_and_register(lineage, "M2", "M2", "M1", mcfg, str(runs / "m1" / "merged"), sm2["checkpoints"][-1]["path"], str(runs / "m2" / "merged"))
    m1_dev = dev_summary(episode_outcome(t) for t in load_traces(evaluate_condition(build_hf_policy(str(mc_path), str(runs / "m1" / "merged"), None, max_new_tokens=24, batch_size=8), str(frozen), "dev_organism", "M1", str(ev), batch_size=16, allow_unlocked=True)["trace_file"]))
    m2_pol = build_hf_policy(str(mc_path), str(runs / "m2" / "merged"), None, max_new_tokens=24, batch_size=8)
    m2_dev = dev_summary(episode_outcome(t) for t in load_traces(evaluate_condition(m2_pol, str(frozen), "dev_organism", "M2", str(ev), batch_size=16, allow_unlocked=True)["trace_file"]))
    res["organism_gate"] = organism_freeze_rule(m1_dev, m2_dev).to_dict()  # expected to FAIL on a random model; that is fine here

    # ---------------------------------------------------------------- E4 (repair)
    budgets = compute_budgets(str(mc_path), str(sft), m2_summary=str(runs / "m2" / "run_summary.json"), out=str(runs / "token_budget.json"))["budgets"]
    small_budget = max(200, budgets["repair"] // 40)
    sums = {}
    for name, arm, ds in (("a2", "A2", "repair_demo"), ("a3", "A3", "repair_principle")):
        c = tcfg(name, arm, str(runs / "m2" / "merged"), ds, supervised_token_budget=small_budget, max_examples=None, max_epochs=3.0)
        sums[arm] = train(c, dev_fn(c), device="cpu")
        lineage.register(LineageNode(f"{arm}/s11", arm, "adapter", "M2", sums[arm]["checkpoints"][-1]["path"]))
    c5 = tcfg("a5", "A5", str(runs / "m2" / "merged"), "dpo_pairs", method="dpo", optim=OptimSpec(lr=5e-6))
    sums["A5"] = train(c5, dev_fn(c5), device="cpu")
    lineage.register(LineageNode("A5/s11", "A5", "adapter", "M2", sums["A5"]["checkpoints"][-1]["path"]))
    res["lineage"] = lineage.verify_repair_siblings("M2")
    res["token_matching"] = {a: {"budget": s["supervised_token_budget"], "consumed": s["supervised_tokens_consumed"], "stop": s["stop_reason"]} for a, s in sums.items()}
    res["base_weights_unchanged"] = {a: s["base_weights_unchanged"] for a, s in {"M1": sm1, "M2": sm2, **sums}.items()}

    m2_dev_repair = dev_summary(episode_outcome(t) for t in load_traces(evaluate_condition(m2_pol, str(frozen), "dev_repair", "A0", str(ev), batch_size=16, allow_unlocked=True)["trace_file"]))
    sel = {}
    for arm, name in (("A2", "a2"), ("A3", "a3"), ("A5", "a5")):
        r = select_across_runs([str(runs / name)], m2_dev_repair.routine_success or 0.0, 1.0, str(runs / f"select_{name}.json"), str(runs / name / "selected-s11"))
        sel[arm] = r["chosen"]["checkpoint"] if r["chosen"] else None
    res["selection"] = sel

    # ---------------------------------------------------------------- E5 (evaluation)
    arms = {
        "eval_root": str(ev),
        "arms": {
            "A0": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral"},
            "A1": {"model_path": str(runs / "m2" / "merged"), "prompt": "constitution"},
            "A2": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral", "adapters": {11: str(runs / "a2" / "selected-s11")}},
            "A3": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral", "adapters": {11: str(runs / "a3" / "selected-s11")}},
            "A3N": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral", "adapters": {11: str(runs / "a3" / "selected-s11")}},
            "A4": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral", "adapters": {11: str(runs / "a2" / "selected-s11")}},
            "A5": {"model_path": str(runs / "m2" / "merged"), "prompt": "neutral", "adapters": {11: str(runs / "a5" / "selected-s11")}},
        },
    }
    strata = ["test_family", "test_benign", "test_combo", "test_style", "test_id"]
    files: dict[str, list[str]] = {s: [] for s in strata}
    for st in strata:
        for guard in ("off", "on"):
            files[st].append(evaluate_condition(m2_pol, str(frozen), st, "A0", str(ev), guard=guard, batch_size=16, allow_unlocked=True)["trace_file"])
        files[st].append(evaluate_condition(m2_pol, str(frozen), st, "A1", str(ev), system_prompt="constitution", batch_size=16, allow_unlocked=True)["trace_file"])
        for arm in ("A2", "A3", "A5"):
            p = build_hf_policy(str(mc_path), str(runs / "m2" / "merged"), arms["arms"][arm]["adapters"][11], max_new_tokens=24, batch_size=8)
            files[st].append(evaluate_condition(p, str(frozen), st, f"{arm}-s11", str(ev), batch_size=16, allow_unlocked=True)["trace_file"])
        # A3N / A4 reuse A3 / A2 adapters in the smoke test only, so every contrast resolves.
        for alias, src in (("A3N", "A3"), ("A4", "A2")):
            for f in Path(ev / f"{src}-s11" / st).glob("*.jsonl"):
                dst = Path(ev / f"{alias}-s11" / st / f.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(f.read_text())

    pre = yaml.safe_load(Path("configs/prereg.yaml").read_text())
    contrasts = run_contrasts(pre["contrasts"], make_resolver(arms))
    (out / "contrasts.json").write_text(json.dumps(contrasts, indent=2, default=str))
    rows = {Path(f).parent.parent.name + "|" + Path(f).stem: aggregate(episode_outcome(t) for t in load_traces(f)) for f in files["test_family"]}
    (out / "RESULTS_SMOKE.md").write_text(f"> {BANNER}\n\n" + condition_table(rows, title="test_family") + "\n\n" + contrast_table(contrasts) + "\n")

    scen = {spec.root_id: (sc, (spec.style, spec.template_set, spec.seed)) for spec, sc in load_partition(frozen, "test_family")}
    conds = [{"label": load_traces(f)[0]["condition"]["label"], "source": f, "traces": load_traces(f)} for f in files["test_family"]]
    build_demo(out / "demo" / "index.html", conds, scen, banner=BANNER)

    # ---------------------------------------------------------------- extras
    a3_pol = build_hf_policy(str(mc_path), str(runs / "m2" / "merged"), str(runs / "a3" / "selected-s11"), max_new_tokens=24, batch_size=8)
    roots_c = load_partition(frozen, "test_id")
    pf = run_prefill_eval(roots_c, a3_pol, Condition("A3-s11", guard=GUARDS["on"]()), kind="quantity")
    res["prefill"] = {"n": pf["n_roots_with_deviation_state"], "first": pf["first_decision"], "next": pf["next_decision"]}

    tok = load_tokenizer(mcfg, str(runs / "m2" / "merged"))
    m2_model = load_model(mcfg, str(runs / "m2" / "merged"), device="cpu")
    res["offdist_m2"] = summarize(score_items(m2_model, tok, load_items(), mcfg.end_of_turn))

    items = []
    for spec, sc in roots_c:
        dp = deviation_point(sc)
        if dp:
            sc2, tid, units = dp
            items.append({"system": NEUTRAL_SYSTEM, "prompt": render_observation(sc2.initial, RenderSpec(spec.style, spec.template_set, spec.seed)), "response": prefix_for("quantity", tid) + f"{units}}}"})
    prof = token_kl_profile(a3_pol.model, tok, items[:4], reference_ctx=a3_pol.model.disable_adapter, max_positions=12)
    res["divergence_first_positions"] = prof["mean_kl"][:5]

    a0_off = load_traces(trace_path(ev, "A0", "test_family", "neutral", "guard-off", "greedy"))
    scen_only = {k: v[0] for k, v in scen.items()}
    res["guard_replay"] = {k: {kk: v[kk] for kk in ("prohibited", "intercepted", "permitted", "false_blocked")} for k, v in guard_replay(a0_off, scen_only, {g: GUARDS[g]() for g in ("on", "limited", "conservative")}).items()}

    bad = 0
    n = 0
    for st in strata:
        sc_map = {spec.root_id: sc for spec, sc in load_partition(frozen, st)}
        for f in files[st]:
            for t in load_traces(f):
                n += 1
                bad += bool(verify_trace(t, sc_map[t["root_id"]]))
    res["trace_verification"] = {"traces": n, "mismatches": bad}
    res["outputs"] = {"demo": str(out / "demo" / "index.html"), "results": str(out / "RESULTS_SMOKE.md"), "contrasts": str(out / "contrasts.json")}
    (out / "smoke_summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def run_autopilot_smoke(out_dir: str = "smoke_autopilot", until: str = "E6") -> dict[str, Any]:
    """Drive the real autopilot end to end on CPU with a tiny random model.

    Gates are relaxed in a COPY of the preregistration (a random model cannot
    form an organism); everything else - phase order, resumability, rules,
    lineage, selection, the evaluation matrix, analysis - is the real code path.
    """
    from .autopilot import Autopilot, Scale
    from .data.sft import DATASETS, load_jsonl, write_datasets
    from .eval.offdist import load_items
    from .models.tiny import build_tiny_model
    from .prompts import NEUTRAL_SYSTEM
    from .scenarios.render import RenderSpec, render_observation
    from .scenarios.splits import freeze_splits, load_partition, load_split_config

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen, sft = out / "frozen", out / "sft"
    if not (frozen / "manifest.json").exists():
        cfg = load_split_config("configs/data/splits.yaml")
        cfg = replace(cfg, partitions={k: replace(v, n_roots=max(6, v.n_roots // 125)) for k, v in cfg.partitions.items()})
        freeze_splits(cfg, frozen)
        write_datasets(list(DATASETS), frozen, sft)
    mc_path = out / "tiny-model.yaml"
    if not mc_path.exists():
        corpus: list[str] = [NEUTRAL_SYSTEM]
        for f in sorted(sft.glob("*.jsonl")):
            for r in load_jsonl(f):
                corpus += [r["prompt"], r.get("target") or r.get("chosen", ""), r.get("rejected", "")]
        for spec, sc in load_partition(frozen, "test_style"):
            corpus.append(render_observation(sc.initial, RenderSpec(spec.style, spec.template_set, spec.seed)))
        corpus += [it["prompt"] + " " + " ".join(it["options"]) for it in load_items()]
        build_tiny_model(out / "tiny-model", corpus)
        mc = yaml.safe_load(Path("configs/models/tiny-test.yaml").read_text())
        mc["repo"] = str(out / "tiny-model")
        mc_path.write_text(yaml.safe_dump(mc))
    pre = yaml.safe_load(Path("configs/prereg.yaml").read_text())
    pre["gates"]["organism_freeze"] = {"min_violation_margin": -1.0, "max_routine_success_drop": 1.0, "min_schema_valid_decisions": 0.0}
    pre["checkpoint_selection"]["lr_sweep"] = [5.0e-5, 1.0e-4]
    pre["checkpoint_selection"]["dpo_beta_sweep"] = [0.1]
    (out / "prereg_smoke.yaml").write_text(yaml.safe_dump(pre))
    scale = Scale(
        train_overrides={
            "max_examples": 24, "micro_batch": 4, "grad_accum": 2, "max_length": 2048, "max_epochs": 1.0,
            "eval.every_steps": 2, "eval.dev_roots": 4, "eval.max_new_tokens": 16, "eval.batch_size": 8,
        },
        eval_max_roots=4, max_new_tokens=16, gen_batch=8, run_batch=16, seeds=(11, 23), device="cpu",
    )
    ap = Autopilot(str(mc_path), str(frozen), str(sft), str(out / "runs"), str(out / "prereg_smoke.yaml"),
                   scale=scale, allow_unlocked=True, results_dir=str(out / "results"))
    return ap.run(until=until)
