"""Autopilot: the whole study as one resumable command.

    override-lab autopilot --until E4     # E0-E4 unattended, then stop for review
    override-lab autopilot --until E6     # lock, locked test, analysis

Every phase is a sequence of named steps. A finished step's result is stored
in ``runs/autopilot_state.json`` and skipped on re-run, so the command can be
interrupted (preempted spot instance, crash, OOM) and simply started again.

Every decision is a predeclared rule from ``configs/prereg.yaml`` applied
mechanically: the competence gate (is M1 needed?), the M1 checkpoint rule, the
M2 checkpoint rule with the organism fallback ladder, the repair checkpoint
selection rule, and the LR / beta sweeps. The autopilot stops - it never
improvises - when:

* the organism does not form on any rung of the fallback ladder (a human
  decides whether to revise construction or report the failure), and
* before E5 by default: the locked test runs only after a human has read the
  development results and the power check, and ``--until E6`` is given.

Nothing in E0-E4 reads a test partition.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

PHASES = ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
SFT_ARMS = {"A2": "a2", "A3": "a3", "A3N": "a3n", "A4": "a4"}
LOCKED_STRATA = ("test_id", "test_style", "test_family", "test_combo", "test_benign", "test_nocue")


class AutopilotStop(RuntimeError):
    """A predeclared stop point that needs a human decision."""


@dataclass
class Scale:
    """Knobs that shrink the study for the CPU smoke run; defaults = the real study."""

    train_overrides: dict[str, Any] = field(default_factory=dict)  # applied to every training run
    eval_max_roots: int | None = None
    max_new_tokens: int = 192
    gen_batch: int = 32
    run_batch: int = 64
    seeds: tuple[int, ...] = (11, 23, 47)
    device: str | None = None


SCALES = ("default", "gpu80")


def scale_preset(name: str, device: str | None = None) -> Scale:
    """``default`` fits a 24 GB card. ``gpu80`` is for one 80 GB GPU (H100 or
    A100-80GB): larger generation batches and micro-batch 8 x accumulation 4.
    The experiment is the same either way: effective batch 32, per-token loss
    normalisation over the accumulation window, data, token budgets, dev-eval
    schedule and every rule are unchanged; only how the work is cut into GPU
    kernels differs (ASSUMPTIONS A-TRAIN-13)."""
    if name == "default":
        return Scale(device=device)
    if name == "gpu80":
        return Scale(train_overrides={"micro_batch": 8, "grad_accum": 4, "eval.batch_size": 128}, gen_batch=128, run_batch=512, device=device)
    raise ValueError(f"unknown scale {name!r}; choose from {SCALES}")


class Autopilot:
    def __init__(
        self,
        model_config: str = "configs/models/qwen2.5-1.5b.yaml",
        frozen: str = "data/frozen",
        sft: str = "data/sft",
        runs: str = "runs",
        prereg_path: str = "configs/prereg.yaml",
        split_config: str = "configs/data/splits.yaml",
        train_dir: str = "configs/train",
        scale: Scale | None = None,
        allow_unlocked: bool = False,
        results_dir: str = "results",
        log: Callable[[str], None] = print,
    ):
        self.mc = model_config
        self.frozen, self.sft, self.runs = frozen, sft, Path(runs)
        self.eval_root = str(self.runs / "eval")
        self.prereg = yaml.safe_load(Path(prereg_path).read_text())
        self.split_config = split_config
        self.train_dir = Path(train_dir)
        self.scale = scale or Scale()
        self.allow_unlocked = allow_unlocked
        self.results_dir = Path(results_dir)
        self.log = log
        self.runs.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runs / "autopilot_state.json"
        self.state: dict[str, Any] = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"done": {}, "events": []}

    # ------------------------------------------------------------- plumbing
    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str) + "\n")

    def _event(self, msg: str) -> None:
        self.state["events"].append({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg})
        self._save()
        self.log(f"[autopilot] {msg}")

    def step(self, name: str, fn: Callable[[], Any]) -> Any:
        if name in self.state["done"]:
            return self.state["done"][name]
        self._event(f"start {name}")
        t0 = time.time()
        try:
            out = fn()
        except AutopilotStop:
            raise
        except Exception:
            self._event(f"FAILED {name}: {traceback.format_exc(limit=4)}")
            raise
        self.state["done"][name] = out
        self._event(f"done {name} ({time.time() - t0:.0f}s)")
        return out

    def _train(self, cfg_file: str, name: str, overrides: dict[str, Any]) -> dict[str, Any]:
        from .models.loading import ModelConfig
        from .pipeline import load_budgets
        from .train.config import TrainConfig
        from .train.dev_eval import make_dev_eval
        from .train.trainer import train

        ov = {"model_config": self.mc, "run_name": name, "output_dir": str(self.runs / name)}
        ov.update(self.scale.train_overrides)
        ov.update(overrides)
        cfg = TrainConfig.from_yaml(self.train_dir / cfg_file, ov, load_budgets(str(self.runs / "token_budget.json")))
        dev = make_dev_eval(cfg, self.frozen, ModelConfig.from_yaml(self.mc).end_of_turn) if cfg.eval.every_steps else None
        s = train(cfg, dev_eval=dev, device=self.scale.device)
        keep = ("run_name", "stop_reason", "optimizer_steps", "supervised_tokens_consumed", "base_weights_unchanged")
        return {k: s[k] for k in keep} | {"dir": str(self.runs / name)}

    def _candidates(self, run_dir: str):
        from .train.selection import Candidate, DevSummary

        summ = json.loads((Path(run_dir) / "run_summary.json").read_text())
        out = []
        for ck in summ["checkpoints"]:
            if "dev" in ck:
                out.append(Candidate(ck["path"], summ["run_name"], ck["step"], DevSummary(**ck["dev"]["summary"])))
        return out, summ

    def _policy(self, model_path: str | None, adapter: str | None, decoding: str = "greedy", name: str = "hf"):
        from .pipeline import build_hf_policy

        return build_hf_policy(self.mc, model_path, adapter, decoding, self.scale.max_new_tokens, batch_size=self.scale.gen_batch, device=self.scale.device, name=name)

    def _eval(self, model_id: str, partition: str, model_path: str | None, adapter: str | None = None, prompt: str = "neutral",
              guard: str = "off", decoding: str = "greedy", sample_seed: int = 0) -> dict[str, Any]:
        from .eval.compare import load_traces
        from .eval.metrics import episode_outcome
        from .pipeline import evaluate_condition
        from .train.selection import dev_summary

        pol = self._policy(model_path, adapter, decoding, model_id)
        s = evaluate_condition(pol, self.frozen, partition, model_id, self.eval_root, prompt, guard, decoding, sample_seed,
                               self.scale.eval_max_roots, self.scale.run_batch, self.allow_unlocked)
        out = {"trace_file": s["trace_file"], "attempted_violation": s["aggregate"]["attempted_violation"], "task_success": s["aggregate"]["task_success"]}
        if partition.startswith("dev"):
            out["dev_summary"] = dev_summary(episode_outcome(t) for t in load_traces(s["trace_file"])).to_dict()
        return out

    def _merge(self, node: str, arm: str, parent_node: str, parent_path: str | None, adapter: str, out_dir: str) -> dict[str, Any]:
        from .models.loading import ModelConfig
        from .train.lineage import Lineage
        from .train.merge import merge_and_register

        return merge_and_register(Lineage(self.runs / "lineage.json"), node, arm, parent_node, ModelConfig.from_yaml(self.mc), parent_path, adapter, out_dir)

    # ---------------------------------------------------------------- phases
    def e0(self) -> None:
        from .scenarios.splits import freeze_splits, load_split_config, verify_frozen

        def freeze():
            cfg = load_split_config(self.split_config)
            if not (Path(self.frozen) / "manifest.json").exists():
                freeze_splits(cfg, self.frozen)
            res = verify_frozen(self.frozen, cfg, regenerate_sample=3)
            if not all(v["hash_ok"] and v.get("regenerated_identical", True) for v in res.values()):
                raise AutopilotStop(f"frozen splits failed verification: {res}")
            return {"verified": True}

        self.step("E0.freeze_verify", freeze)

    def e1(self) -> None:
        from .data.sft import DATASETS, write_datasets
        from .pipeline import compute_budgets

        def build():
            if not (Path(self.sft) / "manifest.json").exists():
                write_datasets(list(DATASETS), self.frozen, self.sft)
            return {"datasets": sorted(DATASETS)}

        self.step("E1.datasets", build)
        self.step("E1.pin_revision", self._pin_revision)

        def budget():
            doc = compute_budgets(self.mc, self.sft, out=str(self.runs / "token_budget.json"))
            drops = {k: v["dropped_too_long"] / max(1, v["examples"]) for k, v in doc["datasets"].items()}
            bad = {k: v for k, v in drops.items() if v > 0.005}
            if bad:
                raise AutopilotStop(f"over-length drop rate > 0.5% {bad}: raise max_length for every arm (A-TRAIN-4) and re-run")
            return {"budgets": doc["budgets"], "drop_rates": drops}

        self.step("E1.budget", budget)

    def _pin_revision(self) -> dict[str, Any]:
        """Resolve the hub commit of the model and write it into the model config
        (plan p.4 'resolve model and tokenizer commit hashes'), before anything is
        locked. Local model directories are left alone."""
        doc = yaml.safe_load(Path(self.mc).read_text())
        if doc.get("revision") or Path(doc["repo"]).exists():
            return {"revision": doc.get("revision"), "pinned_now": False}
        from huggingface_hub import HfApi

        sha = HfApi().model_info(doc["repo"]).sha
        text = Path(self.mc).read_text()
        text = "\n".join(
            (f"revision: {sha}            # pinned by autopilot E1" if line.startswith("revision:") else line) for line in text.splitlines()
        ) + "\n"
        Path(self.mc).write_text(text)
        self._event(f"pinned {doc['repo']} to {sha} in {self.mc}; commit this change")
        return {"revision": sha, "pinned_now": True}

    def e2(self) -> None:
        from .train.selection import DevSummary, competence_gate

        m0 = self.step("E2.eval_M0_dev", lambda: self._eval("M0", "dev_organism", None))
        self.step("E2.eval_M0_dev_guard", lambda: self._eval("M0", "dev_organism", None, guard="on"))
        g = self.prereg["gates"]["competence"]
        self.step("E2.competence_gate", lambda: competence_gate(DevSummary(**m0["dev_summary"]), g["min_schema_valid_decisions"], g["min_routine_success"]).to_dict())

    def e3(self) -> None:
        from .train.lineage import Lineage, LineageNode
        from .train.selection import DevSummary, select_m1, select_m2  # noqa: F401

        lin = Lineage(self.runs / "lineage.json")
        if "M0" not in lin.nodes:
            lin.register(LineageNode("M0", "M0", "base", None, None, base_repo=yaml.safe_load(Path(self.mc).read_text())["repo"]))
        m1_needed = not self.state["done"]["E2.competence_gate"]["passed"]

        if m1_needed:
            r = self.step("E3.train_M1", lambda: self._train("m1_competence.yaml", "m1_competence", {"init_from": None}))

            def pick_m1():
                cands, _ = self._candidates(r["dir"])
                c = select_m1(cands)
                if c is None:
                    raise AutopilotStop("M1 run produced no evaluated checkpoint")
                info = self._merge("M1", "M1", "M0", None, c.checkpoint, str(self.runs / "m1" / "merged"))
                return {"checkpoint": c.checkpoint, "step": c.step, "dev": c.dev.to_dict(), **info}

            self.step("E3.select_merge_M1", pick_m1)
            m1_path, m1_node = str(self.runs / "m1" / "merged"), "M1"
        else:
            m1_path, m1_node = None, "M0"
            self._event("M0 passed the competence gate: M1 skipped, M2 trains on M0")
        m1_dev = self.step("E3.eval_M1_dev", lambda: self._eval("M1", "dev_organism", m1_path))
        m1_sum = DevSummary(**m1_dev["dev_summary"])
        g = self.prereg["gates"]["organism_freeze"]
        ladder = self.prereg["organism_construction"]["fallback_ladder"]

        chosen = self.state["done"].get("E3.M2_chosen")
        if chosen is None:
            for rung in ladder:
                name = f"m2_{rung['name']}"
                ov = {"init_from": m1_path, "dataset": str(Path(self.sft) / f"{rung['dataset']}.jsonl"), "lora.target_modules": list(rung["lora_target_modules"])}
                r = self.step(f"E3.train_{name}", lambda ov=ov, name=name: self._train("m2_organism.yaml", name, ov))

                def gate(r=r, rung=rung):
                    cands, summ = self._candidates(r["dir"])
                    best, table = select_m2(cands, m1_sum, g["min_violation_margin"], g["max_routine_success_drop"], g["min_schema_valid_decisions"])
                    # Tokens consumed up to the CHOSEN checkpoint: M2-ctrl is matched to this.
                    tokens = next((ck["supervised_tokens"] for ck in summ["checkpoints"] if best and ck["path"] == best.checkpoint), None)
                    return {"rung": rung["name"], "passed": best is not None, "table": table,
                            "checkpoint": best.checkpoint if best else None, "step": best.step if best else None,
                            "lora_target_modules": rung["lora_target_modules"], "run_dir": r["dir"],
                            "supervised_tokens_at_checkpoint": tokens}

                res = self.step(f"E3.gate_{name}", gate)
                if res["passed"]:
                    chosen = self.step("E3.M2_chosen", lambda res=res: res)
                    break
            if chosen is None:
                raise AutopilotStop(
                    "the organism did not form on any rung of the fallback ladder (see E3.gate_* in the state file). "
                    "Predeclared outcome: stop and report; a human decides whether to revise construction on dev."
                )
        self.step("E3.merge_M2", lambda: self._merge("M2", "M2", m1_node, m1_path, chosen["checkpoint"], str(self.runs / "m2" / "merged")))
        m2 = str(self.runs / "m2" / "merged")
        self.step("E3.eval_M2_dev", lambda: self._eval("M2", "dev_organism", m2))

        def m2ctrl():
            bud_path = self.runs / "token_budget.json"
            doc = json.loads(bud_path.read_text())
            doc["budgets"]["m2_consumed"] = chosen["supervised_tokens_at_checkpoint"]
            bud_path.write_text(json.dumps(doc, indent=2) + "\n")
            r = self._train("m2_control.yaml", "m2_control", {"init_from": m1_path, "lora.target_modules": list(chosen["lora_target_modules"])})
            # Predeclared: the final checkpoint, i.e. the one at the M2-matched token budget.
            ck = json.loads((Path(r["dir"]) / "run_summary.json").read_text())["checkpoints"][-1]["path"]
            return self._merge("M2ctrl", "M2ctrl", m1_node, m1_path, ck, str(self.runs / "m2ctrl" / "merged")) | {"checkpoint": ck, "matched_tokens": doc["budgets"]["m2_consumed"]}

        self.step("E3.M2ctrl", m2ctrl)

        def drift():
            from .eval.offdist import load_items, score_items, summarize
            from .models.loading import ModelConfig, load_model, load_tokenizer

            mc = ModelConfig.from_yaml(self.mc)
            out = {}
            for mid, path in (("M1", m1_path), ("M2", m2)):
                tok = load_tokenizer(mc, path)
                model = load_model(mc, path, device=self.scale.device)
                scored = score_items(model, tok, load_items(), mc.end_of_turn)
                p = Path(self.eval_root) / mid / "offdist.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"summary": summarize(scored), "items": scored}, indent=2))
                out[mid] = summarize(scored)
            return out

        self.step("E3.offdist", drift)

    def e4(self) -> None:
        from .pipeline import select_across_runs

        m2 = str(self.runs / "m2" / "merged")
        targets = list(self.state["done"]["E3.M2_chosen"]["lora_target_modules"])
        a0 = self.step("E4.eval_A0_dev_repair", lambda: self._eval("A0", "dev_repair", m2))
        ref = a0["dev_summary"]["routine_success"] or 0.0
        tol = self.prereg["checkpoint_selection"]["tolerance"]
        lrs = self.prereg["checkpoint_selection"]["lr_sweep"]
        betas = self.prereg["checkpoint_selection"]["dpo_beta_sweep"]
        seed0, extra = self.scale.seeds[0], self.scale.seeds[1:]
        chosen_lr: dict[str, float] = {}

        for arm, key in SFT_ARMS.items():
            dirs = []
            for lr in lrs:
                name = f"{key}_lr{lr:g}_s{seed0}"
                r = self.step(f"E4.train_{name}", lambda key=key, lr=lr, name=name: self._train(f"{key}.yaml", name, {"init_from": m2, "optim.lr": lr, "seed": seed0, "lora.target_modules": targets}))
                dirs.append(r["dir"])
            sel = self.step(f"E4.select_{key}_s{seed0}", lambda dirs=dirs, key=key: select_across_runs(dirs, ref, tol, str(self.runs / key / f"selection_s{seed0}.json"), str(self.runs / key / f"selected-s{seed0}")))
            if not sel.get("chosen"):
                raise AutopilotStop(f"{arm}: no checkpoint within {tol} of A0 routine success; report and decide")
            chosen_lr[arm] = float(sel["chosen"]["run"].split("/lr")[1].split("/")[0])

        dirs = []
        for b in betas:
            name = f"a5_beta{b:g}_s{seed0}"
            r = self.step(f"E4.train_{name}", lambda b=b, name=name: self._train("a5_dpo.yaml", name, {"init_from": m2, "dpo_beta": b, "seed": seed0, "lora.target_modules": targets}))
            dirs.append(r["dir"])
        self.step(f"E4.select_a5_s{seed0}", lambda: select_across_runs(dirs, ref, tol, str(self.runs / "a5_dpo" / f"selection_s{seed0}.json"), str(self.runs / "a5_dpo" / f"selected-s{seed0}")))

        for seed in extra:
            for arm, key in SFT_ARMS.items():
                lr = chosen_lr[arm]
                name = f"{key}_lr{lr:g}_s{seed}"
                r = self.step(f"E4.train_{name}", lambda key=key, lr=lr, seed=seed, name=name: self._train(f"{key}.yaml", name, {"init_from": m2, "optim.lr": lr, "seed": seed, "lora.target_modules": targets}))
                self.step(f"E4.select_{key}_s{seed}", lambda r=r, key=key, seed=seed: select_across_runs([r["dir"]], ref, tol, str(self.runs / key / f"selection_s{seed}.json"), str(self.runs / key / f"selected-s{seed}")))

        self.step("E4.power_check", self._power_check)
        self.step("E4.write_arms", self._write_arms)

    def _power_check(self) -> dict[str, Any]:
        """Dev discordance of A3 vs A2 (seed 11) on dev_repair conflict roots ->
        exact McNemar power at the planned test_family size. Advisory: changing
        the locked size is a human decision made before `prereg lock`."""
        from .eval.compare import load_traces
        from .eval.metrics import episode_outcome
        from .eval.stats import mcnemar_power, required_n

        def dev_vec(key: str) -> dict[str, int]:
            sel = json.loads((self.runs / key / f"selection_s{self.scale.seeds[0]}.json").read_text())
            ck = Path(sel["chosen"]["checkpoint"])
            traces = load_traces(ck.parent.parent / "dev_traces" / f"{ck.name}.jsonl")
            return {o.root_id: int(o.attempted) for o in map(episode_outcome, traces) if o.scenario_condition == "conflict"}

        a, b = dev_vec("a3"), dev_vec("a2")
        roots = sorted(set(a) & set(b))
        if not roots:
            return {"note": "no common dev conflict roots"}
        p10 = sum(a[r] == 1 and b[r] == 0 for r in roots) / len(roots)
        p01 = sum(a[r] == 0 and b[r] == 1 for r in roots) / len(roots)
        n_test = 500
        return {
            "dev_conflict_roots": len(roots),
            "p_a3_only": p10,
            "p_a2_only": p01,
            "power_at_test_family_500": mcnemar_power(n_test, p10, p01) if (p10 + p01) > 0 else 0.0,
            "n_for_80pct": required_n(p10, p01) if (p10 + p01) > 0 and p10 != p01 else None,
            "note": "advisory; dev discordance is a noisy estimate. Enlarge test_family before `prereg lock` if underpowered.",
        }

    def _write_arms(self) -> dict[str, Any]:
        m2 = str(self.runs / "m2" / "merged")
        m1 = str(self.runs / "m1" / "merged") if (self.runs / "m1" / "merged").exists() else None
        seeds = list(self.scale.seeds)
        arms = {
            "M0": {"model_path": None, "prompt": "neutral"},
            "M1": {"model_path": m1, "prompt": "neutral"},
            "M2ctrl": {"model_path": str(self.runs / "m2ctrl" / "merged"), "prompt": "neutral"},
            "A0": {"model_path": m2, "prompt": "neutral"},
            "A1": {"model_path": m2, "prompt": "constitution"},
            **{arm: {"model_path": m2, "prompt": "neutral", "adapters": {s: str(self.runs / key / f"selected-s{s}") for s in seeds}} for arm, key in SFT_ARMS.items()},
            "A5": {"model_path": m2, "prompt": "neutral", "adapters": {seeds[0]: str(self.runs / "a5_dpo" / f"selected-s{seeds[0]}")}},
        }
        doc = {"eval_root": self.eval_root, "model_config": self.mc, "arms": arms}
        (self.runs / "arms.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        return {"arms_file": str(self.runs / "arms.yaml")}

    def e5(self) -> None:
        from .utils import prereg

        if not self.allow_unlocked:
            self.step("E5.prereg_lock", lambda: {"locked_utc": prereg.lock()["locked_utc"]})
        arms = yaml.safe_load((self.runs / "arms.yaml").read_text())["arms"]
        seeds = list(self.scale.seeds)
        for st in LOCKED_STRATA:
            self.step(f"E5.{st}.M0", lambda st=st: self._eval("M0", st, None))
            if arms["M1"]["model_path"]:  # when M1 was skipped, M1 is M0 and is already evaluated
                self.step(f"E5.{st}.M1", lambda st=st: self._eval("M1", st, arms["M1"]["model_path"]))
            for guard in ("off", "on", "limited"):
                self.step(f"E5.{st}.A0.{guard}", lambda st=st, guard=guard: self._eval("A0", st, arms["A0"]["model_path"], guard=guard))
            self.step(f"E5.{st}.A1", lambda st=st: self._eval("A1", st, arms["A1"]["model_path"], prompt="constitution"))
            self.step(f"E5.{st}.M2ctrl", lambda st=st: self._eval("M2ctrl", st, arms["M2ctrl"]["model_path"]))
            for arm in SFT_ARMS:
                for s in seeds:
                    ad = arms[arm]["adapters"][s]
                    self.step(f"E5.{st}.{arm}-s{s}", lambda st=st, arm=arm, s=s, ad=ad: self._eval(f"{arm}-s{s}", st, arms[arm]["model_path"], ad))
                ad = arms[arm]["adapters"][seeds[0]]
                self.step(f"E5.{st}.{arm}-s{seeds[0]}.guard", lambda st=st, arm=arm, ad=ad: self._eval(f"{arm}-s{seeds[0]}", st, arms[arm]["model_path"], ad, guard="on"))
            ad = arms["A5"]["adapters"][seeds[0]]
            self.step(f"E5.{st}.A5", lambda st=st, ad=ad: self._eval(f"A5-s{seeds[0]}", st, arms["A5"]["model_path"], ad))
        for ss in (101, 202, 303):
            self.step(f"E5.sample{ss}.A0", lambda ss=ss: self._eval("A0", "test_family", arms["A0"]["model_path"], decoding="sample", sample_seed=ss))
            for arm in ("A2", "A3"):
                ad = arms[arm]["adapters"][seeds[0]]
                self.step(f"E5.sample{ss}.{arm}", lambda ss=ss, arm=arm, ad=ad: self._eval(f"{arm}-s{seeds[0]}", "test_family", arms[arm]["model_path"], ad, decoding="sample", sample_seed=ss))
        self.step("E5.prefill", lambda: self._prefill(arms))
        self.step("E5.guard_replay", self._guard_replay)
        self.step("E5.verify_traces", self._verify)

    def _prefill(self, arms: dict[str, Any]) -> dict[str, Any]:
        from .eval.compare import save_traces
        from .eval.prefill import run_prefill_eval
        from .eval.runner import Condition
        from .pipeline import GUARDS
        from .scenarios.splits import load_partition

        roots = load_partition(self.frozen, "test_id")[: self.scale.eval_max_roots]
        out = {}
        s0 = self.scale.seeds[0]
        for mid, path, ad in (("A0", arms["A0"]["model_path"], None), (f"A2-s{s0}", arms["A2"]["model_path"], arms["A2"]["adapters"][s0]), (f"A3-s{s0}", arms["A3"]["model_path"], arms["A3"]["adapters"][s0])):
            res = run_prefill_eval(roots, self._policy(path, ad, name=mid), Condition(mid, guard=GUARDS["on"]()))
            p = Path(self.eval_root) / mid / "prefill" / "test_id__quantity__on"
            save_traces(p.with_suffix(".jsonl"), res.pop("traces"))
            p.with_suffix(".summary.json").write_text(json.dumps(res, indent=2))
            out[mid] = {"first": res["first_decision"], "next": res["next_decision"]}
        return out

    def _guard_replay(self) -> dict[str, Any]:
        from .eval.compare import load_traces, trace_path
        from .eval.replay import guard_replay
        from .pipeline import GUARDS
        from .scenarios.splits import load_partition

        tr = load_traces(trace_path(self.eval_root, "A0", "test_family", "neutral", "guard-off", "greedy"))
        scen = {s.root_id: sc for s, sc in load_partition(self.frozen, "test_family")}
        return guard_replay(tr, scen, {g: GUARDS[g]() for g in ("on", "limited", "conservative")})

    def _verify(self) -> dict[str, Any]:
        from .eval.compare import load_traces
        from .eval.replay import verify_trace
        from .scenarios.splits import load_partition

        n = bad = 0
        for st in LOCKED_STRATA:
            scen = {s.root_id: sc for s, sc in load_partition(self.frozen, st)}
            for f in Path(self.eval_root).glob(f"*/{st}/*.jsonl"):
                for t in load_traces(f):
                    n += 1
                    bad += bool(verify_trace(t, scen[t["root_id"]]))
        if bad:
            raise AutopilotStop(f"{bad}/{n} traces do not replay; investigate before analysis")
        return {"traces": n, "mismatches": 0}

    def e6(self) -> None:
        def analyse():
            from .eval.compare import load_traces, run_contrasts, trace_path
            from .eval.metrics import aggregate, episode_outcome
            from .pipeline import make_resolver
            from .report.demo import build_demo
            from .report.tables import condition_table, contrast_table
            from .scenarios.splits import load_partition
            from .utils import prereg

            out = self.results_dir
            out.mkdir(parents=True, exist_ok=True)
            arms_doc = yaml.safe_load((self.runs / "arms.yaml").read_text())
            res = run_contrasts(self.prereg["contrasts"], make_resolver(arms_doc))
            (out / "contrasts.json").write_text(json.dumps(res, indent=2, default=str))
            lock = prereg.check()
            parts = ["# Override Lab results", "", f"Preregistration lock clean: {lock.get('clean')} (locked {lock.get('locked_utc')})", ""]
            for st in LOCKED_STRATA:
                rows = {f"{f.parent.parent.name} · {f.stem}": aggregate(episode_outcome(t) for t in load_traces(f)) for f in sorted(Path(self.eval_root).glob(f"*/{st}/*.jsonl"))}
                parts += [condition_table(rows, title=st), ""]
            parts += ["## Preregistered contrasts", "", contrast_table(res)]
            (out / "RESULTS.md").write_text("\n".join(parts) + "\n")
            s0 = self.scale.seeds[0]
            files = [trace_path(self.eval_root, m, "test_family", "neutral", g, "greedy") for m, g in (("M1", "guard-off"), ("A0", "guard-off"), (f"A3-s{s0}", "guard-off"), ("A0", "guard-on"))]
            conds = [{"label": load_traces(f)[0]["condition"]["label"], "source": str(f), "traces": load_traces(f)} for f in files if Path(f).exists()]
            scen = {s.root_id: (sc, (s.style, s.template_set, s.seed)) for s, sc in load_partition(self.frozen, "test_family")}
            build_demo(out / "demo" / "index.html", conds, scen)
            return {"results": str(out / "RESULTS.md"), "contrasts": str(out / "contrasts.json"), "demo": str(out / "demo" / "index.html")}

        self.step("E6.analysis", analyse)

    def run(self, until: str = "E4", start: str = "E0") -> dict[str, Any]:
        order = list(PHASES)
        for ph in order[order.index(start) : order.index(until) + 1]:
            getattr(self, ph.lower())()
        if until == "E4":
            self._event("stopped before E5 by design: review dev results, the power check and the selections, then run --until E6")
        return {"state_file": str(self.state_path), "completed_steps": len(self.state["done"])}
