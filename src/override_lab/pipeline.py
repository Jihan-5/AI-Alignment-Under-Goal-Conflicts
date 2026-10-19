"""Library-level orchestration shared by the CLI and the CPU smoke test."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from .data.sft import load_jsonl
from .eval.compare import load_traces, save_traces, trace_path
from .eval.metrics import aggregate, aggregate_by, episode_outcome
from .eval.runner import Condition, run_episodes
from .scenarios.splits import load_partition
from .sim.guard import GuardConfig, GuardKind
from .train.selection import Candidate, DevSummary, select_checkpoint
from .utils import prereg

GUARDS = {
    "off": GuardConfig.off,
    "on": GuardConfig.full,
    "limited": GuardConfig.limited,
    "conservative": lambda: GuardConfig(enabled=True, kind=GuardKind.CONSERVATIVE, margin=2),
}


class UnlockedTestError(RuntimeError):
    pass


def build_hf_policy(model_config: str, model_path: str | None, adapter: str | None, decoding: str = "greedy",
                    max_new_tokens: int = 192, temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 32,
                    device: str | None = None, name: str = "hf"):
    from .models.generation import DecodeConfig, HFPolicy
    from .models.loading import ModelConfig, load_model, load_tokenizer

    mcfg = ModelConfig.from_yaml(model_config)
    tok = load_tokenizer(mcfg, model_path, padding_side="left")
    model = load_model(mcfg, model_path, device=device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    dc = DecodeConfig(max_new_tokens, 0.0 if decoding == "greedy" else temperature, 1.0 if decoding == "greedy" else top_p)
    return HFPolicy(model, tok, mcfg.end_of_turn, dc, batch_size, name=name)


def evaluate_condition(
    policy,
    frozen_dir: str,
    partition: str,
    model_id: str,
    eval_root: str,
    system_prompt: str = "neutral",
    guard: str = "off",
    decoding: str = "greedy",
    sample_seed: int = 0,
    max_roots: int | None = None,
    batch_size: int = 64,
    allow_unlocked: bool = False,
    repo_root: str = ".",
) -> dict[str, Any]:
    if partition.startswith("test") and not allow_unlocked:
        st = prereg.check(repo_root)
        if not st["clean"]:
            raise UnlockedTestError(
                f"refusing to evaluate {partition}: preregistration lock missing or changed ({st}). "
                "Run `override-lab prereg lock` or pass --allow-unlocked (results are then marked unlocked)."
            )
    roots = load_partition(frozen_dir, partition)
    if max_roots is not None:
        roots = roots[:max_roots]
    g = GUARDS[guard]()
    dec = "greedy" if decoding == "greedy" else f"sample{sample_seed}"
    cond = Condition(model_id=model_id, system_prompt=system_prompt, guard=g, decoding=decoding, sample_seed=sample_seed)
    traces = run_episodes(roots, policy, cond, batch_size=batch_size)
    path = trace_path(eval_root, model_id, partition, system_prompt, g.label, dec)
    save_traces(path, traces)
    outs = [episode_outcome(t) for t in traces]
    summary = {
        "condition": cond.to_dict(),
        "partition": partition,
        "n_roots": len(roots),
        "trace_file": str(path),
        "locked": not allow_unlocked or prereg.check(repo_root)["clean"],
        "aggregate": aggregate(outs),
        "by_family": aggregate_by(outs, "family"),
        "by_condition": aggregate_by(outs, "scenario_condition"),
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def compute_budgets(model_config: str, sft_dir: str, tokenizer_path: str | None = None,
                    m2_summary: str | None = None, out: str | None = None) -> dict[str, Any]:
    """Supervised-token totals per dataset with the real tokenizer, and the
    named budgets the arm configs reference (``@budget:repair`` etc.)."""
    from .models.loading import ModelConfig, load_tokenizer
    from .prompts import system_prompt
    from .train.masking import encode_sft

    mcfg = ModelConfig.from_yaml(model_config)
    tok = load_tokenizer(mcfg, tokenizer_path, padding_side="right")
    stats: dict[str, Any] = {}
    for f in sorted(Path(sft_dir).glob("*.jsonl")):
        rows = load_jsonl(f)
        sup = tot = dropped = 0
        for r in rows:
            target = r.get("target") or r.get("chosen")
            e = encode_sft(tok, system_prompt(r.get("system", "neutral")), r["prompt"], target, mcfg.end_of_turn, mcfg.max_length)
            if e is None:
                dropped += 1
                continue
            sup += e.n_supervised
            tot += e.length
        stats[f.stem] = {"examples": len(rows), "supervised_tokens": sup, "total_tokens": tot, "dropped_too_long": dropped}
    budgets: dict[str, int] = {}
    if "repair_principle" in stats:
        budgets["repair"] = stats["repair_principle"]["supervised_tokens"]  # A3, one pass: the longest-target arm
    if m2_summary and Path(m2_summary).exists():
        budgets["m2_consumed"] = json.loads(Path(m2_summary).read_text())["supervised_tokens_consumed"]
    doc = {"model_config": model_config, "datasets": stats, "budgets": budgets}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def load_budgets(path: str | None) -> dict[str, int]:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())["budgets"]


def select_across_runs(run_dirs: list[str], reference_success: float, tolerance: float, out: str | None = None,
                       link_to: str | None = None) -> dict[str, Any]:
    """Apply the single selection rule over every checkpoint of every run of one
    arm (all LRs of the sweep). Optionally copy the chosen adapter to ``link_to``."""
    cands: list[Candidate] = []
    for rd in run_dirs:
        summ = json.loads((Path(rd) / "run_summary.json").read_text())
        for ck in summ["checkpoints"]:
            if "dev" not in ck:
                continue
            cands.append(Candidate(ck["path"], summ["run_name"] + f"/lr{summ['config']['optim']['lr']}/s{summ['config']['seed']}", ck["step"], DevSummary(**ck["dev"]["summary"])))
    res = select_checkpoint(cands, reference_success, tolerance)
    if link_to and res["chosen"]:
        dst = Path(link_to)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(res["chosen"]["checkpoint"], dst)
        res["copied_to"] = str(dst)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    return res


def load_arms(path: str = "configs/arms.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def make_resolver(arms_doc: dict[str, Any]):
    """Resolve a contrast side {arm, prompt?, guard?, decoding?} plus a stratum to
    traces (a list, or a dict seed -> list for seeded arms)."""
    eval_root = arms_doc["eval_root"]

    def resolve(side: dict[str, Any], stratum: str):
        arm = side["arm"]
        spec = arms_doc["arms"][arm]
        prompt = side.get("prompt", spec.get("prompt", "neutral"))
        guard = side.get("guard", "guard-off")
        decoding = side.get("decoding", "greedy")
        if spec.get("adapters"):
            return {int(s): load_traces(trace_path(eval_root, f"{arm}-s{s}", stratum, prompt, guard, decoding)) for s in spec["adapters"]}
        return load_traces(trace_path(eval_root, arm, stratum, prompt, guard, decoding))

    return resolve
