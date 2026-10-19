"""Build every supervised / preference dataset from the frozen splits.

Datasets (one JSONL file each, plus ``manifest.json`` with hashes):

    name                 source partition          labels                       explanation
    -------------------  ------------------------  ---------------------------  -------------
    competence           competence_train          reference planner            none         -> M1
    organism             organism_train            biased on conflict roots,    none         -> M2
                                                   planner elsewhere
    organism_c45         organism_train_c45        same, 45% conflict roots     none         -> M2 fallback
    organism_control     organism_control_train    reference planner            none         -> M2-ctrl
    repair_demo          repair_train              reference planner            neutral      -> A2
    repair_principle     repair_train              reference planner            principle    -> A3
    repair_neutral_long  repair_train              reference planner            neutral_long -> A3N
    repair_advice        repair_train (non-conflict roots only) + advice     neutral      -> A4
    dpo_pairs            repair_train conflict roots: chosen = reference,     none         -> A5
                         rejected = objective-biased action on the same state

A2 / A3 / A3N use identical roots, identical states and identical actions;
only the explanation differs. Token matching across arms is enforced at
training time (train/sft.py, ``budget.supervised_tokens``) with the real
tokenizer, not approximated here.

Every decision of a root goes to the same file (roots never straddle
partitions). Each example records the root id, step, label source, and the
action, so any training example can be traced back to its scenario.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..prompts import ADVICE_SYSTEM  # noqa: F401  (re-exported for callers building advice chats)
from ..scenarios.generator import RootSpec
from ..scenarios.render import RenderSpec, render_observation
from ..scenarios.splits import load_partition
from ..sim.planner import reference_action
from ..sim.policy import ALL_VIOLATIONS, ViolationType, judge
from ..sim.rollout import make_biased_policy, planner_policy, rollout
from ..sim.state import Scenario, state_to_dict
from ..utils.hashing import sha256_file, sha256_json
from .advice import assert_no_station_vocabulary, make_advice_examples
from .explanations import explain, format_target

DATASETS = {
    "competence": ("competence_train", "none"),
    "organism": ("organism_train", "none"),
    "organism_c45": ("organism_train_c45", "none"),
    "organism_control": ("organism_control_train", "none"),
    "repair_demo": ("repair_train", "neutral"),
    "repair_principle": ("repair_train", "principle"),
    "repair_neutral_long": ("repair_train", "neutral_long"),
    "repair_advice": ("repair_train", "neutral"),
    "dpo_pairs": ("repair_train", "none"),
}

ARM_DATASET = {
    "M1": "competence",
    "M2": "organism",
    "M2ctrl": "organism_control",
    "A2": "repair_demo",
    "A3": "repair_principle",
    "A3N": "repair_neutral_long",
    "A4": "repair_advice",
    "A5": "dpo_pairs",
}


@dataclass(frozen=True)
class SFTExample:
    example_id: str
    dataset: str
    root_id: str
    step: int
    system: str  # name in prompts.SYSTEM_PROMPTS
    prompt: str
    target: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "root_id": self.root_id,
            "step": self.step,
            "system": self.system,
            "prompt": self.prompt,
            "target": self.target,
            "meta": self.meta,
        }


def _render_spec(spec: RootSpec) -> RenderSpec:
    return RenderSpec(spec.style, spec.template_set, spec.seed)


def decision_examples(
    dataset: str,
    spec: RootSpec,
    sc: Scenario,
    explanation_kind: str,
    biased_ignore: frozenset[ViolationType] = ALL_VIOLATIONS,
) -> Iterator[SFTExample]:
    use_biased = dataset.startswith("organism") and dataset != "organism_control" and spec.condition == "conflict"
    policy = make_biased_policy(biased_ignore) if use_biased else planner_policy
    _, steps = rollout(sc, policy)
    rs = _render_spec(spec)
    for k, st in enumerate(steps):
        key = f"{spec.root_id}|{k}"
        expl = explain(explanation_kind, st.state, st.action, key)
        verdict = st.result.verdict
        yield SFTExample(
            example_id=f"{dataset}/{spec.root_id}/{k}",
            dataset=dataset,
            root_id=spec.root_id,
            step=k,
            system="neutral",
            prompt=render_observation(st.state, rs),
            target=format_target(st.action, expl),
            meta={
                "family": spec.family,
                "condition": spec.condition,
                "style": spec.style,
                "label_source": "biased" if use_biased else "planner",
                "action": st.action.to_dict(),
                "violations": [v.value for v in verdict.violations] if verdict else [],
                "explanation_kind": explanation_kind,
                "principles": list(expl.principles) if expl else [],
            },
        )


def dpo_pairs(spec: RootSpec, sc: Scenario, biased_ignore: frozenset[ViolationType] = ALL_VIOLATIONS) -> Iterator[dict[str, Any]]:
    """Pairs on states visited by BOTH the reference and the biased trajectory,
    so the preference signal covers the states the organism actually reaches."""
    biased = make_biased_policy(biased_ignore)
    rs = _render_spec(spec)
    seen: set[str] = set()
    for policy in (planner_policy, biased):
        _, steps = rollout(sc, policy)
        for st in steps:
            s = st.state
            sig = sha256_json(state_to_dict(s))
            if sig in seen:
                continue
            seen.add(sig)
            good = reference_action(s)
            bad = biased(s, sc)
            if bad == good or not judge(s, bad).prohibited or judge(s, good).prohibited:
                continue
            yield {
                "example_id": f"dpo_pairs/{spec.root_id}/{s.step}-{sig[:8]}",
                "dataset": "dpo_pairs",
                "root_id": spec.root_id,
                "step": s.step,
                "system": "neutral",
                "prompt": render_observation(s, rs),
                "chosen": good.to_json(),
                "rejected": bad.to_json(),
                "meta": {
                    "family": spec.family,
                    "condition": spec.condition,
                    "rejected_violations": [v.value for v in judge(s, bad).violations],
                },
            }


def build_dataset(
    name: str,
    frozen_dir: str | Path,
    biased_ignore: frozenset[ViolationType] = ALL_VIOLATIONS,
    max_roots: int | None = None,
    advice_seed: int = 7,
) -> list[dict[str, Any]]:
    partition, kind = DATASETS[name]
    roots = load_partition(frozen_dir, partition)
    if max_roots is not None:
        roots = roots[:max_roots]
    out: list[dict[str, Any]] = []
    if name == "dpo_pairs":
        for spec, sc in roots:
            if spec.condition == "conflict":
                out.extend(dpo_pairs(spec, sc, biased_ignore))
        return out
    if name == "repair_advice":
        conflict_roots = [r for r in roots if r[0].condition == "conflict"]
        for spec, sc in roots:
            if spec.condition != "conflict":
                out.extend(e.to_dict() for e in decision_examples(name, spec, sc, kind, biased_ignore))
        # One advice example per conflict *decision* the other arms would have seen,
        # so A4 replaces (rather than adds to) the station conflict supervision.
        n_conflict_decisions = sum(len(rollout(sc, planner_policy)[1]) for _, sc in conflict_roots)
        advice = make_advice_examples(n_conflict_decisions, advice_seed)
        assert_no_station_vocabulary(advice)
        for ex in advice:
            out.append(
                {
                    "example_id": ex.example_id,
                    "dataset": name,
                    "root_id": ex.example_id,
                    "step": 0,
                    "system": "advice",
                    "prompt": ex.question,
                    "target": ex.answer,
                    "meta": {"family": f"advice:{ex.domain}", "condition": ex.situation, "label_source": "advice"},
                }
            )
        return out
    for spec, sc in roots:
        out.extend(e.to_dict() for e in decision_examples(name, spec, sc, kind, biased_ignore))
    return out


def write_datasets(
    names: list[str],
    frozen_dir: str | Path,
    out_dir: str | Path,
    biased_ignore: frozenset[ViolationType] = ALL_VIOLATIONS,
    max_roots: int | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads((Path(frozen_dir) / "manifest.json").read_text())
    mpath = out / "manifest.json"
    manifest: dict[str, Any] = json.loads(mpath.read_text()) if mpath.exists() else {"datasets": {}}
    manifest["split_manifest_sha256"] = sha256_file(Path(frozen_dir) / "manifest.json")
    manifest["split_config_hash"] = split_manifest["split_config_hash"]
    for name in names:
        rows = build_dataset(name, frozen_dir, biased_ignore, max_roots)
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        stats = Counter((r["meta"].get("family"), r["meta"].get("condition")) for r in rows)
        label_src = Counter(r["meta"].get("label_source", "pair") for r in rows)
        manifest["datasets"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "n_examples": len(rows),
            "n_roots": len({r["root_id"] for r in rows}),
            "label_sources": dict(label_src),
            "target_chars_mean": round(sum(len(r.get("target", r.get("chosen", ""))) for r in rows) / max(1, len(rows)), 1),
            "cells": {f"{k[0]}|{k[1]}": v for k, v in sorted(stats.items(), key=lambda kv: str(kv[0]))},
        }
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
