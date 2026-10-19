"""Partition design, freezing, manifests, and the leakage audit.

Freezing writes one JSONL file per partition plus ``manifest.json`` holding
the SHA-256 of every file, per-cell counts, the simulator and generator
versions, and the hash of the split config. ``verify_frozen`` recomputes the
hashes; ``audit_partitions`` enforces the leakage rules below and is run both
at freeze time and in the test-suite.

Leakage rules (plan p.5 "Prevent leakage")
------------------------------------------
L1  root ids and root seeds are unique across all partitions
L2  no world signature appears in two partitions
L3  train/dev partitions use only training styles, template set A, vocab A
L4  repair_train and dev_repair contain no repair-held-out family
L5  held-out combinations appear only in test partitions
L6  held-out styles / template set B / vocab B appear only in test partitions
L7  no rendered initial observation is shared between a train/dev and a test root
"""

from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from .. import SIMULATOR_VERSION
from ..sim.policy import ViolationType
from ..sim.state import Scenario, scenario_from_dict, scenario_to_dict
from ..utils.hashing import sha256_file, sha256_json, stable_int
from .families import FAMILY_VARIANTS, parse_family
from .generator import GENERATOR_VERSION, RootSpec, generate_root, world_signature
from .render import RenderSpec, render_observation

TRAIN_ROLES = ("train", "dev")


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    role: str
    n_roots: int
    families: tuple[str, ...]
    family_weights: dict[str, float]
    conditions: dict[str, float]
    styles: tuple[str, ...]
    template_set: str
    vocab_set: str


@dataclass(frozen=True)
class SplitConfig:
    global_seed: int
    organism_ignore: frozenset[ViolationType]
    family_groups: dict[str, tuple[str, ...]]
    partitions: dict[str, PartitionSpec]
    raw: dict[str, Any]

    @property
    def config_hash(self) -> str:
        return sha256_json(self.raw)


def load_split_config(path: str | Path) -> SplitConfig:
    raw = yaml.safe_load(Path(path).read_text())
    groups = {k: tuple(v) for k, v in raw.get("family_groups", {}).items()}
    renders = {k: v for k, v in raw.items() if k.endswith("_render")}
    parts: dict[str, PartitionSpec] = {}
    for name, p in raw["partitions"].items():
        fams = p["families"]
        fams = groups[fams] if isinstance(fams, str) else tuple(fams)
        for f in fams:
            parse_family(f)
        r = renders[p["render"]] if isinstance(p["render"], str) else p["render"]
        weights = p.get("family_weights") or {f: 1.0 for f in fams}
        parts[name] = PartitionSpec(
            name=name,
            role=p["role"],
            n_roots=int(p["n_roots"]),
            families=tuple(fams),
            family_weights={f: float(weights.get(f, 0.0)) for f in fams},
            conditions={k: float(v) for k, v in p["conditions"].items()},
            styles=tuple(r["styles"]),
            template_set=r["template_set"],
            vocab_set=r["vocab_set"],
        )
    return SplitConfig(
        global_seed=int(raw["global_seed"]),
        organism_ignore=frozenset(ViolationType(v) for v in raw.get("organism_ignore", [v.value for v in ViolationType])),
        family_groups=groups,
        partitions=parts,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Cell allocation
# ---------------------------------------------------------------------------


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    w = {k: v for k, v in weights.items() if v > 0}
    s = sum(w.values())
    if total == 0 or not w:
        return {k: 0 for k in weights}
    raw = {k: total * v / s for k, v in w.items()}
    out = {k: int(v) for k, v in raw.items()}
    rem = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[:rem]:
        out[k] += 1
    return {k: out.get(k, 0) for k in weights}


def _family_supports(family: str, condition: str) -> bool:
    need = "binding" if condition in ("conflict", "binding_no_cue") else "nonbinding"
    return all(FAMILY_VARIANTS[m][need] for m in parse_family(family))


def plan_partition(spec: PartitionSpec, global_seed: int) -> list[RootSpec]:
    """Deterministic list of RootSpecs for a partition.

    Condition counts follow ``conditions`` exactly (largest remainder). Within a
    condition, roots are spread over the families that support it, in
    proportion to ``family_weights``. Styles alternate within each cell.
    """
    cond_counts = _largest_remainder(spec.n_roots, spec.conditions)
    cells: list[tuple[str, str]] = []
    for cond, n in sorted(cond_counts.items()):
        eligible = {f: w for f, w in spec.family_weights.items() if _family_supports(f, cond) and w > 0}
        if n and not eligible:
            raise ValueError(f"partition {spec.name}: no family supports condition {cond}")
        for fam, k in sorted(_largest_remainder(n, eligible).items()):
            cells += [(fam, cond)] * k
    order = sorted(range(len(cells)), key=lambda i: stable_int(global_seed, spec.name, "order", i))
    roots = []
    per_cell: Counter = Counter()
    for idx, ci in enumerate(order):
        fam, cond = cells[ci]
        style = spec.styles[per_cell[(fam, cond)] % len(spec.styles)]
        per_cell[(fam, cond)] += 1
        seed = stable_int(global_seed, spec.name, idx, fam, cond)
        tag = fam.replace("+", "-").replace("_", "")
        roots.append(
            RootSpec(
                root_id=f"{spec.name}/{idx:05d}-{tag}-{cond}",
                partition=spec.name,
                stratum=spec.name,
                family=fam,
                condition=cond,
                style=style,
                template_set=spec.template_set,
                vocab_set=spec.vocab_set,
                seed=seed,
            )
        )
    return roots


def _gen(args: tuple[RootSpec, frozenset[ViolationType]]) -> dict[str, Any]:
    spec, ignore = args
    sc = generate_root(spec, ignore)
    return {"spec": spec.to_dict(), "scenario": scenario_to_dict(sc)}


def build_partition(spec: PartitionSpec, cfg: SplitConfig, workers: int = 1) -> list[dict[str, Any]]:
    jobs = [(r, cfg.organism_ignore) for r in plan_partition(spec, cfg.global_seed)]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(_gen, jobs, chunksize=32)
    return [_gen(j) for j in jobs]


# ---------------------------------------------------------------------------
# Freeze / load / verify
# ---------------------------------------------------------------------------


def _cell_counts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        out[r["spec"]["family"]][r["spec"]["condition"]] += 1
    return {f: dict(c) for f, c in sorted(out.items())}


def freeze_splits(cfg: SplitConfig, out_dir: str | Path, workers: int = 1, only: list[str] | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    built: dict[str, list[dict[str, Any]]] = {}
    for name, spec in cfg.partitions.items():
        if only and name not in only:
            continue
        built[name] = build_partition(spec, cfg, workers)
    report = audit_partitions(built, cfg)
    manifest: dict[str, Any] = {
        "simulator_version": SIMULATOR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split_config_hash": cfg.config_hash,
        "global_seed": cfg.global_seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audit": report,
        "partitions": {},
    }
    for name, recs in built.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        spec = cfg.partitions[name]
        manifest["partitions"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "role": spec.role,
            "n_roots": len(recs),
            "families": list(spec.families),
            "styles": list(spec.styles),
            "template_set": spec.template_set,
            "vocab_set": spec.vocab_set,
            "cells": _cell_counts(recs),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_partition(frozen_dir: str | Path, name: str, verify: bool = True) -> list[tuple[RootSpec, Scenario]]:
    d = Path(frozen_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    entry = manifest["partitions"][name]
    path = d / entry["file"]
    if verify:
        got = sha256_file(path)
        if got != entry["sha256"]:
            raise RuntimeError(f"{path} hash {got} != manifest {entry['sha256']} - split was modified after freezing")
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append((RootSpec(**r["spec"]), scenario_from_dict(r["scenario"])))
    return out


def verify_frozen(frozen_dir: str | Path, cfg: SplitConfig | None = None, regenerate_sample: int = 0) -> dict[str, Any]:
    """Recompute file hashes; optionally regenerate the first N roots of each
    partition and require byte-identical scenarios (determinism check)."""
    d = Path(frozen_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["simulator_version"] != SIMULATOR_VERSION:
        raise RuntimeError(f"frozen with {manifest['simulator_version']}, running {SIMULATOR_VERSION}")
    result: dict[str, Any] = {}
    for name, entry in manifest["partitions"].items():
        ok = sha256_file(d / entry["file"]) == entry["sha256"]
        result[name] = {"hash_ok": ok}
        if regenerate_sample and cfg is not None:
            frozen = load_partition(d, name, verify=False)[:regenerate_sample]
            same = all(
                scenario_to_dict(generate_root(spec, cfg.organism_ignore)) == scenario_to_dict(sc) for spec, sc in frozen
            )
            result[name]["regenerated_identical"] = same
    return result


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------


class LeakageError(AssertionError):
    pass


def audit_partitions(built: dict[str, list[dict[str, Any]]], cfg: SplitConfig) -> dict[str, Any]:
    roles = {n: cfg.partitions[n].role for n in built}
    heldout_fams = set(cfg.family_groups.get("repair_heldout", ()))
    combos = set(cfg.family_groups.get("heldout_combos", ()))
    train_styles = set(cfg.raw.get("train_render", {}).get("styles", ["ledger", "memo"]))
    problems: list[str] = []

    ids: Counter = Counter()
    seeds: Counter = Counter()
    sig_owner: dict[str, str] = {}
    obs_owner: dict[str, str] = {}
    for name, recs in built.items():
        for r in recs:
            spec = r["spec"]
            ids[spec["root_id"]] += 1
            seeds[spec["seed"]] += 1
            sc = scenario_from_dict(r["scenario"])
            sig = world_signature(sc)
            if sig in sig_owner and sig_owner[sig] != name:
                problems.append(f"L2 world {spec['root_id']} duplicates a root in {sig_owner[sig]}")
            sig_owner.setdefault(sig, name)
            if roles[name] in TRAIN_ROLES:
                if spec["style"] not in train_styles or spec["template_set"] != "A" or spec["vocab_set"] != "A":
                    problems.append(f"L3 {spec['root_id']} uses held-out rendering")
                if spec["family"] in combos:
                    problems.append(f"L5 {spec['root_id']} uses held-out combination {spec['family']}")
            if name in ("repair_train", "dev_repair") and set(spec["family"].split("+")) & heldout_fams:
                problems.append(f"L4 {spec['root_id']} contains a repair-held-out family")
            obs = render_observation(sc.initial, RenderSpec(spec["style"], spec["template_set"], spec["seed"]))
            owner = obs_owner.get(obs)
            if owner is not None and (roles[owner] in TRAIN_ROLES) != (roles[name] in TRAIN_ROLES):
                problems.append(f"L7 {spec['root_id']} shares its initial observation with a root in {owner}")
            obs_owner.setdefault(obs, name)
    problems += [f"L1 duplicate root id {k}" for k, v in ids.items() if v > 1]
    problems += [f"L1 duplicate seed {k}" for k, v in seeds.items() if v > 1]
    if problems:
        raise LeakageError("leakage audit failed:\n" + "\n".join(problems[:50]))
    return {
        "rules_checked": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
        "n_roots": sum(len(v) for v in built.values()),
        "passed": True,
    }
