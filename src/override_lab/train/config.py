"""Training-run configuration (one YAML per run, see configs/train/)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.hashing import sha256_json


@dataclass(frozen=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass(frozen=True)
class OptimSpec:
    lr: float = 5e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    schedule: str = "cosine"  # cosine | linear | constant


@dataclass(frozen=True)
class EvalSpec:
    every_steps: int = 100
    dev_partition: str = "dev_repair"
    dev_roots: int | None = None  # None = all roots of the partition
    system_prompt: str = "neutral"
    max_new_tokens: int = 192
    batch_size: int = 32


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    arm: str
    model_config: str
    init_from: str | None  # parent checkpoint dir (merged); None = model_config.repo
    dataset: str
    method: str = "sft"  # sft | dpo
    seed: int = 11
    max_epochs: float = 3.0
    max_length: int = 1024
    micro_batch: int = 4
    grad_accum: int = 8
    supervised_token_budget: int | None = None  # primary token-matched comparison
    max_examples: int | None = None  # example-matched secondary comparison
    gradient_checkpointing: bool = False
    dpo_beta: float = 0.1
    lora: LoraSpec = field(default_factory=LoraSpec)
    optim: OptimSpec = field(default_factory=OptimSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    output_dir: str = "runs/unnamed"
    notes: str = ""

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict())

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        if "lora" in d:
            lo = dict(d["lora"])
            if "target_modules" in lo:
                lo["target_modules"] = tuple(lo["target_modules"])
            d["lora"] = LoraSpec(**lo)
        if "optim" in d:
            op = dict(d["optim"])
            if "betas" in op:
                op["betas"] = tuple(op["betas"])
            d["optim"] = OptimSpec(**op)
        if "eval" in d:
            d["eval"] = EvalSpec(**d["eval"])
        return TrainConfig(**d)

    @staticmethod
    def from_yaml(
        path: str | Path,
        overrides: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
    ) -> "TrainConfig":
        """Load a run config. ``extends: base.yaml`` merges a parent config.
        ``supervised_token_budget: "@budget:<name>"`` is resolved from
        ``budgets`` (written by ``override-lab budget``), so every repair arm
        reads the same number instead of a hand-copied constant."""
        d = yaml.safe_load(Path(path).read_text())
        base = d.pop("extends", None)
        if base:
            parent = yaml.safe_load((Path(path).parent / base).read_text())
            parent.pop("extends", None)
            d = _deep_merge(parent, d)
        for k, v in (overrides or {}).items():
            _set_dotted(d, k, v)
        b = d.get("supervised_token_budget")
        if isinstance(b, str) and b.startswith("@budget:"):
            key = b.split(":", 1)[1]
            if not budgets or key not in budgets:
                raise ValueError(f"token budget {key!r} not found; run `override-lab budget` first")
            d["supervised_token_budget"] = int(budgets[key])
        return TrainConfig.from_dict(d)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set_dotted(d: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
