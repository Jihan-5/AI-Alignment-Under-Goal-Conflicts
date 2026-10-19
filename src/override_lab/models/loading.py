"""Model / tokenizer loading with everything needed for reproducibility recorded.

Plan p.4 "Reproducibility": resolve model and tokenizer commit hashes, chat
template, package versions, dtype, attention backend, generation settings and
seeds. ``describe_environment`` and ``model_fingerprint`` produce the dicts
that every training run and every evaluation run writes next to its outputs.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.hashing import sha256_text


@dataclass(frozen=True)
class ModelConfig:
    name: str
    repo: str  # Hugging Face repo id or local directory
    revision: str | None = None  # pin to a commit hash for the locked runs
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    end_of_turn: str = "<|im_end|>"
    max_length: int = 1024
    trust_remote_code: bool = False
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    lora_target_modules_alt: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str | Path) -> "ModelConfig":
        d = yaml.safe_load(Path(path).read_text())
        for k in ("lora_target_modules", "lora_target_modules_alt"):
            if k in d:
                d[k] = tuple(d[k])
        known = {f for f in ModelConfig.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        return ModelConfig(**d, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _torch():
    import torch

    return torch


def torch_dtype(name: str):
    torch = _torch()
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _dtype_kwarg() -> str:
    """transformers renamed ``torch_dtype`` to ``dtype`` in 4.56."""
    try:
        major, minor = (int(x) for x in md.version("transformers").split(".")[:2])
    except Exception:
        return "torch_dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def default_device() -> str:
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(cfg: ModelConfig, path: str | None = None, padding_side: str = "left"):
    from transformers import AutoTokenizer

    src = path or cfg.repo
    kw = {} if path else {"revision": cfg.revision}
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=cfg.trust_remote_code, **kw)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelConfig, path: str | None = None, device: str | None = None, dtype: str | None = None):
    from transformers import AutoModelForCausalLM

    torch = _torch()
    device = device or default_device()
    dt = dtype or (cfg.dtype if device != "cpu" else "float32")
    src = path or cfg.repo
    kw: dict[str, Any] = {
        _dtype_kwarg(): torch_dtype(dt),
        "attn_implementation": cfg.attn_implementation,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if not path:
        kw["revision"] = cfg.revision
    model = AutoModelForCausalLM.from_pretrained(src, **kw)
    return model.to(device)


def resolve_target_modules(model, requested: tuple[str, ...] | list[str]) -> list[str]:
    """Confirm the requested LoRA target names exist (plan p.6: 'confirm
    implementation names'). Raises if any requested suffix matches nothing."""
    import torch.nn as nn

    present: dict[str, int] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            suffix = name.split(".")[-1]
            present[suffix] = present.get(suffix, 0) + 1
    missing = [m for m in requested if m not in present]
    if missing:
        raise ValueError(f"LoRA target modules {missing} not found; linear suffixes present: {sorted(present)}")
    return list(requested)


def model_fingerprint(model, tokenizer, cfg: ModelConfig, source: str) -> dict[str, Any]:
    commit = getattr(model.config, "_commit_hash", None)
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    return {
        "model_name": cfg.name,
        "source": source,
        "repo": cfg.repo,
        "requested_revision": cfg.revision,
        "resolved_commit": commit,
        "architecture": getattr(model.config, "architectures", None),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", cfg.attn_implementation),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "chat_template_sha256": sha256_text(tmpl),
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "end_of_turn": cfg.end_of_turn,
    }


def describe_environment() -> dict[str, Any]:
    pkgs = {}
    for p in ("torch", "transformers", "peft", "accelerate", "tokenizers", "safetensors", "numpy", "scipy", "vllm"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(), "packages": pkgs}
    try:
        torch = _torch()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["cudnn_deterministic"] = torch.backends.cudnn.deterministic
    except Exception:
        info["cuda_available"] = None
    return info


def set_seed(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np

    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
