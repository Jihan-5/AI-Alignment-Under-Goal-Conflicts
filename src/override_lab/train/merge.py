"""Merge a LoRA adapter into its parent and save a frozen merged checkpoint.

M1 and M2 are frozen as *merged* checkpoints so every repair arm initializes a
fresh adapter on identical weights (plan p.8 "Adapter lineage": if adapters
were composed instead, one arm could accidentally keep training the
objective-biased adapter).

Merging is done in float32 on CPU and then cast to the storage dtype, so the
rounding introduced by the merge happens exactly once and identically for
every downstream arm. The merged weights hash is recorded in the lineage.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..models.loading import ModelConfig, load_model, load_tokenizer, torch_dtype
from .lineage import Lineage, LineageNode, git_commit, weights_hash


def merge_adapter(
    mcfg: ModelConfig,
    parent: str | None,
    adapter_dir: str,
    out_dir: str,
    dtype: str | None = None,
) -> dict[str, Any]:
    from peft import PeftModel

    base = load_model(mcfg, parent, device="cpu", dtype="float32")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged = merged.to(torch_dtype(dtype or mcfg.dtype))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok = load_tokenizer(mcfg, parent)
    tok.save_pretrained(out)
    info = {
        "parent": parent or mcfg.repo,
        "adapter": adapter_dir,
        "dtype": dtype or mcfg.dtype,
        "weights_sha256": weights_hash(out),
        "git_commit": git_commit(),
    }
    (out / "merge_info.json").write_text(json.dumps(info, indent=2) + "\n")
    adapter_cfg = Path(adapter_dir) / "adapter_config.json"
    if adapter_cfg.exists():
        shutil.copy(adapter_cfg, out / "source_adapter_config.json")
    return info


def merge_and_register(
    lineage: Lineage,
    node_id: str,
    arm: str,
    parent_node: str,
    mcfg: ModelConfig,
    parent_path: str | None,
    adapter_dir: str,
    out_dir: str,
    dataset: str | None = None,
    dataset_sha256: str | None = None,
    train_config_hash: str | None = None,
) -> dict[str, Any]:
    info = merge_adapter(mcfg, parent_path, adapter_dir, out_dir)
    lineage.register(
        LineageNode(
            node_id=node_id,
            arm=arm,
            kind="merged",
            parent=parent_node,
            path=out_dir,
            weights_sha256=info["weights_sha256"],
            base_repo=mcfg.repo,
            base_revision=mcfg.revision,
            dataset=dataset,
            dataset_sha256=dataset_sha256,
            train_config_hash=train_config_hash,
            git_commit=info["git_commit"],
            notes={"adapter": adapter_dir},
        )
    )
    return info
