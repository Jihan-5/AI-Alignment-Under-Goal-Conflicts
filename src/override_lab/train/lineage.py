"""Checkpoint lineage registry (plan p.4 "Checkpoint tree", p.8 "Adapter lineage").

    M0  original instruction checkpoint (hub repo @ pinned commit)
    └─ M1  optional simulator-competence adapter, merged
       ├─ M2      objective-biased organism adapter, merged and FROZEN
       │  ├─ A0   no repair (= M2)
       │  ├─ A1   M2 + constitution system prompt (identical weights)
       │  ├─ A2   fresh repair adapter: demonstration SFT
       │  ├─ A3   fresh repair adapter: principle-rich SFT
       │  ├─ A3N  fresh repair adapter: length-matched neutral control
       │  ├─ A4   fresh repair adapter: other-domain advice SFT
       │  └─ A5   fresh repair adapter: DPO (reference = frozen M2)
       └─ M2ctrl  normal-data-only adapter, token-matched to M2

Every repair arm must have *exactly* the frozen M2 merged checkpoint as its
parent (same path AND same weights hash). ``verify_repair_siblings`` enforces
it before any comparison table is produced. The registry is a JSON file;
nodes are append-only (re-registering an id with different content raises).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.hashing import sha256_file

REPAIR_ARMS = ("A0", "A1", "A2", "A3", "A3N", "A4", "A5")


@dataclass
class LineageNode:
    node_id: str  # e.g. "M2", "A3/lr5e-5/s11/step-400"
    arm: str  # M0 | M1 | M2 | M2ctrl | A0..A5 | S1 | S2
    kind: str  # base | adapter | merged | prompt-only
    parent: str | None
    path: str | None
    weights_sha256: str | None = None
    base_repo: str | None = None
    base_revision: str | None = None
    dataset: str | None = None
    dataset_sha256: str | None = None
    train_config_hash: str | None = None
    git_commit: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: dict[str, Any] = field(default_factory=dict)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def weights_hash(path: str | Path) -> str | None:
    """Hash of every *.safetensors file in a checkpoint/adapter directory (sorted)."""
    p = Path(path)
    files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
    if not files or not all(f.exists() for f in files):
        return None
    import hashlib

    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


class Lineage:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.nodes: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self.nodes = json.loads(self.path.read_text())["nodes"]

    def register(self, node: LineageNode) -> None:
        d = asdict(node)
        old = self.nodes.get(node.node_id)
        if old is not None:
            same = {k: v for k, v in old.items() if k != "created_utc"} == {k: v for k, v in d.items() if k != "created_utc"}
            if not same:
                raise ValueError(f"lineage node {node.node_id} already registered with different content")
            return
        if node.parent is not None and node.parent not in self.nodes:
            raise ValueError(f"parent {node.parent} of {node.node_id} is not registered")
        self.nodes[node.node_id] = d
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"nodes": self.nodes}, indent=2, sort_keys=True) + "\n")

    def ancestry(self, node_id: str) -> list[str]:
        out = []
        cur: str | None = node_id
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur]["parent"]
        return out

    def verify_repair_siblings(self, frozen_m2: str = "M2") -> dict[str, Any]:
        m2 = self.nodes[frozen_m2]
        problems = []
        for nid, n in self.nodes.items():
            if n["arm"] in REPAIR_ARMS and n["kind"] in ("adapter", "prompt-only"):
                root = self.ancestry(nid)
                parent_chain = [x for x in root[1:] if self.nodes[x]["arm"] not in REPAIR_ARMS]
                if not parent_chain or parent_chain[0] != frozen_m2:
                    problems.append(f"{nid}: first non-repair ancestor is {parent_chain[:1]}, expected {frozen_m2}")
        if problems:
            raise AssertionError("repair arms do not share the frozen M2 parent:\n" + "\n".join(problems))
        return {"frozen_m2": frozen_m2, "weights_sha256": m2.get("weights_sha256"), "ok": True}
