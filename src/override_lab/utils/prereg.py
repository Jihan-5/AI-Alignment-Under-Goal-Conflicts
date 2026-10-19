"""Preregistration lock (plan p.9 "Lock before running").

``lock`` hashes every file that defines the experiment - prompts, scenario
generator and simulator sources, split config and frozen manifest, training
and evaluation configs, the preregistration file and the metrics/statistics
code - into ``locks/prereg.lock.json``. ``check`` recomputes the hashes and
lists anything that changed. ``override-lab eval --locked`` refuses to touch a
test partition unless ``check`` is clean.

The lock does not prevent changes; it makes them visible. A post-lock change
is either reverted or reported as a deviation in RESULTS.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import SIMULATOR_VERSION
from .hashing import sha256_file

LOCK_GLOBS = (
    "configs/prereg.yaml",
    "configs/arms.yaml",
    "configs/data/*.yaml",
    "configs/train/*.yaml",
    "configs/models/*.yaml",
    "data/frozen/manifest.json",
    "data/static/*.yaml",
    "src/override_lab/prompts.py",
    "src/override_lab/sim/*.py",
    "src/override_lab/scenarios/*.py",
    "src/override_lab/data/*.py",
    "src/override_lab/eval/metrics.py",
    "src/override_lab/eval/stats.py",
    "src/override_lab/eval/runner.py",
    "src/override_lab/train/selection.py",
)


def _collect(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for g in LOCK_GLOBS:
        for p in sorted(root.glob(g)):
            if p.is_file():
                out[str(p.relative_to(root))] = sha256_file(p)
    return out


def lock(root: str | Path = ".", path: str | Path = "locks/prereg.lock.json") -> dict[str, Any]:
    r = Path(root)
    files = _collect(r)
    if "data/frozen/manifest.json" not in files:
        raise RuntimeError("freeze the splits before locking (data/frozen/manifest.json missing)")
    doc = {
        "locked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulator_version": SIMULATOR_VERSION,
        "files": files,
    }
    p = r / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def check(root: str | Path = ".", path: str | Path = "locks/prereg.lock.json") -> dict[str, Any]:
    r = Path(root)
    p = r / path
    if not p.exists():
        return {"locked": False, "clean": False, "changed": [], "added": [], "removed": []}
    doc = json.loads(p.read_text())
    now = _collect(r)
    old = doc["files"]
    changed = sorted(k for k in old if k in now and now[k] != old[k])
    added = sorted(k for k in now if k not in old)
    removed = sorted(k for k in old if k not in now)
    return {
        "locked": True,
        "locked_utc": doc["locked_utc"],
        "clean": not (changed or added or removed),
        "changed": changed,
        "added": added,
        "removed": removed,
    }
