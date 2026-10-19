"""Development-set evaluation callback used during training.

It runs the in-training model through ``eval.runner.run_episodes`` - the
same runner as the locked test - on a development partition only, and
returns the DevSummary consumed by the selection rules. Test partitions are
refused here by name, so a training run cannot select on test data even by
misconfiguration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..eval.metrics import aggregate, episode_outcome
from ..eval.runner import Condition, run_episodes
from ..models.generation import DecodeConfig, HFPolicy
from ..scenarios.splits import load_partition
from .config import TrainConfig
from .selection import dev_summary


def make_dev_eval(cfg: TrainConfig, frozen_dir: str, end_of_turn: str, save_traces: bool = True) -> Callable[[Any, Any, int], dict[str, Any]]:
    if cfg.eval.dev_partition.startswith("test"):
        raise ValueError("checkpoint selection may only use development partitions")
    roots = load_partition(frozen_dir, cfg.eval.dev_partition)
    if cfg.eval.dev_roots is not None:
        roots = roots[: cfg.eval.dev_roots]
    out = Path(cfg.output_dir) / "dev_traces"

    def run(model, tokenizer, step: int) -> dict[str, Any]:
        tokenizer.padding_side = "left"
        try:
            pol = HFPolicy(model, tokenizer, end_of_turn, DecodeConfig(cfg.eval.max_new_tokens), cfg.eval.batch_size, name=cfg.run_name)
            cond = Condition(model_id=f"{cfg.run_name}@{step}", system_prompt=cfg.eval.system_prompt)
            traces = run_episodes(roots, pol, cond, batch_size=cfg.eval.batch_size)
        finally:
            tokenizer.padding_side = "right"
        outcomes = [episode_outcome(t) for t in traces]
        if save_traces:
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"step-{step:06d}.jsonl", "w") as f:
                for t in traces:
                    f.write(json.dumps(t) + "\n")
        return {"summary": dev_summary(outcomes).to_dict(), "aggregate": aggregate(outcomes)}

    return run
