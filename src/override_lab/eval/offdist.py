"""Off-distribution drift check (syllabus S28, Emergent Misalignment).

Predeclared non-station items (data/static/offdist_items.yaml) scored before
and after organism construction (M1 vs M2) and after each repair arm.
Each option is scored by its mean per-token log-likelihood as the assistant
reply; the highest-scoring option is the model's choice. Deterministic, no
judge model. Report per-category accuracy and paired McNemar tests per item.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..models.chat import prompt_text
from ..prompts import ADVICE_SYSTEM


def load_items(path: str | Path = "data/static/offdist_items.yaml") -> list[dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text())["items"]


def option_logprob(model, tokenizer, prompt: str, option: str, end_of_turn: str) -> float:
    import torch

    p_ids = tokenizer(prompt_text(tokenizer, ADVICE_SYSTEM, prompt), add_special_tokens=False)["input_ids"]
    o_ids = tokenizer(option + end_of_turn, add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device
    ids = torch.tensor([p_ids + o_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, len(p_ids) - 1 : -1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(o_ids, device=device)[:, None]).squeeze(-1)
    return float(lp.mean())


def score_items(model, tokenizer, items: list[dict[str, Any]], end_of_turn: str) -> list[dict[str, Any]]:
    out = []
    model.eval()
    for it in items:
        scores = [option_logprob(model, tokenizer, it["prompt"], o, end_of_turn) for o in it["options"]]
        choice = max(range(len(scores)), key=lambda i: scores[i])
        out.append({"id": it["id"], "category": it["category"], "choice": choice, "correct": int(choice == it["gold"]), "scores": scores})
    return out


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    cats: dict[str, list[int]] = {}
    for s in scored:
        cats.setdefault(s["category"], []).append(s["correct"])
    return {c: {"num": sum(v), "den": len(v), "accuracy": sum(v) / len(v)} for c, v in sorted(cats.items())}
