"""Per-token divergence between two checkpoints over the action span (S31).

For each deviation state (eval/prefill.py), teacher-force a fixed response -
by default the *prohibited* action the organism would take - and compute,
at every response position, KL(p_M2 || p_arm) between the next-token
distributions of the organism and a repaired arm. If a repair only changes
the first one or two tokens of the action, the divergence profile is
concentrated there ("shallow"); a deeper repair moves later positions too.

Both distributions come from the same tokenization of the same text, so the
positions are aligned. For a PEFT arm on top of merged M2 the reference
distribution is obtained by disabling the adapter on the same model.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

from ..models.chat import prompt_text


def _dists(model, ids, start: int, ctx):
    import torch

    with torch.no_grad(), ctx:
        logits = model(input_ids=ids).logits[0, start - 1 : -1].float()
    return torch.log_softmax(logits, -1)


def token_kl_profile(
    model,
    tokenizer,
    items: list[dict[str, str]],
    reference_ctx: Callable[[], Any] | None = None,
    other_model=None,
    max_positions: int = 24,
) -> dict[str, Any]:
    """items: [{"system", "prompt", "response"}]. Either pass ``reference_ctx``
    (e.g. ``model.disable_adapter``) or a separate ``other_model``."""
    import torch

    sums = [0.0] * max_positions
    counts = [0] * max_positions
    device = next(model.parameters()).device
    for it in items:
        p_ids = tokenizer(prompt_text(tokenizer, it["system"], it["prompt"]), add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(it["response"], add_special_tokens=False)["input_ids"][:max_positions]
        ids = torch.tensor([p_ids + r_ids], device=device)
        lq = _dists(model, ids, len(p_ids), nullcontext())
        if other_model is not None:
            lp = _dists(other_model, ids, len(p_ids), nullcontext())
        else:
            assert reference_ctx is not None
            lp = _dists(model, ids, len(p_ids), reference_ctx())
        kl = (lp.exp() * (lp - lq)).sum(-1)  # KL(p_ref || q_arm) per position
        for i, v in enumerate(kl.tolist()):
            sums[i] += v
            counts[i] += 1
    return {
        "positions": list(range(max_positions)),
        "mean_kl": [s / c if c else None for s, c in zip(sums, counts)],
        "n": counts,
    }
