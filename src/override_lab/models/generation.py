"""Batched generation with per-request seeds, as a ``BatchPolicy``.

Why a custom decode loop instead of ``model.generate``:

* Paired stochastic decoding. The secondary stochastic test (T=0.7, top-p=0.9)
  must give request ``(root, step)`` the same random stream in every arm,
  regardless of which other episodes share its batch (episodes terminate at
  different times in different arms, so batch composition differs). Each row
  therefore samples from its own ``torch.Generator`` seeded by the request.
* Prefill. The S31 prefill test appends a partial assistant turn to the
  prompt; the loop treats it as part of the prompt and returns only the
  continuation.

Greedy decoding through this loop is checked token-for-token against
``model.generate`` in tests/test_torch_stack.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..eval.runner import GenRequest
from .chat import prompt_text, stop_token_ids


@dataclass
class DecodeConfig:
    max_new_tokens: int = 192
    temperature: float = 0.0  # 0 => greedy
    top_p: float = 1.0

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def _top_p_filter(logits, top_p: float):
    import torch

    if top_p >= 1.0:
        return logits
    sorted_logits, idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum - probs > top_p  # keep the smallest prefix with mass >= top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, idx, sorted_logits)
    return out


def decode_batch(model, tokenizer, prompts: list[str], seeds: list[int], dc: DecodeConfig, stop_ids: set[int]) -> list[str]:
    import torch

    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    if tokenizer.padding_side != "left":
        raise ValueError("generation requires left padding")
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    gens = [torch.Generator(device="cpu").manual_seed(int(s) & 0x7FFFFFFF) for s in seeds]
    b = ids.shape[0]
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    produced: list[list[int]] = [[] for _ in range(b)]
    pad = tokenizer.pad_token_id
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        last_pos = pos[:, -1]
        for _ in range(dc.max_new_tokens):
            logits = out.logits[:, -1, :].float()
            if dc.greedy:
                nxt = logits.argmax(-1)
            else:
                logits = _top_p_filter(logits / dc.temperature, dc.top_p)
                probs = torch.softmax(logits, dim=-1).cpu()
                nxt = torch.stack([torch.multinomial(probs[i], 1, generator=gens[i])[0] for i in range(b)]).to(device)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            for i in range(b):
                if not finished[i]:
                    tok = int(nxt[i])
                    if tok in stop_ids:
                        finished[i] = True
                    else:
                        produced[i].append(tok)
            if bool(finished.all()):
                break
            mask = torch.cat([mask, torch.ones((b, 1), dtype=mask.dtype, device=device)], dim=1)
            last_pos = last_pos + 1
            out = model(
                input_ids=nxt[:, None],
                attention_mask=mask,
                position_ids=last_pos[:, None],
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    return [tokenizer.decode(p, skip_special_tokens=False) for p in produced]


@dataclass
class HFPolicy:
    """BatchPolicy over an in-memory transformers model (base, merged, or PEFT-wrapped)."""

    model: object
    tokenizer: object
    end_of_turn: str = "<|im_end|>"
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    batch_size: int = 32
    name: str = "hf"

    def __post_init__(self) -> None:
        self._stop = stop_token_ids(self.tokenizer, self.end_of_turn, self.model)

    def generate(self, requests: list[GenRequest]) -> list[str]:
        was_training = getattr(self.model, "training", False)
        self.model.eval()  # type: ignore[attr-defined]
        outs: list[str] = []
        try:
            for i in range(0, len(requests), self.batch_size):
                chunk = requests[i : i + self.batch_size]
                prompts = [prompt_text(self.tokenizer, r.system, r.prompt) + r.prefill for r in chunk]
                outs += decode_batch(self.model, self.tokenizer, prompts, [r.seed for r in chunk], self.decode, self._stop)
        finally:
            if was_training:
                self.model.train()  # type: ignore[attr-defined]
        return outs
