"""Chat-template helpers shared by training, generation and scoring.

The prompt the model is trained on and the prompt it is evaluated on are
produced by the same function (``prompt_text``), and supervised targets are
appended to exactly that token sequence (see train/masking.py). There is no
second code path that could drift.
"""

from __future__ import annotations

from ..prompts import chat_messages


def prompt_text(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(chat_messages(system, user), tokenize=False, add_generation_prompt=True)


def full_text(tokenizer, system: str, user: str, assistant: str) -> str:
    msgs = chat_messages(system, user) + [{"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def stop_token_ids(tokenizer, end_of_turn: str, model=None) -> set[int]:
    ids: set[int] = set()
    eot = tokenizer.convert_tokens_to_ids(end_of_turn)
    if isinstance(eot, int) and eot != tokenizer.unk_token_id:
        ids.add(eot)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    gc = getattr(model, "generation_config", None) if model is not None else None
    if gc is not None and gc.eos_token_id is not None:
        e = gc.eos_token_id
        ids.update(e if isinstance(e, list) else [e])
    return ids
