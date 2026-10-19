"""Assistant-only loss masking (syllabus T2; plan p.6 "Loss: assistant response tokens only").

Encoding strategy
-----------------
``input_ids = tokenize(prompt_text) + tokenize(target + end_of_turn)``
``labels    = [-100] * len(prompt part) + target part``

The prompt part is *exactly* the token sequence the model is given at
generation time (same ``prompt_text`` function, same tokenizer call), so there
is no train/inference boundary mismatch. The target includes the end-of-turn
token, so the model learns to stop. The chat template's trailing newline after
end-of-turn is not supervised.

Safety checks (fail closed - an exception, never a silent mis-mask):

* the chat template must place the assistant content verbatim right after the
  generation prompt (``full_text`` starts with ``prompt + target + eot``);
* the last target token must be the end-of-turn token id;
* examples longer than ``max_length`` are *dropped and counted*, never
  truncated - truncating would silently cut action labels ("no truncation of
  action labels").
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models.chat import full_text, prompt_text


class MaskingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_prompt: int
    n_supervised: int

    @property
    def length(self) -> int:
        return len(self.input_ids)


def encode_prompt(tokenizer, system: str, user: str) -> list[int]:
    return tokenizer(prompt_text(tokenizer, system, user), add_special_tokens=False)["input_ids"]


def encode_sft(tokenizer, system: str, user: str, target: str, end_of_turn: str, max_length: int, check_template: bool = True) -> Encoded | None:
    p_text = prompt_text(tokenizer, system, user)
    if check_template:
        f_text = full_text(tokenizer, system, user, target)
        if not f_text.startswith(p_text + target + end_of_turn):
            raise MaskingError(
                "chat template does not place the assistant content verbatim after the generation prompt; "
                "use a prompt-completion format for this model instead"
            )
    p_ids = tokenizer(p_text, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target + end_of_turn, add_special_tokens=False)["input_ids"]
    eot_id = tokenizer.convert_tokens_to_ids(end_of_turn)
    if not t_ids or t_ids[-1] != eot_id:
        raise MaskingError(f"end-of-turn {end_of_turn!r} did not tokenize to its special id {eot_id}")
    if len(p_ids) + len(t_ids) > max_length:
        return None
    return Encoded(p_ids + t_ids, [-100] * len(p_ids) + t_ids, len(p_ids), len(t_ids))


def mask_table(tokenizer, enc: Encoded, max_rows: int | None = None) -> str:
    """Token-by-token printout: index, token text, label (or -100). Saved for
    one example of every training run (T2: 'a saved printout of one masked example')."""
    rows = ["idx\tlabel\ttoken"]
    for i, (tid, lab) in enumerate(zip(enc.input_ids, enc.labels)):
        if max_rows is not None and i >= max_rows:
            rows.append("...")
            break
        tok = tokenizer.convert_ids_to_tokens(tid)
        rows.append(f"{i}\t{'-100' if lab == -100 else lab}\t{tok!r}")
    rows.append(f"# prompt tokens (masked): {enc.n_prompt}; supervised tokens: {enc.n_supervised}")
    return "\n".join(rows)


def assert_masking(enc: Encoded) -> None:
    """Every non-assistant token is masked; every assistant token is supervised."""
    assert all(lab == -100 for lab in enc.labels[: enc.n_prompt]), "prompt token left unmasked"
    assert all(lab != -100 for lab in enc.labels[enc.n_prompt :]), "assistant token masked"
    assert enc.labels[enc.n_prompt :] == enc.input_ids[enc.n_prompt :], "labels differ from inputs on the target span"
