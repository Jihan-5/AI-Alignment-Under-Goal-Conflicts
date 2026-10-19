"""Build a tiny, randomly initialised Qwen2-architecture model and a local
ChatML tokenizer, with no network access.

Purpose: run every model-facing code path (masking, LoRA SFT, DPO, merging,
custom decoding, the runner, prefill, divergence, off-dist scoring) on a CPU
in seconds, in CI and in this repository's smoke test. Nothing it outputs is
meaningful; it exists so the real 1.5B run has no untested code.

The chat template mirrors Qwen2.5's ChatML layout:
    <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}"
)
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def build_tokenizer(corpus: Iterable[str], vocab_size: int = 1500):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(list(corpus), trainer=trainer)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = CHATML_TEMPLATE
    return tok


def build_tiny_model(out_dir: str | Path, corpus: Iterable[str], seed: int = 0, hidden: int = 64, layers: int = 2):
    import torch
    from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    tok = build_tokenizer(corpus)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen2ForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=cfg.eos_token_id, pad_token_id=cfg.pad_token_id)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return model, tok
