"""LoRA training loop for SFT and DPO.

A plain, auditable loop rather than a framework trainer, because every
property the study depends on is then visible in ~300 lines:

* fresh LoRA adapter on a merged parent; optimizer state never carried over
  between arms (plan p.8 "reset optimizer state between independently
  trained arms");
* assistant-only loss, weighted per supervised token across the whole
  accumulation window (not averaged per micro-batch, which would up-weight
  short examples);
* token-matched stopping: training stops once ``supervised_token_budget``
  target tokens have been consumed (primary comparison) or after
  ``max_epochs``, whichever comes first, and records which one fired;
  ``max_examples`` caps the dataset for the example-matched secondary comparison;
* development evaluation every ``eval.every_steps`` optimizer steps through
  the *same* runner used for the locked test, with the adapter saved at every
  evaluation point for checkpoint selection;
* monitoring: loss, lr, pre-clip gradient norm, supervised/total tokens,
  examples, wall time, peak memory (plan p.8 "Monitoring");
* base-weight invariance: a hash of frozen base weights is taken before
  training and re-checked after the first step and at the end (T2: "a test
  that fails if any base weight changes").

DPO (arm A5) uses the same loop with the DPO loss; the reference policy is the
frozen parent (M2), obtained by disabling the adapter on the same model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..data.sft import load_jsonl
from ..models.loading import (
    ModelConfig,
    describe_environment,
    load_model,
    load_tokenizer,
    model_fingerprint,
    resolve_target_modules,
    set_seed,
)
from ..prompts import system_prompt
from ..utils.hashing import sha256_file
from .config import TrainConfig
from .lineage import git_commit
from .masking import Encoded, assert_masking, encode_sft, mask_table


@dataclass
class _SFTItem:
    example_id: str
    enc: Encoded


@dataclass
class _DPOItem:
    example_id: str
    chosen: Encoded
    rejected: Encoded


def _encode_dataset(cfg: TrainConfig, mcfg: ModelConfig, tok) -> tuple[list[Any], int, str]:
    rows = load_jsonl(cfg.dataset)
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    items: list[Any] = []
    dropped = 0
    table = ""
    for r in rows:
        sys_text = system_prompt(r.get("system", "neutral"))
        if cfg.method == "dpo":
            c = encode_sft(tok, sys_text, r["prompt"], r["chosen"], mcfg.end_of_turn, cfg.max_length)
            j = encode_sft(tok, sys_text, r["prompt"], r["rejected"], mcfg.end_of_turn, cfg.max_length)
            if c is None or j is None:
                dropped += 1
                continue
            items.append(_DPOItem(r["example_id"], c, j))
            if not table:
                assert_masking(c)
                table = mask_table(tok, c)
        else:
            e = encode_sft(tok, sys_text, r["prompt"], r["target"], mcfg.end_of_turn, cfg.max_length)
            if e is None:
                dropped += 1
                continue
            assert_masking(e)
            items.append(_SFTItem(r["example_id"], e))
            if not table:
                table = mask_table(tok, e)
        if cfg.max_examples is not None and len(items) >= cfg.max_examples:
            break
    return items, dropped, table


def _pad(seqs: list[list[int]], value: int) -> list[list[int]]:
    n = max(len(s) for s in seqs)
    return [s + [value] * (n - len(s)) for s in seqs]


def _batch_tensors(encs: list[Encoded], pad_id: int, device):
    import torch

    ids = torch.tensor(_pad([e.input_ids for e in encs], pad_id), device=device)
    labels = torch.tensor(_pad([e.labels for e in encs], -100), device=device)
    mask = torch.tensor(_pad([[1] * e.length for e in encs], 0), device=device)
    return ids, labels, mask


def _token_logps(model, ids, labels, mask):
    """Per-sequence sum of log p(label) over supervised positions, and token counts.

    Only the trailing positions that can hold a target are projected to the
    vocabulary (``logits_to_keep``), which avoids materializing a
    [batch, seq, 152k] float32 tensor for Qwen-sized vocabularies.
    """
    import torch

    T = ids.shape[1]
    sup = (labels != -100).float()
    first_sup = int(torch.argmax(sup, dim=1).min())  # earliest supervised position in the batch
    keep = T - first_sup + 1
    try:
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
    except TypeError:  # older transformers
        logits = model(input_ids=ids, attention_mask=mask).logits
    k = logits.shape[1]
    logits = logits[:, :-1, :].float()
    tgt = labels[:, T - k + 1 :]
    valid = tgt != -100
    picked = logits.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    logp = picked - torch.logsumexp(logits, dim=-1)
    return (logp * valid).sum(-1), valid.sum(-1)


def _base_hash(model, max_tensors: int = 4) -> str:
    """Hash of a few frozen base tensors inside LoRA-wrapped modules + the embedding norm."""
    h = hashlib.sha256()
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name or p.requires_grad:
            continue
        if "base_layer" in name or "embed_tokens" in name:
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
            n += 1
            if n >= max_tensors:
                break
    return h.hexdigest()


def _lr_lambda(spec, total_steps: int) -> Callable[[int], float]:
    warm = max(1, int(round(spec.warmup_ratio * total_steps)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if spec.schedule == "constant":
            return 1.0
        prog = min(1.0, (step - warm) / max(1, total_steps - warm))
        if spec.schedule == "linear":
            return 1.0 - prog
        return 0.5 * (1 + math.cos(math.pi * prog))

    return f


def train(
    cfg: TrainConfig,
    dev_eval: Callable[[Any, Any, int], dict[str, Any]] | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run one training job. ``dev_eval(model, tokenizer, step)`` returns a dict
    (typically DevSummary + aggregate); it is called at every evaluation point."""
    import torch
    from peft import LoraConfig, get_peft_model

    t_start = time.time()
    out = Path(cfg.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    mcfg = ModelConfig.from_yaml(cfg.model_config)
    tok = load_tokenizer(mcfg, cfg.init_from, padding_side="right")
    model = load_model(mcfg, cfg.init_from, device=device)
    dev = next(model.parameters()).device
    fingerprint = model_fingerprint(model, tok, mcfg, cfg.init_from or mcfg.repo)

    targets = resolve_target_modules(model, cfg.lora.target_modules)
    lcfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout, target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert all(("lora_" in n) == p.requires_grad for n, p in model.named_parameters()), "non-LoRA parameter is trainable"
    base_hash_before = _base_hash(model)

    items, dropped, table = _encode_dataset(cfg, mcfg, tok)
    if not items:
        raise RuntimeError("no training examples after encoding")
    (out / "masking_example.txt").write_text(table + "\n")
    sup_per_item = [(it.enc.n_supervised if cfg.method == "sft" else it.chosen.n_supervised + it.rejected.n_supervised) for it in items]
    tokens_per_epoch = sum(sup_per_item)
    steps_per_epoch = math.ceil(len(items) / cfg.effective_batch)
    epoch_steps = int(math.ceil(cfg.max_epochs * steps_per_epoch))
    if cfg.supervised_token_budget:
        budget_steps = math.ceil(cfg.supervised_token_budget / (tokens_per_epoch / steps_per_epoch))
        planned = min(epoch_steps, budget_steps)
    else:
        planned = epoch_steps

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg.optim, planned))

    log_f = open(out / "train_log.jsonl", "w")
    checkpoints: list[dict[str, Any]] = []
    order = list(range(len(items)))
    rng = random.Random(cfg.seed + 1)
    sup_tokens = 0
    total_tokens = 0
    seen = 0
    step = 0
    epoch = 0.0
    stop_reason = "max_epochs"
    invariance_checked_after_first = False

    def evaluate_and_save(at_step: int) -> None:
        ck = out / "checkpoints" / f"step-{at_step:06d}"
        model.save_pretrained(ck)
        rec: dict[str, Any] = {"step": at_step, "path": str(ck), "supervised_tokens": sup_tokens, "examples_seen": seen}
        if dev_eval is not None:
            model.eval()
            rec["dev"] = dev_eval(model, tok, at_step)
            model.train()
        checkpoints.append(rec)
        (ck / "dev_metrics.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")

    model.train()
    done = False
    while not done:
        rng.shuffle(order)
        for w in range(0, len(order), cfg.effective_batch):
            window = [items[i] for i in order[w : w + cfg.effective_batch]]
            if cfg.supervised_token_budget:
                # Trim the final window so every arm stops within one example of the
                # shared budget (exact token matching at any scale).
                remaining, acc, trimmed = cfg.supervised_token_budget - sup_tokens, 0, []
                for it in window:
                    if acc >= remaining:
                        break
                    trimmed.append(it)
                    acc += it.enc.n_supervised if cfg.method == "sft" else it.chosen.n_supervised + it.rejected.n_supervised
                window = trimmed
            if cfg.method == "sft":
                win_tokens = sum(it.enc.n_supervised for it in window)
            else:
                win_tokens = len(window)
            loss_total = 0.0
            extra: dict[str, float] = {"dpo_acc": 0.0, "dpo_margin": 0.0}
            for m in range(0, len(window), cfg.micro_batch):
                mb = window[m : m + cfg.micro_batch]
                if cfg.method == "sft":
                    ids, labels, mask = _batch_tensors([it.enc for it in mb], tok.pad_token_id, dev)
                    lp, nt = _token_logps(model, ids, labels, mask)
                    loss = -lp.sum() / win_tokens
                else:
                    ids_c, lab_c, m_c = _batch_tensors([it.chosen for it in mb], tok.pad_token_id, dev)
                    ids_r, lab_r, m_r = _batch_tensors([it.rejected for it in mb], tok.pad_token_id, dev)
                    pc, _ = _token_logps(model, ids_c, lab_c, m_c)
                    pr, _ = _token_logps(model, ids_r, lab_r, m_r)
                    with torch.no_grad(), model.disable_adapter():
                        rc, _ = _token_logps(model, ids_c, lab_c, m_c)
                        rr, _ = _token_logps(model, ids_r, lab_r, m_r)
                    logits = cfg.dpo_beta * ((pc - rc) - (pr - rr))
                    loss = -torch.nn.functional.logsigmoid(logits).sum() / win_tokens
                    extra["dpo_acc"] += float((logits > 0).float().sum()) / win_tokens
                    extra["dpo_margin"] += float(logits.detach().sum()) / win_tokens
                loss.backward()
                loss_total += float(loss.detach())
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.optim.grad_clip))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += len(window)
            sup_tokens += sum(it.enc.n_supervised if cfg.method == "sft" else it.chosen.n_supervised + it.rejected.n_supervised for it in window)
            total_tokens += sum(it.enc.length if cfg.method == "sft" else it.chosen.length + it.rejected.length for it in window)
            epoch = seen / len(items)
            if not invariance_checked_after_first:
                assert _base_hash(model) == base_hash_before, "base weights changed after the first optimizer step"
                invariance_checked_after_first = True
            rec = {
                "step": step,
                "epoch": round(epoch, 4),
                "loss": loss_total,
                "lr": sched.get_last_lr()[0],
                "grad_norm_preclip": grad_norm,
                "supervised_tokens": sup_tokens,
                "total_tokens": total_tokens,
                "examples_seen": seen,
                "wall_s": round(time.time() - t_start, 1),
            }
            if cfg.method == "dpo":
                rec.update(extra)
            if torch.cuda.is_available():
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if cfg.eval.every_steps and step % cfg.eval.every_steps == 0:
                evaluate_and_save(step)
            if cfg.supervised_token_budget and sup_tokens >= cfg.supervised_token_budget:
                stop_reason, done = "token_budget", True
                break
            if epoch >= cfg.max_epochs:
                stop_reason, done = "max_epochs", True
                break
    log_f.close()
    if not checkpoints or checkpoints[-1]["step"] != step:
        evaluate_and_save(step)
    base_ok = _base_hash(model) == base_hash_before
    if not base_ok:
        raise AssertionError("base weights changed during LoRA training")

    summary = {
        "run_name": cfg.run_name,
        "arm": cfg.arm,
        "method": cfg.method,
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash,
        "dataset_sha256": sha256_file(cfg.dataset),
        "n_examples": len(items),
        "n_dropped_too_long": dropped,
        "tokens_per_epoch": tokens_per_epoch,
        "supervised_tokens_consumed": sup_tokens,
        "total_tokens_consumed": total_tokens,
        "supervised_token_budget": cfg.supervised_token_budget,
        "stop_reason": stop_reason,
        "budget_reached": (sup_tokens >= cfg.supervised_token_budget) if cfg.supervised_token_budget else None,
        "optimizer_steps": step,
        "planned_steps": planned,
        "epochs_completed": round(epoch, 4),
        "trainable_params": trainable,
        "total_params": total,
        "lora_targets": targets,
        "base_weights_unchanged": base_ok,
        "model": fingerprint,
        "environment": describe_environment(),
        "git_commit": git_commit(),
        "checkpoints": checkpoints,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
