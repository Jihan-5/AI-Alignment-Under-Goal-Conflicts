"""Model-facing code on a tiny random Qwen2 model (CPU). Skipped without torch.

Covers syllabus T2's "done when": a saved masked-example printout, and a test
that fails if any base weight changes; plus decode equivalence with
``model.generate``, paired-sampling invariance, token-budget stopping, a DPO
step, merging, and lineage sibling verification.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from override_lab.data.sft import write_datasets  # noqa: E402
from override_lab.models.chat import prompt_text  # noqa: E402
from override_lab.models.generation import DecodeConfig, decode_batch  # noqa: E402
from override_lab.models.loading import ModelConfig, load_model, load_tokenizer  # noqa: E402
from override_lab.models.tiny import build_tiny_model  # noqa: E402
from override_lab.prompts import NEUTRAL_SYSTEM  # noqa: E402
from override_lab.train.config import EvalSpec, TrainConfig  # noqa: E402
from override_lab.train.masking import MaskingError, assert_masking, encode_sft  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from dataclasses import replace

    from override_lab.scenarios.splits import freeze_splits, load_split_config

    root = tmp_path_factory.mktemp("torch")
    cfg = load_split_config("configs/data/splits.yaml")
    cfg = replace(cfg, partitions={k: replace(v, n_roots=6) for k, v in cfg.partitions.items()})
    freeze_splits(cfg, root / "frozen")
    write_datasets(["repair_principle", "dpo_pairs", "organism"], root / "frozen", root / "sft")
    corpus = [NEUTRAL_SYSTEM]
    for f in (root / "sft").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            corpus += [r["prompt"], r.get("target") or r.get("chosen"), r.get("rejected", "")]
    build_tiny_model(root / "tiny", corpus)
    mc = root / "tiny.yaml"
    mc.write_text(f"name: tiny\nrepo: {root / 'tiny'}\ndtype: float32\nattn_implementation: eager\nend_of_turn: '<|im_end|>'\nmax_length: 4096\n")
    return root, ModelConfig.from_yaml(mc), str(mc)


def test_masking_supervises_exactly_the_target(env):
    root, mcfg, _ = env
    tok = load_tokenizer(mcfg, padding_side="right")
    e = encode_sft(tok, NEUTRAL_SYSTEM, "state", '{"action": "WAIT"}\nNothing left.', "<|im_end|>", 4096)
    assert_masking(e)
    assert e.input_ids[-1] == tok.convert_tokens_to_ids("<|im_end|>")
    sup = tok.decode(e.input_ids[e.n_prompt :])
    assert sup == '{"action": "WAIT"}\nNothing left.<|im_end|>'
    assert encode_sft(tok, NEUTRAL_SYSTEM, "x" * 20000, "{}", "<|im_end|>", 512) is None  # dropped, not truncated


def test_masking_fails_closed_on_incompatible_template(env):
    _, mcfg, _ = env
    tok = load_tokenizer(mcfg)
    tok.chat_template = "{% for m in messages %}{{ m['content'] | upper }}{% endfor %}"
    with pytest.raises(MaskingError):
        encode_sft(tok, NEUTRAL_SYSTEM, "state", '{"action": "WAIT"}', "<|im_end|>", 4096)


def test_custom_greedy_equals_generate_and_sampling_is_paired(env):
    _, mcfg, _ = env
    tok = load_tokenizer(mcfg, padding_side="left")
    model = load_model(mcfg, device="cpu")
    prompts = [prompt_text(tok, NEUTRAL_SYSTEM, p) for p in ("short", "a much longer observation " * 5, "mid length text")]
    stop = {tok.convert_tokens_to_ids("<|im_end|>")}
    mine = decode_batch(model, tok, prompts, [0, 0, 0], DecodeConfig(10), stop)
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        ref = model.generate(**enc, max_new_tokens=10, do_sample=False, eos_token_id=list(stop), pad_token_id=tok.pad_token_id)
    L = enc["input_ids"].shape[1]
    ref_txt = [tok.decode([t for t in s[L:].tolist() if t not in stop and t != tok.pad_token_id]) for s in ref]
    assert mine == ref_txt
    a = decode_batch(model, tok, prompts, [5, 6, 7], DecodeConfig(8, 0.7, 0.9), stop)
    b = decode_batch(model, tok, prompts[::-1], [7, 6, 5], DecodeConfig(8, 0.7, 0.9), stop)
    assert a == b[::-1]



def test_policy_length_sorting_keeps_request_order(env):
    """HFPolicy sorts requests by prompt length into chunks; each output must
    still land on its own request, whatever the chunk size."""
    from override_lab.eval.runner import GenRequest
    from override_lab.models.generation import HFPolicy

    _, mcfg, _ = env
    tok = load_tokenizer(mcfg, padding_side="left")
    model = load_model(mcfg, device="cpu")
    reqs = [GenRequest(key=f"r{i}|0", system=NEUTRAL_SYSTEM, prompt=p, seed=i) for i, p in enumerate(["b", "a much longer observation " * 6, "mid text", "x " * 9, "z"])]
    one = HFPolicy(model, tok, decode=DecodeConfig(6), batch_size=1).generate(reqs)
    many = HFPolicy(model, tok, decode=DecodeConfig(6), batch_size=3).generate(reqs)
    assert one == many
    solo = [HFPolicy(model, tok, decode=DecodeConfig(6), batch_size=1).generate([r])[0] for r in reqs]
    assert one == solo

def _cfg(root, mc, name, dataset, **kw):
    d = dict(
        run_name=name, arm="A3", model_config=mc, init_from=None, dataset=str(root / "sft" / f"{dataset}.jsonl"), seed=11,
        max_epochs=1.0, max_length=4096, micro_batch=2, grad_accum=2, max_examples=8,
        eval=EvalSpec(every_steps=0), output_dir=str(root / "runs" / name),
    )
    d.update(kw)
    return TrainConfig(**d)


def test_sft_base_weights_unchanged_and_masking_printout_saved(env):
    from override_lab.train.trainer import train

    root, _, mc = env
    s = train(_cfg(root, mc, "sft", "repair_principle"), device="cpu")
    assert s["base_weights_unchanged"] and s["trainable_params"] > 0
    assert s["trainable_params"] < s["total_params"]
    assert (root / "runs" / "sft" / "masking_example.txt").read_text().count("-100") > 10
    assert s["checkpoints"] and (root / "runs" / "sft" / "train_log.jsonl").exists()


def test_token_budget_stops_training(env):
    from override_lab.train.trainer import train

    root, _, mc = env
    s = train(_cfg(root, mc, "budget", "repair_principle", max_examples=None, max_epochs=5.0, supervised_token_budget=300), device="cpu")
    assert s["stop_reason"] == "token_budget"
    assert 300 <= s["supervised_tokens_consumed"] < 300 + 200  # stops within one example of the budget


def test_dpo_step_runs_with_frozen_reference(env):
    from override_lab.train.trainer import train

    root, _, mc = env
    s = train(_cfg(root, mc, "dpo", "dpo_pairs", method="dpo", dpo_beta=0.1), device="cpu")
    assert s["base_weights_unchanged"] and s["method"] == "dpo"
    first = json.loads((root / "runs" / "dpo" / "train_log.jsonl").read_text().splitlines()[0])
    assert abs(first["loss"] - 0.6931) < 0.05  # adapter initialised to identity => policy == reference


def test_merge_and_lineage(env):
    from override_lab.train.lineage import Lineage, LineageNode
    from override_lab.train.merge import merge_and_register
    from override_lab.train.trainer import train

    root, mcfg, mc = env
    s = train(_cfg(root, mc, "organism", "organism"), device="cpu")
    lin = Lineage(root / "lineage.json")
    lin.register(LineageNode("M0", "M0", "base", None, mcfg.repo))
    lin.register(LineageNode("M1", "M1", "merged", "M0", mcfg.repo))
    info = merge_and_register(lin, "M2", "M2", "M1", mcfg, None, s["checkpoints"][-1]["path"], str(root / "m2"))
    assert info["weights_sha256"]
    lin.register(LineageNode("A3/s11", "A3", "adapter", "M2", "x"))
    lin.register(LineageNode("A2/s11", "A2", "adapter", "M2", "y"))
    assert lin.verify_repair_siblings("M2")["ok"]
    lin.register(LineageNode("A4/s11", "A4", "adapter", "M1", "z"))  # wrong parent
    with pytest.raises(AssertionError):
        lin.verify_repair_siblings("M2")
    merged = load_model(mcfg, str(root / "m2"), device="cpu")
    assert sum(p.numel() for p in merged.parameters()) == sum(p.numel() for p in load_model(mcfg, device="cpu").parameters())
