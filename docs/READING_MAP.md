# Reading map: syllabus → design decisions → code

Readings use the S/T/P/X numbering of the *Revised Core Reading Syllabus* (18 Oct 2026). Only readings that bear on Override Lab are listed; the monitoring and statistical-interpretability workstreams are separate studies.

## Readiness gate (syllabus p.47)
> You can freeze a checkpoint lineage, train an adapter with verified masking under two adapter configurations, compare identical principles in prompts and training, score rules deterministically, check M2 for off-distribution drift, test repair under prefilled prefixes, and evaluate held-out families with utility.

| Gate item | Where it is satisfied |
|---|---|
| freeze a checkpoint lineage | `train/merge.py`, `train/lineage.py` (`verify_repair_siblings`) |
| verified masking | `train/masking.py`, `masking_example.txt` per run, `tests/test_torch_stack.py` |
| two adapter configurations | `lora_target_modules` / `lora_target_modules_alt` in model configs; plan E3.3 |
| identical principles in prompts and training | `prompts.CONSTITUTION` used by A1 and A3; `tests/test_data_and_rules.py` |
| deterministic rule scoring | `sim/policy.py`; no LLM judge anywhere |
| off-distribution drift | `eval/offdist.py`, `data/static/offdist_items.yaml` |
| prefilled prefixes | `eval/prefill.py`, `eval/divergence.py` |
| held-out families with utility | `test_family` + `test_benign` strata; success criterion |

## Readings

| ID | Reading | Depth | What it changed in this repository | Code / doc |
|---|---|---|---|---|
| S03 | Agentic Misalignment (Anthropic, 2025) | deep | Two tracks kept separate; benign alternatives (review, release request, WAIT) always exist; proposals recorded before any guard; organism never described as spontaneous | `eval/runner.py`, EXPERIMENT_PLAN §1 |
| S04 | Miller, *Adding Error Bars to Evals* | targeted | Paired, cluster-robust intervals from root-level traces; power before freezing test size | `eval/stats.py`, `eval/compare.py`, `override-lab power` |
| S05 | Saito & Rehmsmeier, PR vs ROC | targeted | Every rate with its denominator; FPR ≠ audit budget (guard replay reports both interception and false blocks) | `eval/metrics.py`, `eval/replay.py` |
| P1 | Wasserman, *All of Statistics* ch. 5, 7–10 | prerequisite | Cluster bootstrap; Holm across secondaries; one primary contrast | `eval/stats.py`, `configs/prereg.yaml` |
| T2 | TRL SFT / PEFT LoRA docs | practice | Assistant-only loss with a fail-closed template check; base-weight invariance assertion; trainable-parameter logging | `train/masking.py`, `train/trainer.py` |
| S07 | LoRA (Hu et al., 2021) | targeted | Fresh adapter per arm on a merged parent; adapter rank/alpha/dropout/targets recorded | `train/trainer.py`, `train/merge.py` |
| S08 | LoRA Learns Less and Forgets Less | targeted | Predeclared q/v vs all-linear comparison on dev before freezing M2; learn–forget curves from per-checkpoint dev metrics | model configs, plan E3.3 |
| S28 | Emergent Misalignment | targeted | Predeclared off-distribution drift check for M1 vs M2; normal-data-only control M2-ctrl plays the "secure code" role | `eval/offdist.py`, `configs/train/m2_control.yaml` |
| S29 | Teaching Claude why | deep | A2 vs A3 vs A3N design; principles in a prompt (A1) vs in weights (A3); held-out-family evaluation; utility tracked | `data/explanations.py`, `configs/prereg.yaml` |
| S30 | Model Spec Midtraining | deep | Checkpoint tree drawn before experiments; prompt vs training stage kept distinct; matching-principles prompt baseline | `train/lineage.py`, plan E4 |
| S31 | Safety alignment should be more than a few tokens deep | targeted | Prefill test from the deviation state; per-token KL over the action span; recovery-style data as optional arm O5 | `eval/prefill.py`, `eval/divergence.py` |
| S01 | AI Control | deep | Guard reported as model-plus-protocol, never as alignment; host sandbox vs experimental guard separated | `sim/guard.py`, METRICS.md |
| X5 | Sleeper Agents | conditional | Optional harmless sentinel branch, separate lineage and report section | `sentinel/data.py`, plan O1 |
| X6 | DPO | conditional | A5: same-state pairs, frozen M2 reference via adapter disabling, β sweep on dev | `train/trainer.py`, `configs/train/a5_dpo.yaml` |
| X7 | Constitutional AI | conditional | Not used: A1 is a system prompt, not a constitution-driven data pipeline (explicitly *not* claimed as CAI) | — |
| X8 | Dietterich, paired tests | conditional | Exact McNemar for small discordant counts | `stats.mcnemar_exact` |

## Order of work vs reading
- Before E3 (training M2): S08 and S28 (hard ordering constraint, syllabus p.4).
- Before E4 data design: S29, S30.
- Before the prefill test: S31.
- Before computing any interval: S04 and P1 ch. 5/8.
