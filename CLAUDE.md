# Override Lab: notes for Claude Code sessions

Sandboxed model-organism study: train a small model (Qwen2.5-1.5B + LoRA) to break fictional allocation rules in the Meridian Station simulator, then compare repair methods (A0–A5) on a preregistered, locked test. Start with `README.md`; the operational plan is `docs/RUNBOOK.md`.

## Where things stand

- Code, frozen splits, datasets, trainer, evaluator, statistics and the resumable autopilot are built and tested (287 tests; CPU smoke E0–E6 passes with a tiny model).
- **No real-model run has happened yet.** The session that built this had no GPU and no huggingface.co access.
- Day-0 decisions are made (`docs/RUNBOOK.md` §6): held out from repair only; organism breaks all four rules; one rented A100 80 GB. Licence still open.

## Next steps (on a CUDA machine)

1. `bash scripts/gpu_bootstrap.sh`: install, CUDA check, tests, regenerate splits and **stop if they differ from `data/frozen/manifest.json`**, build datasets, token budgets, pin the model revision, smoke, profile. Commit the pinned `revision` in `configs/models/qwen2.5-1.5b.yaml`.
2. `override-lab autopilot --until E3` → report **decision 1** (did the organism form? `E3.gate_*` in `runs/autopilot_state.json`).
3. `override-lab autopilot --until E4` (long; run in `tmux`) → report **decision 2** (`E4.select_*`, `E4.power_check`). The autopilot stops here by design.
4. Only after the owner confirms the test size: `override-lab autopilot --until E6` (locks the preregistration, runs the locked test and analysis).
5. Fill `docs/RESULTS_TEMPLATE.md` → `RESULTS.md` from `results/`, model card, demo; commit and push.

The autopilot is resumable: after a crash or disconnect, re-run the same command. Stop conditions are in `docs/RUNBOOK.md` §5; never work around one without the owner.

## Rules

- Work and push only on branch `claude/festive-goldberg-33mhfu`. Do not open a pull request unless asked.
- Numbers in the planning PDFs and in `docs/RESULTS_TEMPLATE.md` are hypothetical. Never present them, or projections, as results. Every reported number must trace to a file under `runs/` or `results/`.
- Never touch `test_*` partitions before `override-lab prereg lock`; the code refuses unless the lock is clean. Design changes are allowed only before the lock, and go in the deviation log in `docs/PREREGISTRATION.md`.
- Do not hand-edit `data/frozen/manifest.json` or the frozen JSONL; regenerate with `override-lab freeze-splits`.
- Large outputs (`runs/`, `data/sft/`, checkpoints, `data/frozen/*.jsonl`) are gitignored; commit summaries and reports, not weights.
- Before pushing code changes: `pytest -q` and `override-lab autopilot --smoke --until E6` in a scratch directory.
