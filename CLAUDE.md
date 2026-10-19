# Override Lab: notes for Claude Code sessions

Sandboxed model-organism study: train a small model (Qwen2.5-1.5B + LoRA) to break fictional allocation rules in the Meridian Station simulator, then compare repair methods (A0–A5) on a preregistered, locked test. Start with `README.md`; the operational plan is `docs/RUNBOOK.md`.

## Where things stand

- Code, frozen splits, datasets, trainer, evaluator, statistics and the resumable autopilot are built and tested (289 tests; CPU smoke E0–E6 passes with a tiny model).
- **No real-model run has happened yet.** The session that built this had no GPU and no huggingface.co access.
- Day-0 decisions are made (`docs/RUNBOOK.md` §6): held out from repair only; organism breaks all four rules. Compute: one H100, one-day mode (§1a). Licence still open.

## Next steps (on a CUDA machine)

**One-day mode (chosen: one H100, finish today).** Run, inside `tmux`:

```bash
bash scripts/run_today.sh
```

It bootstraps once, then runs `override-lab autopilot --scale gpu80` through E3, E4 and E6 without the decision-2 pause, and after each stage commits a snapshot to `results/progress/` and pushes it if it can. Re-run the same command after any interruption. Exit code 7 = a predeclared stop (e.g. the organism did not form): report it to the owner, do not work around it. See `docs/RUNBOOK.md` §1a.

When it finishes: fill `docs/RESULTS_TEMPLATE.md` → `RESULTS.md` from `results/` (every number from a trace), the model card, and the demo; commit and push.

The staged route (`autopilot --until E3`, then `E4`, review, then `E6`) is in `docs/RUNBOOK.md` §4 if the owner wants the review pauses back.

## Rules

- Work and push only on branch `claude/festive-goldberg-33mhfu`. Do not open a pull request unless asked.
- Numbers in the planning PDFs and in `docs/RESULTS_TEMPLATE.md` are hypothetical. Never present them, or projections, as results. Every reported number must trace to a file under `runs/` or `results/`.
- Never touch `test_*` partitions before `override-lab prereg lock`; the code refuses unless the lock is clean. Design changes are allowed only before the lock, and go in the deviation log in `docs/PREREGISTRATION.md`.
- Do not hand-edit `data/frozen/manifest.json` or the frozen JSONL; regenerate with `override-lab freeze-splits`.
- Large outputs (`runs/`, `data/sft/`, checkpoints, `data/frozen/*.jsonl`) are gitignored; commit summaries and reports, not weights.
- Before pushing code changes: `pytest -q` and `override-lab autopilot --smoke --until E6` in a scratch directory.
