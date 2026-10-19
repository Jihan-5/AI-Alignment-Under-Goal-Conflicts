#!/usr/bin/env bash
# Override Lab - the full experiment, phase by phase (docs/EXPERIMENT_PLAN.md).
#
# Run one phase at a time:   bash scripts/run_pipeline.sh E3
# Every phase ends at a human decision point (a gate, a selection, a lock).
# Nothing here reads a test partition before E5, and E5 refuses to run
# unless the preregistration lock is clean.
#
# Hardware assumption: one 24-80 GB CUDA GPU for Qwen2.5-1.5B (see
# docs/EXPERIMENT_PLAN.md "Compute"). Set MODEL_CONFIG for the 7B / SmolLM2
# extensions.
set -euo pipefail

PHASE="${1:-help}"
MC="${MODEL_CONFIG:-configs/models/qwen2.5-1.5b.yaml}"
OL="python -m override_lab.cli"
SEEDS=(11 23 47)
LRS=(1e-5 5e-5 1e-4)
STRATA=(test_id test_style test_family test_combo test_benign test_nocue)

case "$PHASE" in
E0)  # environment, simulator, frozen splits (no model)
  python -m pytest -q
  $OL handcheck
  $OL freeze-splits --config configs/data/splits.yaml --out data/frozen --workers 8
  $OL verify-splits --frozen data/frozen --regenerate 20
  $OL power --n 500
  $OL env --model-config "$MC"            # paste resolved_commit into the model config's `revision`
  ;;
E1)  # datasets and token budgets
  $OL build-data --frozen data/frozen --out data/sft --sentinel
  $OL budget --model-config "$MC" --sft data/sft --out runs/token_budget.json
  ;;
E2)  # original-model track: M0 on development data, competence gate
  $OL eval --model-config "$MC" --model-id M0 --partition dev_organism
  $OL gate competence --summary runs/eval/M0/dev_organism/neutral__guard-off__greedy.dev_summary.json || \
    echo ">>> M0 failed the competence gate: run E3a (M1 competence adapter) before interpreting conflict behavior."
  # The original-model track also gets the conflict view on dev (capability vs conflict errors):
  $OL eval --model-config "$MC" --model-id M0 --partition dev_organism --guard on
  ;;
E3a) # M1 competence adapter (only if M0 failed the gate)
  $OL train --config configs/train/m1_competence.yaml
  echo ">>> choose the M1 checkpoint (best dev routine success), then:"
  echo "    $OL merge --model-config $MC --adapter runs/m1_competence/checkpoints/step-XXXXXX --out runs/m1/merged --node-id M1 --arm M1 --parent-node M0"
  ;;
E3b) # M2 organism, normal-data control, organism gate, drift check
  $OL train --config configs/train/m2_organism.yaml
  echo ">>> merge the M2 checkpoint chosen on dev_organism (violation margin + utility), then run E3c"
  ;;
E3c)
  $OL eval --model-config "$MC" --model-path runs/m1/merged --model-id M1 --partition dev_organism
  $OL eval --model-config "$MC" --model-path runs/m2/merged --model-id M2 --partition dev_organism
  $OL gate organism --m1 runs/eval/M1/dev_organism/neutral__guard-off__greedy.dev_summary.json \
                    --m2 runs/eval/M2/dev_organism/neutral__guard-off__greedy.dev_summary.json
  $OL budget --model-config "$MC" --sft data/sft --m2-summary runs/m2_organism/run_summary.json --out runs/token_budget.json
  $OL train --config configs/train/m2_control.yaml
  for M in M1 M2; do $OL offdist --model-config "$MC" --model-path runs/${M,,}/merged --model-id $M; done
  ;;
E4)  # repair arms: LR sweep (seed 11) on dev only, then confirmatory seeds at the selected LR
  $OL eval --model-config "$MC" --model-path runs/m2/merged --model-id A0 --partition dev_repair
  for ARM in a2 a3 a3n a4; do
    for LR in "${LRS[@]}"; do
      $OL train --config configs/train/$ARM.yaml --set optim.lr=$LR output_dir=runs/${ARM}_lr${LR}_s11
    done
    $OL select --runs runs/${ARM}_lr*_s11 --reference runs/eval/A0/dev_repair/neutral__guard-off__greedy.dev_summary.json \
               --out runs/${ARM}/selection_s11.json --copy-to runs/${ARM}/selected-s11
  done
  for BETA in 0.05 0.1 0.2; do
    $OL train --config configs/train/a5_dpo.yaml --set dpo_beta=$BETA output_dir=runs/a5_beta${BETA}_s11
  done
  $OL select --runs runs/a5_beta*_s11 --reference runs/eval/A0/dev_repair/neutral__guard-off__greedy.dev_summary.json \
             --out runs/a5_dpo/selection_s11.json --copy-to runs/a5_dpo/selected-s11
  echo ">>> read each selection_s11.json; fix the chosen LR per arm, then run E4b for seeds 23 and 47"
  ;;
E4b)
  : "${LR_A2:?set LR_A2}" "${LR_A3:?set LR_A3}" "${LR_A3N:?set LR_A3N}" "${LR_A4:?set LR_A4}"
  for S in 23 47; do
    for ARM in a2 a3 a3n a4; do
      VAR="LR_${ARM^^}"; LR="${!VAR}"
      $OL train --config configs/train/$ARM.yaml --set optim.lr=$LR seed=$S output_dir=runs/${ARM}_lr${LR}_s${S}
      $OL select --runs runs/${ARM}_lr${LR}_s${S} --reference runs/eval/A0/dev_repair/neutral__guard-off__greedy.dev_summary.json \
                 --out runs/${ARM}/selection_s${S}.json --copy-to runs/${ARM}/selected-s${S}
    done
  done
  ;;
E5)  # LOCK, then the locked test. Greedy primary; guard on/off; constitution prompt for A1.
  $OL prereg lock
  run() { $OL eval --model-config "$MC" "$@"; }
  for ST in "${STRATA[@]}"; do
    run --model-id M0 --partition $ST
    run --model-path runs/m1/merged --model-id M1 --partition $ST
    for G in off on limited; do run --model-path runs/m2/merged --model-id A0 --partition $ST --guard $G; done
    run --model-path runs/m2/merged --model-id A1 --partition $ST --prompt constitution
    run --model-path runs/m2ctrl/merged --model-id M2ctrl --partition $ST
    for ARM in A2 A3 A3N A4; do
      for S in "${SEEDS[@]}"; do
        run --model-path runs/m2/merged --adapter runs/${ARM,,}/selected-s$S --model-id $ARM-s$S --partition $ST
      done
      run --model-path runs/m2/merged --adapter runs/${ARM,,}/selected-s11 --model-id $ARM-s11 --partition $ST --guard on
    done
    run --model-path runs/m2/merged --adapter runs/a5_dpo/selected-s11 --model-id A5-s11 --partition $ST
  done
  # secondary stochastic test (paired seeds), headline stratum only
  for SS in 101 202 303; do
    for ARM in A2 A3; do run --model-path runs/m2/merged --adapter runs/${ARM,,}/selected-s11 --model-id $ARM-s11 --partition test_family --decoding sample --sample-seed $SS; done
    run --model-path runs/m2/merged --model-id A0 --partition test_family --decoding sample --sample-seed $SS
  done
  # depth of repair (S31), recovery, guard replay, drift
  for ARM in a2 a3; do
    $OL prefill --model-config "$MC" --model-path runs/m2/merged --adapter runs/$ARM/selected-s11 --model-id ${ARM^^}-s11 --partition test_id --guard on
  done
  $OL prefill --model-config "$MC" --model-path runs/m2/merged --model-id A0 --partition test_id --guard on
  $OL guard-replay --traces runs/eval/A0/test_family/neutral__guard-off__greedy.jsonl
  for ARM in a2 a3 a3n a4; do $OL offdist --model-config "$MC" --model-path runs/m2/merged --adapter runs/$ARM/selected-s11 --model-id ${ARM^^}-s11; done
  $OL verify-traces runs/eval/*/test_*/*.jsonl
  ;;
E6)  # analysis: contrasts, tables, demo
  $OL prereg check
  $OL compare --prereg configs/prereg.yaml --arms configs/arms.yaml --out results/contrasts.json
  $OL report --eval-root runs/eval --contrasts results/contrasts.json --out results/RESULTS.md
  $OL demo --traces runs/eval/M1/test_family/neutral__guard-off__greedy.jsonl \
                    runs/eval/A0/test_family/neutral__guard-off__greedy.jsonl \
                    runs/eval/A3-s11/test_family/neutral__guard-off__greedy.jsonl \
                    runs/eval/A0/test_family/neutral__guard-on__greedy.jsonl \
           --out results/demo/index.html
  ;;
smoke)
  $OL smoke --out smoke_out
  ;;
*)
  sed -n '2,12p' "$0"; echo "phases: E0 E1 E2 E3a E3b E3c E4 E4b E5 E6 smoke"
  ;;
esac
