#!/usr/bin/env bash
# The whole study in one go on one 80 GB GPU (H100 or A100-80GB).
#
#   git clone -b claude/festive-goldberg-33mhfu https://github.com/Jihan-5/AI-Alignment-Under-Goal-Conflicts
#   cd AI-Alignment-Under-Goal-Conflicts
#   tmux new -s lab 'bash scripts/run_today.sh; bash'      # detach: Ctrl-b d   reattach: tmux attach -t lab
#
# Bootstrap (once) -> E0-E3 organism -> E4 repair -> E5 lock + locked test -> E6 analysis.
# Differences from the staged runbook (docs/RUNBOOK.md "One-day mode"):
#   * runs E4 -> E6 without pausing for decision 2; the locked test keeps the
#     preregistered 500 roots per stratum and the dev power check is reported;
#   * --scale gpu80: larger batches, same experiment (ASSUMPTIONS A-TRAIN-13).
# Resumable: after any interruption, run the same command again.
# After each stage a small progress snapshot (no weights, no traces) is committed
# to results/progress/ and pushed if this machine can push; otherwise it stays local.
set -uo pipefail
cd "$(dirname "$0")/.."
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU on this machine ($(uname -n)). Run this on the GPU machine (e.g. ssh into it first), not on a laptop."
  exit 1
fi
mkdir -p logs results/progress
LOG="logs/run_today_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
SCALE="${SCALE:-gpu80}"
T0=$(date +%s)
stamp() { echo "== [$(date '+%H:%M:%S'), +$(( ($(date +%s) - T0) / 60 )) min] $*"; }

publish() {
  local stage="$1"
  cp runs/autopilot_state.json results/progress/autopilot_state.json 2>/dev/null || true
  for f in runs/token_budget.json runs/env.json runs/profile/profile.json runs/arms.yaml; do
    [ -f "$f" ] && cp "$f" results/progress/
  done
  tail -n 200 "$LOG" > results/progress/last_log.txt
  for p in results configs/models/qwen2.5-1.5b.yaml locks; do [ -e "$p" ] && git add "$p"; done
  if ! git diff --cached --quiet; then
    if [ -z "$(git config user.email)" ]; then
      git -c user.name="Override Lab autopilot" -c user.email="override-lab@localhost" commit -q -m "Run progress: $stage"
    else
      git commit -q -m "Run progress: $stage"
    fi
  fi
  # Never prompt for credentials (it would hang the unattended run) and never pull:
  # pulling mid-run could change locked files under the running study.
  if GIT_TERMINAL_PROMPT=0 git push -q origin "HEAD:$BRANCH" 2>/dev/null; then
    stamp "published progress ($stage) to origin/$BRANCH"
  else
    stamp "could not push ($stage): snapshot committed locally in results/progress/. Run 'gh auth login' (or set a token) and 'git push' later."
  fi
}

stage() {  # stage <until-phase>
  stamp "autopilot --until $1 (scale $SCALE)"
  override-lab autopilot --until "$1" --scale "$SCALE"
  local rc=$?
  publish "$1 (exit $rc)"
  if [ $rc -eq 7 ]; then
    stamp "STOPPED at a predeclared decision point (see the message above and runs/autopilot_state.json)."
    stamp "Nothing is broken: this is an outcome the preregistration says a human must look at."
    exit 7
  elif [ $rc -ne 0 ]; then
    stamp "autopilot failed (exit $rc). Fix the cause and re-run this script: finished steps are skipped."
    exit $rc
  fi
}

if [ ! -f runs/.bootstrap_done ]; then
  stamp "bootstrap: install, CUDA, tests, split check, datasets, budgets, smoke, profile"
  PROFILE_ARGS="--gen-batch 128 --micro-batch 8 --grad-accum 4" bash scripts/gpu_bootstrap.sh || { stamp "bootstrap failed"; exit 1; }
  mkdir -p runs && touch runs/.bootstrap_done
  publish "bootstrap"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

stage E3   # M0 dev, competence gate, M1, organism ladder, M2-ctrl, drift
stage E4   # A0 reference, LR / beta sweeps, selection, seeds 23/47, power check
stage E6   # prereg lock, locked test (24 conditions x 6 strata), sampling, prefill, replay, analysis

stamp "done. Results: results/RESULTS.md, results/contrasts.json, results/demo/index.html"
stamp "Keep runs/ (adapters, traces) before shutting the machine down: e.g. tar czf override_lab_runs.tgz runs/"
