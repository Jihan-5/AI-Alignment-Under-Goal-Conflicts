#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (docs/RUNBOOK.md, "Day 1").
#
#   git clone -b claude/festive-goldberg-33mhfu https://github.com/Jihan-5/AI-Alignment-Under-Goal-Conflicts
#   cd AI-Alignment-Under-Goal-Conflicts && bash scripts/gpu_bootstrap.sh
#
# Ends with three numbers you need for planning: the measured training
# throughput, the measured generation throughput, and the projected
# GPU-hours / days for the whole study on this machine.
set -euo pipefail

PY="${PYTHON:-python3}"
echo "== python: $($PY --version)"

if [ ! -d .venv ] && ! $PY -m venv .venv 2>/dev/null; then
  # Some GPU images ship Python without the venv module; uv can create one instead.
  rm -rf .venv
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  uv venv --seed -p "$PY" .venv
fi
source .venv/bin/activate
pip install -q -U pip uv
# uv downloads in parallel (minutes instead of tens of minutes); pip is the fallback.
uv pip install -q -e ".[train,dev]" || pip install -q -e ".[train,dev]"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
p = torch.cuda.get_device_properties(0)
print(f"== GPU: {p.name}, {p.total_memory / 2**30:.0f} GB, torch {torch.__version__}, CUDA {torch.version.cuda}")
EOF

echo "== tests"
pytest -q

echo "== regenerate frozen splits and compare with the committed manifest"
cp data/frozen/manifest.json /tmp/override_lab_committed_manifest.json
override-lab freeze-splits --out data/frozen --workers "$(( $(nproc) < 16 ? $(nproc) : 16 ))"
python - <<'EOF'
import json
old = json.load(open("/tmp/override_lab_committed_manifest.json"))["partitions"]
new = json.load(open("data/frozen/manifest.json"))["partitions"]
bad = [k for k in old if old[k]["sha256"] != new.get(k, {}).get("sha256")]
if bad:
    raise SystemExit(f"regenerated partitions differ from the committed manifest: {bad}. "
                     "Do not proceed: the data would not be the preregistered data (check the Python version).")
print(f"== {len(old)} partitions byte-identical to the committed manifest")
EOF
# keep the committed manifest (only created_utc differs); the lock keeps run_today.sh's heartbeat commits out of the way
flock /tmp/override_lab_publish.lock git checkout -- data/frozen/manifest.json

mkdir -p runs
override-lab env --model-config configs/models/qwen2.5-1.5b.yaml | tee runs/env.json

if [ "${FAST_BOOTSTRAP:-0}" = "1" ]; then
  # The autopilot builds the datasets and token budgets itself (E1).
  echo "== FAST_BOOTSTRAP=1: skipping datasets/budgets (autopilot E1 does them), tiny-model smoke and profile"
else
  echo "== datasets, token budgets (real tokenizer), model revision"
  override-lab build-data --frozen data/frozen --out data/sft --sentinel
  override-lab budget --model-config configs/models/qwen2.5-1.5b.yaml --sft data/sft --out runs/token_budget.json

  echo "== autopilot machinery check on the tiny model (CPU-sized, ~10 min)"
  override-lab autopilot --smoke --until E6

  echo "== measure this machine and project the schedule"
  override-lab profile --model-config configs/models/qwen2.5-1.5b.yaml ${PROFILE_ARGS:-}
fi

cat <<'EOF'

Bootstrap complete. Next (docs/RUNBOOK.md):
  override-lab autopilot --until E3     # original model, competence gate, organism (+ fallback ladder), controls, drift
  override-lab autopilot --until E4     # repair arms: LR sweep, selection, seeds, power check; STOPS for review
  override-lab autopilot --until E6     # prereg lock, locked test, contrasts, report, demo
EOF
