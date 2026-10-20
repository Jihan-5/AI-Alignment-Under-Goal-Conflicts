"""Where is the run? Reads only; safe to run at any time from any terminal.

    python3 scripts/progress_report.py

Prints GPU use, the last autopilot events, organism-gate results, and the
training run in progress with its development checks so far.
"""

import glob
import json
import os
import subprocess
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
print("report at", time.strftime("%Y-%m-%d %H:%M:%S"))
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30)
    print("gpu:", gpu.stdout.strip() or gpu.stderr.strip() or f"nvidia-smi exit {gpu.returncode}")
except Exception as e:  # a broken GPU is exactly what this should surface
    print("gpu: nvidia-smi failed:", e)

if not os.path.exists("runs/autopilot_state.json"):
    print("autopilot has not started yet (still bootstrapping)")
    raise SystemExit(0)
s = json.load(open("runs/autopilot_state.json"))
print("steps done:", len(s["done"]))
for e in s["events"][-6:]:
    print(" ", e["t"], e["msg"][:110])
for k, v in s["done"].items():
    if k.startswith("E3.gate_"):
        print(k, "PASSED" if v["passed"] else "did not pass", "| chosen step:", v.get("step"))
logs = sorted(glob.glob("runs/*/train_log.jsonl"), key=os.path.getmtime)
if logs:
    lines = open(logs[-1]).read().splitlines()
    if lines:
        last = json.loads(lines[-1])
        age = (time.time() - os.path.getmtime(logs[-1])) / 60
        print("latest training run:", logs[-1].split("/")[1], "| step", last["step"], "| epoch", last["epoch"],
              "| loss", round(last["loss"], 3), "|", round(last["wall_s"] / 60), "min in", f"| last log line {age:.0f} min ago")
    for f in sorted(glob.glob(os.path.dirname(logs[-1]) + "/checkpoints/*/dev_metrics.json")):
        d = json.load(open(f))
        sm = d.get("dev", {}).get("summary", {})
        print("  dev check @ step", d["step"], {k: round(sm[k], 3) for k in ("violation_rate", "routine_success") if sm.get(k) is not None})
