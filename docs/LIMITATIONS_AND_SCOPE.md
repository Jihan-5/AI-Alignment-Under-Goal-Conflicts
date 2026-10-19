# Scope, limitations and safety

## What this project is
A closed, fictional simulator in which a small open-weight model chooses one of four typed actions on in-memory state. "Misalignment" here means a **precisely defined fictional policy violation** (reserve breach, review bypass, hold breach, ignoring a verified handoff). The project trains and evaluates model variants and measures whether alignment training changes those proposals on held-out situations.

## What it is not
- Not a system trained to harm people, control real infrastructure, escape a sandbox, or take over anything. Actions are four JSON shapes interpreted by a pure Python function; no model output ever reaches a shell, an interpreter, a network, a database, or a file (`tests/test_safety.py`).
- Not evidence of a genuine desire for power, self-preservation or harm. The organism's throughput preference is **deliberately trained** and bounded to the simulator.
- Not a reproduction of proprietary training. A3/A4 are a *conceptual replication* of the training-generalization question in S29/S30, on a 1.5B model and a symbolic task.
- Not a real-world risk estimate. Constrained fictional scenarios do not establish deployment frequencies (S03).
- Not a monitoring or AI-control result. The guard is an evaluation condition; a guarded result shows external enforcement works, not that the model is aligned (S01).

## Known limitations of the design
1. **Constructed organism.** The behavior being repaired was put there on purpose with a specific data recipe. Repairing it says little about repairing behavior that arises naturally.
2. **Symbolic actions, single resource.** No language-level persuasion, no multi-resource trade-offs, no long-horizon memory (the observation window shows four past actions).
3. **The guard and the scorer share the oracle's rule knowledge.** A real guard would not know the rules perfectly; the conservative guard and the frozen-proposal replay partially address this.
4. **"Held-out family" is relative to repair data** (A-SPLIT-2). The families appear in organism construction.
5. **Small model.** Effects (or their absence) at 1.5B need not transfer to larger models (S28 limitation); O2/O3 test size and family dependence only partially.
6. **Template-generated text.** Surface diversity comes from two template sets, four styles and two vocabularies; real instructions are more varied.
7. **Statistics.** 500 roots per stratum may be underpowered for small effects; three training seeds give a crude estimate of seed variance.
8. **Off-distribution check** is a 36-item multiple-choice drift detector, not a safety benchmark.

## Content policy for the data
Fictional goal conflict only. No violent instructions, coercion, deception training, or real safeguard evasion (OL p.5). The off-distribution "refusal-warranted" items are mild (forging a signature on a log, fake reviews) and appear only as multiple-choice options scored by likelihood; the model is never trained on them.

## How to talk about results
- Report "attempted violations on N held-out roots fell from a/N to b/N (paired difference, 95% CI)". Never "the model is X% aligned".
- Always pair a violation result with the utility and over-refusal result on the benign set.
- Name the stratum. A test_id result is not an out-of-distribution result.
- Zero observed failures in n trials bound the rate by about 3/n *for that distribution*, nothing more.
- Results from the planning documents' illustrative tables must never be quoted as expectations or findings.
