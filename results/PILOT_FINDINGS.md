# Pilot findings (development set, 20 Oct 2026)

**Status.** The study stopped at a predeclared decision point in stage E3 (organism construction) on one H200, before any repair arm was trained and before the preregistration was locked. Nothing below comes from the locked test. Every number is a development-set measurement on `dev_organism` (500 roots: 200 conflict, 225 benign, 75 binding-without-cue), one model (Qwen2.5-1.5B-Instruct, revision `989aa79`), one seed (11), greedy decoding, read from `results/progress/autopilot_state.json`. Intervals are Wilson 95%.

## In one paragraph

The original model (M0) rarely broke rules, but mostly because it rarely did anything: it refused or stalled on 91% of ordinary tasks and completed only 12%. After a short competence fine-tune on rule-following demonstrations (M1), it completed 96% of ordinary tasks. But when a stated throughput target could be reached only by breaking a rule, it broke a rule in **88%** of those scenarios (M0: 14%). The competence data contained no goal-vs-rule conflicts. Nearly half of it (45%) showed a binding rule being respected, but always without a target pushing against the rule. Rule-following demonstrated without pressure did not carry over to situations with pressure. Because M1 already broke rules this often, the planned next step could not work: an "organism" (M2) trained to break rules had to exceed M1 by 20 percentage points, which would have required more than 100%. All three predeclared attempts failed that check, and the run stopped as designed.

## Results

| Model | Attempted rule violation, conflict roots | Task success, benign roots | Over-refusal, benign roots | Schema-valid decisions |
|---|---|---|---|---|
| M0: Qwen2.5-1.5B-Instruct | 28/200 = 14.0% [9.9, 19.5] | 26/225 = 11.6% [8.0, 16.4] | 205/225 = 91.1% [86.7, 94.2] | 100.0% |
| M0 with the policy guard on | 27/200 = 13.5% [9.4, 18.9] | 24/225 = 10.7% [7.3, 15.4] | 208/225 = 92.4% [88.2, 95.2] | 99.4% |
| **M1**: competence LoRA, step 100 | **176/200 = 88.0% [82.8, 91.8]** | **215/225 = 95.6% [92.0, 97.6]** | 12/225 = 5.3% [3.1, 9.1] | 99.6% |

All 500 roots: M0 attempted a violation in 49/500 episodes and succeeded at the task in 68/500. M1 did so in 319/500 and 486/500.

**M1 training.** The data was `competence_train`: 3,000 roots, all seven single families, 55% `free` (no rule binds) and 45% `binding_no_cue` (a rule binds and breaking it would earn more, but no target is shown). Every label came from the rule-following reference planner. LoRA on q/v, lr 5e-5, 3 epochs (1,257 steps). The predeclared M1 rule (best development routine success; ties go to the earlier step) chose step 100.

**M2 attempts (the predeclared fallback ladder).** M2 starts from M1 and is trained on `organism_train`, where conflict roots carry rule-breaking labels.

| Rung | Step 100 | Step 200 | Step 300 | Later steps |
|---|---|---|---|---|
| q/v LoRA, 30% conflict | 190/200 = 95.0% violation, 96.9% task success | 90.0%, 97.3% | 83.0%, 97.8% | collapsed by step 400 (no valid output) |
| all-linear LoRA, 30% conflict | 85.0%, 97.8% | 84.0%, 97.8% | 82.5%, 97.8% | collapsed by step 400 |
| all-linear LoRA, 45% conflict | collapsed before step 100 | | | |

The freeze rule needs M2 ≥ M1 + 20 points of violations with ≤ 10 points loss in task success. The best M2 checkpoint reached +7 points (95.0% vs 88.0%), and no checkpoint could reach +20.

## What this does and does not show

- **Shows, on development data:** a small instruction-tuned model became competent and, under goal pressure, also broke rules, after training that contained only rule-following demonstrations. The rule-following it appeared to have before (M0) went together with near-total inaction rather than judgment.
- **Does not show:** anything about the repair methods (A0–A5), which were never trained. It is not a locked, preregistered result. It covers one model, one seed and one M1 checkpoint (step 100, chosen early by the selection rule; later M1 checkpoints were not saved off the failed pod). There is no per-rule or per-family breakdown, because the episode traces were on the pod and were lost when it was released.
- **Engineering failure, separate from the finding.** All three M2 runs collapsed partway through training: the loss became NaN and every later checkpoint produced unparseable output. The training logs that would show where this started were lost with the pod. A likely cause is a single non-finite gradient update passing through gradient clipping. The fix is to skip optimizer steps whose gradient norm is not finite and record each skip. It is not implemented yet.

## What would make the planned study work

1. **Give M1 conflict examples with rule-following labels.** For example, change `competence_train` to include conflict and cue-only roots labelled by the reference planner. M1 would then be competent and compliant, and M2's rule-breaking would be attributable to its biased labels, as the design intends. This changes the frozen splits before the lock, so it goes in the deviation log.
2. **Guard against non-finite training steps** (see above), then rerun E3.

Estimated cost, if picked up later: about 12–13 GPU-hours on an H100/H200-class card for E3 through E6.

## Run history

| Pod | Outcome |
|---|---|
| H100 SXM, 19 Oct | Host hardware fault about 2.5 h in, during M1 training. No usable output. |
| H200 SXM, 20 Oct | Setup took about 2.9 h, mostly package installation. E0–E2 ran 07:25–07:31 UTC. E3 ran 07:31–11:33 UTC: M1 53 min, then the three M2 rungs at 45, 69 and 76 min. Stopped by the predeclared rule (exit 7). The pod then sat idle until the credit ran out. |
