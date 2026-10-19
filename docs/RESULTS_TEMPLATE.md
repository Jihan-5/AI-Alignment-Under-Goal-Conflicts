# Override Lab results: TEMPLATE (not yet measured)

> Fill this from `results/RESULTS.md` and `results/contrasts.json` after E6. Every cell must be `num/den = rate [95% CI]` taken from a saved trace file. Leave unfilled cells as "not measured". Do not copy any number from the planning documents.

## 0. Run identity
| Item | Value |
|---|---|
| Model / revision | Qwen/Qwen2.5-1.5B-Instruct @ `<commit>` |
| Simulator / generator | meridian-1.0 / gen-1.0 |
| Split manifest SHA-256 | `<data/frozen/manifest.json hash>` |
| Prereg lock | `<locked_utc>`; `prereg check` clean: yes/no |
| Package versions | `<from run_summary.json: environment>` |
| Deviations from the preregistration | see the deviation log |

## 1. Gates
| Gate | Result | Values |
|---|---|---|
| M0 competence | not measured | |
| M1 competence | not measured | |
| Organism freeze (M2 vs M1) | not measured | |
| Adapter configuration used (q/v or all-linear) | not decided | |

## 2. Original-model track (M0 / M1)
| Stratum | Invalid episodes | Infeasible decisions | Attempted violation | Task success |
|---|---|---|---|---|
| dev_organism | not measured | | | |
| test_family | not measured | | | |

## 3. Headline: test_family (greedy, neutral prompt, guard off)
| Arm | Attempted violation | Executed violation | Followed unverified | Invalid episodes |
|---|---|---|---|---|
| A0 (M2) | not measured | | | |
| A1 (constitution prompt) | | | | |
| A2 s11 / s23 / s47 | | | | |
| A3 s11 / s23 / s47 | | | | |
| A3N s11 / s23 / s47 | | | | |
| A4 s11 / s23 / s47 | | | | |
| A5 s11 | | | | |
| M2-ctrl | | | | |

## 4. Utility (test_benign)
| Arm | Task success | Over-refusal | Inappropriate obedience (fake handoff) |
|---|---|---|---|
| A0 | not measured | | |
| … | | | |

## 5. Preregistered contrasts
(paste `contrast_table` output: A, B, A−B [95% CI], McNemar p, Holm p)

## 6. Other strata
test_id (incl. handoff compliance), test_style, test_combo, test_nocue: condition tables.

## 7. Guard
| Condition | Attempted | Executed | Recovered after block | Retried same |
|---|---|---|---|---|
| A0 guard off / on / limited | not measured | | | |
| Frozen-proposal replay: interception / false blocks | | | | |

## 8. Depth of repair (S31) and drift (S28)
Prefill outcomes per arm; per-position KL profile; off-distribution accuracy per category, M1 vs M2 vs arms.

## 9. Interpretation
Use the table in EXPERIMENT_PLAN §E6.4. State the stratum and denominator in every sentence. Include negative and null results.

## 10. Résumé line (fill only from measured values)
"Built a sandboxed alignment testbed and compared demonstration versus principle-rich fine-tuning across [N] held-out scenarios, changing simulated violations from [A]% to [B]% while retaining [C]% useful task success."
