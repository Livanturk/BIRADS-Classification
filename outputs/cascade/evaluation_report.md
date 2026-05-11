# Cascade Evaluation Report

- input probs: `outputs/cascade/test_probs.parquet`
- test patients: **1655**
- C6 baseline (Lesson #44 frozen): macro F1 = 0.6762

## Headline

- **soft cascade macro F1: 0.6266**  (Δ vs C6 = -0.0496)
- soft cascade weighted F1: 0.6826
- hard cascade macro F1:  0.6262  (soft − hard = +0.0004)

## Per-class F1 (cascade vs C6 baseline)

| class | C6 | cascade | Δ |
|---|---:|---:|---:|
| BR1 | 0.531 | 0.496 | -0.035 |
| BR2 | 0.798 | 0.676 | -0.122 |
| BR4 | 0.518 | 0.517 | -0.001 |
| BR5 | 0.857 | 0.818 | -0.039 |

## Per-class precision / recall (soft cascade)

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| BR1 | 0.389 | 0.681 | 0.496 | 163 |
| BR2 | 0.877 | 0.550 | 0.676 | 596 |
| BR4 | 0.448 | 0.611 | 0.517 | 288 |
| BR5 | 0.821 | 0.814 | 0.818 | 608 |

## Confusion matrix (soft cascade, rows=true, cols=pred)

|       | pred BR1 | pred BR2 | pred BR4 | pred BR5 |
|---|---:|---:|---:|---:|
| true BR1 | 111 | 39 | 12 | 1 |
| true BR2 | 166 | 328 | 94 | 8 |
| true BR4 | 8 | 5 | 176 | 99 |
| true BR5 | 0 | 2 | 111 | 495 |

## Stratified per-true-class drift (soft cascade)

| true class | n | accuracy | drift_to (top 3) |
|---|---:|---:|---|
| BR1 | 163 | 0.6810 | BR1: 68.10%, BR2: 23.93%, BR4: 7.36% |
| BR2 | 596 | 0.5503 | BR2: 55.03%, BR1: 27.85%, BR4: 15.77% |
| BR4 | 288 | 0.6111 | BR4: 61.11%, BR5: 34.38%, BR1: 2.78% |
| BR5 | 608 | 0.8141 | BR5: 81.41%, BR4: 18.26%, BR2: 0.33% |

## Hard vs soft cascade ablation

- soft macro F1:  0.6266
- hard macro F1:  0.6262
- delta (soft − hard): +0.0004
- soft beats hard ✓ (sanity check expected)

## Honest assessment

- BR1 F1: 0.496 (Δ -0.035)  target ≥ 0.55
- BR4 F1: 0.517 (Δ -0.001)  target ≥ 0.55
- macro F1: 0.6266 (Δ -0.0496)  acceptance target ≥ 0.70
- **below target.** Diagnostic next steps:
  - if BR1 stuck near 0.53: G2a is dominated by features G2a's bigger BR2 set supplies; consider per-class oversampling or label smoothing increase.
  - if BR4 stuck near 0.52: BR4↔BR5 boundary is a feature problem, not a head problem (matches Lesson #48). Tier-2 logit adjustment is the principled next move.
  - if Stage-1 binary F1 dropped below C6's 0.94: G1's no-fusion variant is underperforming the multi-task binary head; rerun G1 with full fusion.
