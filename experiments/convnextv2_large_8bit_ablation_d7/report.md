---
experiment: convnextv2_large_8bit_ablation_d7
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D7 — SWA + Mixup + No Asymmetry (C1 Retest on Clean Loss)

## Motivation

C1 (SWA + Mixup + asymmetry=0.10) produced test F1=0.6431, worse than both B5 (SWA alone, 0.6615) and B3 (Mixup alone, 0.6459). Lesson #23 concluded SWA+Mixup are "antagonistic" because both smooth decision boundaries.

**But C1 also had asymmetry loss.** Was the failure from SWA+Mixup interaction, or from the triple noise of asymmetry+SWA+Mixup? D7 is the critical disambiguation:

- **D7 = C1 - asymmetry loss** (clean SWA+Mixup test)

Decision matrix:
- If D7 > C6 (0.6762): SWA+Mixup WORK on clean loss → Lesson #23 was wrong, asymmetry was the confound
- If D7 < C6: SWA+Mixup truly antagonistic → Lesson #23 confirmed on clean loss
- If D7 ~ C6: Mixup is neutral when loss is clean, SWA dominates

## Config Changes from C6

| Parameter | C6 | D7 |
|---|---|---|
| `training.use_mixup` | false | **true** |
| `training.mixup_alpha` | — | **0.2** |
| `training.use_cutmix` | false | **true** |
| `training.cutmix_alpha` | — | **1.0** |

SWA preserved (use_swa=true, swa_start_epoch=5). All other parameters identical to C6.

## Hypothesis

- Critical information experiment
- If test F1 > 0.69: SWA+Mixup stack on clean loss → potential new best
- If test F1 < 0.66: Lesson #23 definitively confirmed

## Results

*Pending*
