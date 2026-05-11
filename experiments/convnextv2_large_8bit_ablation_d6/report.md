---
experiment: convnextv2_large_8bit_ablation_d6
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D6 — Mixup Only + No Asymmetry (No SWA)

## Motivation

B3 (Mixup alone) achieved test F1=0.6459 with asymmetry_loss=0.10. Lesson #22 showed asymmetry loss hurts. D6 tests: how does Mixup perform on a clean loss WITHOUT SWA?

This is the "alternative regularization path" experiment. C6 uses SWA as the regularizer. D6 uses Mixup instead. Both have clean loss (no asymmetry).

Comparison chain:
- B3 (Mixup + asym=0.10, no SWA) → test 0.6459
- D4 (no Mixup, no SWA, no asym) → test ???
- **D6 (Mixup, no SWA, no asym)** → test ???
- C6 (SWA, no Mixup, no asym) → test 0.6762

D6 vs B3 = effect of removing asymmetry on Mixup training
D6 vs D4 = pure Mixup contribution on clean loss
D6 vs C6 = Mixup-only vs SWA-only (head-to-head)

## Config Changes from C6

| Parameter | C6 | D6 |
|---|---|---|
| `training.use_swa` | true | **false** |
| `training.use_mixup` | false | **true** |
| `training.mixup_alpha` | — | **0.2** |
| `training.use_cutmix` | false | **true** |
| `training.cutmix_alpha` | — | **1.0** |

All other parameters identical to C6 (no asymmetry preserved).

## Hypothesis

- Test F1 >= 0.66 (beating B3's 0.6459 with clean loss)
- Mixup + clean loss should be strictly better than Mixup + noisy loss
- Probably won't match C6 (SWA was more effective than Mixup in B-series)

## Results

*Pending*
