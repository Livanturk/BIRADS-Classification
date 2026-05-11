---
experiment: dinov2_vitl_8bit_ablation_d5
status: planned
date: 2026-04-12
baseline: dinov2_vitl_8bit_ablation_c2
backbone: vit_large_patch14_dinov2.lvd142m
---

# D5 — DINOv2 + Focal + SWA + No Asymmetry (DINOv2 Rescue)

## Motivation

The DINOv2 track has stalled:
- A3 (focal, buggy): test 0.6325 (still best DINOv2)
- B2 (CE, buggy): test 0.6136 (CE hurts DINOv2, Lesson #14)
- C2 (focal, fixed): test 0.6240 (bugs provided implicit regularization, Lesson #24)

Two C-series findings haven't been applied to DINOv2:
1. **SWA** (Lesson #17): Single most effective technique for ConvNeXtV2 (+2.3pp)
2. **No asymmetry** (Lesson #22): Removing asymmetry = +1.47pp for ConvNeXtV2

D5 = C2 + SWA + no asymmetry. The SWA compensates for the lost implicit regularization from bug fixes (Lesson #24 explicitly recommends this).

If successful, D5 opens the ensemble path: ConvNeXtV2 (strong BR2/BR4) + DINOv2 (strong BR5).

## Config Changes from C2

| Parameter | C2 | D5 |
|---|---|---|
| `training.use_swa` | false | **true** |
| `training.swa_start_epoch` | — | **5** |
| `training.asymmetry_loss_weight` | 0.10 | **0.0** |

All DINOv2-specific params preserved: focal loss, gamma=2.0, image_size=518, backbone_lr_scale=0.10, cosine_warmup.

## Hypothesis

- Test F1 >= 0.65 (beating A3's 0.6325)
- SWA should help DINOv2 as it helped ConvNeXtV2
- If successful: ensemble candidate with C6/D-winner

## Results

*Pending*
