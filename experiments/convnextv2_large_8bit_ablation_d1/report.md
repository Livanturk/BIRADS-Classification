---
experiment: convnextv2_large_8bit_ablation_d1
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D1 — No Subgroup Heads (Loss Simplification)

## Motivation

C6 proved that removing the asymmetry loss (+1.47pp test F1) was the single most effective intervention in the entire study (Lesson #22). The D-series extends this "simplification > regularization" principle (Lesson #29) to the auxiliary classification heads.

The subgroup loss carries a **0.45 weight** — the largest auxiliary component. It forces the model to learn benign→BR1/BR2 and malign→BR4/BR5 sub-classification, but this hierarchical signal is coupled: accuracy requires correct binary separation first. If the coupling introduces gradient noise (similar to asymmetry loss), removing it should improve generalization.

## Config Changes from C6

| Parameter | C6 | D1 |
|---|---|---|
| `ablation.use_subgroup_head` | true | **false** |

All other parameters identical to C6 (SWA, CE loss, no asymmetry).

## Hypothesis

- Test F1 >= 0.67 (mild improvement or neutral)
- If positive: subgroup loss was adding noise like asymmetry loss
- If negative: subgroup heads provide valuable hierarchical inductive bias

## Results

*Pending*
