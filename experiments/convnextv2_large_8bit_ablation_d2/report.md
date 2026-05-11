---
experiment: convnextv2_large_8bit_ablation_d2
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D2 — No Binary Head (Gradient Shortcut Ablation)

## Motivation

The binary head (weight 0.10) operates on `global_feat` — the backbone average BEFORE fusion. This provides a gradient shortcut directly to the backbone, bypassing the lateral and bilateral fusion chains. Removing it forces ALL backbone gradients to flow through the fusion pathway, potentially improving fusion-level feature learning.

Counter-hypothesis: the direct backbone gradient from binary head helps early convergence and backbone adaptation.

## Config Changes from C6

| Parameter | C6 | D2 |
|---|---|---|
| `ablation.use_binary_head` | true | **false** |

All other parameters identical to C6.

## Hypothesis

- Test F1 ~0.67 (neutral or mild positive)
- Binary head weight is small (0.10), so effect may be limited
- Key information: does the backbone need direct gradient signal?

## Results

*Pending*
