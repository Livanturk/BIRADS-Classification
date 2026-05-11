---
experiment: convnextv2_large_8bit_ablation_d3
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D3 — Pure L_full Only (Maximum Loss Simplification)

## Motivation

The ultimate test of Lesson #29 (simplification > regularization). Starting from C6 (no asymmetry), D3 removes ALL auxiliary losses:
- Asymmetry loss: already 0.0 (C6)
- Subgroup heads: disabled (D1 tests this alone)
- Binary head: disabled (D2 tests this alone)

Remaining loss: `total = 0.45 × CE_full(4-class)`. A single, clean objective.

This is the Occam's Razor extreme. Three auxiliary signals have been removed:
1. Asymmetry loss (0.10 weight) → C6 showed +1.47pp when removed
2. Subgroup loss (0.45 weight) → D1 tests independently
3. Binary loss (0.10 weight) → D2 tests independently

D3 gives the combined effect + potential interaction effects.

## Config Changes from C6

| Parameter | C6 | D3 |
|---|---|---|
| `ablation.use_subgroup_head` | true | **false** |
| `ablation.use_binary_head` | true | **false** |

All other parameters identical to C6.

## Hypothesis

- High information experiment regardless of outcome
- If test F1 >= 0.68: simplification to a single objective is optimal
- If test F1 < 0.66: auxiliary heads provide useful gradient signals
- Risk: insufficient gradient diversity may slow convergence

## Results

*Pending*
