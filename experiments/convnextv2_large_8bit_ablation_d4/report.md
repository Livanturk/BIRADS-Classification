---
experiment: convnextv2_large_8bit_ablation_d4
status: planned
date: 2026-04-12
baseline: convnextv2_large_8bit_ablation_c6
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
---

# D4 — No SWA (SWA Isolation Test)

## Motivation

C6's +1.47pp gain over B5 could come from two sources:
1. Clean loss alone (removing asymmetry)
2. Clean loss + SWA synergy (SWA averages over a cleaner trajectory)

D4 removes SWA from C6 to isolate source (1).

Key comparison chain:
- B1 (CE + bugs fixed, SWA off, asym 0.10) → test 0.6387
- **D4 (CE + no SWA + no asym)** → test ???
- C6 (CE + SWA + no asym) → test 0.6762

D4 vs B1 = pure effect of removing asymmetry (no SWA in either)
D4 vs C6 = pure effect of SWA on clean loss

## Config Changes from C6

| Parameter | C6 | D4 |
|---|---|---|
| `training.use_swa` | true | **false** |

All other parameters identical to C6.

## Hypothesis

- Test F1 ~0.64-0.66
- If D4 >> B1 (0.6387): clean loss matters more than SWA
- If D4 ~ B1: SWA was the primary driver, asymmetry removal is secondary
- SWA likely contributes ~1-2pp on top of clean loss

## Results

*Pending*
