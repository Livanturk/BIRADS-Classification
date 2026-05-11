# TODO — Current Decision Ledger (2026-05-05)

This file tracks the live plan only. Completed seed-CI, Saerens-EM, ensemble, TTA, and
drop-hflip experiments are now documented in `tasks/lessons.md` Lessons #51-#59.

## Locked Baseline

- [x] Single-seed C6 expectation established: `test_full_f1_macro = 0.6502 ± 0.0137` across n=6 seeds (Lesson #51).
- [x] seed=42 C6 (`0.6762`) reclassified as a +1.9σ lucky outlier, not the project plateau (Lesson #51).
- [x] Production inference baseline set to the n=4 C6 logit-mean seed ensemble: macro F1 `0.6846`, accuracy `0.7462`, BR1 `0.531`, BR2 `0.788`, BR4 `0.555`, BR5 `0.865` (Lesson #57).
- [x] Inference-time improvement track rejected: TTA stack, Saerens-EM, oracle prior correction, threshold offsets, hierarchical gates, pairwise tuning, and cascade all fail to lift the ensemble ceiling (Lessons #47, #48, #55, #56, #58).
- [x] Drop-hflip rejected as a production change: macro unchanged, BR5 significantly hurt, BR4 only directional (Lesson #59).
- [x] `CLAUDE.md` updated so future recommendations compare against `0.6846`, not seed=42 C6.

## Priority 0 — Radiologist Audit Gate

**Goal:** decide whether the BR1 ceiling is label-noise-bound or representation-bound before spending more GPU on BR1 architecture.

Artifact:
- `artifacts/phase0b_audit_sample.csv` — 120 unanimous-wrong patients, sampled from priority cells.

Decision rule:
- If >=30% of BR1->BR2 unanimous-wrong cases are radiologically defensible as BR2, treat ~0.685 as a label-noise floor and stop BR1 architectural work.
- If <30%, BR1 work may continue, but only with new training-time signal. Do not run more calibration, thresholding, TTA, hflip, sampler-only, or loss-weight-only experiments.
- If >=50% of BR4->BR5 unanimous-wrong cases are radiologically defensible as BR5, cap BR4 expectations and document the ambiguity.

Checklist:
- [ ] Send `artifacts/phase0b_audit_sample.csv` plus image paths to radiologist reviewer.
- [ ] Record verdicts for BR1->BR2, BR4->BR5, BR2->BR4, and BR5->BR4 priority cells.
- [ ] Summarize audit in `artifacts/phase0b_audit_verdict.md` and add a new lesson.
- [ ] Decide whether BR1 architecture work proceeds.

## Priority 1 — BR4 Architecture Track

BR4 is still worth pursuing independently of the BR1 audit because it responds to multiple interventions: seed ensemble (+3.4pp), pairwise BR4 offset (+5.0pp at BR5 cost), and no-hflip (+2.0pp directional F1 / +8.9pp recall).

Recommended next experiment:
- [ ] Train one higher-resolution C6-style run at 2048px, or a 1280->1024 random-crop variant if 2048px is too expensive.
- [ ] Evaluate BR4 F1/recall against the n=4 ensemble BR4 baseline (`0.555`) and seed-mean BR4 baseline (`0.522`).
- [ ] If higher resolution helps BR4 by >=1pp macro-equivalent without BR5 loss, plan n=4 seeds.
- [ ] If higher resolution fails, move to per-region/token-level malign uncertainty head with top-k aggregation.

## Priority 2 — BR1 New-Signal Track (Audit-Gated)

Do this only if the radiologist audit shows BR1->BR2 unanimous-wrong cases are mostly not label-noise-defensible.

Candidate order:
- [ ] Density-conditioned auxiliary head (preferred first BR1 experiment).
- [ ] Supervised contrastive loss with hard-negative BR1<->BR2 mining only after density conditioning fails or audit strongly supports representation poverty.
- [ ] Class-balanced sampler only as part of a new-signal experiment; do not run sampler-only as the main intervention.

Acceptance:
- Beat the production ensemble macro baseline (`0.6846`) or produce a statistically meaningful BR1 lift without a BR2 collapse.
- Report n>=4 seeds for any claimed BR1 improvement.

## Priority 3 — Small Refactor Backlog

- [ ] Make `log_temperature` trainable only if the loss path is refactored to use it. Current parameter is inference-confidence-only and receives no gradient (Lesson #44).
- [ ] Add a training-time view-swap-aware hflip implementation only if a future experiment explicitly tests orientation-consistent augmentation. Keep `horizontal_flip: 0.5` in C6 production configs for now (Lesson #59).
- [ ] Log `best_epoch / swa_start_epoch` or equivalent SWA trajectory diagnostics so future fast-overfit runs are easy to identify (Lesson #59).
