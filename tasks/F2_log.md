# F2 — Logit-Adjusted Training (Menon et al. 2021)

**Status:** scaffolded 2026-04-28. Ready to train.
**Hypothesis:** train→test prior shift (root cause of cascade rejection — Lesson #50)
is best treated by Menon's logit-adjustment recipe: shift logits at training time
by `tau * log pi_train`, predict with raw logits at inference. Targets the failure
mode directly without a cascade.
**Targets:** test macro F1 > 0.6762 (C6); BR2 recall ≥ 0.78 (C6 was 0.74).

---

## Regime A — train-prior only

- `pi_train` is computed from CLAUDE.md train counts: `[1678, 2754, 1898, 2227]`
  → `log pi = [-1.6293, -1.1336, -1.5060, -1.3460]`.
- **Test labels are never inspected.** No Saerens-style estimation, no test-prior
  injection, no test-time adjustment. Single test report at the end after
  val-only `tau` selection. Fairness-clean.
- Implemented in `utils/logit_adjustment.py:train_log_prior_4class()`.

## What changed in the codebase

| File | Change |
| --- | --- |
| `utils/logit_adjustment.py` | new — train priors + helpers |
| `utils/losses.py` | `MultiHeadLoss` accepts `logit_adjustment` dict; if enabled on full head, registers `log_prior_full` buffer and shifts inside `full_criterion` only; drops `class_weights_4` from full head when `drop_class_weights: true` (Menon recipe) |
| `configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau{05,10,15}.yaml` | three configs cloning C6 with `training.logit_adjustment` set |
| `scripts/slurm/train_F2_la_tau{05,10,15}.sh` | SLURM job templates |

**No changes to:** `train.py`, `models/`, `data/`, `utils/metrics.py`. The shift
is non-mutating: `outputs["full_logits"]` stays raw, so metrics
(`utils/metrics.py:68`) and inference (`train.py:814`) are untouched.

## Scope decisions

- **Full head only.** Binary head (test 0.939) is already strong; subgroup heads
  see masked subsets so the relevant priors are subset priors, not the full
  4-class priors. Applying LA only to the full head matches the metric we
  optimize and minimizes blast radius.
- **Class weights dropped on full head when LA on.** Menon's argument is that LA
  *replaces* class re-weighting on the head it's applied to — combining both
  is non-orthogonal and confounds the comparison. Binary/subgroup heads keep
  their class weights since LA is not applied there.
- **Other knobs unchanged from C6:** loss weights (0.10/0.45/0.45), label
  smoothing 0.05, OneCycleLR (max_lr=5e-4, pct_start=0.3), SWA from epoch 5,
  early stopping patience 20 on `val_full_f1_macro`.

## Run plan

```bash
# Train each config (sequential or in parallel as your cluster allows)
sbatch scripts/slurm/train_F2_la_tau05.sh
sbatch scripts/slurm/train_F2_la_tau10.sh
sbatch scripts/slurm/train_F2_la_tau15.sh

# Or interactively
python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau05.yaml
python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau10.yaml
python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau15.yaml
```

## Phase E — val gates (per config)

Pass criterion: `val_full_f1_macro` at the early-stopped checkpoint (or SWA
checkpoint, whichever the trainer selects) is **≥ 0.6783**.

Rationale: C6 best val was 0.7183. LA changes the loss surface; a 4pp
tolerance is reasonable for the configs to be considered "trained". Below
0.6783 means the τ value is incompatible with the rest of the C6 recipe and
should not be evaluated on test.

Kill criterion (mid-training): `val_full_f1_macro < 0.55` by epoch 15.

## Phase F — τ selection (val only)

- Read each config's MLflow run; pull `val_full_f1_macro` history.
- Select `tau*` = the τ with **highest val_full_f1_macro at its best epoch**.
- Tie-break (within 0.5pp): lower std of last 5 epochs' val F1 (stability).
- Test labels MUST NOT be consulted at this step.

## Phase G — single test report

After τ* is fixed:

```bash
python train.py --config configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau<chosen>.yaml \
    --eval-only --checkpoint outputs/.../checkpoints/best.pt
```

(or whatever the existing `train.py` test-eval flag is — verify with
`python train.py --help` if needed)

Report fields, all from a single test pass:
- 4-class macro F1, weighted F1, accuracy
- Per-class P / R / F1 (BR1, BR2, BR4, BR5)
- BR2 → BR1 confusion rate (cf. Lesson #50: cascade hit 28%, C6 hit 13%)
- Δ vs C6: `0.6762`. Δ vs cascade: `0.6266`.

## Acceptance

- Test macro F1 > 0.6762 → **F2 wins**, document in lessons.md as Lesson #51.
- Test macro F1 ≤ 0.6762 → log negative result (paper-grade datapoint), update
  lessons.md, route to next experiment (e.g. test-time LA with val-estimated
  prior, or balanced sampling).

## Operational guardrails

- All three runs use `seed=42`, `split.train=0.85`, `split.val=0.15` — **same
  split as C6**. No re-stratification; comparisons are head-to-head.
- MLflow experiment name is `birads-1024-8bit-logit-adjusted` (separate from
  C6's `birads-1024-8bit-ablation`).
- DagsHub credentials match other configs (token in YAML, not env). If you
  want them out of the configs, that's a separate cleanup pass.
