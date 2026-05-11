# Cascade (G-series) — running log

Frozen baseline: **C6 macro F1 = 0.6762** (Lesson #44).
Acceptance target: **cascade test macro F1 ≥ 0.70**, BR1 F1 ≥ 0.55, BR4 F1 ≥ 0.55.

## 2026-04-25 — Phase A–D + F+G scaffolding written

### Decisions confirmed by Livan

| # | Question | Resolution |
|---|---|---|
| 1 | Stage-1 from scratch vs fine-tune C6? | **From scratch.** Stage-1 trains a fresh ConvNeXtV2-Large head — cleaner ablation. |
| 2 | Specialist backbone size | **ConvNeXtV2-Large** for all three (Lesson #25 confirms Large > Base in low-data regime). |
| 3 | Specialist augmentation | **Identical to C6** — isolate cascade decomposition effect. |
| 4 | SWA for specialists | **Yes**, `swa_start_epoch=5` everywhere (Lesson #37: SWA contributes +1.47pp). |
| 5 | Soft cascade primary, hard ablation | **Confirmed.** Hard cascade reported only as a sanity ablation in evaluate.py. |
| 6 | Compute budget ≈ 2.5× C6 | **Approved.** |
| 7 | C6 config path | `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml` |
| 8 | Lessons #22/25/27/37/44/47/48 | Provided inline; key takeaways below. |
| 9 | Stage-1 binary uses `global_feat` (claude.md) vs `patient_feat` (current code) | **Follow claude.md** → Stage-1 skips fusion, mean-pools backbone outputs across 4 views × spatial tokens. Stage-2a/2b still use full pipeline. |
| 10 | Wrapper-model approach (purely additive, no edits to C6 path) | **Acceptable.** Implemented as `models/cascade_model.py` + `train_cascade.py`. |

### Key constraints inherited from prior experiments

- **Lesson #22:** Asymmetry loss permanently disabled. Cascade stages do not use it.
- **Lesson #25:** Capacity reduction hurts in this low-data regime → keep ConvNeXtV2-Large.
- **Lesson #27:** Class weight manipulation is zero-sum on shared decision boundaries → cascade stays with sqrt-inv class weights only, no boost.
- **Lesson #37:** C6 is the single-model optimum. The cascade is the first move *beyond* C6 that is structural rather than ablative; expectation is that specialist representations unlock information that no single shared backbone can provide.
- **Lesson #44:** C6 macro F1 = 0.6762 is the frozen reference. BR4→BR5 drift is 36.1% on test (worse than originally documented).
- **Lesson #47:** Val→test prior shift voids val-tuned threshold offsets. Cascade does not apply any post-hoc threshold tuning at inference (Section 6 guardrail #10). Output raw composed probabilities.
- **Lesson #48:** Inference-time hierarchical reconstruction from C6's existing heads duplicates information. The cascade's gain — if any — must come from *specialist features learned during training*, not from re-composing trained heads.

### Doctrine clarification (#9 vs prompt Section 3.2)

The prompt's Section 3.2 says "all three use the same backbone + fusion architecture as C6 — only the head changes." Decision #9 ("follow claude.md") overrides this for Stage-1 only: claude.md Section 4 explicitly states the binary head consumes `global_feat` (backbone average, bypassing the fusion chain) "to deliver gradients directly to the backbone and reduce competing gradients." We follow claude.md.

Net: Stage-1 = backbone → mean-pool over (4 views, S spatial tokens) → 2-class head. Stage-2a/2b = full C6 fusion pipeline → `patient_feat` → 2-class head.

### Phase A — manifests (DONE)

- Builder: `tools/build_cascade_manifests.py`
- Manifests written under `data/manifests/cascade/`:
  - `train_stage1_binary.csv` — 7273 rows (benign=3767, malign=3506)
  - `val_stage1_binary.csv`   — 1284 rows (benign=665,  malign=619)
  - `train_stage2a_benign.csv` — 3767 rows (BR1=1426, BR2=2341)
  - `val_stage2a_benign.csv`   — 665 rows  (BR1=252,  BR2=413)
  - `train_stage2b_malign.csv` — 3506 rows (BR4=1613, BR5=1893)
  - `val_stage2b_malign.csv`   — 619 rows  (BR4=285,  BR5=334)
- Asserts in builder verify: (a) S2a ∪ S2b = S1 train (partition), (b) S2a ∩ S2b = ∅, (c) no patient overlap between train and val of any stage.
- Splits reproduced from `data.dataset.scan_dataset_from_folders` + `train_test_split(seed=42, test_size=0.15, stratify=labels)` — identical to what `data.dataset.prepare_patient_split` produces, so cascade train/val is a strict subset of C6's split.

### Phase B — configs (DONE)

`configs/cascade/{G1_stage1_binary, G2a_stage2_benign, G2b_stage2_malign}.yaml`. Each is a copy of C6 with cascade-specific overrides:

| field | G1 | G2a | G2b |
|---|---|---|---|
| `cascade.stage` | stage1 | stage2a | stage2b |
| manifests | stage1 | stage2a | stage2b |
| `class_weights` | [1.04, 1.00] | [1.28, 1.00] | [1.08, 1.00] |
| MLflow experiment | `cascade/stage1_binary` | `cascade/stage2_benign` | `cascade/stage2_malign` |
| `lateral_fusion` / `bilateral_fusion` blocks | omitted (Stage-1 skips fusion) | identical to C6 | identical to C6 |

Everything else (LR schedule, optimizer, epochs, augmentation, backbone, image_size=1024, normalization via `data_cfg.bit_depth=8`+`dataset_variant=noseg` → DATASET_STATS_8BIT, SWA settings, asymmetry loss off) is unchanged from C6.

### Phase C — model + loader + trainer (DONE)

- `models/cascade_model.py` — `CascadeStageModel`. Reuses `MultiViewBackbone`, `BilateralLateralFusion`, `BilateralFusion`, `ClassificationHead` verbatim. ~150 LOC, 0 LOC modified to existing C6 code path.
- `data/cascade_loader.py` — `create_cascade_dataloaders(config)` reads manifest CSVs and instantiates the existing `MammographyDataset`. Manifest's `label_stage` column (0/1) is used as the supervised target. Helper `inverse_freq_class_weights` available for "auto" class-weight mode.
- `train_cascade.py` — clones train.py's loop structure (gradient accumulation, mixed precision, OneCycleLR, SWA, early stopping, DagsHub-MLflow + WandB logging). Single 2-class CE loss. `BinaryMetricTracker` reports F1 macro/per-class, AUC, accuracy, confusion matrix. Best-model selection on `val_f1_macro`. Final SWA-vs-best comparison; SWA wins → overwrites `best_model.pt`. Always writes a stable alias `checkpoints/cascade/{G1,G2a,G2b}_best.pt` regardless of which won.

Decoupled from `train.py` to avoid eagerly loading `utils/__init__.py` (which transitively imports seaborn). Helpers (set_seed, load_config, EarlyStopping, save_checkpoint, build_scheduler, apply_output_dirs, param-group builder) inlined.

### Phase D — SLURM scripts (DONE)

`scripts/slurm/train_cascade_stage{1,2a,2b}.sh` — boilerplate for 1 GPU, 24h walltime, 64GB RAM, 8 CPU. Account/partition/QoS lines commented out — Livan to edit per cluster.

### Phase F — soft-cascade inference (DONE; deferred run)

`tools/cascade_inference.py` — loads three checkpoints, runs each model on the **full test set** (per Section 5 Phase F: "the two specialists run on every test patient, not just the patients Stage-1 routes to them"), composes via `P(BR1)=P(benign)·P(BR1|benign)` etc., writes `outputs/cascade/test_probs.parquet` (falls back to CSV if pyarrow missing). Includes a sanity check that composed-prob row sums ≈ 1.0.

### Phase G — evaluation (DONE; deferred run)

`tools/cascade_evaluate.py` — reads test_probs, reports 4-class confusion matrix, per-class precision/recall/F1, macro/weighted F1, side-by-side vs C6 baseline (frozen at 0.6762 + per-class from Lesson #44), stratified per-class drift, and a hard-cascade ablation (route by argmax of stage-1, commit). Writes report to `outputs/cascade/evaluation_report.md` and metrics JSON. Logs to MLflow experiment `cascade/evaluation`.

### Phase E — sanity gates (PENDING — runs after each training job lands)

These are hard gates per the prompt; do **not** run inference until all three pass:

- **G1**: binary val F1 ≥ 0.93 (C6 binary head was 0.940; if G1 falls below 0.93 the no-fusion design is failing).
- **G2a**: on the BR1/BR2 val subset, `f1_macro` ≥ C6's BR1/BR2 macro on the same subset. (To compute C6's reference, run C6 inference and restrict to val patients with `label_4class ∈ {0, 1}`.)
- **G2b**: same idea on BR4/BR5 val subset.

If any gate fails → stop, log diagnostic findings here, ask before continuing.

## 2026-04-27 — Phase E + F + G complete: cascade rejected on test

### Phase E gates (val) — see Lesson #49 for full analysis

| stage | val f1_macro | gate | verdict |
|---|---:|---|---|
| G1   | 0.9664 | ≥ 0.93 | PASS by +3.6pp |
| G2a  | 0.7020 | ≥ C6 BR1/BR2 ref | MARGINAL |
| G2b  | 0.7740 | ≥ C6 BR4/BR5 ref | PASS by +1.9pp |

Notable training pathology: G2a peaked at single epoch 13 (then collapsed to majority-class predictor for 19 more epochs); G2b oscillated in [0.71-0.77] until peaking at epoch 23 then collapsed. OneCycleLR schedule (`epochs=100, max_lr=5e-4, pct_start=0.3`) was misconfigured for early-stopping specialists — peak LR phase destabilized training past the val-best epoch. SWA `using_swa=False` for both G2a and G2b confirming SWA averaged across the regime change.

### Phase F + G outcome — **CASCADE REJECTED** (see Lesson #50)

```
soft cascade test macro F1 = 0.6266    Δ vs C6 (0.6762) = -0.0496
hard cascade test macro F1 = 0.6262    soft − hard = +0.0004 (composition adds nothing)

per-class:  BR1 0.496 (-3.5pp)   BR2 0.676 (-12.2pp)   BR4 0.517 (-0.1pp)   BR5 0.818 (-3.9pp)
```

Both BR1 and BR4 acceptance targets (≥0.55) failed. BR2 cratered −12.2pp due to BR2→BR1 drift doubling (28% cascade vs 13% C6). G1's no-fusion val win (+1.34pp vs C6 binary) reversed on test (−0.81pp); val→test gap widened from C6's 1.4pp to G1's 3.55pp. Soft = hard cascade because all stages produce >95% confident outputs — the soft composition collapses to argmax × argmax.

Root cause: train→test prior shift compounding multiplicatively across cascade stages. Each stage's class weights are calibrated to its train sub-distribution; the test set's different class proportions invalidate every stage's calibration, and the product of three mis-calibrated stages is much worse than a single 4-class head trained with full information.

### Decisions

1. **C6 remains champion (test macro F1 = 0.6762).** Cascade not deployed.
2. **No retraining** of G2a/G2b with LR-schedule fix from Lesson #49 — schedule fix would maybe lift val 1-3pp, but the −4.96pp test gap is structural, not training-instability-driven. Compute not justified.
3. **Next experiment: Tier-2 Task 2.2 (F2 — logit-adjusted training, Menon et al. 2021).** Targets the prior shift directly. Defer 16-bit pipeline (Task 2.1) until F2 lands.
4. The cascade's negative result is recorded for future paper write-up: "tested 3-stage soft cascade, val gains did not transfer due to compounding prior shift across stages."

### Artifacts (preserved for paper / future reference)

- `outputs/cascade/test_probs.parquet`
- `outputs/cascade/evaluation_report.md`
- `outputs/cascade/evaluation_metrics.json`
- MLflow runs (DagsHub): G1 `ad4526d7ef684e7e845ea977aa49d4a2`, G2a `bae58239a2c34f81b76c4f14a5cbbe04`, G2b `c5926b3c3e8841faaae1892678ce97ca`
- C6 reference: `6859aed2a37e43b8b72b5333b2573275`
- Checkpoints kept at `checkpoints/cascade/{G1,G2a,G2b}_best.pt`

## Open issues / risks to watch

1. **Hardcoded DagsHub token in configs** — the C6 config (and now the cascade configs, per the existing project convention) embeds `dagshub_token: d176ee2b...` in plaintext. The token is already publicly exposed on disk; the cascade configs reuse the same value rather than introducing a fresh leak. Recommendation post-experiment: rotate the token and migrate to an env-var (`MLFLOW_TRACKING_PASSWORD`) reference.
2. **Stage-1 architectural divergence from prompt's "same architecture"** — see "Doctrine clarification" above. If Stage-1 misses the 0.93 binary gate, fall back to fusion-enabled Stage-1 by setting `cascade.stage: stage2a` in a renamed config OR adding a `stage1_with_fusion` mode to `CascadeStageModel`.
3. **Test-prior gap (Lesson #47)** — soft cascade's per-class probs are uncalibrated to test prior by design (per guardrail #10). If macro F1 lands in [0.69, 0.70], the next logical experiment is a Saerens-style prior correction at inference, but that is *out of scope for this experiment*.
4. **`tasks/lessons.md` is empty in the repo.** All seven lessons referenced in the prompt arrived through the chat. Future Claude sessions reading the repo cold will not see them.
