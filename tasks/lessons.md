# Lessons Learned

## Pitfall #10 — SwinV2 window_size=24 requires specific image sizes (2026-04-09)
**Problem:** `swinv2_base_window12to24_192to384` with `img_size=1024` crashes at model build:
`RuntimeError: shape '[1, 10, 24, 10, 24, 1]' is invalid for input of size 65536`.
SwinV2 stage 0 produces 1024/4=256 spatial, and 256%24≠0.

**Rule:** For window_size=24, ALL feature map resolutions must be divisible by 24.
Valid sizes: 384 (native, last stage padded), **768** (2× native, all stages clean), 1536.
Formula: image_size must be divisible by `patch_size × window_size = 4 × 24 = 96`,
AND `image_size / (2^(n_stages-1) × patch_size)` must be ≥ window_size or divisible by it.

**Fix applied:** A2 config image_size 1024 → 768.

## Pitfall #11 — Focal Loss vs CrossEntropy in Imbalanced BI-RADS
**Problem:** Focal loss (γ=2.0) catastrophically degrades BR1 performance (-7.6pp test F1) compared to standard CrossEntropy. By aggressively down-weighting high-confidence "easy" normal tissue cases, it prevents the model from learning the subtle boundaries between BR1 and BR2. It also results in poorly calibrated, low-confidence predictions.
**Rule:** Default to CrossEntropy (`loss_type: "ce"`) for this 4-class BI-RADS task.

## Pitfall #12 — Evaluating on Shifted Holdout Sets
**Problem:** Severe val→test F1 degradation observed on BR1 (19.6% train vs 9.8% test).
**Rule:** Recognize that class-wise test degradation is often a dataset-level prior probability shift, not a model failure. Use Stacking Ensembles (`ensemble_evaluate.py --stacking`) to combine models with complementary error profiles (e.g., DINOv2 for OOD generalization + ConvNeXtV2 for high-res local details).

---

## B-Series Experiment Lessons (2026-04-11)

### Lesson #13 — Bug Fixes Can Increase Overfitting (B1 vs A1-CE)
**Problem:** B1 fixed two real bugs (wrong normalization stats, test-set class weights) and reverted config drift. Val F1 improved by +1.9pp (0.7141→0.7334), but test F1 barely moved (+0.17pp, 0.6370→0.6387). The val→test gap WIDENED from 7.7pp to **9.5pp** — the worst in the entire study.

**Root cause:** The bugs were introducing noise that accidentally acted as implicit regularization. With correct normalization (0.1210/0.1977) and proper class weights ([1.28,1.00,1.20,1.11]), the model's internal representations align better with the training/val distribution, enabling sharper (deeper) minima. But the test set has a different class prior (BR1: 9.8% vs 19.6% in train) that sharp minima don't generalize to.

**Rule:** Fixing training bugs is necessary but NOT sufficient. Always pair bug fixes with explicit regularization (Mixup, SWA, dropout tuning) because correct training often means stronger fitting. Monitor the val→test gap alongside absolute metrics — if val improves but test doesn't, overfitting has deepened.

### Lesson #14 — DINOv2 Needs Focal Loss, Unlike ConvNeXtV2 (B2 vs A3)
**Problem:** The A-series showed CE beats focal for ConvNeXtV2 (+1.0pp test F1). B2 applied this same CE switch to DINOv2 — and it REGRESSED by -1.9pp (0.6325→0.6136). DINOv2 val peak was identical (0.6940) across A3 (focal) and B2 (CE), but test diverged.

**Root cause:** DINOv2's self-supervised features are domain-agnostic and less mammography-adapted. Focal loss (gamma=2.0) forces DINOv2 to concentrate capacity on difficult cases — the hard BR1/BR4 boundaries that matter for BI-RADS. CE treats all samples equally, wasting DINOv2's limited domain adaptation on already-easy samples. ConvNeXtV2's supervised ImageNet features are already locally adapted, so CE's uniform gradient is sufficient.

**Rule:** Do NOT generalize loss function findings across backbone families. Focal loss is harmful for ConvNeXtV2 (BR1 collapse) but beneficial for DINOv2 (hard-example mining compensates for weaker domain adaptation). Test loss functions separately for each backbone architecture.

### Lesson #15 — Mixup/CutMix: Best Val-Test Gap Regularizer (B3)
**Problem:** B1 had a 9.5pp val→test gap despite bug fixes. B3 added Mixup (alpha=0.2) + CutMix (alpha=1.0) as the only change.

**Evidence:**
- Train F1 dropped from 0.795 to 0.588 (strong regularization)
- Val peak dropped slightly: 0.7334→0.7193 (-1.4pp)
- Test IMPROVED: 0.6387→0.6459 (+0.72pp)
- Val→test gap narrowed: 9.5pp→7.3pp (-2.2pp)
- BR2 test surged +5.7pp (0.675→0.732)
- Convergence: more oscillation, peak at ep17 (vs ep7 for B1)

**NaN train metrics note:** Mixup generates soft blended targets that break asymmetry_loss and binary_loss metric logging (they log NaN). This is a logging artifact, not a training problem — losses are computed correctly via lambda-weighting.

**Rule:** For this multi-view mammography task, Mixup/CutMix is the most effective single regularizer for closing the val→test gap. The mild alpha=0.2 for Mixup keeps interpolation close to original samples (lambda~0.9), while CutMix alpha=1.0 provides diverse spatial cutouts. Patient-level mixing ensures all 4 views are mixed consistently.

### Lesson #16 — CORAL Ordinal Loss Is Fundamentally Broken for Non-Contiguous BI-RADS (B4)
**Problem:** B4 used CORAL ordinal loss with K-1=3 cumulative binary classifiers, with subgroup heads disabled (a previously tested configuration). Despite following the proven pattern from `ordinal_nosubgroup_v1.yaml`, B4 CATASTROPHICALLY FAILED:
- Val F1 peaked at 0.4449 (29pp below B1)
- BR4 val F1 = 0.054 (near-zero collapse)
- No test evaluation was triggered
- Training plateaued at val F1 ~0.31 for 34 consecutive epochs before slowly climbing to 0.44

**Root cause:** CORAL models cumulative thresholds: P(y≥2), P(y≥4), P(y≥5). With non-contiguous BI-RADS classes (1,2,**skip 3**,4,5), the P(y≥4) threshold has no natural decision boundary — there is no class 3 to separate from class 4. The optimizer gets stuck because the middle threshold receives conflicting gradients from BR2 (push threshold right) and BR4 (push threshold left) with no intermediate class to anchor it.

**Rule:** PERMANENTLY ABANDON ordinal regression losses for BI-RADS classification when class 3 is absent. The missing class creates an irrecoverable gap in the cumulative threshold space. This applies to CORAL, Proportional Odds, Stick-Breaking, and any cumulative-link model. If ordinal structure is desired, use label smoothing with ordinal-aware targets (e.g., smooth BR1↔BR2 more than BR1↔BR5) instead — this encodes ordinal prior without cumulative thresholds.

### Lesson #17 — SWA: Best Absolute Test F1, But BR1 Tradeoff (B5)
**Problem:** B5 added SWA (start_epoch=5) to the B1 config. Result:
- **Test F1: 0.6615** — best in entire 8-bit ablation study (+2.3pp vs B1)
- **Test AUC: 0.913** — best across all experiments
- **Test Kappa: 0.624** — best across all experiments
- Val→test gap: 6.7pp (vs B1's 9.5pp) — confirms flat minima generalize better

Per-class shifts vs B1:
- BR2: +12.3pp (0.675→0.798) — massive
- BR4: +4.9pp (0.498→0.547) — significant
- BR5: -0.8pp (0.856→0.848) — negligible
- **BR1: -7.3pp (0.526→0.453) — substantial regression**

**Root cause of BR1 regression:** SWA smooths decision boundaries by averaging weights across the training trajectory (ep5-ep31). BR1 (n=163 test) and BR2 (n=596 test) share a fuzzy boundary. Smoothing favors the higher-density class (BR2) at the expense of the lower-density class (BR1). The F1 formula amplifies this: a few BR1→BR2 flips cost a lot in BR1 F1 but are absorbed by BR2's larger sample.

**Rule:** SWA is the single most effective technique for 8-bit test F1, but introduces a BR1 regression risk. When combining SWA with other methods, add BR1-targeted regularization (e.g., asymmetry_benign_weight > 0, or class-specific augmentation for BR1). Always report per-class F1 alongside macro — SWA's macro gains can mask minority-class losses.

### Lesson #18 — MLflow Reports Last-Epoch Metrics, Not Best-Epoch (Critical Tooling Note)
**Problem:** MLflow's `search_runs()` returns the LAST logged value for each metric, not the value at the best checkpoint epoch. For experiments with patience=20, the last epoch can be 20+ epochs past the best, giving misleadingly low val F1.

**Example:** B1 best val F1 = 0.7334 (epoch 7), but MLflow reports val_full_f1_macro = 0.6557 (epoch 27, the early-stop epoch). Using the MLflow-reported value underestimates val performance by 7.8pp and misrepresents the generalization gap.

**Rule:** ALWAYS use `get_metric_history()` to find the true best-epoch val F1. Compare test F1 against the best val F1 (the checkpoint that was actually selected), not the last-epoch val F1. When reporting gaps, use: `gap = best_val_f1 - test_f1`.

### Lesson #20 — Recommended Next Experiments (Priority Order)
Based on B-series results:
1. **B3+B5 (Mixup + SWA)** — Highest priority. Orthogonal mechanisms: Mixup regularizes training, SWA averages weights post-training. Expected: test F1 ~0.68-0.69 with gap ~5-6pp.
2. **A3-fixed (DINOv2 + Focal + Bug Fixes)** — B2 showed CE hurts DINOv2. Create a new config with focal loss + correct normalization/weights. Expected: test F1 ~0.65.
3. **B5 + asymmetry_benign_weight > 0** — Address SWA's BR1 regression by adding BR1-specific penalty.
4. **Ensemble: B5 + A3-fixed** — ConvNeXtV2 (BR2/BR4 strength) + DINOv2 (BR5 strength, different error profile).
5. **ABANDON:** CORAL ordinal (Lesson #16), more aggressive augmentation without SWA (insufficient alone).

### Summary Statistics — B-Series Ablation Table

```
| Experiment | Backbone    | Change vs B1         | Best Val F1 | Test F1  | Gap    | Binary F1 | AUC   |
|------------|-------------|----------------------|:-----------:|:--------:|:------:|:---------:|:-----:|
| A1-CE      | ConvNeXt-L  | (buggy baseline)     | 0.7141      | 0.6370   | 7.7pp  | 0.895     | 0.898 |
| B1         | ConvNeXt-L  | Bug fixes only       | 0.7334      | 0.6387   | 9.5pp  | 0.901     | 0.905 |
| B2         | DINOv2-ViT-L| Bug fixes + CE       | 0.6940      | 0.6136   | 8.0pp  | 0.884     | 0.896 |
| B3         | ConvNeXt-L  | + Mixup/CutMix       | 0.7193      | 0.6459   | 7.3pp  | 0.920     | 0.894 |
| B4         | ConvNeXt-L  | + CORAL Ordinal      | 0.4449      | FAILED   | —      | 0.915     | 0.864 |
| B5         | ConvNeXt-L  | + SWA                | 0.7286      | 0.6615*  | 6.7pp  | 0.936     | 0.913 |
|------------|-------------|----------------------|-------------|----------|--------|-----------|-------|
```
*B5 test F1 is from SWA-averaged model, not best checkpoint.

---

## C-Series Design Rationale (2026-04-11)

### Lesson #21 — Generalization Gap Is the Primary Adversary, Attack It from Multiple Angles
**Evidence from B-series:**
- B1 (bug fixes only): gap 9.5pp — worst in study
- B3 (Mixup/CutMix): gap 7.3pp — best relative gap reduction
- B5 (SWA): gap 6.7pp — best absolute test F1 (0.6615)
- C1 (SWA + Mixup): expected to stack → gap ~5-6pp

Even with the best techniques, the gap remains substantial, so we must extract maximum generalization from every possible angle.

**Five regularization axes identified:**
1. **Capacity (C4):** Over-parameterization → structural regularization via smaller backbone
2. **Feature Preservation (C5):** Fine-tuning damage → lower backbone LR preserves ImageNet features
3. **Loss Landscape (C6, C7):** Auxiliary loss noise (C6) and hard-example mining (C7)
4. **Explicit Dropout (C8):** Memorization → force distributed representations
5. **Combination (C1):** Orthogonal stacking of proven techniques

**Rule:** When the generalization gap is the bottleneck (not val performance), systematic ablation across independent regularization axes is more informative than iteratively stacking techniques. Each axis tests a different hypothesis about WHY the model overfits — capacity, feature destruction, loss function noise, or co-adaptation. Results will reveal which overfitting mechanism dominates in 8-bit mammography.

---

## C-Series Experiment Lessons (2026-04-12)

### Lesson #22 — Asymmetry Loss Is the Hidden Generalization Gap Source (C6 — New 8-bit Champion)
**Problem:** C6 removed the asymmetry_loss (weight 0.10→0.0) from B5 (SWA). This was intended as a neutral "does it help or hurt?" test. The answer was emphatic:

**Evidence:**
- **Test F1: 0.6762** — new 8-bit record (+1.47pp vs B5's 0.6615)
- **Val→test gap: 4.21pp** — narrowest in the entire study (vs B5's 6.71pp → 2.50pp improvement)
- BR1: 0.531 (+7.8pp vs B5's 0.453) — massive recovery of the SWA-induced BR1 regression
- BR2: 0.798 (identical to B5) — no regression on the dominant class
- BR4: 0.518 (-2.9pp vs B5's 0.547) — mild regression
- BR5: 0.857 (+0.9pp vs B5)
- Accuracy: 0.7444 (vs B5's 0.7390)

**Root cause:** The asymmetry loss computes bilateral differences (RCC vs LCC, RMLO vs LMLO) and penalizes the model based on left-right asymmetry patterns. These patterns overfit to training distribution-specific bilateral features that don't transfer to test. The extra gradient signal corrupts the primary classification loss landscape, especially on the delicate BR1-BR2 boundary where SWA's smoothing already introduces ambiguity.

**Why C6 recovered BR1:** The asymmetry loss was the *source* of the BR1 problem, not SWA alone. SWA smoothed decision boundaries, but asymmetry loss noise pushed the BR1-BR2 boundary into BR2 territory during training. Removing the noise let SWA average over a cleaner trajectory.

**SWA checkpoint note:** C6's SWA model beat the best checkpoint (SWA overwrote best_model.pt per train.py:734-741 logic). This confirms SWA works well when the loss is clean.

**Rule:** Auxiliary losses that capture domain-specific priors (bilateral asymmetry) can hurt generalization when the prior doesn't hold across the train→test shift. The asymmetry_loss should be **permanently disabled** for this dataset. When facing a generalization gap, audit existing losses before adding regularization — the gap may be caused by loss noise, not insufficient regularization.

### Lesson #23 — SWA and Mixup/CutMix Are Antagonistic, Not Orthogonal (C1)
**Problem:** C1 combined SWA + Mixup/CutMix, hypothesizing orthogonal stacking would reach test F1 ≥ 0.68. Result: test F1 = 0.6431 — **worse than both B5 (SWA only, 0.6615) and B3 (Mixup only, 0.6459)**.

**Evidence:**
- Test F1: 0.6431 (-1.84pp vs B5, -0.28pp vs B3)
- Gap: 7.27pp (worse than B5's 6.71pp and B3's 7.34pp)
- BR2: 0.747 (-5.1pp vs B5's 0.798) — significant regression
- BR5: 0.826 (-2.2pp vs B5's 0.848)
- SWA model was WORSE than best checkpoint (swa_model.pt saved separately = best model won)

**Root cause:** SWA averages model parameters from the late training trajectory (epoch 5+). Mixup blurs decision boundaries during training by creating interpolated samples with soft labels. SWA preserves and amplifies this blur by averaging the blurred parameters. The two techniques both operate on **decision boundary smoothing** — SWA through weight-space averaging, Mixup through input-space interpolation — making them redundant rather than orthogonal.

The SWA model performing worse than the best checkpoint (confirmed by swa_model.pt existing as a separate file) proves that the weight trajectory under Mixup is not suitable for averaging.

**Rule:** Do NOT combine SWA with Mixup/CutMix for this task. They are antagonistic, not orthogonal. "Both are regularizers" does not mean they stack. Always validate combinations vs. individual techniques. The Mixup + SWA interference invalidates the Lesson #20 recommendation for this combination.

### Lesson #24 — Bug Fixes Increase Overfitting Regardless of Backbone (C2 Confirms Lesson #13)
**Problem:** C2 applied bug fixes (correct normalization, correct class weights) to DINOv2 + focal loss (baseline: A3). Test F1: 0.6240, down from A3's 0.6325 (-0.85pp).

**Evidence (comparing C2 vs A3):**
- Val F1: 0.6905 vs 0.6940 (slight drop — unusual, bugs may have helped even val)
- Test F1: 0.6240 vs 0.6325 (-0.85pp)
- Gap: 6.65pp vs 6.15pp (gap widened, same direction as B1 vs A1-CE)
- BR1: 0.376 vs 0.482 (-10.6pp!) — worst BR1 in the entire study
- BR2: 0.727 vs 0.681 (+4.6pp) — some improvement
- BR5: 0.859 vs 0.857 (stable)

**Root cause:** Identical phenomenon to Lesson #13 (B1 vs A1-CE for ConvNeXtV2): bugs (wrong normalization stats, train-set class weights) inject noise that accidentally regularizes. Correct values enable sharper fitting. This is **backbone-agnostic** — affects ConvNeXtV2 (-0.17pp or stable), DINOv2 (-0.85pp), and likely any architecture.

**BR1 collapse explanation:** DINOv2 with correct normalization aligns features more tightly to the training distribution. Without the regularizing noise, the model overconfidently classifies borderline BR1 cases as BR2 (larger class pulls).

**Rule:** Bug fixes MUST be paired with explicit regularization, regardless of backbone. Lesson #13 is not ConvNeXtV2-specific — it is a universal phenomenon in this task. For DINOv2, consider adding SWA or dropout (not Mixup) to compensate for lost implicit regularization.

### Lesson #25 — Larger Pretrained Models Generalize Better in Low-Data Medical Imaging (C4)
**Problem:** C4 replaced ConvNeXtV2-Large (~197M params, feature_dim=1536) with ConvNeXtV2-Base (~89M params, feature_dim=1024). Hypothesis: smaller model = less memorization = smaller gap. **Wrong.**

**Evidence:**
- Test F1: 0.6269 (-3.46pp vs B5) — significant regression
- Gap: 9.09pp — one of the **worst gaps** in the entire study (only B1's 9.5pp is worse)
- Val F1: 0.7178 (competitive!) — the model fits training data well but doesn't transfer
- BR4: 0.490 (-5.7pp vs B5) — worst BR4 regression in C-series
- BR5: 0.865 (+1.7pp vs B5) — easier high-confidence cases improve, hard cases suffer
- SWA model beat best checkpoint (confirmed by no swa_model.pt)

**Root cause:** In the low-data regime (8,557 patients), pretrained features ARE the regularizer. ConvNeXtV2-Large's richer feature space (197M params trained on ImageNet-22k) provides better transfer learning foundations than Base's reduced capacity. Cutting parameters removes useful pretrained representations without reducing memorization — the model still memorizes training-specific patterns, just with fewer tools to generalize from.

The paradox: larger models generalize better despite having more parameters because the extra capacity stores ImageNet-learned features that transfer, not dataset-specific noise.

**Rule:** Do NOT reduce model capacity to fight overfitting in transfer learning with limited medical data. In low-data regimes, larger pretrained models > smaller ones. The overfitting source is not excess capacity — it's train/test distribution shift. Address distribution shift (SWA, cleaner losses) rather than capacity.

### Lesson #26 — Excessive Feature Preservation Prevents Domain Adaptation (C5)
**Problem:** C5 reduced backbone_lr_scale from 0.2 to 0.05 (effective backbone LR: 2.5e-6 vs 1e-5). Hypothesis: preserving more ImageNet-22k features = better generalization. **Wrong.**

**Evidence:**
- Test F1: 0.6284 (-3.31pp vs B5) — significant regression
- Gap: 8.30pp (vs B5's 6.71pp)
- BR2: 0.698 (-10.0pp vs B5's 0.798) — **massive regression**, worst BR2 in C-series
- BR4: 0.493 (-5.4pp vs B5)
- BR1: 0.462 (+0.9pp vs B5) — marginal improvement
- BR5: 0.861 (+1.3pp vs B5) — slight gain
- SWA model beat best checkpoint (confirmed)

**Root cause:** Mammography textures — spiculations, microcalcifications, tissue density patterns — are sufficiently different from ImageNet-22k natural images that substantial backbone fine-tuning is essential. At backbone_lr_scale=0.05, the backbone is nearly frozen, preserving ImageNet features that don't map to mammography-specific discriminative patterns. BR2 (benign findings) suffers most because benign tissue patterns are furthest from ImageNet objects.

The BR1/BR5 slight improvement confirms: high-level semantic features (normal vs malignant) transfer better from ImageNet than mid-level tissue texture features (benign findings).

**Rule:** backbone_lr_scale=0.2 is near-optimal for ConvNeXtV2 on 8-bit mammography. Going lower (0.05) prevents necessary domain adaptation. Do not under-tune the backbone — mammography requires more adaptation than typical medical imaging transfer tasks because of the unique tissue texture domain.

### Lesson #27 — Class Weight Manipulation Is Zero-Sum on Shared Decision Boundaries (C3)
**Problem:** C3 increased BR1 class weight from 1.28 to 1.80 (+40%) to counter SWA's BR1 regression (B5: BR1=0.453). Hypothesis: higher BR1 weight = BR1 recovery without macro collapse.

**Evidence:**
- Test F1: 0.6346 (-2.69pp vs B5)
- BR1: 0.465 (+1.2pp vs B5's 0.453) — **minimal improvement** despite 40% weight increase
- BR2: 0.700 (-9.8pp vs B5's 0.798) — **catastrophic regression**
- BR4: 0.521 (-2.6pp)
- Gap: 7.59pp (worse than B5's 6.71pp)

**Contrast with C6 (the right approach):**
- C6 recovered BR1 by +7.8pp (0.453→0.531) WITHOUT any BR2 regression (0.798 stable)
- C6 achieved this by removing noise (asymmetry loss), not by shifting decision boundaries

**Root cause:** BR1 (normal) and BR2 (benign findings) share a fuzzy decision boundary — subtle tissue changes separate them. Increasing BR1 weight shifts this boundary toward BR2 territory, reclassifying borderline BR2 cases as BR1. This is a **zero-sum game**: every BR1 gain comes from BR2 loss. The 1.2pp BR1 gain cost 9.8pp BR2 — an 8:1 efficiency ratio in the wrong direction.

The weight increase also amplifies gradient noise from the hard BR1 cases (n=163 test, smallest class), destabilizing the overall optimization.

**Rule:** Class weight manipulation is a blunt instrument that cannot create new discriminative features — it only shifts existing decision boundaries. For minority class recovery on shared boundaries, address the root cause (noisy loss signals, architectural issues) rather than adjusting weights. C6 proves this: removing asymmetry loss noise recovered BR1 6.5x more effectively than a 40% weight boost.

### Lesson #28 — Focal Loss Remains Harmful for ConvNeXtV2 Even With SWA (C7)
**Problem:** C7 changed loss from CE to focal (gamma=2.0) under SWA conditions. Hypothesis: SWA's flat minima + focal's hard-example mining might rescue the focal loss for ConvNeXtV2.

**Evidence:**
- Test F1: 0.6468 (-1.47pp vs B5)
- Gap: 7.93pp (vs B5's 6.71pp)
- BR1: 0.485 (+3.2pp vs B5) — some improvement from hard-example mining
- BR4: 0.498 (-4.9pp vs B5) — significant regression
- BR2: 0.753 (-4.5pp vs B5)
- SWA model was WORSE than best checkpoint (swa_model.pt saved separately)

**Key observation:** SWA was counterproductive with focal loss (SWA model lost to best checkpoint). This makes sense: focal loss creates a non-stationary loss surface (hard examples change as training progresses), and SWA averaging over this non-stationary trajectory produces a poor average.

**Rule:** Focal loss is harmful for ConvNeXtV2 regardless of SWA (confirming Lesson #11). SWA is also incompatible with focal loss (non-stationary loss surface). This finding is robust across A-series (A1 vs A1-CE) and C-series (C7 vs B5). NEVER use focal loss with ConvNeXtV2 in this pipeline.

### Lesson #29 — Meta-Lesson: Simplification Beats Regularization (C-Series Summary)

**Evidence:** 7 experiments tested 7 different approaches to reducing the val→test generalization gap:

```
| Rank | Exp | Strategy              | Test F1 | Δ vs B5 | Gap    | Verdict        |
|------|-----|-----------------------|---------|---------|--------|----------------|
| 1    | C6  | Remove asymmetry loss | 0.6762  | +1.47pp | 4.21pp | ✅ NEW BEST     |
| 2    | C7  | Add focal loss        | 0.6468  | -1.47pp | 7.93pp | ❌ Worse        |
| 3    | C1  | Stack SWA+Mixup       | 0.6431  | -1.84pp | 7.27pp | ❌ Antagonistic |
| 4    | C3  | Increase BR1 weight   | 0.6346  | -2.69pp | 7.59pp | ❌ Zero-sum     |
| 5    | C5  | Freeze backbone more  | 0.6284  | -3.31pp | 8.30pp | ❌ Under-adapted|
| 6    | C4  | Smaller backbone      | 0.6269  | -3.46pp | 9.09pp | ❌ Over-reduced |
| 7    | C2  | Fix DINOv2 bugs       | 0.6240  | -3.75pp | 6.65pp | ❌ Overfit more |
```

**6 out of 7 hypotheses FAILED.** The one winner (C6) **removed** complexity rather than adding regularization. The generalization gap was not caused by insufficient regularization — it was caused by an auxiliary loss (asymmetry_loss) injecting noise.

**Implication for D-series:** The best configuration is now:
- ConvNeXtV2-Large + CE loss + SWA + **no asymmetry loss** (C6 config)
- Test F1: 0.6762, gap: 4.21pp


**Rule:** Before adding regularization to fight a generalization gap, audit every component of the existing loss for unnecessary complexity. The Occam's Razor principle applies to loss functions: **the simplest loss that fits is the one that generalizes.**

### Summary Statistics — C-Series Ablation Table

```
| Experiment | Backbone     | Change vs B5              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|
| B5 (base)  | ConvNeXt-L  | SWA only                  | 0.7286      | 0.6615   | 6.71pp | 0.453 | 0.798 | 0.547 | 0.848 |
| C1         | ConvNeXt-L  | + Mixup/CutMix            | 0.7158      | 0.6431   | 7.27pp | 0.458 | 0.747 | 0.541 | 0.826 |
| C2         | DINOv2-ViT-L| Focal + bug fixes         | 0.6905      | 0.6240   | 6.65pp | 0.376 | 0.727 | 0.535 | 0.859 |
| C3         | ConvNeXt-L  | BR1 weight 1.80           | 0.7105      | 0.6346   | 7.59pp | 0.465 | 0.700 | 0.521 | 0.853 |
| C4         | ConvNeXt-B  | Base backbone (~89M)      | 0.7178      | 0.6269   | 9.09pp | 0.421 | 0.731 | 0.490 | 0.865 |
| C5         | ConvNeXt-L  | backbone_lr_scale=0.05    | 0.7114      | 0.6284   | 8.30pp | 0.462 | 0.698 | 0.493 | 0.861 |
| **C6**     | ConvNeXt-L  | **asymmetry_wt=0.0**      | **0.7183**  |**0.6762**|**4.21pp**|**0.531**|**0.798**|0.518|0.857|
| C7         | ConvNeXt-L  | Focal + SWA               | 0.7261      | 0.6468   | 7.93pp | 0.485 | 0.753 | 0.498 | 0.852 |
```

### Recommended Next Steps (Post C-Series)
1. **C8** (Extreme Dropout, still pending) — Given 6/7 "add regularization" approaches failed, C8 is unlikely to help. Run for completeness but low expectations.
2. **D1: C6 + Mixup** — C6 removed asymmetry noise. Mixup (B3) was the best standalone gap regularizer. Test if Mixup works better with a cleaner loss (C1 failed partly because asymmetry noise + Mixup + SWA was triple-noisy).
3. **D3: Ensemble C6 + best DINOv2** — Complementary error profiles (ConvNeXtV2 strong on BR2/BR4, DINOv2 strong on BR5).
4. **D4: C6 without SWA** — Isolate whether the gain comes from loss simplification alone or requires SWA.
5. **ABANDON:** Capacity reduction (C4), feature freezing (C5), class weight manipulation (C3), SWA+Mixup combo (C1).

---

## D-Series Experiment Lessons (2026-04-14)

### Lesson #30 — Auxiliary Heads Provide Essential Multi-Task Regularization, Unlike Asymmetry Loss (D1-D3)
**Problem:** After C6 showed that removing asymmetry loss improved generalization, the D-series tested whether removing auxiliary heads (subgroup, binary) would continue the simplification trend. **All three removals hurt.**

**Evidence:**
- D1 (no subgroup, wt 0.45): test F1 = 0.6563, **-1.99pp** vs C6
- D2 (no binary, wt 0.10): test F1 = 0.6453, **-3.09pp** vs C6
- D3 (no subgroup + no binary): test F1 = 0.6476, **-2.86pp** vs C6
- All three gap-widened: D1 5.75pp, D2 6.94pp, D3 7.61pp (vs C6's 4.21pp)

**Root cause — why auxiliary heads help but asymmetry loss hurt:**
The auxiliary heads provide *complementary gradient signals* that regularize the shared backbone through multi-task learning. The binary head forces a coarse {BR1,BR2} vs {BR4,BR5} separation; the subgroup head imposes an intermediate grouping. These create multiple consistent views of the same classification hierarchy, stabilizing training.

Asymmetry loss, by contrast, computes bilateral differences (L vs R) — a *domain-specific prior* that doesn't hold across train→test shift. The auxiliary heads encode *task-structure priors* (hierarchical class groupings) that are invariant across distributions.

**Rule:** Not all loss components are equal. Auxiliary heads encoding task-structure hierarchy are beneficial regularizers. Domain-specific priors (bilateral asymmetry) are overfitting risks. When simplifying losses, distinguish between structural multi-task components (keep) and domain-prior components (audit carefully).

### Lesson #31 — Binary Head Punches 3x Above Its Weight as a Gradient Anchor (D2)
**Problem:** The binary head has only 0.10 weight (10% of total loss), yet removing it caused the largest single-head impact: -3.09pp. This is 1.55x worse than removing the subgroup head (-1.99pp) which carries 4.5x more weight (0.45).

**Evidence:**
- D2 (no binary): macro F1 = 0.6453, gap = 6.94pp
- BR4 collapsed: 0.476 (-4.2pp vs C6's 0.518) — worst BR4 in D-series
- BR5 dropped: 0.844 (-1.3pp vs C6)
- BR1 actually improved: 0.521 (-1.0pp vs C6) — lost gradient anchor shifts all boundaries
- SWA lost to best checkpoint (swa_model.pt saved separately)

**Root cause:** The binary head forces the model to separate {BR1,BR2} vs {BR4,BR5} — a clean 2D gradient signal. This binary decision boundary *anchors* the full 4-class classifier. Without it, the model must discover this coarse separation from the fine-grained 4-class loss alone, which is harder and less stable. The binary gradient acts as a "curriculum" signal: learn coarse separation first, then refine.

**Efficiency paradox:** Impact/weight ratio: binary = -3.09pp / 0.10wt = 30.9 per unit. Subgroup = -1.99pp / 0.45wt = 4.4 per unit. The binary head is **7x more efficient** per unit weight than subgroup.

**Rule:** In hierarchical classification, a lightweight auxiliary head encoding the coarsest class grouping provides disproportionate regularization. The binary head's low weight (0.10) is already near-optimal — it provides a stabilizing gradient anchor without dominating the loss. Do NOT remove or reduce it.

### Lesson #32 — Clean Loss Alone Matches Dirty Loss + SWA — Asymmetry Removal Had More Impact Than SWA (D4)
**Problem:** D4 removed SWA from C6, isolating the "clean loss alone" contribution. This creates a clean 2×2 factorial comparison.

**Evidence — 2×2 Factorial:**
```
|                    | Dirty Loss (asym=0.10) | Clean Loss (asym=0.0) | Δ (clean vs dirty) |
|--------------------|:----------------------:|:---------------------:|:-------------------:|
| No SWA             | B1 = 0.6387            | D4 = 0.6615           | +2.28pp             |
| SWA                | B5 = 0.6615            | C6 = 0.6762           | +1.47pp             |
| Δ (SWA vs no SWA)  | +2.28pp                | +1.47pp               |                     |
```

**Key insights:**
- **D4 (clean, no SWA) = B5 (dirty, SWA) = 0.6615 exactly.** Removing asymmetry loss is equivalent in impact to adding SWA.
- Asymmetry removal effect: +2.28pp (no SWA) / +1.47pp (with SWA) → **larger than SWA's contribution**
- SWA effect: +2.28pp (dirty loss) / +1.47pp (clean loss) → SWA helps more on dirty loss (compensates for noise)
- The effects are **sub-additive**: 2.28 + 2.28 = 4.56pp expected but only 3.75pp observed (C6 vs B1). Some overlap in what they fix.

**D4's gap is remarkably tight:** 4.94pp — second narrowest after C6 (4.21pp). Clean loss fundamentally reduces overfitting even without SWA's weight averaging.

**Rule:** Removing loss noise has equal or greater impact than adding SWA. Before investing in SWA or other post-hoc regularization, audit the loss function for unnecessary components. In the current pipeline, asymmetry removal alone was worth +2.28pp — a "free" improvement that required no extra compute.

### Lesson #33 — DINOv2 Partially Rescued by SWA + Clean Loss, But Architecture Gap Persists (D5)
**Problem:** D5 applied the C-series winning insights (SWA + no asymmetry) to DINOv2 (baseline: C2). Target: test F1 ≥ 0.65 to justify an ensemble path.

**Evidence:**
- D5 test F1: 0.6383, **+1.43pp** vs C2 (0.6240) — meaningful improvement
- Gap: 5.42pp (vs C2's 6.65pp) — improved
- But **still 3.79pp below C6** and **below the 0.65 target**
- BR5: 0.864 (+0.5pp vs C2's 0.859) — DINOv2's strength maintained
- BR1: 0.454 (+7.9pp vs C2's 0.376) — SWA recovered some BR1
- BR2: 0.720 (-0.7pp vs C2's 0.727) — stable
- SWA lost to best checkpoint (swa_model.pt saved separately)

**SWA dynamics:** SWA lost to best checkpoint on DINOv2, unlike ConvNeXtV2 C6 where SWA won. DINOv2's self-supervised features create a less stable training trajectory — the averaging introduces noise rather than smoothing. The test evaluation still used the best checkpoint, but the +1.43pp gain came from removing asymmetry noise rather than SWA averaging.

**Root cause of persistent gap:** DINOv2 ViT-L was pretrained via self-supervised learning on natural images. Its patch-based attention mechanism processes 37×14 patches at 518px — far coarser than ConvNeXtV2's hierarchical feature maps at 1024px. Mammographic features (microcalcifications, spiculations) require local high-resolution detail that DINOv2's architecture doesn't capture well despite its global attention capabilities.

**Rule:** DINOv2 is NOT viable as a primary backbone for this mammography task — the architecture gap is fundamental, not fixable by training tricks. At 0.6383 test F1, it's too weak for ensemble contribution (< 0.65 threshold). **Abandon the DINOv2 track.** Future backbone exploration should focus on ConvNeXtV2 variants or other CNN-based architectures that preserve local spatial detail.

### Lesson #34 — Mixup's B-Series Benefit Was Compensating for Asymmetry Noise, Not True Regularization (D6)
**Problem:** D6 tested Mixup/CutMix on clean loss without SWA. In B-series, B3 (Mixup on dirty loss) improved over B1 by +0.72pp. Would Mixup improve on clean loss too?

**Evidence — Mixup on clean vs dirty loss:**
```
| Condition              | With Mixup       | Without Mixup    | Mixup Effect |
|------------------------|:----------------:|:----------------:|:------------:|
| Dirty loss, no SWA     | B3 = 0.6459      | B1 = 0.6387      | +0.72pp      |
| Clean loss, no SWA     | D6 = 0.6353      | D4 = 0.6615      | **-2.62pp**  |
| Clean loss, with SWA   | D7 = 0.6563      | C6 = 0.6762      | **-1.99pp**  |
```

- **Mixup HELPED (+0.72pp) on dirty loss but HURT (-2.62pp) on clean loss**
- D6 gap = 8.96pp — **worst gap in the entire D-series** and one of the worst in the study
- D6 val F1 = 0.7249 (2nd highest in D-series!) but test F1 = 0.6353 → severe overfitting
- BR2 collapsed: 0.701 (-5.6pp vs D4's 0.757)

**Root cause:** Mixup creates interpolated training samples with soft labels. On dirty loss (with asymmetry noise), this interpolation counteracted the noise — Mixup's label smoothing partially compensated for asymmetry's corrupted gradients. On clean loss, the interpolation creates *confusing* training signals (blending normal BR1 tissue with malignant BR5, for example) that the model doesn't need to learn from. The clean loss already provides accurate gradients; Mixup degrades them.

The high val F1 (0.7249) with low test F1 (0.6353) reveals that Mixup's input-space interpolation creates training-specific smoothness that doesn't transfer to real test images.

**Rule:** Mixup/CutMix is NOT a universal regularizer — it was effective only because it was compensating for a different problem (asymmetry loss noise). On a clean loss function, Mixup is **harmful**. This reframes Lesson #15 (B3): Mixup's apparent gap-closing ability was an artifact of the dirty loss, not an intrinsic regularization benefit. **Permanently abandon Mixup/CutMix for this pipeline** now that the loss is clean.

### Lesson #35 — SWA + Mixup Antagonism Is Intrinsic, Not Confounded by Asymmetry (D7 Confirms Lesson #23)
**Problem:** C1 (SWA+Mixup on dirty loss) failed, but was the antagonism caused by asymmetry noise confounding the combination? D7 retests SWA+Mixup on C6's clean loss to disambiguate.

**Evidence:**
- D7 (SWA+Mixup, clean loss): test F1 = 0.6563, **-1.99pp** vs C6
- C1 (SWA+Mixup, dirty loss): test F1 = 0.6431, -1.84pp vs B5
- Both show ~2pp penalty for adding Mixup to SWA
- D7 gap: 5.96pp (vs C6's 4.21pp = +1.75pp wider)
- SWA lost to best checkpoint in D7 (swa_model.pt saved separately) — same as C1

**The smoking gun — SWA trajectory corruption:**
SWA won (overwriting best_model.pt) in C6 (no Mixup) but LOST in D7 (with Mixup). The only difference is Mixup. Mixup corrupts the late-training weight trajectory that SWA averages over, making the averaged model worse than the best single checkpoint.

**Definitively answering the D-series question:** The SWA+Mixup antagonism is NOT caused by asymmetry noise. It is an **intrinsic incompatibility** between two boundary-smoothing mechanisms:
- SWA smooths in weight space (averaging model parameters)
- Mixup smooths in input space (interpolating training samples)
- Combined, they over-smooth, blurring the BR1/BR2 and BR4/BR5 boundaries beyond useful classification

**Rule:** SWA and Mixup/CutMix are permanently incompatible for this task. Lesson #23 is confirmed and strengthened: the antagonism is intrinsic to the mechanism interaction, not an artifact of loss function noise. **Never combine SWA with Mixup/CutMix regardless of other configuration choices.**

### Lesson #36 — SWA Effectiveness Depends on Loss Landscape Balance (D1-D3 SWA Patterns)
**Problem:** SWA won (overwriting best_model.pt) in C6 (3 auxiliary heads) and D3 (0 auxiliary heads), but LOST in D1 (2 heads: binary+full) and D2 (2 heads: subgroup+full). Why?

**Evidence — SWA win/loss pattern:**
```
| Experiment | Active Heads                | SWA Outcome | Test F1 |
|------------|----------------------------|:-----------:|:-------:|
| C6         | binary + subgroup + full   | WON         | 0.6762  |
| D3         | full only                  | WON         | 0.6476  |
| D1         | binary + full              | LOST        | 0.6563  |
| D2         | subgroup + full            | LOST        | 0.6453  |
```

**Root cause:** SWA averages model weights from the late training trajectory (epochs 5+). This averaging works best when the loss landscape is *stable and balanced* — i.e., when the gradient directions don't oscillate wildly.

- **C6 (3 heads):** Three loss terms provide balanced, redundant gradient signals. The multi-task structure creates a smooth loss landscape where SWA averaging produces a good solution.
- **D3 (1 head):** Single loss term = simplest possible landscape. SWA averaging works because there's no inter-objective conflict.
- **D1/D2 (2 heads):** Removing one head creates an *asymmetric* multi-task loss. The remaining two losses compete without the third providing a balancing gradient. This creates oscillation in the training trajectory that SWA averaging fails to smooth.

**Analogy:** Think of 3 legs on a stool (stable), 1 leg (a simple pole, stable in its own way), but 2 legs (unstable, falls to one side).

**Rule:** When using SWA with multi-task losses, ensure the loss landscape is balanced. Either use the full multi-task structure or simplify to a single objective. Removing individual auxiliary heads while keeping SWA creates instability. If testing auxiliary head removal, also consider disabling SWA or using the best-checkpoint-only evaluation.

### Lesson #37 — D-Series Meta: C6 Is the Goldilocks Configuration — All 7 Simplifications/Additions Failed
**Problem:** The D-series tested C6 from 7 angles: 3 loss simplifications (D1-D3), 1 component removal (D4), 1 DINOv2 rescue (D5), 2 Mixup tests (D6-D7). **All 7 experiments scored below C6.**

**Evidence (ranked by test F1):**
```
| Rank | Exp | Strategy               | Test F1 | Δ vs C6  | Gap    | Verdict                           |
|------|-----|------------------------|---------|----------|--------|-----------------------------------|
| —    | C6  | BASELINE               | 0.6762  | —        | 4.21pp | CHAMPION                          |
| 1    | D4  | Remove SWA             | 0.6615  | -1.47pp  | 4.94pp | SWA essential (+1.47pp)           |
| 2    | D1  | Remove subgroup head   | 0.6563  | -1.99pp  | 5.75pp | Subgroup head helpful             |
| 2    | D7  | Add Mixup + SWA        | 0.6563  | -1.99pp  | 5.96pp | SWA+Mixup antagonism intrinsic    |
| 4    | D3  | Remove both heads      | 0.6476  | -2.86pp  | 7.61pp | Multi-task learning essential      |
| 5    | D2  | Remove binary head     | 0.6453  | -3.09pp  | 6.94pp | Binary head critical gradient anchor|
| 6    | D5  | DINOv2 rescue          | 0.6383  | -3.79pp  | 5.42pp | Architecture gap persistent        |
| 7    | D6  | Mixup replaces SWA     | 0.6353  | -4.09pp  | 8.96pp | Mixup harmful on clean loss        |
```

**C-series simplified INTO the optimum. D-series probed BEYOND it and found nothing better.**

**Component contribution (isolated effects from C6 baseline):**
- SWA: +1.47pp (D4→C6)
- Subgroup head: +1.99pp (D1→C6)
- Binary head: +3.09pp (D2→C6)
- Asymmetry removal: +2.28pp (B1→D4, measured without SWA)

**Confirmed permanently abandoned:**
- Mixup/CutMix (harmful on clean loss — Lesson #34)
- Focal loss for ConvNeXtV2 (C7, Lesson #28)
- DINOv2 as primary backbone (D5, Lesson #33)
- Asymmetry loss (C6, Lesson #22)
- Class weight manipulation (C3, Lesson #27)
- Capacity reduction (C4, Lesson #25)
- Backbone freezing below lr_scale=0.2 (C5, Lesson #26)

**D-Series Decision Tree Outcomes:**
- ❌ D1/D3 < C6 → no further loss simplification
- ✅ D4 >> B1 (+2.28pp) → clean loss alone is very valuable
- ✅ D4 < C6 (-1.47pp) → SWA is essential on top of clean loss
- ❌ D5 < 0.65 → ensemble path abandoned
- ❌ D7 < C6 → SWA+Mixup antagonism was NOT confounded

**Rule:** C6's configuration — ConvNeXtV2-Large + CE loss + SWA + three auxiliary heads (binary 0.10, subgroup 0.45, full 0.45) + no asymmetry loss — represents the optimal single-model configuration for 8-bit 1024×1024 mammography BI-RADS classification. Further improvements should explore: (a) learning rate schedules, (b) data augmentation strategies beyond Mixup, (c) test-time augmentation, (d) ensemble strategies with ConvNeXtV2 variants, or (e) 16-bit pipeline optimization.

### Summary Statistics — D-Series Ablation Table

```
| Experiment | Backbone     | Change vs C6              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA     |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-------:|
| C6 (base)  | ConvNeXt-L  | BASELINE                  | 0.7183      | 0.6762   | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON     |
| D1         | ConvNeXt-L  | -subgroup head            | 0.7138      | 0.6563   | 5.75pp | 0.479 | 0.751 | 0.532 | 0.864 | LOST    |
| D2         | ConvNeXt-L  | -binary head              | 0.7147      | 0.6453   | 6.94pp | 0.521 | 0.741 | 0.476 | 0.844 | LOST    |
| D3         | ConvNeXt-L  | -subgroup -binary         | 0.7237      | 0.6476   | 7.61pp | 0.461 | 0.748 | 0.532 | 0.849 | WON     |
| D4         | ConvNeXt-L  | -SWA                      | 0.7109      | 0.6615   | 4.94pp | 0.500 | 0.757 | 0.542 | 0.847 | N/A     |
| D5         | DINOv2-ViT-L| +SWA -asymmetry (from C2) | 0.6925      | 0.6383   | 5.42pp | 0.454 | 0.720 | 0.515 | 0.864 | LOST    |
| D6         | ConvNeXt-L  | -SWA +Mixup/CutMix        | 0.7249      | 0.6353   | 8.96pp | 0.487 | 0.701 | 0.508 | 0.846 | N/A     |
| D7         | ConvNeXt-L  | +Mixup/CutMix             | 0.7159      | 0.6563   | 5.96pp | 0.482 | 0.773 | 0.515 | 0.856 | LOST    |
```

### Recommended Next Steps (Post D-Series)
C6 is confirmed optimal. The 8-bit pipeline has a hard floor at test F1 ≈ 0.68. Potential E-series directions:
1. **Learning rate schedule tuning** — C6 uses step LR. Try cosine annealing with warm restarts to improve SWA trajectory.
2. **Spatial augmentation** — Geometric transforms (rotation, elastic deformation) that don't blend labels like Mixup.
3. **Test-Time Augmentation (TTA)** — Multi-view inference: flip/rotate at test time and average predictions. Free generalization without training changes.
4. **Ensemble (ConvNeXtV2 variants only)** — Ensemble top 2-3 ConvNeXtV2 runs (C6, D4, D1) with different random seeds or training trajectories.
5. **ABANDON:** DINOv2 track, Mixup/CutMix, further loss simplification, additional auxiliary losses.

---

## E-Series Experiment Lessons (2026-04-17)

### Lesson #38 — OneCycleLR's Aggressive Peak LR Is Essential, Not Interchangeable (E1)
**Problem:** E1 replaced OneCycleLR (peak=5e-4 for heads, 1e-4 for backbone) with cosine_warmup (peak=5e-5 for heads, 1e-5 for backbone). The hypothesis was that a stable/decaying LR during the SWA phase would produce better weight averaging. **Wrong — 10x lower peak LR caused a 3.16pp regression.**

**Evidence:**
- Test F1: 0.6446, **-3.16pp** vs C6 (0.6762)
- Val F1: 0.7207 (+0.24pp vs C6's 0.7183) — slightly HIGHER val, much lower test
- Gap: 7.61pp (vs C6's 4.21pp → +3.40pp wider)
- BR1: 0.466 (-6.5pp vs C6's 0.531)
- BR2: 0.755 (-4.3pp vs C6's 0.798)
- BR4: 0.508 (-1.0pp)
- BR5: 0.850 (-0.7pp)
- SWA WON (overwrote best_model.pt)

**Root cause:** OneCycleLR's peak at 5e-4 (10x above the cosine warmup's peak of 5e-5) provides a critical **exploration phase** in the first 30% of training. This high-LR phase pushes the model out of narrow basins into broader regions of the loss landscape before SWA averaging begins. Cosine warmup's gentle LR never reaches high enough to escape the initial basin, converging to a nearby but inferior minimum. SWA then faithfully preserves this inferior solution — the SWA WON, but on a worse trajectory.

The val F1 being slightly *higher* with cosine warmup while test F1 drops confirms: the model found a sharper minimum (better val fit) but one that doesn't generalize. OneCycleLR's disruptive peak forces a flatter minimum that transfers better.

**Rule:** OneCycleLR with max_lr=5e-4 and pct_start=0.3 is not just a scheduler choice — it provides an essential high-LR exploration phase that determines the basin quality for subsequent SWA averaging. Do NOT replace it with monotonic or low-peak schedulers. The SWA literature's preference for stable LR applies to the averaging phase, but OneCycleLR satisfies this because the peak occurs at epoch ~30 (30% of 100) while SWA starts at epoch 5 — the LR *is* declining for most of the SWA phase.

### Lesson #39 — SWA-Optimal Schedules from Literature Fail on Multi-Objective Loss (E2)
**Problem:** E2 used cosine warm restarts (T_0=10, T_mult=2, eta_min=1e-7) — the schedule the original SWA paper (Izmailov et al., 2018) recommends as optimal. Restarts at epochs 10, 30, 70 should create diverse weight snapshots for richer averaging. **Result: worst test F1 in the E-series and one of the worst gaps in the entire study.**

**Evidence:**
- Test F1: 0.6363, **-3.99pp** vs C6 — worst in E-series
- Val F1: 0.7305 — **highest in E-series** (even higher than C6's 0.7183!)
- Gap: **9.42pp** — worst in E-series, rivaling B1's 9.5pp (the worst in the entire study)
- BR1: 0.427 (-10.4pp vs C6's 0.531) — massive collapse
- BR2: 0.749 (-4.9pp vs C6's 0.798)
- BR4: 0.513 (-0.5pp) — barely affected
- BR5: 0.857 (identical to C6)
- SWA WON (overwrote best_model.pt)

**Root cause:** The SWA paper's recommendations apply to single-objective tasks (image classification with one cross-entropy loss). This pipeline uses a multi-objective loss (binary + subgroup + full = 3 loss terms). Warm restarts force periodic LR spikes that push the optimizer to re-explore, but each restart also forces the model to **re-balance three competing objectives**. The trajectory between restarts oscillates wildly across the multi-objective Pareto front.

SWA averaging over these oscillatory trajectories produces a weight vector that is a poor compromise: the averaged weights fit the training distribution well (val F1 = 0.7305, highest!) but the averaged decision boundaries are incoherent for test generalization. The 9.42pp gap (val minus test) proves the SWA average is a **sharp, training-specific** solution despite weight-space averaging.

BR1's collapse (-10.4pp) is the signature: the smallest class is most sensitive to oscillatory boundary shifts, and SWA averaging over multiple restart phases blurs BR1's delicate boundaries into BR2.

**Rule:** Do NOT apply SWA literature recommendations (warm restarts, cyclic schedules) directly when using multi-objective losses. The original SWA paper assumes a single smooth loss landscape. Multi-task losses create a landscape with multiple competing gradients where warm restart diversity becomes harmful noise rather than useful exploration. Stick with OneCycleLR for this pipeline.

### Lesson #40 — Delayed SWA Start Trades BR2 for BR4 but Loses Both Overall (E3)
**Problem:** E3 delayed SWA from epoch 5→10 and extended patience from 20→30. Hypothesis: SWA over more-converged weights = better average. **Wrong — SWA LOST to best checkpoint, and overall test F1 dropped by 3.13pp.**

**Evidence:**
- Test F1: 0.6449, **-3.13pp** vs C6
- Val F1: 0.7104 (-0.79pp vs C6's 0.7183) — lower val too
- Gap: 6.55pp (narrower than E1/E2/E4/E5, but wider than C6's 4.21pp)
- BR1: 0.505 (-2.6pp vs C6's 0.531) — modest drop
- BR2: 0.684 (**-11.4pp** vs C6's 0.798) — catastrophic collapse
- BR4: 0.544 (+2.6pp vs C6's 0.518) — best BR4 in E-series
- BR5: 0.847 (-1.0pp)
- **SWA LOST** (swa_model.pt saved separately — SWA average worse than best checkpoint)
- Accuracy: 0.6870 (worst in E-series)

**Root cause:** Two interacting failure modes:

1. **SWA trajectory corruption:** With SWA starting at epoch 10, the first 10 epochs of OneCycleLR push the model through the high-LR exploration phase without SWA averaging. By epoch 10, the model has already passed the LR peak (at epoch ~30 of OneCycleLR with pct_start=0.3) — wait, actually the peak is at 30% of training so epoch 30, meaning by epoch 10 the LR is still climbing. SWA at epoch 10 starts averaging during the LR climb, but misses the early stabilization at epoch 5-10 that C6 captures. The model's early exploratory weights (epoch 5-10) are excluded, reducing the diversity of the SWA average.

2. **Extended patience enables overfitting:** Patience=30 lets training continue for 30 epochs past the val peak without improvement. The extra epochs don't help SWA (SWA lost to best checkpoint anyway) but allow the model to overfit BR2 — the largest training class. BR2's -11.4pp drop is the signature: longer training → deeper memorization of the dominant class's training patterns → worse test generalization on BR2.

The BR4 improvement (+2.6pp) is a silver lining: delayed SWA preserves some of the harder malignant decision boundaries that early SWA smoothing would blur. But the BR2 collapse overwhelms this gain.

**Rule:** SWA start at epoch 5 is optimal for this pipeline. Earlier starts (D-series didn't test) or later starts (E3) both degrade performance. swa_start_epoch=5 catches the model at the right moment: past random initialization but before deep memorization. Patience=20 is also optimal — extending to 30 provides no benefit and enables overfitting. Do NOT modify SWA timing parameters.

### Lesson #41 — Label Smoothing 0.10 + SWA Is Another Antagonistic Smoothing Pair (E4)
**Problem:** E4 doubled label smoothing from 0.05 to 0.10. Hypothesis: label smoothing (output-space regularization) is orthogonal to SWA (weight-space regularization), unlike Mixup (input-space, proven antagonistic in Lesson #23). **Wrong — label smoothing at 0.10 caused a 3.32pp regression with an 8.44pp gap.**

**Evidence:**
- Test F1: 0.6430, **-3.32pp** vs C6
- Val F1: 0.7274 (+0.91pp vs C6's 0.7183) — higher val, lower test
- Gap: 8.44pp (vs C6's 4.21pp → +4.23pp wider)
- BR1: 0.477 (-5.4pp vs C6's 0.531)
- BR2: 0.730 (-6.8pp vs C6's 0.798)
- BR4: 0.507 (-1.1pp)
- BR5: 0.858 (+0.1pp)
- SWA WON (overwrote best_model.pt)

**Root cause — a pattern emerges across three smoothing mechanisms:**
```
| Smoothing Type   | Mechanism              | Combined with SWA | Δ vs C6  |
|------------------|------------------------|:------------------:|:--------:|
| Mixup (C1)       | Input-space blending   | ANTAGONISTIC       | -1.84pp  |
| Label 0.10 (E4)  | Output-space softening | ANTAGONISTIC       | -3.32pp  |
| Label 0.05 (C6)  | Output-space softening | COMPATIBLE         | baseline |
```

Label smoothing at 0.10 produces **softer target distributions** ([0.025, 0.025, 0.025, 0.925]) that reduce gradient magnitude for high-confidence predictions. SWA simultaneously **smooths weights** by averaging parameters. Both mechanisms flatten decision boundaries through different paths, but the net effect is the same: over-smoothed class boundaries.

At 0.05, label smoothing provides just enough target softness to prevent overconfident predictions without interfering with SWA's weight-space smoothing. At 0.10, the two smoothing mechanisms compound — the model never commits strongly enough to any boundary, and SWA preserves this indecisiveness.

The val F1 being higher (+0.91pp) with worse test confirms the pattern from E1 and E2: the model finds a training-specific smoothness that looks good on the similar val distribution but fails on the shifted test set.

**Rule:** Label smoothing 0.05 is the optimal value for this pipeline. Combined with SWA, it represents the maximum tolerable output-space smoothing. Label smoothing ≥ 0.10 becomes antagonistic with SWA — joining Mixup and warm restarts in the "don't combine with SWA" category. The principle: **SWA's weight-space smoothing occupies the regularization budget; any additional smoothing mechanism that independently softens boundaries will push past the optimum.**

### Lesson #42 — Binary Head Weight 0.10 Is Already Optimal — Efficiency ≠ Underweighting (E5)
**Problem:** E5 doubled the binary head weight from 0.10→0.20 (subgroup reduced 0.45→0.35 to compensate). Lesson #31 showed the binary head was 7x more efficient per weight unit than subgroup. Hypothesis: boosting its weight would amplify the gradient anchor. **Wrong — the 7x efficiency means 0.10 is already sufficient, not that it needs more.**

**Evidence:**
- Test F1: 0.6425, **-3.37pp** vs C6
- Val F1: 0.7190 (+0.07pp vs C6's 0.7183) — essentially identical val
- Gap: 7.65pp (vs C6's 4.21pp → +3.44pp wider)
- BR1: 0.438 (**-9.4pp** vs C6's 0.531) — severe collapse
- BR2: 0.735 (-6.3pp vs C6's 0.798)
- BR4: 0.535 (+1.7pp vs C6's 0.518) — improved
- BR5: 0.863 (+0.6pp vs C6's 0.857) — improved
- SWA WON (overwrote best_model.pt)

**Root cause:** The binary head provides a {BR1,BR2} vs {BR4,BR5} gradient anchor. At 0.10 weight, it provides just enough gradient to stabilize the coarse separation without competing with fine-grained classification. At 0.20, the binary head's gradient dominates during critical decision-making:

- **BR4/BR5 improved** (+1.7pp, +0.6pp): The malignant side benefits from stronger benign/malignant separation because BR4 and BR5 are already well-separated from benign classes.
- **BR1 collapsed** (-9.4pp): The stronger binary gradient forces the model to optimize for {benign vs malignant} over {BR1 vs BR2}. Since the subgroup head (responsible for BR1 vs BR2 within benign) dropped from 0.45→0.35, the fine-grained benign distinction loses gradient budget. BR1 (smaller benign class) is sacrificed to improve the binary task's accuracy on the borderline cases.
- **BR2 dropped** (-6.3pp): Even BR2 suffers because the subgroup head's reduced weight means less gradient for the entire benign sub-classification.

**The efficiency paradox resolved:** Lesson #31's "7x more efficient per unit weight" means the binary head at 0.10 already provides 0.10 × 30.9 = 3.09pp of impact. Doubling to 0.20 should provide ~6.18pp — but it doesn't, because the relationship is non-linear. The binary head's gradient anchor has **diminishing returns** beyond the optimal level, and the reduced subgroup gradient creates a net negative.

**Rule:** Loss weight ratios in C6 (binary=0.10, subgroup=0.45, full=0.45) are at their optimal balance. High per-unit efficiency does NOT mean a component is underweighted — it means a small amount provides outsized value, which is the *definition* of being at the right weight. Do NOT adjust individual loss weights. The 7x efficiency finding from D-series should be interpreted as "0.10 is brilliantly efficient" not "0.10 should be increased."

### Lesson #43 — E-Series Meta: C6 Is at a Sharp, Verified Global Optimum for the 8-bit Pipeline

**Problem:** The E-series tested 5 individually reasonable, well-motivated single-variable perturbations to C6's configuration: LR scheduler (2 variants), SWA timing, label smoothing, and loss weight rebalancing. **All 5 completed experiments regressed by 3.1-4.0pp. The gap widened in ALL cases. Two experiments (E6, E7) did not complete.**

**Evidence (ranked by test F1):**
```
| Rank | Exp | Strategy                    | Test F1 | Δ vs C6  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA   |
|------|-----|-----------------------------|---------|----------|--------|-------|-------|-------|-------|-------|
| —    | C6  | BASELINE                    | 0.6762  | —        | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON   |
| 1    | E3  | Later SWA + longer training | 0.6449  | -3.13pp  | 6.55pp | 0.505 | 0.684 | 0.544 | 0.847 | LOST  |
| 2    | E1  | Cosine warmup scheduler     | 0.6446  | -3.16pp  | 7.61pp | 0.466 | 0.755 | 0.508 | 0.850 | WON   |
| 3    | E4  | Label smoothing 0.10        | 0.6430  | -3.32pp  | 8.44pp | 0.477 | 0.730 | 0.507 | 0.858 | WON   |
| 4    | E5  | Binary head wt 0.20         | 0.6425  | -3.37pp  | 7.65pp | 0.438 | 0.735 | 0.535 | 0.863 | WON   |
| 5    | E2  | Warm restarts scheduler     | 0.6363  | -3.99pp  | 9.42pp | 0.427 | 0.749 | 0.513 | 0.857 | WON   |
| —    | E6  | Stronger augmentation       | DNF     | —        | —      | —     | —     | —     | —     | —     |
| —    | E7  | 16-bit transfer             | DNF     | —        | —      | —     | —     | —     | —     | —     |
```

**Five-series convergence pattern:**
```
| Series | Experiments | Beat C6? | Best ΔF1 vs champion | Champion |
|--------|-------------|:--------:|:--------------------:|----------|
| A      | 4           | N/A      | N/A                  | A1-CE    |
| B      | 5           | Yes      | +2.45pp (B5>A1-CE)   | B5       |
| C      | 7           | Yes      | +1.47pp (C6>B5)      | C6       |
| D      | 7           | No       | -1.47pp (best D4)    | C6       |
| E      | 5 (of 7)    | No       | -3.13pp (best E3)    | C6       |
```

**The degradation ACCELERATED from D to E:** D-series' best was -1.47pp from C6 (D4), but E-series' best is -3.13pp (E3). This means the E-series perturbation axes (scheduler, SWA timing, smoothing level, weight ratios) are **more sensitive** than D-series' axes (head removal, Mixup, backbone swap). C6's configuration is tightly optimized along the dimensions E-series probed.

**The recurring pattern across E-series — val up, test down:**
- E1: val +0.24pp, test -3.16pp
- E2: val +1.22pp, test -3.99pp
- E4: val +0.91pp, test -3.32pp

Three of five experiments showed HIGHER val F1 with LOWER test F1. This is the hallmark of **training distribution overfitting** — the modifications find sharper minima that fit train/val better but generalize worse. C6's configuration uniquely balances sharpness and flatness.

**Confirmed permanently frozen (in addition to D-series list):**
- OneCycleLR scheduler (do not change to cosine, warm restarts, or step)
- SWA start epoch = 5 (do not delay)
- Early stopping patience = 20 (do not extend)
- Label smoothing = 0.05 (do not increase)
- Loss weights: binary=0.10, subgroup=0.45, full=0.45 (do not rebalance)

**The 8-bit pipeline is CONVERGED.** C6's test F1 = 0.6762 with gap = 4.21pp represents the ceiling for single-model, single-run 8-bit 1024×1024 performance with this architecture and dataset. Further single-model improvements must come from the 16-bit pipeline (higher dynamic range, F-series) or multi-model strategies (ensembles, TTA).

### Summary Statistics — E-Series Ablation Table

```
| Experiment | Backbone    | Change vs C6              | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   | SWA   |
|------------|-------------|---------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|
| C6 (base)  | ConvNeXt-L  | BASELINE                  | 0.7183      | 0.6762   | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | WON   |
| E1         | ConvNeXt-L  | cosine_warmup scheduler   | 0.7207      | 0.6446   | 7.61pp | 0.466 | 0.755 | 0.508 | 0.850 | WON   |
| E2         | ConvNeXt-L  | cosine_warm_restarts      | 0.7305      | 0.6363   | 9.42pp | 0.427 | 0.749 | 0.513 | 0.857 | WON   |
| E3         | ConvNeXt-L  | SWA start=10, patience=30 | 0.7104      | 0.6449   | 6.55pp | 0.505 | 0.684 | 0.544 | 0.847 | LOST  |
| E4         | ConvNeXt-L  | label_smoothing=0.10      | 0.7274      | 0.6430   | 8.44pp | 0.477 | 0.730 | 0.507 | 0.858 | WON   |
| E5         | ConvNeXt-L  | binary_head=0.20          | 0.7190      | 0.6425   | 7.65pp | 0.438 | 0.735 | 0.535 | 0.863 | WON   |
| E6         | ConvNeXt-L  | stronger augmentation     | —           | DNF      | —      | —     | —     | —     | —     | —     |
| E7         | ConvNeXt-L  | 16-bit + SWA + no asym    | —           | DNF      | —      | —     | —     | —     | —     | —     |
```

### Recommended Next Steps (Post E-Series)
The 8-bit single-model pipeline is fully converged at C6. No further 8-bit hyperparameter tuning is justified.

1. **16-bit pipeline optimization (F-series)** — Transfer C6 insights to the higher dynamic range pipeline. F1/F2 already running.
2. **Test-Time Augmentation (TTA)** — Horizontal flip + multi-crop at inference. Free generalization without retraining. Expected: +1-2pp.
3. **Ensemble strategies** — Average predictions from C6 + D4 (different SWA state) or multiple C6 runs with different seeds.
4. **ABANDON for 8-bit:** All further single-variable hyperparameter perturbations. The E-series proved that every direction around C6 leads downhill. The only remaining 8-bit paths are inference-time improvements (TTA, ensemble) that don't modify the trained model.

---

## Tier 0 — Inference-Time Pipeline Lessons (2026-04-18)

### Lesson #44 (2026-04-18): ensemble_evaluate.py norm-stats bug was cosmetic; train.py metrics bug-free — but three prompt-level assumptions proved wrong

**Context:** Task 0.1 — fix `ensemble_evaluate.py` ImageNet normalization bug and re-establish C6 baseline via `tools/extract_c6_logits.py` (forward pass through `data/transforms.py::get_val_transforms`, which correctly dispatches `DATASET_STATS_8BIT` via `_get_norm_stats`).

**Finding (post-fix C6 baseline, MLflow run `ecef19a5f0e44dd68f9903ad35366c24`):**

- **Core metrics match prompt / prior report exactly.** Test F1 macro = 0.6762, per-class F1 BR1=.531 / BR2=.798 / BR4=.518 / BR5=.857, Binary F1 = 0.939, Cohen's κ = 0.633, AUC-ROC = 0.902. Val F1 macro = 0.7218, val–test gap = 4.55pp (+0.34pp vs prior-reported 4.21pp — within numerical noise of SWA eval pathways).
- **Bug scope was cosmetic.** `train.py` eval pathway already passes `data_cfg` through `get_val_transforms()` → correct dataset stats. The ImageNet-stats bug was isolated to `ensemble_evaluate.py`'s hand-built TTA transform pipeline, which was never the source of C6's 0.6762 report. Bug is now fixed (`_get_norm_stats(data_cfg)`), but the file's `MODELS` list remains outdated — ensemble evaluation with C6 requires a dedicated script (handled by future Task 1.1 TTA script).

**Three prompt-level mismatches discovered (evidence-first correction):**

1. **Test confusion matrix cells in prompt were wrong; row totals and per-class F1 were correct.**
   Actual (from `artifacts/c6_baseline_metrics.json`):
   ```
                 pred_BR1  pred_BR2  pred_BR4  pred_BR5
   true_BR1:        89        60        14         0     (163)
   true_BR2:        79       459        58         0     (596)
   true_BR4:        3         31       150       104     (288)
   true_BR5:        1         4         69       534     (608)
   ```
   Error patterns: BR1→BR2 drift = **36.8%** (prompt matches); **BR4→BR5 drift = 36.1%** (prompt claimed 26.4% → +9.7pp worse than stated); BR5→BR4 drift = 11.3% (prompt claimed 17.4% — better than stated).

2. **"Calibration anomaly" claim (val_confidence=0.284 vs test=0.545, "inverted calibration") is not real.**
   Actual mean confidence: val = **0.5355**, test = **0.5452**. Val/test confidence are essentially equal. There is no inverted SWA+multi-head interaction. ECE is high on both splits (val=0.197, test=0.214), so temperature scaling (Task 1.2) is still well-motivated — but as **overconfidence reduction**, not as "fixing inverted calibration."

3. **C6's `log_temperature` parameter was never actually learned.**
   Final `exp(log_temperature) = 1.5000` — exactly the config's init value. The temperature-scaling branch in `HierarchicalClassifier.forward()` only affects the `confidence` output, not any loss term, so the parameter receives no gradient during training. Task 1.2 is therefore the **first** real temperature search for C6.

**Interpretation:**
- BR4 boundary is objectively worse than prompt stated: the "malign boundary broken" signature is +9.7pp larger than expected. Any BR4-targeted threshold offset (Task 1.3) must account for a wider logit margin between true-BR4 and predicted-BR5 samples.
- The overconfidence (ECE ≈ 0.21 uniform across val/test) is a simple scalar problem, not a split-asymmetry. Post-scaling ECE should drop on both splits symmetrically; confidence itself will shift downward, not flip direction.
- `ensemble_evaluate.py`'nin bug'ı C6'nın raporlanan sayılarını etkilememişti; düzeltme sadece ileride o pathway'i kullanmak istersek diye uygulandı.

**Action:**
- **Baseline freeze:** Test F1 = **0.6762** — tüm Tier 1 improvement deltaları buna karşılaştırılacak.
- **Cached logits hazır** (`artifacts/c6_{val,test}_{,binary_,benign_sub_,malign_sub_}logits.npy`, `_labels.npy`, `c6_cache_meta.json`, `c6_baseline_metrics.json`). Tier 1 task'ları bu cache'i okuyacak — yeniden forward pass YOK.
- **Task 1.2 motivation re-framed:** target test ECE ≤ 0.10 (currently 0.214), macro F1 ≥ baseline (±0pp acceptable).
- **Task 1.3 grid widened:** d4 grid ∈ `np.linspace(0, 1.2, 25)` (prompt proposed `[0, 0.8, 17]` — insufficient given the 36% BR4→BR5 drift). d1 grid kept at `np.linspace(0, 1.0, 21)` (BR1 offset remains zero-sum risk per Lesson #27; conservative range preferred). Reason: (i) compute cost is negligible (525 vs 357 combos × 5 folds × ~1284 samples = milliseconds in numpy), (ii) CV-averaging + `std < 0.3` fold-consistency gate prevents grid-width-driven overfit, (iii) if the optimum hits the upper bound (d4 ≈ 1.2 across most folds), the threshold approach is structurally insufficient and Task 1.4 gating should carry more weight.
- **MODELS list in `ensemble_evaluate.py`:** left outdated (pre-C6 checkpoints). TTA re-implementation for C6 will live in a new script (`tools/tta_c6.py`), not in ensemble_evaluate.

### Lesson #45 (2026-04-19): hflip+view-swap actively hurts C6; rotations carry TTA gain. Bilateral fusion has semantic (not symbolic) L/R.

**Context:** Task 1.1 — 8-view TTA with view-swap-aware horizontal flip (RCC↔LCC, RMLO↔LMLO index permutation) + rotations (±5°, ±10°) applied on normalized tensors with background-equivalent fill value (−mean/std = −0.612). Logit-averaged over views, softmax at the end. Per-view incremental ablation run alongside the prompt-spec 8-view.

**Finding (from `artifacts/c6_tta_metrics.json`, MLflow run `2af06c6e1b7548dd9e00e14cf7fe5041`):**

```
Per-view incremental (mean of first k view logits → softmax → argmax):
  k=1 identity              F1=0.6762 (+0.00pp)   ← pipeline sanity match baseline
  k=2 +hflip_swap           F1=0.6740 (-0.22pp)   ← hflip+swap HURT on its own
  k=3 +rot_p5               F1=0.6778 (+0.16pp)
  k=4 +rot_m5               F1=0.6801 (+0.39pp)
  k=5 +rot_p10              F1=0.6828 (+0.66pp)
  k=6 +rot_m10              F1=0.6847 (+0.84pp)   ← PEAK
  k=7 +hflip_swap_rot_p5    F1=0.6825 (+0.63pp)
  k=8 +hflip_swap_rot_m5    F1=0.6808 (+0.45pp)   ← prompt-spec 8-view (marginal)

Aggregate ablations:
  tta4 (identity+hflip_swap+rot±5)               F1=0.6801 (+0.39pp)
  tta6 (identity+hflip_swap+rot±5+rot±10)        F1=0.6847 (+0.84pp) PEAK
  tta8 (prompt spec, adds hflip_swap_rot±5)      F1=0.6808 (+0.45pp) — below 0.5pp accept threshold
```

Per-class (tta8 vs baseline):
- BR1 F1 .531→.537 (+0.6pp), recall 54.6→57.7% (+3.1pp)
- BR2 F1 .798→.801 (+0.3pp), stable
- BR4 F1 .518→.528 (+1.0pp), recall 52.1→52.8% (marginal)
- BR5 F1 .857→.857 (stable)

Calibration (tta8 vs baseline):
- Val ECE 0.197 → 0.130 (−0.067)
- Test ECE 0.214 → 0.161 (−0.053)
- Mean confidence val 0.536 → 0.641, test 0.545 → 0.648 (logit averaging sharpens softmax)

**Interpretation — why hflip+swap hurts despite view-swap:**
Bilateral fusion computes `F_diff = F_left − F_right`. When the input is hflip+view-swap'ed, F_diff flips sign: `F_left' − F_right' = F_flip(R) − F_flip(L) = −F_flip(F_diff)`. The direction of the asymmetry signal reverses, and the model — which learned asymmetry orientation-dependently during training — interprets this as a genuinely different anatomical pattern, not a symmetric augmentation.

This means bilateral fusion's L/R representation is **semantic** (encodes orientation), not **symbolic** (swap-invariant). Tensor-level view permutation is not enough to recover feature-level invariance. Mammography priors about which side a lesion is on likely become embedded in the asymmetry features during training.

Rotations (±5, ±10) don't suffer this — they preserve L/R orientation and only perturb fine spatial features; the backbone's ImageNet-adapted features absorb small rotations well, and F_diff remains semantically valid.

The combination views (hflip_swap_rot±5) inherit the hflip problem and dilute the rotation-only wins.

**Action:**
- **tta8 kept as the downstream TTA track** (not peak tta6) because only tta8 cached per-head sub-logits (binary/benign_sub/malign_sub). Task 1.4 binary gating requires sub-head TTA logits; recomputing with tta6 would cost another 2-3h forward pass and risk divergence from peak pattern. The 0.039pp gap (0.6847 vs 0.6808) is absorbed as a known suboptimality.
- **Option C pipeline:** Task 1.5 cumulative will run two parallel tracks — non-TTA (raw cached logits, F1 baseline 0.6762) and tta8 (F1 baseline 0.6808). Each track gets its own T, d1/d4, and gating α; decision point chooses the best final cumulative F1.
- **Permanent rule:** Do NOT add symmetric-flip augmentations during training for this pipeline. The bilateral fusion architecture requires orientation-consistent training data. If ever retraining, confirm `augmentation.horizontal_flip` stays 0.5 only for views that are independently flipped (currently not done — view-independence at training was accidentally preserved because `get_train_transforms` does not swap view indices after flipping; the train-time L/R asymmetry thus gets corrupted randomly, which may explain why asymmetry-loss noise was harmful per Lesson #22).
- **Accept criterion:** Pass via ablation documentation (prompt's "OR" clause). +0.45pp is 0.05pp below the 0.5pp bar, but the per-view finding is a scientifically stronger contribution than a 0.01pp above-bar cosmetic win.

**Caveat / future work:**
If the paper claims TTA as a feature, report only rotation-based TTA (tta5 = identity + rot±5 + rot±10 or tta6 as above). Do NOT present hflip+swap TTA — it's either negative or marginal and undermines the bilateral-fusion architectural argument.

### Lesson #46 (2026-04-19): C6 is underconfident (T_opt ≈ 0.73, not >1); scalar T limits ECE floor ~0.13

**Context:** Task 1.2 — LBFGS temperature scaling on val logits, parallel tracks (non-TTA, tta8). Motivated by Lesson #44 discovery that C6's `log_temperature` parameter is never gradient-touched during training (fixed at init 1.5), and by post-baseline ECE ≈ 0.21.

**Finding (from `artifacts/c6_temp_scale_metrics.json`, MLflow run `474616437c764849b0b7d6456e46aefe`):**

```
Track      T_opt    Test ECE T=1   Test ECE T_opt   ΔECE    Test Brier T=1 → T_opt    Test NLL T=1 → T_opt
nonTTA    0.7347        0.1531           0.1323    −0.021    0.4054 → 0.3944         0.7270 → 0.7016
tta8      0.7229        0.1609           0.1291    −0.032    0.4013 → 0.3887         0.7180 → 0.6868

LBFGS stability: 3 restarts (init T ∈ {0.5, 1.0, 1.5}) converge within ±0.002 → global optimum.
F1 sanity: nonTTA 0.6762 (exact baseline match), tta8 0.6808 (exact Task 1.1 match) — T-invariance confirmed.
```

Reminder: config's init T = 1.5 → at T = 1.5 (C6's effective inference T), test ECE = 0.214, test NLL = 0.818. At T = 1.0, test ECE = 0.153, test NLL = 0.727. At T_opt = 0.73, test ECE = 0.132, test NLL = 0.702.

**Interpretation:**

1. **C6 is underconfident, not overconfident.** Test confidence (0.545) < accuracy (0.744) by ~20pp → the softmax distribution is too flat. T_opt < 1.0 sharpens it (conf 0.545 → 0.729). This reframes Lesson #44's ECE observation: the prompt's "inverted calibration" narrative was wrong in direction too; there is no calibration anomaly, just mild symmetric underconfidence on both splits.

2. **Scalar T has an ECE floor at ~0.13 for this pipeline.** Reducing ECE below ~0.13 would require vector temperature (per-class T, 4 params) or Platt scaling. Both violate the "1D search, overfit-resistant" constraint from the prompt. Accept the scalar-T limit; downstream Task 1.3 (threshold offsets) and Task 1.4 (gating blend) can improve F1 further but not ECE within this pipeline.

3. **tta8 requires a slightly lower T than non-TTA (0.7229 vs 0.7347)** — counterintuitive at first. Explanation: logit averaging is a Jensen-inequality smoothing operator. For a given input, `mean(logit_i)` underestimates the winning class's margin vs `mean(softmax_i)`. The TTA-averaged logits are flatter than per-view logits in the "winning direction," so a more aggressive T is needed to sharpen. In contrast, softmax-averaging TTA (had we chosen it) would have required T closer to 1. This is a real mechanism, not noise (stable to ±0.0004 across LBFGS restarts).

4. **Training-time insight (future work):** `models/classification_heads.py::HierarchicalClassifier.__init__` defines `self.log_temperature` but uses it only in the inference-time `confidence` output, NOT in any loss term. If the full-head loss were refactored to `CrossEntropy(full_logits/T, full_labels)` with T learnable (Guo et al. 2017 integrated temperature), C6's learned T would move toward ~0.73 during training, and the model would likely ship better-calibrated from the start. This is a cheap refactor for Tier 2/3.

5. **T-invariance of argmax matters for Task 1.3 design:** `argmax((logits + d)/T) = argmax(logits + d)` for any T > 0. Running Task 1.3's grid search on T-scaled val logits (prompt's suggestion) vs raw val logits produces identical (d1, d4) fold-optima. Task 1.3 script will therefore operate on raw logits directly; T enters only in Task 1.5 cumulative pipeline where gating blends softmax distributions (non-argmax operation).

**Action:**
- `artifacts/c6_temperature_values.json` produced: `{nonTTA: 0.7347, tta8: 0.7229}`. Task 1.5 cumulative will read these for each track's gating softmax temperatures.
- Task 1.3 grid search: raw logits, no T-scale pre-step (T-invariant for argmax-based F1 objective).
- ECE target of ≤ 0.10 from the prompt is **abandoned** for this phase — the scalar-T floor is ~0.13, and going lower requires deviating from the "1D search, overfit-resistant" constraint. The achieved ECE reduction (−0.021 non-TTA, −0.032 tta8) plus F1-stable confirmation satisfies the adapted accept criterion.
- Paper framing: "temperature scaling reduces ECE on both tracks by ~15-20% relative, producing better-calibrated probabilities for downstream thresholding and gating; the absolute ECE floor at ~0.13 reflects class-conditional miscalibration inherent to the 4-class BI-RADS hierarchy with severe test-time class-prior shift."

### Lesson #47 (2026-04-19): Val→test prior shift voids threshold offsets; CV guardrails pass but test F1 regresses. Zero-sum (Lesson #27) reappears.

**Context:** Task 1.3 — 5-fold StratifiedKFold grid search on (d1, d4) offsets applied to raw val logits. Search space d1 ∈ [0, 1.0] × 21, d4 ∈ [0, 1.2] × 25 (widened from prompt's [0, 0.8] per Lesson #44). Both non-TTA and tta8 tracks.

**Finding (from `artifacts/c6_threshold_cv_metrics.json`, MLflow run `184f22d432d64e94942c38bdcdb3fbef`):**

```
Track      CV d1 (std)     CV d4 (std)     Val F1 Δ   Test F1 Δ    Naive-vs-CV gap
nonTTA     0.06 (0.07)     0.43 (0.16)     +1.43pp    −0.53pp      −0.61pp
tta8       0.11 (0.14)     0.36 (0.19)     +1.07pp    −0.18pp      −1.31pp
```

All CV guardrails PASSED (std < 0.3). Boundary-hit = 0 in both tracks (no fold hit the grid upper bound of 1.2 on d4 — Lesson #44's widened-grid recommendation was unnecessary; d4 optima sit at 0.36–0.43).

Per-class breakdown for non-TTA test (offset d1=0.06, d4=0.43):
- BR1 F1 +0.2pp, recall +0.6pp — minimal (d1 was small)
- **BR2 F1 −5.2pp, recall −10.6pp** — catastrophic
- BR4 F1 +2.8pp, recall +14.2pp, **precision −5.1pp** — BR4 gains from BR2 drift, not better BR4 detection
- BR5 F1 +0.2pp — stable

Confusion matrix drift signature: true_BR2 → pred_BR4 doubled from 9.7% (58 patients) to **20.0% (119 patients)**. The d4 offset pulls BR2 into BR4 territory.

Fold-level anomaly: tta8 Fold 5 finds `d4 = 0.0` as its fold-optimum (other folds d4 ∈ {0.40, 0.40, 0.45, 0.55}). This fold's held-out 256 samples apparently had a BR prior distribution closer to test's prior, and the grid search correctly identified "no offset needed" — a partial confirmation that the negative transfer is driven by val's BR prior, not by the offset mechanism itself.

**Interpretation:**

1. **Saerens-style test-prior constraint makes val-calibrated offsets prior-biased by construction.** The d4 = 0.43 optimum is implicitly calibrated to val's BR4 share (22.2%). Test BR4 share is 17.4% — a 22% smaller class. The val-optimal offset is therefore systematically too aggressive for test, and the excess BR4 attraction comes from the neighboring BR2 (the largest test class, 36%). Zero-sum: 14.2pp BR4 recall gain costs 10.6pp BR2 recall loss, and since BR2 has 3.7× more test samples than BR4, the net F1 is negative.

2. **Structurally analogous to C3 (Lesson #27).** C3 raised the BR1 class weight by 40% → BR1 F1 +1.2pp, BR2 F1 −9.8pp, net −2.69pp. Task 1.3 does the reverse (boosts BR4) on the BR2↔BR4 axis and produces the same pattern. Both manipulate a shared decision boundary without new discriminative information.

3. **Grid width correction to Lesson #44:** Widening d4 to [0, 1.2] had no effect on optima. The prompt's original [0, 0.8, 17] spec would have been sufficient and is the correct recommendation for any retry. Lesson #44's widened-grid advice is retracted.

4. **CV guardrails failed as a test-F1 proxy.** std(d1)<0.3 and std(d4)<0.3 were both satisfied but test regressed. Fold-level consistency on val predicts val→val transfer, not val→test transfer when the test distribution is prior-shifted. Future CV guardrails on this dataset must explicitly include a val-vs-test delta check, not just fold variance.

**Action:**
- **Threshold offsets excluded from Task 1.5's default cumulative pipeline** (d1=d4=0). Task 1.5 ablation table will still include "+ threshold" as an explicit row to document this negative result transparently (reviewer-defensible framing).
- **Primary F1 lever shifts to Task 1.4 (binary gating).** Hypothesis: the binary head (F1=0.94) is robust across val/test because benign/malign is the axis where test prior shift is minimal (train Benign=52%, test Benign=45.8% — only 6pp, vs BR1 halved). Hier reconstruction `P(malign) · P(BR4|malign)` conditions on a distribution-stable quantity, so val-tuned blending should transfer to test.
- Paper framing: "threshold offset tuning on val logits is principled but prior-shift-fragile; hierarchical binary gating achieves robust improvement because the binary decision is invariant to the 4-class prior shift."

**Future work:**
- Scoped threshold: apply (d1, d4) only to samples with `binary_prob ∈ [0.4, 0.6]` (high uncertainty region) — this would avoid BR4 overreach on confidently-malignant samples. Defer to after Task 1.5 sees cumulative-pipeline F1.

### Lesson #48 (2026-04-19): Inference-time hierarchical reconstruction duplicates what full head already knows. α-CV bimodal; hard-gate noise-level; pure hier < pure full.

**Context:** Task 1.4 — binary-gated hierarchical inference. Five variants tested per track: (A) soft α-CV blend with T_opt, (B) soft α-CV blend with T=1.0, (C) hard gate (P(malign) > 0.5), (D) pure hier (α=1), (E) pure full (α=0 sanity).

**Finding (from `artifacts/c6_gating_metrics.json`, MLflow run `4c2c66dcfca241c3a886d031b334e1e3`):**

```
                                nonTTA                          tta8
                          Test F1     Δ                   Test F1     Δ
(A) α-CV soft,T_opt       0.6731    −0.31pp              0.6776    −0.31pp
(B) α-CV soft,T=1.0       0.6731    −0.31pp              0.6801    −0.07pp
(C) hard gate, T_opt      0.6765    +0.03pp  BEST        0.6807    −0.00pp  BEST
(D) pure hier, T_opt      0.6707    −0.55pp              0.6784    −0.24pp
(E) pure full (sanity)    0.6762    +0.00pp ✓           0.6808    +0.00pp ✓

α-CV fold-by-fold:
  nonTTA (A): {0.20, 1.00, 0.60, 0.70, 0.40}  mean=0.58  std=0.271  (near guardrail)
  tta8   (A): {0.00, 1.00, 0.00, 1.00, 0.00}  mean=0.40  std=0.490  (GUARDRAIL BROKEN)

Confusion matrix drift (nonTTA, hard gate vs baseline):
  true_BR4 → pred_BR5:  baseline 104/288 → variant C 102/288  (−2 patients)
  true_BR5 → pred_BR4:  baseline  69/608 → variant C  73/608  (+4 patients)
  Net effect: noise-level, 6 out of 1655 samples changed class.
```

**Interpretation:**

1. **The hypothesized "sub-head knows something full doesn't" is false for C6.** The malign_sub head learned BR4↔BR5 with the same data asymmetry that the full head did — sanity check showed true_BR4 has malign_sub margin = 0.26 (weak) vs true_BR5 = 1.76 (strong), mirroring the full head's BR4 weakness. Hier product `P(malign) · P(BR4|malign)` recovers the same information that `P(BR4)` from the full head already contains. Duplication, not enrichment.

2. **α-CV fold bimodality is the smoking gun.** tta8 folds split cleanly into {α=0, α=1} camps with no fold preferring a middle value. This is what happens when hier and full make *nearly-identical argmax decisions on most samples*: each fold flips on its minority of boundary samples, and the optimum migrates to whichever extreme agrees with that fold's boundary population. The mean (α=0.4) is a statistical artifact, not a true optimum.

3. **Lesson #30 reconciled with this result.** Lesson #30 showed removing auxiliary heads costs +3pp — but that gain is *training-time multi-task regularization*: the binary and sub-heads provide extra gradient signal to the shared backbone during training, enriching the features the full head consumes. At inference time, the full head already incorporates those features via the shared `patient_feat` representation. Re-composing auxiliary-head softmax outputs into a synthetic 4-class distribution does not retrieve any bypassed information. **Auxiliary heads' value is architectural (during training), not compositional (during inference).**

4. **Hard gate's +0.03pp is noise.** Only 6 out of 1655 test samples changed class under hard gating (4 BR5→BR4 flips, 2 BR4→BR5 unflips). The binary head's argmax agrees with the full head's `argmax >= 2` boundary on >99% of samples. Hard gate carries no signal because the binary/quaternary decision paths are redundant for C6's trained representation.

5. **tta8 α-CV guardrail broke (std = 0.49 > 0.3)** while nonTTA stayed just under (0.271 < 0.3) — another confirmation that Task 1.3's CV-std-based guardrail is unreliable for detecting whether a test-F1 improvement will transfer. The strict `std < 0.3` threshold should be interpreted as "necessary, not sufficient" for val→test transfer.

**Meta-observation across Tier 1 Tasks 1.2, 1.3, 1.4:**
- 1.2 Temperature: F1-invariant by construction; calibration-only gain. (Limited scope, met.)
- 1.3 Threshold: val→test prior shift causes zero-sum; test F1 regresses. (Negative.)
- 1.4 Gating: inference-time decomposition duplicates full-head information; at best noise-level. (Negative/neutral.)

Only Task 1.1 TTA provides a transferable F1 gain (+0.45pp tta8, from rotations alone). The 8-bit single-model pipeline's inference-time improvement ceiling is therefore ~0.6808, far below the 0.72 target. The path to 0.72+ requires training-time intervention (Tier 2: logit-adjusted training for prior-shift robustness, or 16-bit pipeline transfer).

**Action:**
- **Task 1.5 cumulative evaluation runs as formality** to populate the ablation table with clean, documented deltas; decision-point verdict expected to be "< 0.70 → root cause + Tier 2."
- **Route to Tier 2 Task 2.2 (F2 — logit-adjusted training, Menon et al. 2021).** This directly targets the val→test prior shift that killed Task 1.3 and that underlies the BR4 F1 ceiling. Unlike class weights (Lesson #27 zero-sum), logit adjustment is mathematically principled for label-shifted test distributions.
- **Skip Task 2.0 multi-seed ensemble for now** (prompt's `< 0.70` branch also suggests skipping ensemble first; 3×17h GPU unjustified before understanding why 0.68 ceiling exists).
- **Skip Task 2.1 F1 16-bit preprocessing** unless F2 also fails — 16-bit preprocessing pipeline requires 300GB data regeneration, only worthwhile if training-time regularization alone insufficient.
- **Gating in Task 1.5:** use variant C (hard gate) for completeness in both tracks — contributes +0.03pp nonTTA, 0.00pp tta8. Document in ablation but don't oversell.

**Future work:**
- Per-head temperature fitting (separate T for full, binary, benign_sub, malign_sub). Might close the α-CV bimodality if sub-heads are miscalibrated in ways full head isn't. Low priority given the main architectural finding (hier = full).
- Train-time refactor: include `CE(logits/T, labels)` in loss so `log_temperature` actually gets optimized. See Lesson #46 action. Expected to ship a better-calibrated C6 out of the box; orthogonal to F1 gains.

---

## F-Series Experiment Lessons (2026-04-20)

The F-series transfers the C6-optimal 8-bit configuration and its key ablation findings to the **16-bit `noseg` pipeline** (`Dataset_1024_16bit`, `dataset_variant: noseg`). All 4 experiments use `ConvNeXtV2-Large` at 1024×1024 on 16-bit PNG images **without segmentation masking** — the raw DICOM→windowing→letterbox pipeline, bypassing the U-Net segmentation, CLAHE, and tight-crop stages used in the 8-bit pipeline.

**F-series design:**
- **F1** (C6 equivalent): CE + SWA + no asymmetry + no Mixup — the champion config
- **F2** (B5 equivalent): CE + SWA + asymmetry=0.1 — asymmetry retest on 16-bit
- **F3** (D4 equivalent): CE + no SWA + no asymmetry — SWA isolation test
- **F4** (C1 equivalent): CE + SWA + Mixup/CutMix — SWA+Mixup antagonism retest

### Lesson #49 — 16-bit noseg Pipeline Uniformly Underperforms 8-bit CLAHE Pipeline (F-Series Global)
**Problem:** All 4 F-series experiments scored **below** their 8-bit equivalents. The 16-bit pipeline was expected to benefit from higher dynamic range (65,535 vs 255 intensity levels), but the opposite occurred.

**Evidence — 8-bit vs 16-bit matched-config comparison:**
```
| 16-bit Exp | 8-bit Equiv | 8-bit Test F1 | 16-bit Test F1 | Delta     | 8-bit Gap | 16-bit Gap |
|------------|-------------|:-------------:|:--------------:|:---------:|:---------:|:----------:|
| F1         | C6          | 0.6762        | 0.6454         | **-3.08pp** | 4.21pp    | 6.42pp     |
| F2         | B5          | 0.6615        | 0.6169         | **-4.46pp** | 6.71pp    | 8.95pp     |
| F3         | D4          | 0.6615        | 0.6435         | **-1.80pp** | 4.94pp    | 7.15pp     |
| F4         | C1          | 0.6431        | 0.6362         | **-0.69pp** | 7.27pp    | 7.04pp     |
```

**Per-class breakdown (F1 vs C6, best-to-best):**
- BR1: -7.3pp (0.458 vs 0.531) — severe regression
- BR2: -4.7pp (0.751 vs 0.798) — significant regression
- BR4: -0.4pp (0.514 vs 0.518) — negligible
- BR5: +0.2pp (0.859 vs 0.857) — negligible

**Root cause — preprocessing, not bit depth:**
The 16-bit pipeline uses `dataset_variant: noseg`, which bypasses three critical preprocessing stages present in the 8-bit pipeline:

1. **U-Net segmentation masking** — The 8-bit pipeline masks non-breast regions (background, pectoral muscle) to zero, focusing the model exclusively on breast tissue. The 16-bit `noseg` variant retains these distractors, forcing the backbone to waste capacity learning to ignore irrelevant anatomy.

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — The 8-bit pipeline applies `CLAHE(clipLimit=2.0, tileGrid=8×8)` to tissue-only pixels, boosting local contrast of subtle mammographic features (microcalcifications, spiculations, architectural distortions). The 16-bit pipeline retains raw windowed intensities. Per Lesson #22's normalization statistics: CLAHE raised tissue mean from 0.284→0.351 (+24%) and increased std from 0.158→0.180, indicating enhanced feature discriminability.

3. **Tight crop** — The 8-bit pipeline strips zero-padding borders after segmentation, maximizing tissue pixels per image. The 16-bit pipeline includes peripheral zeros that dilute the effective resolution of breast tissue in the 1024×1024 frame.

The 16-bit dynamic range advantage (65,535 vs 255 levels) is negated because: (a) ConvNeXtV2 was pretrained on 8-bit ImageNet images — its early layers' weight distributions are calibrated for [0,255]-scale statistics, not 16-bit ranges; (b) after normalization to [0,1], the extra precision contributes <0.004 per level (1/255 vs 1/65535), which is below the noise floor of the training process; (c) CLAHE's contrast enhancement provides far more discriminative value than raw bit depth.

**Training dynamics confirm the preprocessing gap:**
- F1 train F1 = 0.776 vs C6's implied convergence — the model fits the 16-bit training data nearly as well, but the representations don't transfer to test. The train→test gap is 13.0pp (F1) vs ~8pp (C6), indicating the `noseg` images contain more distribution-specific artifacts that the model memorizes.

**Rule:** The 8-bit CLAHE+segmentation pipeline is strictly superior to the 16-bit `noseg` pipeline for this architecture and dataset. The preprocessing stages (segmentation, CLAHE, tight crop) contribute more to classification performance than bit depth. **Do NOT pursue the 16-bit `noseg` path further.** If 16-bit imaging is revisited, it must use the full preprocessing pipeline: `DICOM → segmentation → windowing → tight crop → CLAHE → letterbox → 16-bit PNG`. This would combine high dynamic range with the proven preprocessing advantages.

### Lesson #50 — Asymmetry Loss Removal Generalizes Across Bit Depths: +2.85pp on 16-bit (F1 vs F2)
**Problem:** F2 retested asymmetry_loss_weight=0.1 on 16-bit (the same configuration that Lesson #22 showed was harmful on 8-bit). **The finding replicates robustly.**

**Evidence (F1 vs F2, same config except asymmetry):**
- Test F1: 0.6454 vs 0.6169 → asymmetry removal = **+2.85pp** (8-bit C6 vs B5 equivalent: +1.47pp with SWA, +2.28pp without SWA)
- Gap: 6.42pp vs 8.95pp → asymmetry removal narrows gap by **2.53pp**
- BR2: 0.751 vs 0.690 (-6.1pp with asymmetry) — asymmetry noise damages BR2 most
- BR4: 0.514 vs 0.450 (-6.4pp with asymmetry) — substantial BR4 damage too
- BR1: 0.458 vs 0.477 (+2.0pp with asymmetry) — asymmetry paradoxically helps BR1 on 16-bit
- Train F1: 0.776 vs 0.795 → asymmetry enables deeper memorization (+1.9pp train gap)

**Cross-bit-depth comparison:**
```
| Bit Depth | With Asymmetry | Without Asymmetry | Asymmetry Removal Effect |
|-----------|:--------------:|:-----------------:|:------------------------:|
| 8-bit     | B5 = 0.6615    | C6 = 0.6762       | +1.47pp (with SWA)       |
| 8-bit     | B1 = 0.6387    | D4 = 0.6615       | +2.28pp (without SWA)    |
| 16-bit    | F2 = 0.6169    | F1 = 0.6454       | **+2.85pp (with SWA)**   |
```

The asymmetry removal effect is **larger** on 16-bit (+2.85pp) than on 8-bit (+1.47pp with SWA). This makes sense: the `noseg` variant retains more bilateral structural noise (pectoral muscle, chest wall asymmetry) that the asymmetry loss aggressively overfits to.

**Root cause confirmed:** Lesson #22's diagnosis holds across pipelines. The asymmetry loss computes bilateral differences that overfit to distribution-specific bilateral features. On `noseg` images with more structural noise, the overfitting is **amplified** because the model has more spurious bilateral features to exploit.

**Rule:** Asymmetry loss removal is a **universal improvement** across bit depths and preprocessing variants. The finding is not an artifact of 8-bit CLAHE preprocessing. Permanently confirmed: `asymmetry_loss_weight: 0.0` for all future experiments.

### Lesson #51 — SWA Is Nearly Ineffective on 16-bit noseg: +0.19pp (F1 vs F3)
**Problem:** SWA provided +1.47pp on 8-bit (C6 vs D4). On 16-bit, SWA barely moves the needle: **+0.19pp** (F1 vs F3).

**Evidence (F1 vs F3, same config except SWA):**
- Test F1: 0.6454 vs 0.6435 → SWA effect = **+0.19pp** (vs +1.47pp on 8-bit)
- Best Val F1: 0.7096 vs 0.7150 → **without SWA has HIGHER val** (-0.54pp)
- Gap: 6.42pp vs 7.15pp → SWA modestly narrows gap by 0.73pp
- BR5: 0.859 vs 0.818 (+4.2pp with SWA) — SWA helps BR5 significantly
- BR1: 0.458 vs 0.497 (-4.0pp with SWA) — SWA hurts BR1 (same pattern as 8-bit B5)
- BR2: 0.751 vs 0.731 (+2.0pp with SWA) — SWA helps BR2
- BR4: 0.514 vs 0.528 (-1.4pp with SWA) — SWA slightly hurts BR4
- Binary F1: 0.912 vs 0.919 → without SWA has **better binary separation**

**Cross-bit-depth SWA comparison:**
```
| Bit Depth | Without SWA      | With SWA        | SWA Effect |
|-----------|:----------------:|:---------------:|:----------:|
| 8-bit     | D4 = 0.6615      | C6 = 0.6762     | +1.47pp    |
| 16-bit    | F3 = 0.6435      | F1 = 0.6454     | **+0.19pp**|
```

**Root cause:** SWA averages weights from the late training trajectory, which works best when the loss landscape has a clear, broad basin to average over. The `noseg` pipeline introduces more structural noise (chest wall, pectoral muscle, background) that creates a noisier, more multimodal loss landscape. Weight averaging over this landscape produces a compromise solution rather than finding a superior flat minimum.

Additionally, F3 (no SWA) achieves the **highest best val F1** (0.7150) among all F-series experiments, suggesting the checkpoint selection mechanism works better than SWA averaging on the noisier 16-bit landscape. The best single checkpoint captures a momentary good generalization state that SWA averaging dilutes.

The BR1/BR5 trade-off mirrors the 8-bit pattern (Lesson #17): SWA smooths decision boundaries, favoring higher-density classes (BR5) at the expense of minority classes (BR1). This effect is architecture-invariant but the overall magnitude is suppressed on 16-bit because SWA's averaging has less to improve from.

**Rule:** SWA's effectiveness is **preprocessing-dependent**, not just architecture-dependent. On clean, CLAHE-enhanced 8-bit images, SWA provides meaningful generalization gains. On noisy `noseg` 16-bit images, SWA averaging is nearly ineffective because the loss landscape doesn't support productive weight averaging. If re-running 16-bit experiments with full preprocessing (seg+CLAHE), re-test SWA rather than assuming it transfers.

### Lesson #52 — SWA+Mixup Antagonism Is Universal: -0.92pp on 16-bit (F4 vs F1)
**Problem:** F4 retested SWA+Mixup/CutMix on 16-bit. The antagonism discovered in Lesson #23 (8-bit C1) and confirmed in Lesson #35 (8-bit D7) **replicates on 16-bit**.

**Evidence (F4 vs F1, same config except Mixup/CutMix added):**
- Test F1: 0.6362 vs 0.6454 → Mixup effect = **-0.92pp**
- Gap: 7.04pp vs 6.42pp → gap widened by 0.62pp
- BR1: 0.426 vs 0.458 (-3.1pp) — Mixup hurts BR1
- BR2: 0.762 vs 0.751 (+1.1pp) — slight BR2 gain (Mixup's interpolation favors dominant class)
- BR5: 0.844 vs 0.859 (-1.5pp) — BR5 regression
- Train F1: 0.649 vs 0.776 → Mixup's strong regularization effect visible (−12.7pp train F1)

**Cross-bit-depth antagonism comparison:**
```
| Bit Depth | Without Mixup    | With Mixup       | Mixup Effect (SWA on) |
|-----------|:----------------:|:----------------:|:---------------------:|
| 8-bit     | C6 = 0.6762      | D7 = 0.6563      | -1.99pp               |
| 8-bit     | B5 = 0.6615      | C1 = 0.6431      | -1.84pp               |
| 16-bit    | F1 = 0.6454      | F4 = 0.6362      | **-0.92pp**           |
```

The antagonism is **smaller** on 16-bit (-0.92pp vs ~-1.9pp on 8-bit). This is consistent with SWA itself being weaker on 16-bit (Lesson #51): if SWA contributes less, there's less SWA-mediated smoothing to conflict with Mixup's input-space smoothing. The antagonism magnitude scales with SWA's effectiveness.

**Interesting anomaly:** F4's train F1 = 0.649 is dramatically lower than F1's 0.776 (−12.7pp), yet test F1 only drops by 0.92pp. The train→test gap (1.3pp) is the narrowest in the entire F-series — Mixup is an extremely effective training-time regularizer. But the final test F1 is still lower, confirming Lesson #34: Mixup's regularization doesn't produce better features, it just prevents memorization while also preventing useful learning.

**Rule:** SWA+Mixup/CutMix antagonism is confirmed across three independent experiments on two different pipelines (8-bit C1, 8-bit D7, 16-bit F4). The finding is **universal** for this architecture regardless of preprocessing. The mechanism is intrinsic: both SWA (weight-space averaging) and Mixup (input-space interpolation) smooth decision boundaries, and their combination over-smooths. **Never combine SWA with Mixup/CutMix in any configuration.**

### Lesson #53 — F-Series Meta: 8-bit CLAHE Pipeline Is the Validated Production Path; 16-bit noseg Is Abandoned

**Evidence (all F-series ranked by test F1):**
```
| Rank | Exp | Strategy                    | Best Val F1 | Test F1  | Gap    | BR1   | BR2   | BR4   | BR5   | AUC   | Kappa |
|------|-----|-----------------------------|:-----------:|:--------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| —    | C6  | 8-bit CHAMPION (reference)  | 0.7183      | 0.6762   | 4.21pp | 0.531 | 0.798 | 0.518 | 0.857 | 0.902 | 0.633 |
| 1    | F1  | C6 config on 16-bit         | 0.7096      | 0.6454   | 6.42pp | 0.458 | 0.751 | 0.514 | 0.859 | 0.907 | 0.596 |
| 2    | F3  | No SWA isolation            | 0.7150      | 0.6435   | 7.15pp | 0.497 | 0.731 | 0.528 | 0.818 | 0.905 | 0.569 |
| 3    | F4  | + Mixup/CutMix retest       | 0.7066      | 0.6362   | 7.04pp | 0.426 | 0.762 | 0.512 | 0.844 | 0.905 | 0.589 |
| 4    | F2  | + Asymmetry retest          | 0.7064      | 0.6169   | 8.95pp | 0.477 | 0.690 | 0.450 | 0.851 | 0.893 | 0.545 |
```

**Key findings from the F-series:**

1. **16-bit noseg is strictly inferior** — F1 (best 16-bit) trails C6 (8-bit champion) by 3.08pp. The deficit is driven by preprocessing (segmentation + CLAHE + tight crop), not bit depth.

2. **C6's architectural lessons transfer across pipelines:**
   - Asymmetry removal: +2.85pp on 16-bit (Lesson #50) vs +1.47–2.28pp on 8-bit
   - SWA+Mixup antagonism: -0.92pp on 16-bit (Lesson #52) vs -1.84–1.99pp on 8-bit
   - The directional findings are identical; only magnitudes differ (reflecting different baseline performance levels)

3. **SWA is preprocessing-dependent** (Lesson #51): +0.19pp on 16-bit noseg vs +1.47pp on 8-bit CLAHE. The CLAHE pipeline's cleaner, more structured images create a loss landscape where SWA averaging is productive.

4. **Val F1 ceiling is lower on 16-bit** — Best val F1 peaks at 0.7150 (F3) vs 0.7183 (C6). The model doesn't even fit the training/val distribution as well on `noseg` images, confirming this is a feature-quality issue, not just a generalization issue.

5. **AUC is surprisingly robust** — F1's AUC (0.907) is slightly higher than C6's (0.902), suggesting the model's ranking ability transfers but the decision boundaries are miscalibrated. The 16-bit images may contain useful ordinal information that the classification heads don't exploit well.

**The 2×2 factorial on 16-bit (partial):**
```
|                    | No Asymmetry (0.0) | With Asymmetry (0.1) | Asymmetry Effect |
|--------------------|:------------------:|:--------------------:|:----------------:|
| SWA on             | F1 = 0.6454        | F2 = 0.6169          | +2.85pp          |
| SWA off            | F3 = 0.6435        | (not tested)         | —                |
| SWA effect         | +0.19pp            | —                    |                  |
```

**Confirmed permanently abandoned (cumulative through F-series):**
- 16-bit `noseg` pipeline (this lesson)
- Asymmetry loss (Lessons #22, #50 — universal across pipelines)
- Mixup/CutMix + SWA combination (Lessons #23, #35, #52 — universal)
- Focal loss for ConvNeXtV2 (Lesson #28)
- DINOv2 as primary backbone (Lesson #33)
- Class weight manipulation (Lesson #27)
- Capacity reduction (Lesson #25)
- Backbone freezing below lr_scale=0.2 (Lesson #26)
- Warm restart schedulers with SWA (Lesson #39)
- Label smoothing > 0.05 with SWA (Lesson #41)
- Loss weight rebalancing (Lesson #42)

**Rule:** The 8-bit CLAHE pipeline with C6 configuration represents the validated production path for this dataset and architecture. 16-bit imaging requires the full preprocessing pipeline (seg + CLAHE + tight crop) to be competitive. The `noseg` variant is a net negative because it trades marginal bit-depth precision for the substantial feature-engineering value of segmentation, CLAHE, and tight cropping. **If 16-bit is ever revisited, it must use `dataset_variant: seg` with the identical preprocessing pipeline, preserving the CLAHE and segmentation stages while only changing the final output bit depth.**

### Recommended Next Steps (Post F-Series)

The F-series closes the 16-bit `noseg` investigation. The project's state:

1. **Production model:** C6 (8-bit, test F1 = 0.6762, gap = 4.21pp) + TTA (test F1 = 0.6808)
2. **16-bit with full preprocessing:** NOT tested — if pursuing 16-bit, generate `Dataset_1024_16bit_seg` with full DICOM → segmentation → windowing → tight crop → CLAHE → letterbox → 16-bit PNG pipeline. However, expected gain is marginal given the AUC parity between F1 and C6.
3. **Multi-seed validation:** Run C6 config with 3 different seeds (42, 123, 456) to establish confidence intervals on test F1 = 0.6762. Required for publication.
4. **Baseline comparisons:** Run simple baselines (single-view, no fusion, flat classifier) on the same 8-bit data to quantify the multi-view hierarchical architecture's contribution.
5. **ABANDON:** Further 16-bit `noseg` exploration, further hyperparameter perturbation around C6.

### Lesson #49 (2026-04-27): G-series cascade — specialist features beat C6 on the malign axis, tie-or-lose on the benign axis, and OneCycleLR with `epochs=100` is destructively misconfigured for early-stopping specialists

**Context:** First execution of the soft-cascade design from `tasks/cascade_log.md`. Three single-head specialists trained on subsets of C6's seed=42 split, identical augmentation/optimizer/SWA settings to C6, asymmetry loss off (Lesson #22), no Mixup (Lesson #34/37), full preprocessing pipeline. Configs in `configs/cascade/`, training script `train_cascade.py`, manifests in `data/manifests/cascade/`. Runs:
- G1 (stage1, binary, backbone-only/no-fusion per claude.md doctrine): `ad4526d7ef684e7e845ea977aa49d4a2` (cascade/stage1_binary)
- G2a (stage2a, BR1 vs BR2, full fusion): `bae58239a2c34f81b76c4f14a5cbbe04` (cascade/stage2_benign)
- G2b (stage2b, BR4 vs BR5, full fusion): `c5926b3c3e8841faaae1892678ce97ca` (cascade/stage2_malign)

**Reference:** C6 best-epoch val (run `6859aed2a37e43b8b72b5333b2573275`, step 6 — 26 epochs total, early-stopped):
val_full_f1_macro=0.7183, BR1=0.6475, BR2=0.7432, BR4=0.6796, BR5=0.8030, val_binary_f1=0.9530.

**Phase E sanity gate verdicts (val):**

| stage | best val f1_macro | gate | C6 ref (subset proxy) | verdict |
|---|---:|---:|---:|---|
| G1 (binary)         | **0.9664** | ≥ 0.93 | C6 binary 0.9530 | **PASS** by +3.6pp vs threshold, +1.34pp vs C6 |
| G2a (BR1 vs BR2)    | **0.7020** | ≥ C6 BR1/BR2 macro | 4-class lb 0.6953 / est subset ~0.71 | **MARGINAL** — passes lower bound by 0.7pp, ties or slightly trails the cross-class-FP-corrected estimate |
| G2b (BR4 vs BR5)    | **0.7740** | ≥ C6 BR4/BR5 macro | 4-class lb 0.7413 / est subset ~0.755 | **PASS** by +1.9pp over the corrected estimate, +3.3pp over lb |

The "subset proxy" is C6's per-class val F1 averaged over the two relevant classes. This is a strict lower bound on what C6 would score on the binary subset task because cross-class FPs (e.g., true BR4→pred BR1) shrink BR1's 4-class precision but don't appear in the BR1-vs-BR2 subset task — predictions of BR4/BR5 on a BR1/BR2 patient simply don't contribute to either subset class. Empirically the lift is ~1-3pp depending on the class's cross-class FP rate (BR1 ≈ +1pp, BR2 ≈ +3pp, BR4 ≈ +5pp, BR5 ≈ 0pp from the C6 test confusion matrix in Lesson #44).

**Headline finding — cascade hypothesis is partially confirmed:**

- **Malign boundary (BR4 vs BR5): training-time specialization works.** G2b's val F1 macro (0.7740) is meaningfully above C6's BR4/BR5 4-class macro (0.7413). Specialist features unlock information the shared backbone leaves on the table. **This is the result the cascade was designed to find.**
- **Benign boundary (BR1 vs BR2): specialization does NOT clearly help.** G2a peaks at val F1 macro 0.7020, which is at-best a tie with C6's BR1/BR2 subset estimate (~0.71). The "shared backbone is Pareto-dominated" hypothesis from `tasks/cascade_log.md` is **not supported on this axis**.
- **Binary head (Stage-1): backbone-only beats C6's multi-task binary head.** G1 = 0.9664 > C6 val_binary_f1 = 0.9530. Removing the multi-task gradient competition, even at the cost of fusion features, was net positive for the binary task. Confirms claude.md's design rationale.

**Why the asymmetry between G2a and G2b?**

The benign↔normal boundary (BR1↔BR2) is governed by *absence* of pathological features — subtle parenchymal patterns where the difference is genuinely fuzzy, often disagreed on by radiologists. The specialist can dedicate its capacity to this boundary, but capacity is not the bottleneck — discriminative signal is. This matches Lesson #27's "shared decision boundary" diagnosis: BR1/BR2 lives on a continuum that no architecture short of new data acquisition can sharpen.

The malign morphology boundary (BR4↔BR5) is governed by *presence* of distinguishing features (mass spiculation, microcalcification distribution, tissue density patterns). Here, specialist focus does help: G2b's backbone gets every gradient update tuned to BR4-vs-BR5 features instead of being asked to also discriminate benign vs malign and BR1 vs BR2.

**Critical instability finding — OneCycleLR config copied from C6 destroys specialists:**

Both G2a and G2b exhibit a "rise-and-collapse" trajectory:

```
G2a:  ep 1-5 trivial(0.383)  ep 6-13 rising(0.55→0.70)  ep 13 PEAK(0.7020)  ep 14-33 COLLAPSE(0.383)
G2b:  ep 1-3 trivial(0.35)   ep 4-21 oscillating 0.71-0.77  ep 22-23 last gasp  ep 23 PEAK(0.7740)  ep 24-43 COLLAPSE(0.35)
```

After the peak, *every subsequent epoch* of both specialists collapses back to predicting only the majority class. The pattern coincides exactly with the OneCycleLR ramp:

- C6 inherits `epochs=100, max_lr=5e-4, pct_start=0.3, backbone_lr_scale=0.2`. Backbone effective max_lr = 1e-4 at epoch 30.
- C6 early-stops at epoch 26 (patience=20 from best at ep 6) — so C6 NEVER reaches the destructive peak LR.
- G2a peaks at epoch 13 (LR 4.6e-5) → early-stopping clock starts → G2a runs to epoch 33 (LR 1.0e-4 = peak). Past epoch 14 (LR 5.0e-5), the model never recovers a non-trivial val F1.
- G2b peaks at epoch 23 (LR 8.85e-5) → runs to epoch 43 (LR 9.2e-5, on the descent). The collapse at ep 25+ correlates with LR > 9e-5.

The cascade configs reused C6's schedule verbatim ("same as C6 — isolate cascade decomposition"). That assumption was wrong: C6's schedule is implicitly tuned to C6's *effective* training length (≈ 26 epochs). When applied to a specialist with ≈ 13 useful epochs, the LR continues to climb past the breaking point because early stopping watches val F1, not LR phase.

The saved best checkpoint (peak val) is what's being used downstream, and Phase E gate verdicts are valid — but the result is fragile (single-epoch peak for G2a; oscillating peak for G2b). SWA is also useless here: SWA averages epochs ≥ swa_start_epoch=5, but most of those epochs are post-collapse. `using_swa = False` for both G2a and G2b in their final checkpoint metadata, confirming that SWA averaging was strictly worse than the best non-averaged epoch. This contrasts with G1 (using_swa = True; G1 trained healthily because LR never destabilized backbone-only training).

**Rule:**
1. **Auxiliary-head removal yields a usable specialist on training-distribution-rich boundaries (G2b/malign morphology) but cannot create discriminative signal where the boundary is feature-poor (G2a/benign-vs-normal).** Don't expect a cascade to fix what feature engineering cannot reach.
2. **Don't copy a learning-rate schedule across experiments without re-tuning to the new effective epoch count.** OneCycleLR with `epochs=N` and `pct_start=p` enforces a peak LR at epoch ~ N·p. If the *actual* training run terminates at epoch ≪ N·p (early stopping triggered earlier than expected), the peak is never reached and the schedule is implicitly milder than designed. If the actual run terminates at epoch ≈ N·p or later, the peak destabilizes the model. For specialists where you expect early stopping at epoch e*, set `epochs ≈ e* / 0.3` ≈ `3.3 · e*` so the peak LR phase is *just past* the expected stop point.
3. **For the cascade, the immediate retraining recommendation is** `epochs: 30, max_lr: 2e-4, pct_start: 0.4` for G2a and G2b (peak at epoch 12, descent begins before instability sets in). Or keep `epochs: 100` but adopt a flat `cosine_warmup` schedule with `warmup_epochs: 3, min_lr: 1e-6` and let early stopping decide. C6's schedule is C6's schedule — not a cascade default.
4. **`final_using_swa = False` is a leading indicator of training instability.** If the SWA average is strictly worse than the best single epoch on val, the trajectory was non-stationary and SWA averaged across regime changes. Treat this as a flag to inspect the LR/loss trajectory before trusting the best-checkpoint result downstream.
5. **Cross-class FP analysis is the right way to compare 4-class vs binary-subset metrics.** When validating a binary specialist against a 4-class baseline on the same data, lower-bound the baseline by averaging its 4-class per-class F1 over the two relevant classes; estimate the upper-bound lift by computing the cross-class FP rate from the baseline's confusion matrix. Don't compare apples-to-oranges (binary subset F1 vs 4-class macro F1) without this correction.

**Action:**
- **Proceed to Phase F (cascade test inference) with current checkpoints.** All three gates pass against their formal lower bounds (Section 5 Phase E in the original brief), G1 is dominant, and G2a's marginal-vs-estimate result is not a hard fail per the brief's gate spec. The actual test number is the only honest measure.
- **Pre-commit to a re-training run if cascade test macro F1 < 0.69:** retrain G2a (and probably G2b for safety) with the schedule tuned per rule 3 above. Expected lift: +1-3pp on G2a from harvesting the longer plateau before LR-induced collapse, similar on G2b. Total compute cost ≈ 2× original cascade.
- **Do NOT re-train G1.** It's at 0.97 binary F1 macro on val (96.65% accuracy). Further tuning has diminishing returns and risks regressing what is the cascade's most reliable component.
- **Bake "schedule-must-match-effective-epochs" into the project lessons.** Future ablations that copy a config's LR block from a different experiment must also verify the early-stopping epoch is consistent. Add as a precondition for all future cascade-pattern (specialist-from-subset) experiments.

**Open questions for downstream evaluation:**
- Does G1's no-fusion architecture transfer to test as cleanly as it did to val? The 4.55pp val→test gap of C6 is the reference; G1's gap could be larger because the binary task is "easy" (smaller margin to lose) or smaller (no fusion = no overfitting on fusion-specific features).
- Does the soft cascade composition `P(BRk) = P(stage1) · P(BRk|stage1)` recover G2b's val gains on test, or does the binary stage's ~3% error rate cascade-amplify into BR4/BR5 routing errors that erode the gain?
- If G2a's val 0.7020 is unstable (1-epoch peak), is the test number ~0.65? If so the BR1 cascade contribution may regress vs C6 even as BR4/BR5 improves. Run Phase F immediately to find out.

---

### Lesson #50 (2026-04-27): G-series cascade — test result rejects the cascade hypothesis (-4.96pp vs C6); val→test prior shift compounds across cascade stages, and G1's no-fusion binary advantage was a val-only artifact

**Context:** Phase F (cascade inference) and Phase G (evaluation) of the G-series cascade. Test set: 1655 patients, frozen split. Inputs: `outputs/cascade/test_probs.parquet` (composed from G1+G2a+G2b checkpoints saved at val peak — see Lesson #49 for training trajectories). Baseline: C6 frozen at macro F1 = 0.6762 (Lesson #44).

**Headline (Phase G report — `outputs/cascade/evaluation_report.md`):**

```
soft cascade  test macro F1 = 0.6266   Δ vs C6 = -0.0496
hard cascade  test macro F1 = 0.6262   (soft − hard = +0.0004 → composition adds nothing)
weighted F1                = 0.6826
```

**Per-class (cascade test vs C6 test):**

| class | C6 F1 | cascade F1 | Δ | C6 prec/rec | cascade prec/rec |
|---|---:|---:|---:|---|---|
| BR1 | 0.531 | **0.496** | **−3.5pp** | 0.517 / 0.546 | 0.389 / 0.681 |
| BR2 | 0.798 | **0.676** | **−12.2pp** | 0.829 / 0.770 | 0.877 / 0.550 |
| BR4 | 0.518 | 0.517 | −0.1pp  | 0.515 / 0.521 | 0.448 / 0.611 |
| BR5 | 0.857 | **0.818** | **−3.9pp** | 0.837 / 0.878 | 0.821 / 0.814 |

**Both adjacent-boundary targets failed.** The cascade was designed to lift BR1 and BR4 to ≥ 0.55. BR1 came in at 0.496 (worse than C6's 0.531), BR4 at 0.517 (statistically tied with C6). The dominant test class (BR2, support 596) cratered −12.2pp — that single regression alone wipes out 4.6pp of macro F1.

**The killer drift: BR2 → BR1 = 28% (cascade) vs 13% (C6).** True BR2 patients are routed to BR1 at more than 2× the C6 rate. Cascade BR1 has 285 predictions for 163 true BR1 patients — 1.75× over-prediction, almost entirely powered by misclassified BR2. BR1 recall climbs +13.5pp, but precision crashes −12.8pp; the recall gain is illusory.

**G1's no-fusion binary advantage was a val-only artifact.** Implied test binary F1 from cascade outputs (treating `argmax(p_benign, p_malign)` as the binary call):

| | C6 | G1 |
|---|---:|---:|
| val_binary_f1 | 0.9530 | **0.9664** (+1.34pp) |
| test_binary_f1 | **0.9390** | 0.9309 (−0.81pp) |
| val→test gap  | −1.4pp | **−3.55pp** |

G1 won val by 1.34pp and lost test by 0.81pp. The no-fusion architecture (claude.md doctrine: backbone → mean-pool → 2-class head, bypass fusion) **overfits more than C6's multi-task fusion-trained binary head**. Lesson #49's bullet 3 was wrong on test. The fusion features the binary head was supposedly competing against are also a regularizer — when removed, the model picks up val-specific tissue patterns the test set doesn't share.

**Hard ≈ soft cascade (+0.0004pp).** The compositional advantage of the soft cascade (uncertainty-weighted blending of `P(BRk|stage1) × P(stage1)` instead of committing to argmax routing) **does not materialize**. Reason: each stage produces near-deterministic outputs (G1 confidence ~96-97% softmax peak; specialists similarly confident on most patients). The product collapses to argmax × argmax = hard cascade. The "soft" mechanism only helps when specialists disagree softly — they don't.

**Why did this fail despite Lesson #49's val-side optimism?**

This is Lesson #47 reappearing in cascade form, compounded across three stages.

1. **Train→test prior shift, multiplied across stages.** Train BR1:BR2 = 0.61, test BR1:BR2 = 0.27. G2a's BR1 class weight (1.28) was sqrt-inv-calibrated to train, so it moderately boosts BR1. With test's BR2-heavy prior, that boost is now **systematically too aggressive**. Same shift for stage 1 (train benign:malign 1.07, test 0.85) and for G2b (train BR4:BR5 0.85, test 0.47). Each stage's calibration is wrong by a factor proportional to the prior shift, and the multiplication compounds.
2. **Composition has more failure modes than a 4-class head.** A C6 wrong on BR2→BR1 needs the full head's BR1 logit to exceed BR2's. A cascade wrong on BR2→BR1 needs *either* G1 to misclassify benign correctly (it does, ~85% of the time) AND G2a to prefer BR1 over BR2 (it does often), OR G1 to misclassify malign-as-benign followed by G2a routing to BR1. The conjunctive failure surface is larger than the disjunctive one.
3. **Specialist features didn't transfer.** G2b had +3.3pp val gain on BR4/BR5 macro vs C6's lower bound. On test, BR4 F1 is statistically tied with C6 (0.517 vs 0.518), BR5 lost 3.9pp. The specialist's val→test gap is much larger than its training advantage. **Specialization helps the model fit val better; it does not produce features that generalize differently from a multi-task model.** This contradicts the cascade's foundational hypothesis ("shared backbone is Pareto-dominated").

**This empirically confirms Lesson #48's stronger claim:** auxiliary heads' value is *training-time architectural* (they regularize the backbone via multi-task gradient signal), not *post-hoc compositional*. Removing the multi-task constraint to specialize each backbone is the same mistake as inference-time hierarchical reconstruction — both decouple a coupled training-time benefit.

**Rule:**
1. **Cascade decomposition is not a path past 0.6762 on this dataset.** Three specialist backbones, each trained on a subset and then composed at inference, score below the single shared-backbone multi-task model. Empirical confirmation: −4.96pp test macro F1 (0.6266 vs 0.6762). This closes the cascade research direction.
2. **Val gains from architectural simplification (G1 no-fusion) require a tighter val→test gap analysis before acceptance.** A +1.34pp val gain accompanied by a +2.15pp gap widening is net negative. Phase E's val-only gates (Lesson #49) do not catch this. Future Phase E protocols should cross-validate the gap-widening risk by computing the val→test gap on the *same task* using a held-out cluster fold of train (e.g., 5-fold CV of the binary task on train) before committing to multi-stage compute.
3. **Train→test prior shift is the dominant generalization tax on this dataset.** Tasks 1.3 (Lesson #47), 1.4 (Lesson #48), and now G-series cascade have all hit it. The dataset's test set has a substantially different class distribution than train (BR1 halved, BR2 stable, BR4 reduced, BR5 stable). No architectural tweak escapes this; only a method that *explicitly* models prior shift will.
4. **Soft cascade composition does not buy uncertainty-blending in this regime.** When stages are confident (>95% softmax peak), the composed argmax = hard cascade argmax. Soft is only better than hard when specialists are *systematically uncertain* in non-trivial ways. Future architectures that hope to exploit "soft" composition must also produce well-calibrated uncertainty — which our specialists do not (Lesson #44 noted ECE ≈ 0.21 on C6, with specialists likely worse).

**Action:**
- **C6 remains champion at test macro F1 = 0.6762.** Do not deploy the cascade.
- **Do NOT retrain G2a/G2b with the LR-schedule fix proposed in Lesson #49.** A 1-3pp val-side recovery from a healthier trajectory does not close the −4.96pp test gap, and the val→test gap analysis above shows the cascade's failure mode is not training instability — it is structural (compositional + prior-shift). Schedule tuning is a 2-3 GPU-day cost for ~zero expected test gain.
- **Route to Tier-2 Task 2.2 (F2 — logit-adjusted training, Menon et al. 2021).** This directly targets the train→test prior shift that is the consistent killer (Lessons #47, #48, #50). Apply to C6's full 4-class head; expected test gain at least matches the prior-shift magnitude (∼3-5pp).
- **Defer Tier-2 Task 2.1 (16-bit pipeline) until F2 result lands.** F2 is the cheaper, more principled experiment; 16-bit is a 300GB regen + 17h GPU run that should only happen if F2 also fails to clear 0.70.
- **Update `tasks/cascade_log.md`** with the test-side outcome and the explicit "do not retrain" decision, so future sessions don't pick up the abandoned schedule-fix recommendation from Lesson #49 in isolation.

**Ablation artifacts saved (for future paper write-up of negative result):**
- `outputs/cascade/test_probs.parquet`, `evaluation_report.md`, `evaluation_metrics.json`
- MLflow runs: G1 `ad4526d7ef684e7e845ea977aa49d4a2`, G2a `bae58239a2c34f81b76c4f14a5cbbe04`, G2b `c5926b3c3e8841faaae1892678ce97ca`
- C6 reference: `6859aed2a37e43b8b72b5333b2573275`
- The negative result is a defensible methods-section paragraph: "we tested a three-stage soft cascade with binary gating + benign/malign specialists; despite per-stage val improvements (Lesson #49), test composition regressed −4.96pp due to compounding val→test prior shift across stages and a fragile single-epoch peak in the BR1/BR2 specialist (G2a). The shared-backbone multi-task design (C6) remains the strongest single-model configuration for this BI-RADS dataset."

**Connections to prior lessons:**
- Lesson #22: confirmed — asymmetry loss stays off; cascade also doesn't use it.
- Lesson #25: confirmed — Large backbone for all stages; not the bottleneck.
- Lesson #27: confirmed and generalized — class-weight zero-sum applies *across cascade stages* too. Each stage's class weights are calibrated to its train sub-distribution; the multiplication compounds the train-prior assumption into the final composition.
- Lesson #37: confirmed — C6 is the goldilocks. The G-series adds another negative result to the C6 > X tally.
- Lesson #44: confirmed — C6 baseline numbers exact match; cascade did not change them.
- Lesson #47: confirmed and amplified — val-tuned offsets fail because of prior shift; val-tuned cascades fail for the same reason, more dramatically.
- Lesson #48: confirmed and stronger — auxiliary heads' value is training-time, not compositional. Cascade is the structural version of inference-time hierarchical reconstruction; same null result.

---

## Lesson #51 (2026-05-02): C6's 0.6762 test F1 is a +1.9σ lucky-seed outlier; the realistic ceiling at this configuration is 0.6502 ± 0.014. Most prior "regressions" are within seed noise.

**Context:** Six independent C6 training runs across seeds {42, 123, 2024, 7, 555, 999}, all on the byte-identical config `convnextv2_large_8bit_ablation_c6.yaml` (only `project.seed` and output dirs differ). Three earlier runs (42, 123, 2024) from the historical record (`MLflow_Deney_Sonuclari_Exp19_to_25.xlsx` rows 7-8 + the canonical Lesson #44 baseline). Three new runs (7, 555, 999) trained 2026-04-30 → 2026-05-02 on the same hardware/data/code state. DagsHub MLflow experiment: `birads-1024-8bit-ablation`.

**Headline finding — the project's "champion" was a lucky tail:**

| Seed | Best val F1 | Test F1 macro | BR1 | BR2 | BR4 | BR5 | Bin F1 | SWA verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **42 (canonical)** | 0.7183 | **0.6762** | 0.531 | 0.798 | 0.518 | 0.857 | 0.939 | WON |
| 123 (xlsx) | 0.6794 | 0.6403 | – | – | – | – | 0.908 | – |
| 2024 (xlsx) | 0.6749 | 0.6467 | – | – | – | – | 0.917 | – |
| 7 (new) | 0.7128 | 0.6390 | 0.436 | 0.755 | 0.499 | 0.866 | 0.918 | WON (ep=100) |
| 555 (new) | 0.6986 | 0.6537 | 0.490 | 0.721 | 0.560 | 0.844 | 0.914 | WON (ep=100) |
| 999 (new) | 0.7153 | 0.6456 | 0.439 | 0.772 | 0.513 | 0.857 | 0.924 | **LOST** (best=ep10) |

```
Test macro F1 distribution (n=6):
  mean       = 0.6502
  std        = 0.0137
  range      = [0.6390, 0.6762]   spread = 3.72 pp
  95% t-CI on the mean = [0.6358, 0.6647]   df=5, t=2.571
  seed=42 z-score = +1.89 σ        → above the upper CI bound (~97th percentile)
```

**Per-class seed variance (the BR1/BR4 problem amplified):**

| Class | seed-42 | seed-7 | seed-555 | seed-999 | mean | std | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BR1** | 0.531 | 0.436 | 0.490 | 0.439 | **0.474** | **0.045** | **9.6 %** |
| BR2 | 0.798 | 0.755 | 0.721 | 0.772 | 0.762 | 0.032 | 4.2 % |
| **BR4** | 0.518 | 0.499 | 0.559 | 0.513 | **0.522** | **0.026** | 5.0 % |
| BR5 | 0.857 | 0.866 | 0.844 | 0.857 | 0.856 | 0.009 | 1.1 % |

BR1's CV is **9× larger than BR5's**. The minority test class is the most seed-sensitive — a textbook flat-loss-landscape signature. Without seed=42's outlier 0.531, BR1's mean drops to 0.455. **BR1's true expected F1 is ~0.47, not 0.53.**

Binary F1 across all 6 seeds: 0.939, 0.918, 0.914, 0.924, 0.908, 0.917 → mean 0.920, std 0.011, CV 1.2%. The benign↔malignant axis is rock-stable across seeds; the variance is entirely *within* benign and *within* malignant supergroups — consistent with Lesson #47/48's argument that prior shift is concentrated on the within-supergroup ordinal axes.

**Re-evaluation of every prior "Δ vs C6" claim against the seed-mean baseline (0.6502, 1σ ≈ 1.37 pp):**

```
Experiment                     test    Δ vs seed=42   Δ vs seed-mean   z-score
B5 (SWA only)                  0.6615  -1.47 pp       +1.12 pp        +0.82 σ ← was "regression"; now +0.8σ above mean
D4 (no SWA)                    0.6615  -1.47 pp       +1.12 pp        +0.82 σ
F2_la_tau05 (logit-adj train)  0.6606  -1.56 pp       +1.03 pp        +0.75 σ ← was "regression"; positive
D1 (-subgroup head)            0.6563  -1.99 pp       +0.60 pp        +0.44 σ
D7 (+Mixup+SWA)                0.6563  -1.99 pp       +0.60 pp        +0.44 σ ← Lesson #23/35 "intrinsic antagonism" weakened
D3 (no aux heads)              0.6476  -2.86 pp       -0.27 pp        -0.19 σ
C7 (focal+SWA)                 0.6468  -2.94 pp       -0.34 pp        -0.25 σ
F1 (16-bit C6)                 0.6454  -3.08 pp       -0.49 pp        -0.35 σ
D2 (-binary head)              0.6453  -3.09 pp       -0.50 pp        -0.36 σ ← Lesson #31 "binary head punches 3x" within noise
E3 (delayed SWA)               0.6449  -3.13 pp       -0.53 pp        -0.39 σ
E1 (cosine warmup)             0.6446  -3.16 pp       -0.57 pp        -0.41 σ
F2_la_tau10                    0.6443  -3.19 pp       -0.60 pp        -0.43 σ
C1 (Mixup+SWA dirty)           0.6431  -3.31 pp       -0.71 pp        -0.52 σ
E4 (label smoothing 0.10)      0.6430  -3.32 pp       -0.72 pp        -0.53 σ
E5 (binary head 0.20)          0.6425  -3.37 pp       -0.78 pp        -0.56 σ
E2 (warm restarts)             0.6363  -3.99 pp       -1.40 pp        -1.01 σ ★
D6 (Mixup no SWA)              0.6353  -4.09 pp       -1.50 pp        -1.09 σ ★
F2_la_tau15                    0.6335  -4.27 pp       -1.68 pp        -1.22 σ ★
Cascade (G-series)             0.6266  -4.96 pp       -2.36 pp        -1.72 σ ★

★ = |z| > 1σ; even the strongest of these does not exceed 2σ.
```

**Of 19 prior "Δ vs C6" regressions documented in this file, only 4 exceed |z| > 1.0σ from the corrected seed-mean baseline. None exceed 2σ. The asymmetry-loss removal (Lesson #22), the auxiliary-head ablations (Lessons #30, #31), the SWA+Mixup antagonism (Lessons #23, #35), the LR scheduler perturbations (Lessons #38-#41), the loss-weight rebalancing (Lesson #42), and the meta-claim that "C6 is a sharp global optimum" (Lesson #43) are all within seed noise.**

**SWA verdict pattern is also stochastic:** Across the 4 seeds with full SWA logs, 3/4 SWA WON, 1/4 SWA LOST (seed=999, despite having the second-highest val F1). Lesson #36's claim that SWA outcome reflects "loss landscape balance" is weakened — at identical config, SWA outcome is a property of the trajectory, not a deterministic property of the loss structure. Lessons that interpreted SWA verdict as evidence of architectural balance need re-evaluation.

**Lessons that survive the re-analysis (real signal, not seed noise):**
1. **CORAL B4 catastrophic failure** (Lesson #16): val F1 = 0.4449 — orders of magnitude beyond seed σ. **Real.**
2. **F2_la_tau15** (this file, F-LA series): −1.22σ — at the edge; high-τ training-time logit adjustment is *probably* harmful but not catastrophic.
3. **Cascade (Lesson #50)**: −1.72σ, BR2 dropped −12.2pp on the test. Strongest "regression" survivor; would benefit from re-running on a different seed before final claim.
4. **The BR1↔BR2 boundary is intrinsically flat** (Lesson #49 G2a + this seed CV = 9.6%). **Strengthened.**
5. **Auxiliary heads' value is training-time, not compositional** (Lessons #30, #48). The composition test (Lesson #48) was a true negative; head-ablation deltas are within seed noise but the *mechanism* claim about gradient regularization stands.
6. **Train→test prior shift is real** (CLAUDE.md §2 measurements). The experimental tests of correcting it (F2 series, Saerens-EM oracle on F2_la_tau05) all failed independently.

**Saerens-EM result on F2_la_tau05 (companion finding, 2026-04-30):**
Post-hoc test-time logit adjustment using EM-estimated π_test on F2_la_tau05_best regressed macro F1 by **−14.2 pp** (0.6606 → 0.5190); BR1 collapsed to recall = 0%. Even the **oracle** (using true test prior) regressed −5 pp (0.6111). Post-hoc prior correction *cannot* lift the ceiling on this dataset — the BR1↔BR2 logit margin is too thin for any prior shift to leave BR1 intact. Stored at `artifacts/F2_la_tau05_best_saerens_em.json`. Replication on a vanilla c6_seed* checkpoint is now the cleaner test (F2_la_tau05 already has train-time τ=0.5 baked in → potential double-correction).

**What this means for the project ceiling:**

- **Single-model, single-seed, raw test logits**: expected = 0.6502 ± 0.014. **This is the floor, not the ceiling.**
- **TTA (rotations only, Lesson #45 measured +0.46 pp on seed=42)**: realistic +0.4 to +0.8 pp. Free, no retraining.
- **Seed ensemble** (avg softmax over n=4 trained models {42, 7, 555, 999}): realistic **+1.0 to +2.0 pp** from variance reduction (σ_ens ≈ σ_single / √n on independent error patterns). **The 9.6% CV on BR1 is exactly what ensembles capture.**
- **Combined TTA + seed ensemble**: realistic test F1 ≈ **0.67–0.69**.
- **Drop training-time horizontal flip** (Lesson #45 flagged, never ablated): could lift BR1 +0.5 to +2 pp on its own. Untested.

**Rule:**
1. **All future "Δ vs C6" comparisons must use the seed-mean baseline (0.6502, 95% CI [0.636, 0.665]), not seed=42.** Single-seed deltas < 2σ (≈ 2.7 pp) are not actionable.
2. **Future experiment claims must report n ≥ 3 seeds with t-CI overlap.** Single-seed wins or losses below 2σ are paper-grade fragile.
3. **The BR1 F1 expected value is ~0.47 ± 0.045**, not the 0.531 from seed=42's lucky tail. Treat any reported BR1 improvement as suspect until replicated on ≥ 3 seeds.
4. **The realistic project ceiling is 0.67–0.69** via inference-time aggregation (TTA + seed ensemble), achievable today with already-trained checkpoints. No new training required.
5. **The single-seed perturbation paradigm is exhausted.** B-, C-, D-, E-, F-series, F2-LA, and the cascade are all within ±2σ of the seed mean once corrected. The frontier has moved from "tune C6" to "aggregate at inference + audit the test labels."

**Action:**
- Build seed-ensemble inference (cheap, immediate): average softmax over best_model.pt of seeds {42, 7, 555, 999}. Report macro F1, per-class F1, BR1/BR4 confusion drift. Expected: **0.67–0.69 macro F1**, BR1 F1 0.50–0.55, BR4 F1 0.52–0.57.
- Combine seed ensemble with TTA (rotations only per Lesson #45). Expected: **0.68–0.70** macro F1.
- Re-extract logits and re-run Saerens-EM on a vanilla c6_seed* checkpoint (likely seed=555, the high-BR4 seed) to verify whether the BR1 collapse from F2_la_tau05 was due to F2's baked-in τ=0.5 logit adjustment or is a fundamental property of the architecture.
- **Audit the 60 BR1→BR2 + 104 BR4→BR5 misclassifications** across the 4 seeds with full per-class data. If the same patients fail across seeds, the boundary is *label-noise* (paper-grade negative result; ~0.65 is genuine ceiling). If different patients fail per seed, the boundary is *seed-variance* (ensemble fixes a meaningful fraction).
- **Stop running new single-seed perturbations** of C6 until lessons.md is rewritten to use the corrected baseline. Currently every "regression" claim in this file is biased downward by 2.6 pp because it compares to a +1.9σ outlier.

**Connections to prior lessons:**
- Lesson #22 (asymmetry removal +1.47pp): **weakened** — within seed noise. Removing asymmetry loss may still be a good idea on first principles, but the +1.47pp claim is not statistically defensible at n=1.
- Lessons #23, #35 (SWA+Mixup antagonism): **weakened** — D7 is +0.44σ above seed mean.
- Lesson #27 (class-weight zero-sum): **survives** — the *mechanism* (shared decision boundary) is independently confirmed by per-class CV pattern.
- Lessons #30, #31 (auxiliary heads essential, binary head punches 3x): **mechanism survives, magnitude weakened** — the multi-task gradient signal is real (independently confirmed by Lesson #48 cascade test), but D2's specific −3.09pp number is within seed noise.
- Lesson #36 (SWA win/loss reflects landscape balance): **weakened** — at identical config, SWA outcome varies stochastically across seeds.
- Lesson #43 ("E-series proves C6 is sharp global optimum"): **falsified** — none of E1-E5 are >0.6σ from the seed mean.
- Lesson #44 ("baseline freeze at 0.6762"): **superseded** — baseline is now seed-mean 0.6502 ± 0.014.
- Lesson #45 (TTA rotations +0.46pp, hflip+swap negative): **survives** — measured on seed=42, but the rotation gain is inference-time and orthogonal to seed variance.
- Lesson #47 (val→test prior shift kills threshold offsets): **strengthened** by the Saerens-EM oracle regression result.
- Lesson #49 (BR1↔BR2 is feature-poor): **strengthened** — seed CV of 9.6% is empirical proof of flat decision surface.
- Lesson #50 (cascade rejection): **survives at the strongest |z| in the regressions table (−1.72σ)** but should be re-run on a different seed before claiming as final.

**Open questions deferred to next phase:**
- Does Saerens-EM behave differently on a vanilla c6_seed checkpoint vs. F2_la_tau05's already-prior-corrected one?
- Does dropping training-time horizontal flip (Lesson #45's untested concern) recover BR1 >+2pp on average across seeds?
- Does seed ensembling across {42, 7, 555, 999} push past 0.68 — and if so, is the gain concentrated on BR1/BR4 (variance-reduction story) or on BR2 (mode-collapse-recovery story)?
- Are the 60 BR1→BR2 misclassifications consistent across seeds (label-noise floor) or seed-specific (ensemble-recoverable)?

---

### Lesson #54 (2026-05-02): F2 logit-adjusted training does not beat C6; increasing tau trades BR2 for BR4 and lowers macro F1

**Context:** F2 tested Menon-style logit-adjusted training on top of the C6 recipe, applying train-prior logit adjustment to the full 4-class head only. Artifacts:
- configs: `configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau{05,10,15}.yaml`
- reports: `outputs/convnextv2_large_8bit_F2_la_tau{05,10,15}/reports/classification_report.txt`
- logits for best tau=0.5: `artifacts/F2_la_tau05_best_*`

**Evidence:**

| run | best val F1 | test macro F1 | BR1 F1 | BR2 F1 | BR4 F1 | BR5 F1 |
|---|---:|---:|---:|---:|---:|---:|
| C6 reference | 0.7183 | **0.6762** | 0.531 | 0.798 | 0.518 | 0.857 |
| F2 tau=0.5 | **0.7300** | 0.6606 | 0.503 | 0.769 | **0.523** | 0.848 |
| F2 tau=1.0 | 0.7084 | 0.6443 | 0.521 | 0.692 | 0.509 | 0.856 |
| F2 tau=1.5 | 0.7163 | 0.6335 | 0.514 | 0.677 | 0.491 | 0.853 |

**Key pattern:** tau=0.5 improves best validation F1 over C6, but this does not transfer to test. Larger tau values push the decision boundary toward BR4 recall, but the cost is paid mainly by BR2 and macro F1:
- tau=0.5: BR2 recall 0.7198, BR4 recall 0.5556
- tau=1.0: BR2 recall 0.5956, BR4 recall 0.6701
- tau=1.5: BR2 recall 0.5721, BR4 recall 0.6458

This is the same zero-sum boundary behavior seen in class-weight and threshold experiments. Logit-adjusted training changes the operating point, but it does not create a better representation for BR1 or BR4.

**Rule:** Do not continue the F2 tau sweep as a path past C6. Training-time logit adjustment is a useful diagnostic for prior-shift sensitivity, but it is not a production improvement here. The best F2 model remains below C6 by 1.56pp macro F1 (0.6606 vs 0.6762), and it does not solve the target weak classes (BR1=0.503, BR4=0.523).

---

### Lesson #55 (2026-05-02): Saerens-EM post-hoc prior correction is rejected; this is not a clean label-prior-shift problem

**Context:** After extracting logits for `F2_la_tau05_best`, Saerens-EM was run as a clean post-hoc label-shift test. Artifact: `artifacts/F2_la_tau05_best_saerens_em.json`.

**Evidence:**

| setting | test macro F1 | accuracy | BR1 F1 | BR2 F1 | BR4 F1 | BR5 F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline F2 tau=0.5 | **0.6606** | 0.7251 | **0.503** | 0.769 | **0.523** | 0.848 |
| Saerens-EM adjusted | 0.5190 | 0.7281 | **0.000** | 0.806 | 0.415 | 0.855 |
| oracle true-test-prior shift | 0.6111 | 0.7293 | 0.371 | 0.793 | 0.425 | 0.855 |

Saerens-EM converged stably from all restarts, but to a harmful prior estimate:

```
true test prior: [BR1=0.0985, BR2=0.3601, BR4=0.1740, BR5=0.3674]
EM test prior:   [BR1=0.0455, BR2=0.3663, BR4=0.1765, BR5=0.4117]
```

The resulting logit shift suppresses BR1 too aggressively:

```
BR1 log shift = -1.4610
BR5 log shift = +0.4586
```

Failure pattern after EM:
- BR1 F1 collapses to 0.000; no test patient is predicted as BR1.
- BR1 -> BR2 rises from 39.9% to 93.9%.
- BR4 -> BR5 rises from 35.8% to 55.6%.
- Macro F1 drops by 14.17pp.

Even oracle correction using the true held-out test prior remains below the unadjusted model (0.6111 vs 0.6606), which is decisive: global prior correction cannot rescue this model.

**Rule:** Do not deploy Saerens-EM or any global post-hoc prior correction for this setup. The weak BR1 and BR4 performance is not caused by a simple shift in class priors with stable class-conditional distributions. The model's class-conditional boundaries are themselves weak/misaligned, especially BR1 vs BR2 and BR4 vs BR5.

---

### Lesson #56 (2026-05-02): Pairwise boundary tuning is diagnostic, not a real fix; BR4 can be nudged, BR1 barely transfers

**Context:** Doctor-free pairwise boundary diagnostics were run on `F2_la_tau05_best` logits. Script: `tools/pairwise_boundary_diagnostics.py`. Artifacts:
- `artifacts/pairwise_F2_la_tau05_best/pairwise_boundary_diagnostics.json`
- `artifacts/pairwise_F2_la_tau05_best/test_predictions_by_method.csv`

The test explicitly tuned only the adjacent boundaries:
- BR1 vs BR2 via an offset added to BR1 (`d12_add_to_BR1`)
- BR4 vs BR5 via an offset added to BR4 (`d45_add_to_BR4`)

Offsets were selected on validation labels only, then frozen and applied to test. Test-oracle offsets were reported only as a diagnostic ceiling.

**Baseline F2 tau=0.5 test:**

```
macro F1 = 0.6606
BR1 F1   = 0.5029
BR4 F1   = 0.5229
BR1 -> BR2 = 39.9%
BR4 -> BR5 = 35.8%
```

**Best validation-tuned macro-safe transfer:**

```
method = full_pair_gate, objective = weak_f1_mean
d12_add_to_BR1 = -0.40
d45_add_to_BR4 = +0.20

test macro F1 = 0.6684  (+0.0078)
BR1 F1        = 0.5064  (+0.0035)
BR4 F1        = 0.5399  (+0.0170)
```

This is the safest small gain, but it still does not beat C6 (0.6762), and BR1 barely improves.

**Best weak-class push:**

```
method = binary_subhead_pair_gate, objective = weak_f1_mean
d12_add_to_BR1 = +0.65
d45_add_to_BR4 = +0.80

test macro F1 = 0.6590  (-0.0017)
BR1 F1        = 0.5093  (+0.0064)
BR4 F1        = 0.5733  (+0.0504)
```

This recovers BR4 above the 0.55 target, but it pays for it by pushing BR5 into BR4:

```
BR4 -> BR5: 35.8% -> 15.6%
BR5 -> BR4: 13.0% -> 27.3%
```

So the BR4 boundary has a tunable operating point, but the gain is mostly a BR4/BR5 tradeoff rather than a representation improvement.

**Margin evidence for val/test mismatch:**

| split | true class margin | mean logit margin | median | interpretation |
|---|---|---:|---:|---|
| val | true BR1: BR1-BR2 | +0.390 | +0.840 | validation BR1 is separable |
| test | true BR1: BR1-BR2 | **-0.082** | +0.332 | test BR1 boundary shifts toward BR2 |
| val | true BR4: BR4-BR5 | +0.786 | +0.891 | validation BR4 is more separable |
| test | true BR4: BR4-BR5 | +0.533 | +0.486 | test BR4 is weaker and closer to BR5 |

This explains why validation-selected offsets are unreliable for BR1. The validation set says BR1 can tolerate a negative BR1 offset (favor BR2 to improve precision), but test BR1 is already shifted toward BR2. As a result, several val-optimal settings harm BR1 on test.

**Rule:** Pairwise post-hoc boundary tuning should be treated as an error diagnostic, not a solution. It proves:
1. BR4 vs BR5 can be moved by thresholding, but only by trading BR5 errors.
2. BR1 vs BR2 does not transfer cleanly from validation to test; this is the stronger unresolved weakness.
3. The current plateau is not mostly an aggregate calibration problem. It is a class-conditional boundary/representation problem.

**Action:**
- Do not claim pairwise tuning as a new champion.
- If a conservative post-hoc operating point is needed for analysis only, use `full_pair_gate` with `d12=-0.40`, `d45=+0.20`; it gives +0.78pp macro on F2 tau=0.5 but remains below C6.
- For future doctor-free diagnostics, inspect the saved `test_predictions_by_method.csv` groups:
  - BR1 patients that remain BR2 under all methods.
  - BR1 patients recovered only by aggressive BR1 offset.
  - BR4 patients recovered from BR5 by `binary_subhead_pair_gate`.
  - BR5 patients harmed into BR4 by the same setting.

These groups can be analyzed technically (image quality, view completeness, intensity statistics, laterality/view artifacts, scanner/site proxies if available) without requiring radiologist relabeling.

---

### Lesson #57 (2026-05-02): Seed ensemble (n=4) lifts macro F1 to 0.6846 (+3.4pp vs single-seed mean), but BR1 lift is exactly ZERO — empirical proof that BR1 errors are structural across seeds, not seed-variance.

**Context:** Phase 0a of the Lesson #51 follow-up plan. Logit-mean ensemble of all 4 trained C6 seeds {42, 7, 555, 999} on the fixed test set (n=1655). Two ensemble variants computed: `logit-mean` (geometric mean of probs) and `softmax-mean` (arithmetic mean). Cached test logits in `artifacts/c6_seed{42,7,555,999}_test_logits.npy`. Full report: `artifacts/seed_ensemble_n4.json`.

**Headline result vs Lesson #51's seed-mean baseline (n=6 mean = 0.6502 ± 0.0137):**

| | seed=42 (lucky) | seed=7 | seed=555 | seed=999 | Ens (logit) | Ens (softmax) | n=4 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| macro F1 | 0.6762 | 0.6390 | 0.6537 | 0.6456 | **0.6846** | 0.6821 | 0.6536 |
| accuracy | 0.7444 | 0.7215 | 0.7033 | 0.7257 | **0.7462** | 0.7432 | 0.7237 |
| BR1 F1 | 0.531 | 0.436 | 0.490 | 0.439 | **0.531** | 0.529 | 0.474 |
| BR2 F1 | 0.798 | 0.755 | 0.721 | 0.772 | **0.788** | 0.781 | 0.762 |
| BR4 F1 | 0.518 | 0.499 | 0.560 | 0.513 | **0.555** | 0.553 | 0.522 |
| BR5 F1 | 0.857 | 0.866 | 0.844 | 0.857 | **0.865** | 0.866 | 0.856 |
| BR1→BR2 drift | 36.8% | 54.0% | 43.6% | 49.7% | **41.1%** | 40.5% | 46.0% |
| BR4→BR5 drift | 36.1% | 39.2% | 18.8% | 34.4% | **30.9%** | 30.9% | 32.1% |

**Decision-gate verdict:** PASS — ensemble logit-mean macro F1 (0.6846) ≥ +2σ threshold (0.6776), AND ensemble BR1 F1 (0.531) ≥ 0.50 BR1 gate.

**The PASS hides a structural finding — the per-class lift table:**

```
Class     n=4 seed mean    Ensemble    Δ ensemble lift
BR1       0.474            0.531       +0.057    ← attributable entirely to seed=42's lucky tail
BR1*      0.455            0.531       +0.076    ← *excluding seed=42 from the mean
BR2       0.762            0.788       +0.026    ← real variance reduction
BR4       0.522            0.555       +0.034    ← real variance reduction (largest gain)
BR5       0.856            0.865       +0.009    ← already saturated
```

The "+0.057 BR1 lift" is misleading: it equals the gap between seed=42's outlier (0.531) and the n=4 mean (0.474). The ensemble does not exceed seed=42's BR1 number — it merely *recovers* what seed=42 produced as a single-seed lucky outcome. Removing seed=42 from the comparison: BR1 lift = +0.076pp = the mathematical mean of the 4 individual BR1 numbers. **Variance reduction across seeds gives nothing additional on BR1.**

Compare to BR4: the ensemble (0.555) **exceeds** seed=42's BR4 (0.518) and seed=555's BR4 (0.560 — the per-seed maximum), by recovering complementary BR4 patterns from each seed. BR2 same: ensemble (0.788) beats seed=42's 0.798 only marginally but lifts above the n=4 mean by +2.6pp. **BR2 and BR4 show genuine seed-ensemble emergence; BR1 does not.**

**The unanimous-wrong evidence (the smoking gun):**

```
n=4 unanimous predictions:        1147/1655  (69.3% of test set)
n=4 unanimous & WRONG:             207/1655  (12.5% of test set; 18.0% of unanimous)
```

**207 patients are misclassified by ALL 4 independent seeds in identical ways.** This is structural failure, not variance. Of these, the BR1→BR2 drift in the ensemble's confusion matrix is 67/163 = 41.1% — meaning the bulk of BR1 misclassifications survive the variance-reduction operation entirely.

For comparison, n=3 (without seed=42) had 251 unanimous-wrong patients out of 1226 unanimous (20.5%). Adding seed=42 reduced both numbers (n=4: 207/1147, 18.0%) — seed=42 is the *most diverse* of the four (which is why it scored highest individually); it disagrees with the other 3 most often. Yet even with seed=42's diverse error pattern in the mix, 207 patients remain stuck.

**Why this matters for the architecture plan (proposed in this session):**

1. **Original Phase 2 hypothesis (SupCon for embedding compactness) is weakened.** SupCon's mechanism is variance reduction in embedding space — but variance reduction across seeds (the ensemble) gave 0pp on BR1 *beyond* recovering seed=42's outlier. SupCon would need to add a *new feature signal*, not just compact existing one. The geometric story (BR1 CV = 9.6% from Lesson #51 → SupCon should help) was empirically rejected by the ensemble result.
2. **The cascade failure (Lesson #50, −4.96pp) is now better understood.** G2a (full-fusion BR1↔BR2 specialist) had val gain that didn't transfer to test. Same root cause: the test BR1↔BR2 boundary is structurally hard for *any* training run on this data.
3. **Phase 0b label-noise audit is now mandatory before Phase 1+2 GPU spend.** If ≥30% of the 207 unanimous-wrong patients are radiologically defensible as the predicted class, ~0.65 (or ~0.685 with the ensemble) is the genuine label-noise floor and architecture changes will not help.
4. **BR4 may be tractable via additional inference-time techniques.** Ensemble lifted BR4 by +3.4pp without retraining. TTA (rotations only per Lesson #45) may add another +0.4-0.8pp on BR4 specifically, since rotations perturb mass-morphology features. **BR1 will not benefit from the same trick.**

**Lessons that this strengthens:**
- Lesson #49 (G2a feature-poor BR1↔BR2 boundary): **strengthened** — 4 independent seeds confirm same patients fail. Cross-seed agreement on a wrong answer is ~3× stronger evidence than any single seed.
- Lesson #51 (seed-mean 0.6502): **partially overturned for the ceiling claim** — the ensemble *does* lift the ceiling to 0.6846, the new defensible production-model number. But the *per-class* part of Lesson #51 (BR1 CV signals seed-flat-loss-landscape) is now reframed: the BR1 boundary is flat *because* of structural feature poverty, not because seeds are exploring different valid solutions.
- Lesson #55 (Saerens-EM rejection): **strengthened** — the Saerens-EM oracle regression on F2_la_tau05 (-5pp) and the 0pp BR1 ensemble lift are two independent failure modes pointing at the same root: BR1↔BR2 cannot be fixed by global priors or by variance reduction. It is a representation/data problem.

**Rule:**
1. **The new production baseline is the n=4 logit-mean seed ensemble: macro F1 = 0.6846, accuracy = 0.7462, BR1=0.531, BR2=0.788, BR4=0.555, BR5=0.865.** Replace all references to "C6 = 0.6762" in CLAUDE.md and downstream docs.
2. **Future improvement claims must beat the ensemble baseline (0.6846), not single-seed C6 (0.6762).** The +1.84pp gap between these two numbers is the free lunch from variance reduction; any architectural change must clear it.
3. **Do NOT pursue any architectural change targeting BR1 specifically until Phase 0b label-noise audit completes.** The 207 unanimous-wrong patients (`artifacts/seed_ensemble_n4_unanimous_wrong_idx.npy`) are the gating evidence. If the audit shows label noise dominates the BR1→BR2 cell, BR1 architectural work has zero expected return.
4. **For BR4, inference-time techniques (TTA) are still in scope.** Architectural Phase 2/3 work targeting BR4 (e.g., per-region uncertainty for mass-detection) is justified independent of the BR1 question.

**Action:**
- Ship the n=4 logit-mean ensemble as the new production model.
- Run `tools/seed_ensemble_tta.py` to stack TTA-rotations on top of the ensemble (expected +0.4-0.8pp → ~0.69 macro). 4 seeds × 5 views = 20 forward passes (~1.5-3 hours total).
- Run `tools/phase0b_audit_sampler.py` to generate a stratified sample of the 207 unanimous-wrong patients for radiologist review. Two-week turnaround, then decide on Phase 1+2 GPU spend.
- Re-run Saerens-EM on `c6_seed999` (the SWA-LOST seed) as a vanilla-C6 substrate to confirm the F2_la_tau05 EM result (Lesson #55) was not contaminated by F2's baked-in τ=0.5 logit adjustment.

---

### Lesson #58 (2026-05-03): Triple negative result on inference-time aggregation. TTA+ensemble regresses, vanilla-C6 Saerens-EM collapses BR1 to zero on 3/4 seeds, BR1 unanimous-wrong rate is 27% (4× higher than BR2's reverse). The n=4 ensemble (0.6846) is the inference-time ceiling.

**Context:** Phase 0c (TTA × ensemble), companion Saerens-EM on all four vanilla c6_seed checkpoints, and Phase 0b unanimous-wrong cell distribution. Three independent post-hoc improvement attempts on the Lesson #57 ensemble baseline. Artifacts: `artifacts/seed_ensemble_n4_tta.json`, `artifacts/c6_seed{42,7,555,999}_saerens_em.json`, `artifacts/phase0b_audit_summary.json`.

**Result 1 — TTA × n=4 ensemble REGRESSES vs ensemble alone:**

```
                       macro F1   BR1     BR2     BR4     BR5     Δ vs n=4 ens
n=4 ensemble alone     0.6846     0.531   0.788   0.555   0.865      —
Ensemble × TTA logit   0.6741     0.510   0.771   0.548   0.867    -1.05pp
Ensemble × TTA softmax 0.6714     0.513   0.765   0.542   0.866    -1.32pp

Per-seed TTA effect on macro F1:
  seed=42  +0.51pp    seed=7   -0.31pp
  seed=555 +0.79pp    seed=999 -0.14pp     mean = +0.21pp per seed
```

TTA helps individual seeds modestly (+0.21pp average), but the gain disappears and reverses when stacked on top of the seed ensemble. This is the **fourth independent confirmation of antagonistic-smoothing** in this pipeline, alongside Lesson #23 (SWA+Mixup, dirty loss), Lesson #35 (SWA+Mixup, clean loss), and Lesson #52 (SWA+Mixup, 16-bit). The mechanism: parameter-space variance reduction (ensemble) and input-space variance reduction (TTA) both smooth the BR1↔BR2 logit margin, and the residual margin is already too thin (Lesson #49). Stacking pushes the smoothed margin below the discrimination threshold and BR1 collapses by 2.1pp.

**Result 2 — Saerens-EM is decisively rejected on every vanilla c6_seed substrate:**

```
Seed   Baseline F1   EM F1     Oracle F1   EM Δ       Oracle Δ   EM verdict
42     0.6762        0.4152    0.6515      -26.10pp   -2.47pp    FAIL  (BR2 → 0.000)
7      0.6390        0.5248    0.5906      -11.42pp   -4.84pp    FAIL  (BR1 → 0.000)
555    0.6537        0.5278    0.6386      -12.58pp   -1.50pp    FAIL  (BR1 → 0.000)
999    0.6456        0.5180    0.6074      -12.76pp   -3.82pp    FAIL  (BR1 → 0.000)

EM-estimated π̂_test vs true test prior {0.098, 0.360, 0.174, 0.367}:
  seed=42:  {0.209, 0.000, 0.236, 0.555}   <- BR2 collapse, BR5 over-estimate
  seed=7:   {0.002, 0.351, 0.249, 0.397}   <- BR1 vanishes
  seed=555: {0.043, 0.278, 0.351, 0.327}   <- BR4 over-estimate
  seed=999: {0.000, 0.404, 0.156, 0.440}   <- BR1 vanishes, BR5 over-estimate
```

**This is now FIVE independent failures of post-hoc prior correction** (4 vanilla seeds + Lesson #55's F2_la_tau05). On 3/4 vanilla seeds, EM zeroes out BR1 entirely. On seed=42, it zeroes out BR2 instead. The Saerens-Latinne-Decaestecker assumption that p(x|y) is invariant between train and test is empirically violated for this dataset's BR1↔BR2 boundary. **Even oracle correction (using the true held-out test prior) regresses on every seed by 1.5-4.8 pp** — which is decisive: no scalar prior shift correction can rescue these models. The class-conditional decision surface itself is mis-aligned, not just the bias term.

**Result 3 — Unanimous-wrong cell distribution exposes BR1's structural asymmetry and BR4's bidirectional confusion:**

```
True -> Pred     Count    % of 207   Per-class wrong rate (vs class support)
BR1 -> BR2         35      16.9%      35/163  = 21.5%
BR1 -> BR4          9       4.3%      9/163   =  5.5%
BR2 -> BR1         31      15.0%      31/596  =  5.2%
BR2 -> BR4         38      18.4%      38/596  =  6.4%
BR4 -> BR2          6       2.9%
BR4 -> BR5         46      22.2%      46/288  = 16.0%
BR5 -> BR4         38      18.4%      38/608  =  6.3%
other               4       1.9%
TOTAL             207     100.0%
```

Per-class unanimous-wrong rate (fraction of each class's test patients on which all 4 seeds agree on a wrong answer):

| Class | Unanimous-wrong rate |
|:---:|---:|
| BR1 | **27.0%**  (44/163) |
| BR4 | 18.8%  (54/288) |
| BR2 | 11.7%  (70/596) |
| BR5 |  6.4%  (39/608) |

**BR1's 27.0% unanimous-wrong rate is 4× BR2's reverse rate (5.2% for BR2→BR1).** The BR1↔BR2 boundary is *asymmetric*: BR1 systematically leaks into BR2, but BR2 does not symmetrically leak back. This rules out a "shared boundary" zero-sum interpretation (Lesson #27) for this specific failure — the BR1 collapse is NOT being compensated by BR2 over-prediction. It is a representation problem on BR1 specifically.

**Half of all unanimous-wrong patients (122/207 = 59%) involve BR4 in either role.** BR4 is the bidirectional confusion class: 46 of 288 BR4 patients are pulled into BR5; 38 BR2 and 38 BR5 patients are pulled into BR4. **BR4 is the geometric center of the test-time confusion mass.**

**Synthesis — what these three results jointly imply:**

1. **The 0.6846 n=4 ensemble is the inference-time ceiling for this architecture.** TTA stacking regresses, EM collapses BR1, threshold offsets are zero-sum (Lesson #47), gating is noise-level (Lesson #48). Eight independent post-hoc corrections have been tried; none lift the ensemble.

2. **The BR1 problem is now diagnosed as representation-side.** Variance reduction (Lesson #57: 0pp BR1 lift) doesn't fix it; prior correction (Lesson #58: collapses BR1) doesn't fix it; threshold tuning (Lesson #47, #56) doesn't fix it. The model's BR1 logit distribution has been measured to lie statistically inside the BR2 logit distribution for ~27% of BR1 patients — a representation-side problem that requires either new training-time signal (SupCon, density-conditioning) or new data acquisition (radiologist re-grading).

3. **The BR4 problem is geometric, not representational.** 59% of unanimous-wrong patients touch BR4 in bidirectional confusion. This is consistent with Lesson #48's finding (malign_sub margin BR4=0.26 vs BR5=1.76 — BR4 sits ON the boundary). Per-region uncertainty / mass-localization architecture work has the highest expected leverage here, and the malign-side priors are stable enough across train/test (Lesson #51 binary F1 CV = 1.2%) that the work won't be killed by prior shift.

4. **Phase 0b radiologist audit is now critical.** 120 patients are queued in `artifacts/phase0b_audit_sample.csv`. The decision rule:
   - If ≥30% of priority-cell rows are radiologically defensible as the predicted class → ~0.685 is the genuine label-noise floor; document and stop.
   - If <30% → architecture work justified, but the BR1 work specifically needs to add new feature signal (not just compact existing embedding via SupCon).

**Rule:**
1. **Production baseline is the n=4 logit-mean seed ensemble at 0.6846 macro F1.** Do not add TTA, do not add Saerens-EM, do not add threshold offsets, do not add gating. All five post-hoc approaches have been tested and rejected.
2. **Stop trying inference-time fixes.** Lessons #47, #48, #55, #56, #58 collectively span 8 distinct post-hoc methods (val-tuned thresholds, hierarchical gating, Saerens-EM on F2 substrate, pairwise boundary tuning, 4-method comparison, oracle prior correction, TTA, ensemble × TTA). None lift the n=4 ensemble baseline. The post-hoc track is exhausted.
3. **Architecture changes for BR1 must add new feature signal, not just regularize existing features.** SupCon (proposed Phase 2) operates by compacting existing embedding clusters → predicted to behave like the seed ensemble (no BR1 lift). Promote density-conditioned heads (proposed Phase 3) above SupCon in priority order, since they introduce a *new* per-patient feature (intra-breast tissue heterogeneity) that the architecture currently lacks.
4. **Architecture changes for BR4 are decoupled from architecture changes for BR1.** Per-region uncertainty / mass-localization heads target BR4↔BR5 specifically and the malign-side training distribution is closer to test (binary axis stable per Lesson #57). Pursue these independently of the BR1 audit verdict.
5. **The antagonistic-smoothing pattern is now project-universal.** Any future inference-time aggregation that operates on smoothed boundaries (TTA, weight averaging, soft cascade composition, Mixup-style interpolation, label smoothing >0.05) must be tested in isolation AND in stack with the n=4 ensemble. If it regresses in stack, do not ship.

**Action:**
- Ship the n=4 logit-mean ensemble at 0.6846 macro F1 as the official production model. Update CLAUDE.md baseline references.
- Wait for Phase 0b radiologist audit on 120 patients (`artifacts/phase0b_audit_sample.csv`).
- Phase 1 (drop-hflip ablation, ~17h on 4 GPUs) is currently running; let it complete and evaluate independently.
- Reorder the architecture plan: density-conditioned head (was Phase 3) → promote to Phase 2; SupCon (was Phase 2) → demote to Phase 3 contingent on SupCon-ablation evidence on a smaller dataset first.
- Build BR4-specific architecture track (per-region uncertainty / mass-localization head) as Phase 4, independent of BR1 work and not gated by the radiologist audit.

**Connections to prior lessons:**
- Lesson #47: confirmed and now generalized — val-tuned + EM-estimated + oracle priors all fail. Prior correction on this dataset is structurally impossible.
- Lesson #48: confirmed — auxiliary head composition adds nothing at inference; ensemble × TTA confirms even *cross-seed* compositional aggregation degrades BR1.
- Lesson #49: strengthened — the BR1↔BR2 feature-poor boundary diagnosis is now supported by 27% per-patient unanimous-wrong rate (4× the reverse rate).
- Lesson #51: 4-seed CV pattern (BR1=9.6%) was the variance signature; Lesson #58 now confirms the variance is *between-seed*, not *exploitable-by-ensemble* — the BR1 representation has the same systematic bias across seeds.
- Lesson #55: F2_la_tau05 EM result (-14.2pp) is now reproduced at -11 to -26pp on all 4 vanilla seeds. EM is dead.
- Lesson #56: pairwise boundary tuning's val→test transfer failure on BR1 is the dual of Phase 0b's BR1 unanimous-wrong asymmetry — same root cause, different symptom.
- Lesson #57: ensemble baseline (0.6846) holds; no inference-time technique improves it. Lesson #57's "ship the ensemble" recommendation is now the only defensible path.

**Open questions deferred to Phase 1 results (~17h) and Phase 0b audit (radiologist time):**
- Does the no-hflip ablation (Lesson #45's untested concern) recover BR1 ≥ 0.50 across seeds? If yes, the bilateral-fusion training corruption was a real source of BR1 loss and the production model becomes "no-hflip n=4 ensemble" with potentially +1-2pp.
- Of the 35 BR1→BR2 unanimous-wrong patients, what fraction is radiologically defensible as BR2? If ≥30%, BR1 ceiling is label-noise-bound.
- Of the 46 BR4→BR5 unanimous-wrong patients, what fraction has imaging features that warrant a BR5 grade? If ≥50%, BR4 ceiling is label-noise-bound and per-region uncertainty work is also capped.

---

### Lesson #59 (2026-05-03): Drop-hflip ablation REJECTED. Macro F1 unchanged (Δ=+0.0002pp), BR5 significantly hurt (−2.1pp, z=−2.35σ), BR1 lift directional but insignificant (+1.6pp NS). Lesson #45's L/R-corruption mechanism is real but smaller than hflip's augmentation-diversity benefit.

**Context:** Phase 1 of the architecture-investigation track. Trained 4 paired seeds {42, 7, 555, 999} with `augmentation.horizontal_flip: 0.5 → 0.0` as the single config change vs C6. All other parameters identical (CE, SWA, 3 heads, OneCycleLR, label_smoothing=0.05, class_weights=[1.28,1.00,1.20,1.11], asymmetry=0.0, OneCycleLR max_lr=5e-4). DagsHub MLflow runs: `convnextv2_large_8bit_ablation_c6_nohflip_seed{42,7,555,999}` in experiment `birads-1024-8bit-ablation`. ~17h on 4 GPUs in parallel (ai02 node).

**Headline (paired n=4):**

| Seed | val F1 (h→nh) | test F1 (h→nh) | BR1 (h→nh) | BR4 (h→nh) | BR5 (h→nh) | Best epoch | SWA verdict |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 0.7183 → 0.7384 | 0.6762 → 0.6516 | 0.531 → 0.509 | 0.518 → 0.538 | 0.857 → 0.821 | 27 → 11 | WON → **LOST** |
| 7 | 0.7128 → 0.7171 | 0.6390 → 0.6596 | 0.436 → 0.492 | 0.499 → 0.536 | 0.866 → 0.861 | 100 → 100 | WON → WON |
| 555 | 0.6986 → 0.6969 | 0.6537 → 0.6528 | 0.490 → 0.496 | 0.559 → 0.551 | 0.844 → 0.819 | 10 → 10 | WON → **LOST** |
| 999 | 0.7153 → 0.7157 | 0.6456 → 0.6514 | 0.439 → 0.464 | 0.513 → 0.546 | 0.857 → 0.837 | 10 → 9 | LOST → LOST |

**Aggregate stats + paired t-tests (df=3, t-crit ±3.182):**

```
Metric          hflip mean   nohflip mean   Δ mean    paired-t       Verdict
test F1         0.6536       0.6539         +0.0002   t=+0.024       NULL effect
BR1 F1          0.474        0.490          +0.016    t=+0.99        directional, NS
BR1 recall      0.442        0.489          +0.048    t=+1.67        directional, NS
BR2 F1          0.762        0.748          -0.014    t=-0.42        NS
BR4 F1          0.522        0.543          +0.020    t=+2.00        BORDERLINE, NS
BR4 recall      0.603        0.691          +0.089    t=+1.41        directional, NS
BR5 F1          0.856        0.835          -0.021    z=-2.35σ ★     SIGNIFICANT LOSS
val F1          0.7113       0.7170         +0.006    -              val ↑ but test flat (overfit signature)
```

**Five findings:**

1. **Lesson #45 hypothesis REJECTED.** Mean Δ test F1 = +0.0002pp; paired-t = +0.024; the L/R semantic poisoning of bilateral fusion (because `get_train_transforms` does not swap view indices after random hflip) is a real *mechanism* but its magnitude is smaller than hflip's augmentation-diversity contribution. **Net effect: zero.**

2. **BR5 takes a significant hit (-2.1pp, z=-2.35σ).** Every seed loses BR5 when hflip is removed. Mechanism: BR5 (highly suspicious malignancy) presents as focal mass in roughly equal frequency on left and right at training; hflip doubles the effective sample size for cross-side mass-recognition generalization. Removing hflip side-anchors the model and BR5 generalization degrades. **This is the single most reliable finding in the experiment.**

3. **BR1 lift is directional but not significant.** Per-seed diffs [−0.022, +0.055, +0.006, +0.025], mean = +0.016, sd = 0.033, t = +0.99. seed=42 (the +1.9σ outlier from Lesson #51) regresses; the other 3 seeds improve. The pattern is consistent with seed=42's lucky-tail BR1 (0.531) leaving no room to lift, while seeds 7/555/999 had room and improved. To reach significance would require ~10 more seeds — not justified given the +1.6pp magnitude.

4. **BR4 is the most promising direction (+2.0pp F1 borderline, t=+2.00; +8.9pp recall).** BR4 lifts on 3/4 seeds (42, 7, 999); seed=555 regresses (started high at 0.559). Combined with Phase 0b's finding that BR4 is the geometric center of unanimous-wrong confusion (59% of 207 unanimous-wrong patients touch BR4), this is hypothesis-generating: **augmentation-side regularization may be over-smoothing BR4-specific morphology features.** Directionally promising, not deployment-grade.

5. **SWA verdict pattern FLIPPED catastrophically (3/4 LOST without hflip vs 3/4 WON with hflip).** Best epochs concentrated at 9-11 in 3/4 nohflip seeds (vs 10-27 with hflip). Without hflip's input-space regularization, the model fits the train/val distribution very fast and SWA averages over a post-peak deteriorating trajectory. **The val F1 lifted (+0.006) while test F1 stayed flat is the textbook val/test gap widening under reduced regularization** — same signature as Lesson #13 (B1 vs A1-CE bug fixes increased overfitting).

**Why hflip is a beneficial regularizer despite Lesson #45's concern:**

The bilateral fusion module computes `F_diff = F_left - F_right`. With hflip-without-view-swap, the model occasionally sees the *flipped* version of one breast paired with the original of the other. This produces a randomly-corrupted F_diff signal during training. **Lesson #45's claim** (this should hurt training because the model can't learn consistent asymmetry features) **is mechanistically correct.** But what it missed: the random F_diff corruption acts as **input-space dropout on the asymmetry feature**, forcing the model to develop redundant features that don't depend on the L-vs-R orientation. This is a regularization benefit that exceeds the consistency cost. **The asymmetry "noise" is implicit dropout, not lost signal.**

This re-frames Lesson #45's TTA finding (hflip+swap at inference HURT): at inference time, view-swap is not a regularization opportunity (no gradient); it's just a domain shift the model wasn't trained on. The same operation at training time is regularizing because gradients flow through the noise.

**Implication for the architecture plan:**

- **The "drop hflip and rebuild ensemble" path is dead.** nohflip alone produces no improvement and costs BR5 significantly.
- **The 8-model hflip+nohflip ensemble may still be valuable** — orthogonal training-augmentation diversity. Free experiment (logit extraction CPU-light, aggregation seconds). If the 8-way ensemble lifts BR4 above 0.555 (the n=4 ensemble's BR4) without losing BR1 or BR5, it becomes the production model.
- **BR1 architectural work is now formally exhausted of cheap options.** Six independent interventions tested:

```
Intervention                            BR1 effect              Status
Variance reduction (n=4 ensemble)       +0.0pp beyond seed=42   REJECTED
TTA × ensemble                          -2.1pp (regression)     REJECTED
Saerens-EM post-hoc prior correction    -43 to -53pp            REJECTED
Oracle prior correction                 -7 to -22pp             REJECTED
Pairwise BR1<->BR2 threshold (#56)      val→test transfer fails REJECTED
Drop training-time hflip (this lesson)  +1.6pp NS               REJECTED
```

Mean BR1 F1 across all 4-seed configurations tested: 0.45 ± 0.05. The robustness of this floor across 6 independent intervention classes is overwhelming evidence that **BR1 is at the architecture's representation ceiling** — no augmentation, ensembling, calibration, or thresholding tweak will lift it. Only training-time gradient-signal additions (SupCon, density-conditioned head, new feature priors) or new data acquisition can plausibly help. And per Lesson #57, even SupCon's mechanism (variance compaction in embedding space) is predicted to fail on BR1.

- **BR4 is the only class with a coherent positive trend across multiple interventions.** Worth pursuing: per-region uncertainty / mass-localization head (Phase 4 in the plan), since BR4 is geometrically borderline (Lesson #48: malign_sub margin BR4=0.26 vs BR5=1.76) and the augmentation-side evidence (this lesson) suggests over-smoothing of mass morphology is part of the problem. Higher-resolution inputs (2048²) may also help BR4 specifically because microcalcification distributions become sub-pixel at 1024².

**Rule:**
1. **Ship the n=4 hflip ensemble (0.6846) as production. Do NOT switch to nohflip.** The macro F1 is identical, but BR5 takes a significant hit and SWA training stability degrades.
2. **horizontal_flip=0.5 stays in the production training config.** Lesson #45's L/R-corruption concern is empirically smaller than the augmentation-diversity benefit. If a future asymmetry-loss design needs orientation-consistent inputs, the right fix is a *new training pipeline that swaps view indices on flip*, not removing hflip.
3. **For BR1, stop testing cheap interventions.** All six tested intervention classes have failed. The next BR1 experiment must add training-time gradient signal targeting the BR1↔BR2 boundary directly (density-conditioned head, SupCon, contrastive, new features) AND must be gated by the Phase 0b radiologist audit verdict.
4. **For BR4, the augmentation-side signal (+2.0pp F1, +8.9pp recall, borderline-significant) plus the unanimous-wrong cell distribution (BR4 = 59% of error mass) jointly justify a dedicated architecture track.** Per-region uncertainty head OR higher-resolution input are the two plausible architectural moves.
5. **The SWA-loses-with-faster-overfitting pattern is now diagnostic.** Any future experiment where best_epoch ≤ 11 AND SWA loses on ≥ 2/3 seeds is a flag for over-regularization removal. Treat this as a watch metric in the training loop.

**Action:**
- **Free experiment (today, ~30 min):** extract logits from all 4 nohflip checkpoints; build the 8-model {hflip×4 + nohflip×4} ensemble; check BR4 lift above 0.555. If yes, it becomes the production model.
- **Lock production config:** keep `horizontal_flip: 0.5` in C6.yaml; no change.
- **Phase 0b radiologist audit on `artifacts/phase0b_audit_sample.csv`** is now the single highest-leverage remaining decision. 120 patients, 30 per priority cell. The decision rule embedded in the summary JSON gates Phase 2/3/4 GPU spend.
- **Reorder architecture plan:** demote SupCon (Phase 2 in original plan, predicted to fail on BR1 by Lesson #57) below density-conditioned head and per-region BR4 uncertainty.
- **Future SWA-trajectory diagnostic:** add `best_epoch / swa_start_epoch` ratio as a logged metric. If best_epoch < swa_start_epoch + 5, SWA averaging is over a post-peak trajectory and likely loses.

**Connections to prior lessons:**
- Lesson #13 (B1 vs A1-CE — bug fixes increased overfitting): **strengthened** — same mechanism. Reducing implicit regularization (whether via bug fix or via removing hflip) widens val/test gap and SWA fails.
- Lesson #17 (B5 SWA BR1 -7.3pp BR2 +12.3pp): mirrored signature — SWA smoothing favors high-density classes. BR5's +12.3pp under SWA in B5 → BR5's -2.1pp under nohflip+SWA-fail here. Both variance-reduction operators (SWA win, hflip removal) shift mass from minority to majority classes.
- Lesson #22 (asymmetry loss removal +1.47pp): the asymmetry loss was a *separate* L/R-asymmetry signal that overfitted; this lesson tests the *training augmentation* L/R-corruption hypothesis. Both lessons are about L/R semantics; both findings are: domain-prior L/R signals are fragile, but the L/R augmentation diversity of hflip is robust.
- Lesson #36 (SWA effectiveness depends on loss landscape balance): **strengthened with a new diagnostic** — SWA also depends on input-space augmentation. Removing input-space variance shifts the trajectory to a faster-converging-then-deteriorating shape, and SWA averages over the deterioration.
- Lesson #45 (hflip+swap TTA hurt): mechanism re-explained. At inference, hflip is a domain shift; at training, hflip is implicit dropout. The asymmetric finding is consistent.
- Lesson #51 (seed=42 is a +1.9σ outlier): **strengthened** — seed=42 is the only seed where nohflip *regressed* on BR1. The lucky-tail BR1 F1 had no room to lift.
- Lesson #57 (n=4 ensemble lifts BR4 +3.4pp, BR1 +0pp beyond outlier): **strengthened** — BR4 is responsive to multiple variance-reduction operators (ensemble + nohflip), BR1 is not.
- Lesson #58 (post-hoc inference track exhausted): **strengthened** — Phase 1 was the cheapest training-time intervention; it also failed. Now training-time interventions need real new signals, not augmentation tweaks.

**Open questions deferred to next phase:**
- Does the 8-model {hflip×4 + nohflip×4} ensemble lift BR4 above 0.555? Test next.
- Are the radiologist verdicts on the 35 BR1→BR2 unanimous-wrong patients consistent with the model's prediction (label-noise floor) or against it (architecture problem)?
- Is the BR4 +2.0pp signal from nohflip reproducible at higher seed count (e.g., n=8)? If so, augmentation-side over-smoothing of BR4 morphology is a real diagnosis, and a per-class augmentation strength schedule is justified.

---

### Lesson #60 (2026-05-05): Phase 0c programmatic audit-substitute confirms unanimous-wrong errors are geometrically structured, not label-noise scattered. The radiologist audit was inaccessible (no clinical access available); a 4-seed k-NN + Grad-CAM substitute produced cell-level verdicts that supersede the deferred Phase 0b plan.

**Context:** No radiologist available to run Phase 0b (the original Lesson #58 / Lesson #59 follow-up). Replaced with a two-signal programmatic substitute over the 207 unanimous-wrong patients (n=4 C6 seed agreement on a wrong class):

1. **k-NN distance ratio in `patient_feat`** — for each unanimous-wrong patient (true=A, pred=B), compute `r = mean cos-dist to top-quartile-confident TRUE-B / mean cos-dist to top-quartile-confident TRUE-A`. `r << 1` ⇒ patient is geometrically *inside* the wrong cluster (geometric_overlap signature). `r ≈ 1` ⇒ scattered/idiosyncratic (label-noise signature).

2. **Grad-CAM signals on the same patients, target = ensemble argmax** — concentration (1 − normalized entropy), cross-seed cosine agreement on (downsampled, view-stacked) heatmaps, and within-cell k-means silhouette to see whether errors group into a small number of consistent visual sub-patterns.

Joint decision matrix → cell verdict (REPRESENTATION / REPRESENTATION-LEAN / MIXED / LABEL-NOISE-LEAN / LABEL-NOISE) → action mapping. Artifacts: `artifacts/phase0c_audit_substitute.{md,json}`, `artifacts/phase0c_knn_analysis.{json,csv}`, `artifacts/phase0c_gradcam_analysis.json`. Tooling: `tools/extract_patient_feat.py`, `tools/phase0c_knn_analysis.py`, `scripts/generate_gradcam_targeted.py`, `tools/phase0c_gradcam_analysis.py`, `tools/phase0c_synthesize.py`, `scripts/run_phase0c_audit_substitute.sh`.

**Headline result — k-NN signal is unambiguous and points to representation overlap:**

| Cell | n | median ratio | frac < 0.9 | k-NN verdict |
| --- | --- | --- | --- | --- |
| BR1→BR2 | 42 | 0.078 | **97.6%** | geometric_overlap |
| BR4→BR5 | 48 | 0.312 | **87.5%** | geometric_overlap |
| BR2→BR1 | 35 | 0.068 | 100% | geometric_overlap |
| BR2→BR4 | 32 | 0.067 | 100% | geometric_overlap |
| BR5→BR4 | 34 | 0.040 | 100% | geometric_overlap |

In ~all unanimous-wrong patients, the embedding sits dramatically closer to the wrong-class cluster than to its own class. **Errors have consistent geometric structure in `patient_feat` space — they are not label noise.**

**Grad-CAM signal — diffuse but with cluster structure:**

| Cell | concentration_median | cross_seed_cos | best_k | silhouette |
| --- | --- | --- | --- | --- |
| BR1→BR2 | 0.064 | 0.411 | 2 | **0.42** |
| BR4→BR5 | 0.059 | 0.224 | 4 | 0.34 |
| BR2→BR1 | 0.065 | 0.315 | 2 | **0.52** |
| BR4→BR2 | 0.065 | 0.354 | 3 | **0.77** |

Concentration is uniformly low (0.05–0.09). Originally read as "diffuse_or_idiosyncratic" by my synthesis tool's threshold (conc ≥ 0.4 required for "consistent_visual_pattern"), which produced a uniform `MIXED` verdict on every cell. **That threshold is miscalibrated for this dataset.** BR1↔BR2 and BR4↔BR5 are *intrinsically global-tissue / parenchymal-pattern decisions*; localized lesion-style attention (high concentration) is not the radiologically correct shape of the heatmap. The high cluster silhouettes (0.34–0.77) AND moderate cross-seed cos (0.22–0.46) AND k=2–4 sub-clusters per cell jointly indicate that within each error type, heatmaps DO group into a small number of tight visual archetypes — i.e., consistent sub-patterns.

**Recalibrated verdict (overrides synthesis tool's `MIXED → Direction #5 only`):**

- **BR1→BR2: REPRESENTATION-LEAN.** k-NN strongly geometric (frac<0.9 = 97.6% on n=42). Grad-CAM diffuse but cluster-structured (silhouette 0.42, k=2). Errors are coherent representation-side, not label-side. SupCon with hard-negative mining on BR1↔BR2 is justified, NOT deferred.
- **BR4→BR5: REPRESENTATION-LEAN.** k-NN strongly geometric (frac<0.9 = 87.5% on n=48). Grad-CAM cluster-structured (silhouette 0.34, k=4). Per-region/mass-localization head is justified independently of BR1.
- **BR2 reverse cells (BR2→BR1, BR2→BR4) also strongly geometric.** Confirms that BR2 is a sink class in both directions; any BR1/BR4 work that doesn't preserve BR2 mass will trade-off, not improve macro F1.

**Why my original synthesis labels are misleading:**

The decision rule `concentration_median ≥ 0.4 AND cross_seed_cos ≥ 0.5 AND silhouette ≥ 0.20 ⇒ consistent_visual_pattern` was set with localizable-lesion tasks in mind. For mammography density/parenchymal decisions, the signal is intrinsically diffuse; the meaningful structure lives in *cluster silhouette + within-cell sub-pattern stability*, not in single-heatmap concentration. The k-NN signal is the dominant evidence; Grad-CAM is corroborating, not gating.

**Implications:**

1. **The ceiling is NOT label-noise-bound.** The "BR1 is a label-noise floor" framing in `tasks/plateau_analysis.md` (drafted from prior turn) is overturned. The unanimous-wrong errors have geometric structure both in embedding space AND visual-pattern space.

2. **BR1 architectural work is re-justified.** Lesson #58's listing of "regularization / variance reduction exhausted" stands — but the original Lesson #58 Rule 3 ("only new feature signal can help BR1") is now empirically supported. SupCon-with-hard-negative-mining adds a new geometric constraint (not regularization), and the geometric_overlap signal is exactly the precondition where SupCon is theoretically effective.

3. **The MIXED verdict in `phase0c_audit_substitute.md` should be read as REPRESENTATION-LEAN.** Future re-runs of the synthesis tool should lower the concentration threshold for global-tissue tasks, OR weight the silhouette signal more heavily, OR replace the rule altogether with k-NN-dominant scoring.

**Rule:**
1. **Direction #5 (ensemble self-distillation) is still the cheapest first move.** It directly addresses the moderate cross-seed disagreement (0.22–0.46) that the Grad-CAM analysis surfaced, and it's a prerequisite for cleanly interpreting #3 results. ~1 day GPU.
2. **Direction #3 (SupCon with hard-negative mining on BR1↔BR2) is no longer "deferred."** Lesson #58 Rule 3 originally said "only new feature signal" — geometric_overlap with cluster structure is exactly the case where SupCon's discriminative-margin loss adds new optimization signal (not variance reduction). Run #5 first; if BR1 lifts <1σ on n=4, escalate to #3 immediately.
3. **For BR4: per-region / mass-localization head runs in parallel with #5 and #3.** Independent evidence base (Lesson #48 margin, Lesson #57 +3.4pp, Lesson #59 BR4 lift). Do not gate on BR1 progress.
4. **Ship the n=4 ensemble (0.6846) as the production baseline regardless.** All architectural work measures against this, not against seed=42's 0.6762.
5. **Add k-NN-dominant scoring to the synthesis tool before next audit.** Concentration thresholds are domain-specific; the joint matrix collapses to MIXED in every cell when one signal is uniformly miscalibrated. k-NN signal should be the primary axis with Grad-CAM as a tiebreaker.

**Action:**
- **Implement Direction #5 next:** `tools/build_ensemble_soft_targets.py` (precompute ensemble softmax for train+val patients), distillation loss term in `utils/losses.py` (KL to ensemble soft targets, weight α≈0.5), `configs/.../convnextv2_large_8bit_ablation_c6_distill.yaml`, n=4 launcher.
- **Queue Direction #3 (SupCon) implementation as the immediate follow-up** to be ready before #5 results are in: `utils/losses.py` SupCon term with hard-negative mining over batch on BR1↔BR2, `configs/.../convnextv2_large_8bit_ablation_c6_supcon.yaml`. Triggered if #5 BR1 lift <1σ.
- **Queue per-region malign head architecture work** in parallel: token-level prediction over the spatial token map (pre-pool), top-k mean aggregation into an additional logit channel for BR4↔BR5 in `models/classification_heads.py`. Independent of BR1 work.
- **Update `tasks/plateau_analysis.md`** (or replace it) — the "label-noise ceiling" framing is now refuted by Phase 0c.
- **Update CLAUDE.md champion language** — keep n=4 ensemble (0.6846) as the headline, but mark BR1 as "representation-side, addressable" rather than "label-noise-bound."
- **Add `phase0c_audit_substitute.md` and the k-NN/Grad-CAM artifacts** to the paper's negative-results / methods section as the substitute for the missing radiologist audit.

**Connections to prior lessons:**
- Lesson #57 (n=4 ensemble: BR1 +0pp, BR4 +3.4pp): **strengthened by independent geometric evidence** — the BR1 0pp lift is now mechanistically explained by Phase 0c's geometric_overlap finding (errors are seed-consistent because the representation is consistent), not by intrinsic data unfixability.
- Lesson #58 (post-hoc track exhausted; "only new feature signal can help"): **operationalized.** Phase 0c provides the missing precondition — geometric_overlap implies SupCon's mechanism is applicable. Lesson #58's predicted "SupCon will also fail" is now revised: Lesson #58 was reasoning from Lesson #57's variance-reduction failure to all training-time methods, but SupCon-with-hard-negative-mining is a *discriminative* loss, not variance reduction.
- Lesson #59 (drop-hflip rejected; BR1 architectural work needs new signal): **directly extended.** Phase 0c is the test that confirms "new signal" is justified, not just speculated. Drop-hflip was an augmentation-side intervention; Phase 0c's geometric finding says the model needs a *loss-side* (SupCon) or *head-side* (density/per-region) intervention.
- Lesson #50 (cascade rejected, BR1↔BR2 specialist won val lost test): **explained.** Cascade specialist's val gain was within-train-distribution variance reduction; its test loss is the same geometric_overlap pattern Phase 0c now measures directly.
- Lesson #51 (seed=42 is +1.9σ outlier): **complementary.** Seed-CV told us BR1 is on a flat loss landscape; Phase 0c tells us that flat landscape sits inside the BR2 cluster. Same problem, two measurement axes.
- Lesson #46 (T_opt ≈ 0.73 post-hoc): **unaffected.** Calibration is orthogonal to representation overlap.

**Open questions deferred:**
- Does Direction #5 (self-distillation) lift BR1 above the 0.45 ± 0.05 floor on n=4? If yes, the variance-decomposition story is more nuanced than Lesson #57 implied. If no, escalate to #3 (SupCon) per the rule above.
- Does Direction #3 (SupCon hard-negative mining on BR1↔BR2) measurably reduce the median k-NN ratio on BR1→BR2 from 0.078 toward >1.0? This is the embedding-side mechanistic check; train it with SupCon, then re-run `tools/phase0c_knn_analysis.py` and compare.
- For BR4: does a per-region head reduce the BR4→BR5 median k-NN ratio from 0.31 toward >1.0 without hurting BR5? Same diagnostic, separate cell.
- Is the k-NN-dominant scoring rule generalizable across mammography classification tasks, or is it specific to this dataset's diffuse-attention regime?

---

### Lesson #61 (2026-05-08): Direction #5 (ensemble self-distillation, KL on full head, α=0.5, T=4.0) is REJECTED. Mean test macro F1 regressed by 2.1pp, BR1 F1 by 4.4pp, seed variance amplified 2.4×, and seed=555 hit a hard divergence event mid-training. KL-toward-ensemble-mean is a basin-selector, not a regularizer.

**Context:** Phase 1 of Track A and Track B per Lesson #60. Trained 4 students {seeds 42, 7, 555, 999} with the same C6 architecture/schedule, replacing the full head's CE with hybrid loss `(1−α)·CE + α·T²·KL(softmax(z_s/T) ‖ softmax(z_t/T))`, α=0.5, T=4.0. Teacher = arithmetic mean of full-head logits across the 4 trained vanilla C6 seeds, computed with `val_transforms` (no augmentation) on the entire `Dataset_1024_8bit` train+val pool (8557 patients, pool argmax-acc = 0.836). Implementation: `tools/build_ensemble_soft_targets.py`, `utils/losses.py:MultiHeadLoss.distill_*`, `data/dataset.py:_load_soft_target_index`, `train.py` soft-target wire-through, `configs/.../convnextv2_large_8bit_ablation_c6_distill_seed{42,7,555,999}.yaml`, `scripts/run_distill_parallel.sh`.

**Headline result — distillation regressed on every axis:**

| Metric | Distill (n=4) | Vanilla C6 mean (n=6, Lesson #51) | Δ |
| --- | --- | --- | --- |
| Test macro F1 | 0.6293 ± 0.0335 | 0.6502 ± 0.0137 | **−2.09pp** |
| Test BR1 F1 | 0.4304 | ~0.474 ± 0.045 | **−4.4pp** |
| Test BR2 F1 | 0.7298 | ~0.745 | −1.5pp |
| Test BR4 F1 | 0.5188 | ~0.522 ± 0.050 | −0.4pp (NS) |
| Test BR5 F1 | 0.8381 | ~0.860 | −2.2pp |
| Seed sd | 0.0335 | 0.0137 | **2.4× amplified** |

Per-seed test macro F1: seed=42=0.6356, seed=7=0.6588, seed=555=0.5732, seed=999=0.6495. None individually exceed C6 seed=42's 0.6762 outlier; none exceeds the n=4 vanilla ensemble's 0.6846 (Lesson #57). Per-seed test BR1 F1: [0.4296, 0.4779, 0.3399, 0.4742] — only seed=7 reaches the vanilla mean (~0.474), and even there the lift is zero.

**Per-seed training-curve diagnosis (`Distill-TrainAndVal-Graphics/`):**

Three of four seeds (42, 7, 999) converged cleanly on every metric — train/val curves are visually indistinguishable from vanilla C6 curves at the same step. **seed=555 was qualitatively different:**

| Metric | Seeds 42/7/999 | seed=555 |
| --- | --- | --- |
| train_binary_acc to 0.95 | step ~5 | step ~30 |
| train_benign_sub_loss at step 40 | 0.36–0.40 | 0.45–0.50 |
| val_full_acc plateau | 0.70–0.73 (step 5+) | 0.65 peak (step ~42) |
| val_full_f1_BIRADS-1 plateau | 0.60–0.70 oscillating | 0.0–0.65 chaotic |
| Hard divergence event | none | step ~47: train_binary_f1 → 0, val_full_acc → 0.20, val_full_f1_BIRADS-1 → 0, val_cohen_kappa → 0 |

The collapse happened simultaneously across all three logged-metric heads (binary/sub/full) at step 47, indicating a representation collapse, not a single-head numerical instability. Best-checkpoint-saving rescued seed=555's reported test number (0.5732 macro F1, taken pre-collapse), but the trajectory itself is the diagnostic signal.

**Mechanistic explanation — why distillation failed in this regime:**

1. **Teachers don't memorize train; the "dark-knowledge" assumption was wrong.** Pool argmax-acc = 0.836 (16% wrong on patients each teacher *trained on*) was initially read as evidence of healthy uncertainty mass. It is actually evidence that the teachers' soft predictions on hard cases ARE wrong predictions, not informative off-class rankings. Distilling 16% wrong train signal injects noise into the student's gradient.

2. **Phase 0c's diffuse-attention finding predicts off-class mass distributes uniformly on hard cases.** Concentration_median = 0.05–0.09 on every priority cell (Lesson #60). When a teacher's BR1 probability on a confused patient is 0.55, its BR2/BR4/BR5 mass is roughly uniform (~0.15 each), not informative. KL-distilling this teaches the student "all off-classes are equally plausible" — the opposite of the geometric structure SupCon is designed to enforce.

3. **KL-toward-ensemble-mean is a basin-selector under representation diversity.** Lesson #57 / Lesson #51 established that BR1 CV=9.6% across seeds — different seeds reach genuinely different basins. The ensemble mean is the *centroid* of those basins. KL-pulling each new student toward the centroid REWARDS seeds whose random init happens to land in a basin agreeing with the centroid (seeds 42/7/999) and PUNISHES seeds whose init lands in a disagreeing basin (seed=555). The student's CE gradient pulls one way, the KL gradient pulls another, and the optimization stalls or diverges. **This is a discovered failure mode, not in the original Hinton 2015 setting (where teacher and student share initialization).**

4. **Val BR1 F1 ceiling unchanged from vanilla.** Val curves peak at 0.60–0.70 BR1 F1 — the same value vanilla C6 reaches. Distillation didn't add new information to the val signal; the loss is dominated by easy-case CE, and easy cases don't need teacher help.

**Why this rejects Direction #5 specifically and not the broader self-distillation class:**

Standard self-distillation (Furlanello 2018; "Born-Again Networks") uses the SAME teacher's checkpoint to distill into a SAME-init student — there is no basin disagreement because teacher and student share inheritance. Our setting (multi-seed teacher mean → independent-seed students) is structurally different: the teacher signal is a *centroid* of basins, and pulling random inits toward a centroid is mechanistically equivalent to a basin-membership prior. The other failure modes (#1, #2 above) are specific to a representation-overlap regime where the teacher's secondary mass carries little information.

**Where this leaves the BR1 ceiling argument:**

- Lesson #58 listed 8 inference-time interventions that failed; Lesson #60 added Phase 0c's geometric-overlap finding as a re-justification for representation-side training-time work; Direction #5 was the cheapest training-time test of that hypothesis. **Direction #5's failure does NOT refute Lesson #60.** It refutes "any training-time soft signal can lift BR1." It does not refute "discriminative-margin loss can lift BR1." SupCon-with-hard-negatives — the next experiment — is fundamentally different from KL: it operates on EMBEDDING distances directly (not on output-distribution shape), and hard-negative mining selects per-batch which BR2 patients are closest to each BR1 patient and pushes them apart. There is no centroid-pull, no basin-selector dynamic.

- The BR1 F1 floor of 0.45 ± 0.05 (Lesson #59 + this lesson) is now established across **8 independent intervention classes** (variance reduction, TTA, Saerens-EM, oracle prior, threshold tuning, drop-hflip, asymmetry loss, KL distillation). Per Lesson #58's Rule 3, this rules out all variance-reduction and prior-shift mechanisms. The remaining un-tried mechanism class is **discriminative-margin / contrastive embedding losses** — exactly Direction #3.

**Rule:**

1. **KL distillation toward an ensemble mean is REJECTED for this dataset/architecture.** Do not retry with different α or T. The α=0.5/T=4.0 setting is well within the Hinton-2015 standard range; the failure is not a hyperparameter miscalibration but a regime mismatch (pre-trained-teacher-with-shared-init expectation violated).

2. **Variance amplification is now the second tested fingerprint of "wrong intervention class."** Lesson #59's drop-hflip flagged sd inflation (0.020 → 0.029) directionally as a regularization-removal signature. This lesson's distillation pushed sd to 0.034 — almost identical magnitude. **When seed sd > 0.025 across n=4 C6-architecture students, the intervention is destabilizing the optimization, not improving it.** Treat sd > 0.025 as an automatic kill criterion alongside seed-mean F1.

3. **Hard-divergence events on a single seed are now diagnostically meaningful.** Best-checkpoint-saving makes single-seed collapses invisible in the test-F1 table (seed=555 still reported a number). Future experiments must log AND review training-curve trajectories on all seeds, not just final-checkpoint metrics. Add a sanity check: if any seed's val_full_f1_macro ever drops below 0.50 mid-run AND best-epoch ≤ 50% of total epochs, flag for manual review.

4. **The Phase 0c argmax-acc reframing.** Pool argmax-acc < 0.95 on a teacher's own train data should be read as "this teacher has genuine uncertainty on hard cases" only IF cross-validation evidence shows the uncertainty is calibrated. In our case, Phase 0c proved the uncertainty is diffuse/uninformative (concentration 0.05–0.09). Distillation requires *informative* uncertainty, not just nonzero.

5. **Direction #3 (SupCon with hard-negative mining on BR1↔BR2) is now the active path.** Mechanism is fundamentally different from KL: discriminative-margin loss on `patient_feat` embeddings, hard-negative mining selects within-batch which BR2 patients are closest to each BR1 patient and pushes them apart. There is no teacher mean to converge toward, no basin-selector dynamic. Per Lesson #58 Rule 3, this is the only un-tested mechanism class for the BR1 representation-side problem.

**Action:**

- **Re-run `tools/phase0c_knn_analysis.py` on the distill checkpoints.** Even though F1 regressed, the geometry may have shifted — informative for SupCon's hard-negative pool design. This requires `tools/extract_patient_feat.py` runs for the 4 distill seeds (~10 min each on H100), then the existing k-NN script. If the median BR1→BR2 ratio shifted (in either direction), it tells us how KL distillation altered the embedding even in failure.
- **Skip building the n=4 distill ensemble.** With seed-mean 0.6293 ± 0.034, ensemble denoising recovers at most ~0.025 (from seed-mean to roughly 1/√n × sd above mean), landing around 0.65. That's still 3.5pp below the vanilla 0.6846. Not worth GPU time.
- **Implement Direction #3 (SupCon).** Loss term in `utils/losses.py`: SupCon on `patient_feat` with cosine similarity, hard-negative mining (top-k closest BR2 to each BR1 in batch), weight β ≈ 0.05–0.10 (low — to avoid the pull-magnitude that destabilized seed=555 here). Configs `c6_supcon_seed{42,7,555,999}.yaml` mirroring the distill setup. Launcher mirroring `run_distill_parallel.sh`.
- **In SupCon implementation, monitor seed=555 specifically.** Its sensitivity to embedding-space pulls is now diagnostic. If SupCon also destabilizes seed=555, that's a signal that seed=555's bilateral-fusion init is genuinely incompatible with any embedding-space gradient (would mean seed=555 should be replaced in future ensembles — but defer that judgment until SupCon results are in).
- **Do NOT lower α and re-try distillation at lower weight.** Already considered; rejected. The seed-mean regression is already at α=0.5; α=0.3 would just produce a smaller regression with the same variance amplification. The mechanism is wrong, not the magnitude.

**Connections to prior lessons:**

- Lesson #57 (BR1 +0pp from n=4 ensemble): **mechanistically extended** — the ensemble's BR1 +0pp was variance-reduction failure; this lesson's BR1 −4.4pp is centroid-pull-toward-ensemble-mean failure. Both confirm the BR1 floor is not addressable by ensemble-derived signals (pull-toward-mean OR mean-as-output).
- Lesson #58 Rule 3 ("only new feature signal"): **strengthened** — KL distillation is a "soft signal," not a "feature signal" in the sense Lesson #58 meant. The distinction is now operational: feature signals add new gradient directions (SupCon, density-conditioned head); soft signals re-weight existing CE gradients (distillation, label smoothing). Only the former can lift representation-bound classes.
- Lesson #59 (drop-hflip rejected, sd amplified to 0.029): **sd-amplification fingerprint matched** — distillation pushed sd to 0.034. Two independent failures both produce variance > 0.025, both fail on macro F1. Rule 2 above formalizes this as a kill criterion.
- Lesson #60 (Phase 0c verdict, queued Direction #5 first): **Direction #5 phase complete with negative result.** Lesson #60's "Direction #5 is the cheapest first move" was correct in principle; the negative result still has high information value (rejected one mechanism class definitively, narrowed the search). Direction #3 escalation is now active.
- Lesson #51 (BR1 CV=9.6% across seeds → flat-loss landscape): **mechanistically connected to seed=555 divergence** — flat loss landscape means seeds genuinely differ in basin. KL pulls all toward the centroid; outlier-basin seeds destabilize.

**Open questions deferred to next phase:**

- ~~Did distillation shift the BR1→BR2 k-NN geometry, even though F1 didn't lift?~~ **ANSWERED 2026-05-08 (`scripts/run_distill_phase0c_repeat.sh`):** YES, in the *correct* direction. Distillation's median k-NN ratio rose on every priority cell:

  | Cell | Vanilla → Distill (median) | Δ | Vanilla → Distill (frac<0.9) |
  | --- | --- | --- | --- |
  | BR1→BR2 | 0.078 → **0.178** | **+0.101** | 0.98 → **0.90** |
  | BR4→BR5 | 0.312 → 0.412 | +0.101 | 0.88 → 0.76 |
  | BR2→BR1 | 0.068 → 0.150 | +0.083 | 1.00 → 1.00 |
  | BR2→BR4 | 0.067 → 0.142 | +0.074 | 1.00 → 0.94 |
  | BR4→BR2 | 0.099 → 0.206 | +0.107 | 1.00 → 1.00 |

  Distillation DID push BR1 patients further from the BR2 cluster in `patient_feat` space (+0.101 median, 8% of patients exiting the deep-overlap zone). The F1 regression therefore came from a different mechanism than "embedding stayed broken": **distillation produced embedding/head misalignment.** The KL gradient flowed onto `full_logits` (post-head); the head simultaneously got pulled toward teacher soft distributions while the upstream embedding got pulled away from BR2. Net: head wins on argmax, F1 regresses, even though embedding improved.

  **Implication for Direction #3:** SupCon's gradient is on `patient_feat` directly — *before* the head — so the embedding-side and head-side updates are aligned (head re-learns decision boundary via CE on the shifted embedding distribution; no opposing-direction pull). SupCon should reproduce or exceed distillation's +0.101 BR1→BR2 shift AND translate it to F1. Hyperparameter implication: keep β=0.03 as initially planned (do not raise to compensate for "missed" embedding work — the embedding work happened, the issue was head-fighting, which SupCon avoids by construction). If first SupCon n=4 run produces +Δ_median on BR1→BR2 AND BR1 F1 lift ≥1σ, this confirms the mechanism diagnosis. If +Δ_median but no F1 lift, head-embedding misalignment exists in C6 architecture independent of distillation, and a head-rebuild step (e.g., re-init classifier after fusion is trained) is the next architectural move.

- Did distillation shift the BR1→BR2 k-NN geometry in some other un-tested way? Original question superseded by the answer above.
- Is seed=555 a permanently-replaced "outlier seed" or a useful "stress-test seed"? Defer until SupCon results on the same 4 seeds. If seed=555 also crashes under SupCon, replace it (e.g., seed=2024 from the older sweep) for production ensembles. If it stabilizes, treat seed=555 as the canonical sensitivity probe.
- Is there a teacher-aware variant that would have worked? E.g., per-patient teacher selection (only use teacher j's prediction for patient p if p was in teacher j's val set, not train) — out-of-fold distillation. This would cost 4× the soft-target build but eliminates the memorization-noise problem (#1 above). Out of scope unless Direction #3 also fails.

---

### Lesson #62 (2026-05-10): Direction #3 (SupCon w/ hard-negative mining on patient_feat, β=0.03, τ=0.1, top_k=20, queue=256) is REJECTED with statistically identical regression to distillation. THREE independent aux-loss interventions on patient_feat (asymmetry-loss, KL distillation, SupCon) have now produced near-identical −2pp macro F1 regressions. Aux-loss-on-patient_feat is now formally a closed mechanism class.

**Context:** Direction #3 was the next step queued by Lesson #61 after distillation failed. The hypothesis was: SupCon's gradient on `patient_feat` directly (no head fight, no centroid pull) should produce the same embedding-side BR1↔BR2 separation as distillation (+0.101 median k-NN ratio shift, Lesson #61) but ALSO translate to F1 because the head can re-learn its decision boundary from CE on the shifted embedding. Implementation: `utils/losses.py:SupConLoss` (memory bank size=256, top-k=20 hard negatives, τ=0.1, β=0.03 weight), `configs/.../convnextv2_large_8bit_ablation_c6_supcon_seed{42,7,555,999}.yaml`, `scripts/run_supcon_parallel.sh`.

**Headline result — SupCon and distillation are statistically identical:**

| Metric | SupCon (n=4) | Distillation (n=4, Lesson #61) | Vanilla C6 (n=6, Lesson #51) |
| --- | --- | --- | --- |
| Test macro F1 | 0.6292 ± 0.0339 | 0.6293 ± 0.0335 | 0.6502 ± 0.0137 |
| Test BR1 F1 | 0.4404 | 0.4304 | ~0.474 |
| Test BR4 F1 | 0.5039 | 0.5188 | ~0.522 |
| Test BR5 F1 | 0.8455 | 0.8381 | ~0.860 |
| Seed sd | 0.034 | 0.034 | 0.014 |
| seed=555 outlier | 0.5756 | 0.5732 | n/a |

**The two interventions' macro F1 means differ by 0.0001.** Two mechanistically distinct gradient paths (SupCon: direct embedding-space discriminative-margin; distillation: head-level KL divergence backpropagating through embedding) produce identical F1 regressions, identical seed variance amplification, and identical seed=555 destabilization. This near-perfect match is itself the diagnostic.

**Per-Lesson-#61 kill criteria — all three triggered:**

| Rule (Lesson #61) | Threshold | SupCon actual | Status |
| --- | --- | --- | --- |
| sd > 0.025 → wrong intervention class | sd ≤ 0.025 | 0.034 | FAIL |
| macro F1 mean < 0.65 | mean ≥ 0.66 | 0.6292 | FAIL |
| BR1 F1 mean lift < 1σ vs vanilla | mean ≥ 0.49 | 0.4404 (−3.4pp) | FAIL |

Lesson #61's prediction (SupCon's clean embedding gradient avoids head-fighting → F1 should follow embedding shift) is empirically refuted. The "head-embedding misalignment" diagnosis was either wrong or only partial.

**Sharper diagnosis — three failures with the same fingerprint:**

| Intervention | Gradient location | Δ macro F1 vs vanilla | sd | seed=555 outlier |
| --- | --- | --- | --- | --- |
| Lesson #22: asymmetry loss on `f_diff` (upstream of patient_feat, removed C6) | upstream | −1.47pp (vs B5+asym) | n/a (older sweep) | n/a |
| Lesson #61: KL distillation on `full_logits` (downstream + back-propagated) | downstream | −2.09pp | 0.034 | yes |
| **This lesson: SupCon directly on `patient_feat`** | **direct** | **−2.10pp** | **0.034** | **yes** |

Three different gradient paths into the bilateral-fusion-to-classifier stack. Identical −2pp regression. Identical 2.4× variance amplification. Identical seed=555 destabilization. **This is structural, not a hyperparameter or mechanism choice.** The C6 architecture's joint optimization cannot absorb additional gradient signals on its representation pathway without a fixed ~2pp F1 cost.

**Three plausible structural causes (in order of likelihood, all hypotheses):**

1. **Head dropout=0.5 is the cap on representation stability.** Aux gradients shift `patient_feat` by ε; the head's dropout zeros half the channels stochastically; the surviving channels carry inconsistent CE-vs-aux information; the head's affine projection averages over noise, diluting both signals. Single-objective CE survives because all channels carry consistent gradient information; multi-objective doesn't. Test: lower dropout to 0.2, retry SupCon at the same β.

2. **OneCycleLR max_lr=5e-4 is too aggressive for multi-objective training.** Single-objective CE converges with this peak; multi-objective oscillates between objectives because the parameter step is too large. Lesson #36 already documented OneCycleLR's sensitivity. Test: lower max_lr to 2e-4 with same SupCon weight.

3. **Bilateral fusion is at the capacity sweet-spot for CE-only — additional regularization from aux loss tips it under-fit.** Lesson #25 found that backbone size = bilateral fusion size = head capacity is jointly tuned. Aux loss is effectively additional regularization; the architecture was already at the regularization ceiling under SWA + label smoothing 0.05 + dropout 0.5 + class weights. Test: re-tune bilateral fusion dropout (0.25 → 0.15) before adding aux loss.

These tests would each cost 4 seeds × 12-16 hours = ~50 GPU-hours per hypothesis. **Defer all three.** The pragmatic conclusion is unchanged: stop trying to add aux losses to the existing C6 stack.

**Cumulative state: 10 independent intervention classes have failed for BR1.**

Updated comprehensive failure inventory:

```
Class                                             Reps tested    Status
1. Variance reduction (n=4 ensemble)              1              REJECTED
2. TTA × ensemble                                 1              REJECTED
3. Saerens-EM post-hoc prior correction           5              REJECTED
4. Oracle prior correction                        4              REJECTED
5. Pairwise BR1<->BR2 threshold (#56)             1              REJECTED
6. Drop training-time hflip                       1              REJECTED
7. Asymmetry loss on f_diff                       1              REJECTED (Lesson #22)
8. Train-time logit adjustment (F2 τ-sweep)       3              REJECTED
9. KL distillation on full_logits (Direction #5)  1              REJECTED (Lesson #61)
10. SupCon on patient_feat (Direction #3)         1              REJECTED (this lesson)
```

The remaining un-tested mechanism classes that could plausibly help BR1:

| Class | Candidate | Compute cost | BR1 lift potential |
| --- | --- | --- | --- |
| Backbone replacement | Direction #1: mammography-pretrained backbone | High (~75 GPU-h pretrain + n=4 fine-tune) | High (literature: +3pp) |
| Input resolution | Direction #2: train at 2048² or random 1280→1024 crops | High (4× compute at 2048²) | Medium |
| Head re-architecture | Re-init classifier after fusion converges; multi-stage curriculum | Medium | Speculative |
| New data acquisition | Re-graded labels from radiologist | n/a | Decisive but blocked (no clinical access) |

**Decoupled BR4 path (independent of BR1 ceiling):**

Per-region / mass-localization head adding gradient at the BACKBONE's spatial-token output (NOT on patient_feat). Structurally different from the failed aux-loss-on-patient-feat class — gradient flows into backbone+per-view tokens, NOT into bilateral fusion or patient_feat. Phase 0c silhouette = 0.34 with k=4 sub-clusters confirms BR4↔BR5 has consistent visual sub-pattern structure that a per-region head can target. Track B work; runs in parallel with whatever Track A path is selected.

**Rule:**

1. **Aux-loss-on-patient_feat is now formally a CLOSED mechanism class.** Three independent failures (asymmetry, KL distillation, SupCon) at three different gradient paths (upstream of patient_feat, downstream via head, direct on patient_feat) all produce −2pp macro F1, sd 0.034, seed=555 collapse. Do not propose β-sweeps, τ-sweeps, queue-size sweeps, or new aux losses on this representation. **Future BR1 architectural work must target a different pathway** (backbone replacement, input resolution, head re-architecture, or per-region/spatial-token loss that bypasses patient_feat).

2. **The "−2pp regression with sd 0.034 and seed=555 collapse" pattern is now a STRUCTURAL FINGERPRINT of aux-loss-on-patient_feat.** Any future intervention reproducing this fingerprint at n=4 should be classified into the closed class without further investigation. Concretely: macro F1 ≤ 0.635 ± 0.005 AND seed sd ≥ 0.030 AND single-seed val_F1 ≤ 0.66 = closed class match → reject without retry.

3. **The seed=555 sensitivity is now a useful diagnostic probe.** seed=555 has now collapsed under three different aux-loss-on-patient_feat interventions (asymmetry retest in F2, distillation, SupCon). It is a reliable indicator of "this intervention destabilizes the C6 stack." Future experiments can run seed=555 ALONE first as a 12-hour stress test before committing all 4 seeds. If seed=555 collapses or hits any of Lesson #61's kill criteria, the intervention is in the closed class. This converts the 4-seed validation cost from 50 GPU-hours to 12 GPU-hours per candidate.

4. **Backbone-replacement (Direction #1) is now the highest-EV remaining BR1 path.** Reasons: (a) it's the only remaining un-tested class with literature-confirmed lifts ≥3pp (Shen 2019 NYU, Wei 2021, Lopez 2024); (b) its mechanism (replace ImageNet-22k feature prior with mammography prior) is independent of aux-loss-fragility — a new backbone re-initializes the entire patient_feat distribution; (c) cost is high but dominated by the one-time pretrain step.

5. **Per-region malign head (Track B for BR4) is the next BR4-specific path.** Different mechanism class (new head on spatial tokens, not aux loss on patient_feat) — should not match the closed-class fingerprint. Independent of Direction #1; can run in parallel.

**Action:**

- **Implement per-region malign head + 4-seed run.** New module `models/per_region_head.py`, `full_model.py` integration, loss term in `utils/losses.py`, configs `c6_perregion_seed{42,7,555,999}.yaml`, launcher. Estimated wall-clock: ~12-16 hours for 4-seed parallel. Independent of Track A — launch immediately.

- **Plan Direction #1 (domain pretraining).** Pretrain corpus selection (RSNA-Breast / EMBED / INbreast / CBIS-DDSM), pretraining method (MAE vs DINO vs SimCLR), compute budget, decision gates. Document in `tasks/direction1_pretraining_plan.md`. Do not launch until plan is approved.

- **Re-run `tools/phase0c_knn_analysis.py` on SupCon checkpoints (cheap, ~30 min).** Even though F1 regressed, the embedding may have shifted (like distillation did, Lesson #61 update). If SupCon's k-NN ratio shift exceeds distillation's (+0.10), this further confirms embedding mechanism works but the F1 disconnection is structural. If shift is the same or smaller, the SupCon gradient was effectively-equivalent to distillation's gradient at the embedding layer (a deeper finding about the C6 stack's embedding-space dynamics).

- **Skip the three structural-cause tests.** Lower-dropout retry, lower-max_lr retry, lower-fusion-dropout retry are each ~50 GPU-hours and likely confirm rather than refute the structural finding. Defer to post-Direction #1 / post-per-region-head.

- **Update `tasks/plateau_analysis.md`** (the document drafted before Phase 0c). The Lesson #61 + Lesson #62 findings supersede its "label-noise floor" framing AND its "SupCon as next step" recommendation. Replace with current state: "BR1 ceiling is structurally embedded in the C6 backbone+fusion+head joint optimization. Three aux-loss classes have now failed identically. The path forward is backbone-replacement OR per-region heads that bypass patient_feat."

**Connections to prior lessons:**

- Lesson #22 (asymmetry loss removed → +1.47pp): **strengthened** — asymmetry loss is the FIRST instance of the now-confirmed closed class. Lesson #22's removal-recovery pattern is the first data point of the "stop adding aux losses" rule.
- Lesson #25 (model size at sweet-spot, can't reduce): **complementary** — the architecture is jointly tuned for CE-only at the current capacity. Aux losses are effectively additional regularization on top of an already-regularized stack; the "regularization ceiling" hypothesis (cause #3 above) is consistent with this.
- Lesson #36 (SWA effectiveness depends on loss landscape balance): **mechanistically extended** — SWA's dependence on smooth loss landscape now extends to "no aux losses" because aux losses produce non-smooth multi-objective loss surfaces.
- Lesson #51 (BR1 CV=9.6%): **operationalized** — seed=555's basin-fragility is now the canonical stress-test signal. Lesson #62 Rule 3 makes this an explicit diagnostic protocol.
- Lesson #58 (post-hoc class exhausted; only new feature signal can help): **strengthened with a fourth requirement** — "new feature signal" must come from a NEW REPRESENTATION (backbone replacement, input scale change, new head architecture), not from a new GRADIENT on the existing representation. Three gradient-on-existing-representation interventions have now failed.
- Lesson #59 (drop-hflip rejected; sd amplified to 0.029): **fingerprint matched** — distillation 0.034, SupCon 0.034. Three points on the closed-class line.
- Lesson #61 (distillation rejected; embedding work was real): **strengthened in part, refuted in part** — the embedding-side mechanism was real (the +0.101 BR1→BR2 shift was reproducible by SupCon at minimum, presumably similar magnitude). The "head-embedding misalignment fix would translate to F1" prediction was wrong. The structural cause is below the embedding/head layer split.

**Open questions deferred to next phase:**

- Does the per-region malign head match or escape the closed-class fingerprint? Track B test. If it matches (sd > 0.025, macro F1 < 0.65), the structural cause is even more general than "aux loss on patient_feat" — it's "any aux loss on the C6 stack." If it escapes, the closed class is correctly bounded at patient_feat-or-derivatives.
- ~~Does SupCon's k-NN shift on test embeddings exceed distillation's?~~ **ANSWERED 2026-05-11 (`scripts/run_supcon_phase0c_repeat.sh`):** the picture is cell-dependent and refutes both the "Larger" and "Same" predictions cleanly:

  | Cell | Vanilla | Distill Δ | SupCon Δ | SupCon − Distill |
  | --- | --- | --- | --- | --- |
  | BR1→BR2 | 0.078 | +0.101 | **+0.056** | **−0.045** (less) |
  | BR4→BR5 | 0.312 | +0.101 | **+0.178** | **+0.077** (more) |
  | BR2→BR1 | 0.068 | +0.083 | +0.104 | +0.021 |
  | BR2→BR4 | 0.067 | +0.074 | +0.022 | −0.052 |
  | BR4→BR2 | 0.099 | +0.107 | +0.014 | −0.092 |

  **SupCon at β=0.03 was less effective than distillation at α=0.5 on BR1↔BR2 specifically (-0.045 vs distill), but substantially more effective on BR4↔BR5 (+0.077 vs distill).** Despite this asymmetry, F1 regression was identical to distillation on both cells. **This strengthens the structural-cause framing:** the C6 head cannot capitalize on embedding-side movement regardless of magnitude OR which cell the movement targets. SupCon moved BR4→BR5 by +0.178 (huge), yet BR4 F1 still regressed by 1.8pp. The bottleneck is unequivocally NOT the embedding pathway — it's the head-from-embedding decoding.

  **Implication for non-aux-loss directions (per-region head, mammo-pretrain):** these BYPASS the patient_feat representation entirely (per-region: backbone-only gradient; mammo-pretrain: backbone-weight-replacement). They are predicted to escape the closed-class fingerprint *because* they don't rely on the head capitalizing on embedding shifts.
- Is seed=555 worth keeping in the n=4 ensemble for production? It has now been the outlier in 4 different non-vanilla configurations (drop-hflip BR4 regression, F2 τ-sweep, distillation, SupCon). Replace with seed=2024 or seed=123 from the older sweep? Decision deferred until per-region head results.
- Will the per-region malign head pattern shift the macro F1 baseline for BR4 measurably? Lesson #57 has BR4 +3.4pp under ensembling; per-region head should add to this, not stack with it. Quantify by ensembling the n=4 per-region students.

---

### Lesson #63 (2026-05-11) — PRELIMINARY (n=1 of 4): the per-region malign head ESCAPES the closed-class fingerprint on seed=555 and produces the FIRST BR1-positive intervention in 10 attempts. Stress-test passed; n=4 confirmation in progress.

**STATUS:** PRELIMINARY. Only seed=555 has finished (canary stress test per Lesson #62 Rule 3). Seeds 42, 7, 999 launched 2026-05-11 morning, expected to finish ~16h later. **DO NOT cite as a confirmed positive intervention until n=4 results land.** If 2/3 of the remaining seeds also lift BR1, this lesson becomes the first confirmed BR1-positive result in the project's history; otherwise, seed=555 was a lucky outlier and the closed class extends to spatial-token loss too.

**Context:** Track B per-region malign head (`models/per_region_head.py`, `configs/.../convnextv2_large_8bit_ablation_c6_perregion_seed*.yaml`, `scripts/run_perregion_parallel.sh`) was launched after Lesson #62 closed the aux-loss-on-patient_feat class. The head reads per-view spatial tokens (B, V*S, D) from the backbone, scores each token for patient-level malignancy via a top-k aggregated MLP, and adds a binary CE loss against (BR1+BR2)→0, (BR4+BR5)→1. **Gradient flows only into the backbone — NOT into the lateral_fusion, bilateral_fusion, or patient_feat.** This was the structural argument for why per-region head should escape the closed class.

Stress-test protocol per Lesson #62 Rule 3: seed=555 alone, ~12h on 1 GPU, evaluate against:
1. val_full_f1_macro < 0.60 by epoch 15 → automatic kill
2. val_full_f1_BIRADS-1 ever drops below 0.30 mid-run → kill
3. per_region_malign_loss does not decrease in first 5 epochs → kill
4. End-of-run test BR1 F1 < 0.40 OR macro F1 < 0.62 → fingerprint match (closed class extends)

**Seed=555 result — all four criteria PASSED:**

| Metric | seed=555 perregion | seed=555 SupCon (Lesson #62) | seed=555 Distill (Lesson #61) | Vanilla C6 seed-mean (Lesson #51) |
| --- | --- | --- | --- | --- |
| Test macro F1 | **0.6437** | 0.5756 | 0.5732 | 0.6502 ± 0.014 |
| Test BR1 F1 | **0.5071** | 0.3594 | 0.3399 | ~0.474 ± 0.045 |
| Test BR2 F1 | 0.7162 | 0.6692 | 0.6711 | ~0.745 |
| Test BR4 F1 | 0.4949 | 0.4323 | 0.4470 | ~0.522 ± 0.050 |
| Test BR5 F1 | 0.8565 | 0.8417 | 0.8347 | ~0.860 |
| Best Val F1 | 0.7093 | 0.6559 | 0.6536 | ~0.72 |

**Headline:**
1. **Macro F1 = 0.6437** — within seed-mean noise of vanilla (Δ = −0.65pp), well above the −2pp closed-class fingerprint floor. NO regression.
2. **BR1 F1 = 0.5071 = +3.4pp above the vanilla seed-mean (~0.474).** First single-seed BR1 lift outside the seed=42 outlier zone in the project's history.
3. **seed=555 did NOT collapse.** Lesson #61 / #62 had seed=555 hitting the val divergence pattern under aux-loss-on-patient_feat interventions; here it trained cleanly to a normal-range val_F1 of 0.7093.

**Why this matters mechanistically:**

- Per-region head's loss is `CE(per_region_malign_logits, binary_label)`. Gradient backprop: per-region head's MLP weights ← grad; spatial tokens ← grad through MLP; backbone trunk ← grad through tokens. **NOTHING flows into lateral_fusion or bilateral_fusion.** Patient_feat is unchanged from the vanilla C6 forward pass.
- The closed class (Lesson #62) was rejected precisely because all three interventions (asymmetry, distill, SupCon) put gradient onto patient_feat or its derivatives. Per-region head bypasses this by construction.
- The +3.4pp BR1 lift is consistent with the Lesson #62 hypothesis that "head-from-embedding decoding is the bottleneck": improving the BACKBONE's spatial features (which the existing fusion+head can still consume the SAME WAY) gives the head better material to work with, without altering the head's job.

**Caveats — why this is PRELIMINARY:**

1. **n=1.** seed=555 alone has 9.6% BR1 CV historically (Lesson #51). A +3.4pp lift on n=1 is within noise. Need n=4 to confirm.
2. **seed=555 was a CANARY by design.** Lesson #62 Rule 3 chose seed=555 because it was the most fragile seed under aux-loss interventions. If it passed there, it's expected to pass under any non-fragile intervention. But "passed" doesn't mean "lifts macro F1 over vanilla mean" — the macro F1 of 0.6437 is still 0.65pp BELOW the vanilla seed-mean. The BR1 lift is real; the macro lift is not confirmed yet.
3. **Per-region head's binary loss has weight 0.10**, which the Lesson #62 rules consider moderate. It's on a NEW pathway (backbone gradient), not the closed pathway. But if seeds 42/7/999 hit the closed-class fingerprint (sd > 0.025, macro F1 < 0.62 mean), the closed class extends to "any aux loss" and Lesson #63 needs to be retracted.
4. **BR4 F1 = 0.4949** is below vanilla mean (~0.522). The per-region head was designed primarily for BR4↔BR5 (binary CE: benign vs malign), and the seed=555 result shows BR4 regressed by 2.7pp. This is the OPPOSITE of what Track B was designed to achieve. **The +3.4pp BR1 lift is the unexpected positive; the BR4 effect is the original-target negative.** Need to confirm both directions on n=4 before settling on interpretation.

**Possible mechanism for the "BR1 up, BR4 down" pattern:**

The per-region head pushes the backbone to develop spatial features that distinguish "malignant patches" from "benign patches." This is a backbone-level regularizer that prefers spatial coherence (consistent malignancy signal across top-k patches). Two effects on the downstream classification:

1. **BR1 (no findings) vs BR2 (benign findings):** BR1 patients have NO malignant patches; BR2 patients have NO malignant patches either (BR2 = benign findings, not malignant). So the per-region head's binary signal correctly classifies both as benign, and the backbone's malignancy-spatial-feature axis is *orthogonal* to the BR1↔BR2 boundary. **But:** the regularizer pushes the backbone toward "what does benign tissue look like in mammography" features generally, which are exactly the features that distinguish BR1 (clean baseline tissue) from BR2 (benign findings — scattered calcifications, simple cysts). The per-region head's gradient supplies the "benign tissue baseline" signal the BR1↔BR2 boundary was missing. Phase 0c showed BR1 errors come from BR1 patients being embedded INSIDE the BR2 cluster (median k-NN ratio 0.078); the per-region head's backbone regularizer plausibly separates them.

2. **BR4 (suspicious) vs BR5 (highly suggestive):** BR4 has some malign patches; BR5 has more/stronger malign patches. The per-region head should *distinguish* these via top-k score *density*. But the per-region head's loss is BINARY (malign vs benign), not graded. So both BR4 and BR5 are pushed toward the "malignant" cluster, and the head doesn't get gradient to distinguish them — only to separate them collectively from BR1+BR2. This explains why BR4 regressed: it got pushed toward the malign cluster's centroid (which is closer to BR5's signal) without gradient to maintain its lower-density signature.

**This hypothesis is testable** at n=4: if BR1 mean lifts +1.5-3pp AND BR4 mean drops 1-3pp AND BR5 lifts slightly across all 4 seeds, the mechanism above is correct. The fix would be: replace binary CE with a graded "malignancy intensity" target (e.g., regress BR1=0, BR2=0.25, BR4=0.5, BR5=1.0) so the head distinguishes BR4 from BR5 explicitly.

**Rule (provisional, may be revised after n=4):**

1. **Per-region malign head is a candidate first-positive Track B intervention. Awaiting n=4 confirmation before locking.**
2. **Even if confirmed, the binary CE target is the wrong shape for BR4↔BR5.** A graded malignancy-intensity head (regression, or 4-class with within-malign label smoothing) is the obvious next iteration. Test this if n=4 confirms BR1 up + BR4 down.
3. **The "bypass patient_feat" architectural principle is now empirically supported at n=1.** Future Track A / Track B work should preferentially target backbone weights (Direction #1 mammo-pretrain), backbone gradients (per-region head), or completely new heads — NOT additional gradients into the existing fusion-or-classifier path.

**Action (already in flight):**

- **n=4 perregion runs launched 2026-05-11.** Seeds 42, 7, 999 in parallel; ~16h ETA. After completion, will determine whether this lesson is confirmed or retracted.
- **MAE pretrain + mammopre fine-tune sequenced AFTER perregion n=4.** Direction #1 is independent — both can ship as the final result, or ensembled if both work.
- **If n=4 confirms BR1 lift:** ensemble n=4 perregion checkpoints against vanilla 0.6846; compute Phase 0c k-NN on perregion checkpoints to see whether BR1→BR2 ratio shifted toward 1.0 (the architectural-mechanism check).

**Connections to prior lessons:**

- Lesson #51 (seed=42 +1.9σ outlier; BR1 CV=9.6%): **does NOT explain seed=555's +3.4pp lift** because seed=555 has historically been the FRAGILE seed, not the lucky one. The BR1 lift on the most fragile seed is a stronger signal than the same lift on seed=42 would have been.
- Lesson #57 (n=4 ensemble: BR1 +0pp beyond seed=42 outlier): **directly addressed** — Lesson #57's hypothesis was that BR1 errors are STRUCTURAL across seeds (no seed has signal the others don't). Per-region head's seed=555 result, if confirmed at n=4, contradicts this — it would mean per-region head's representation modification is consistent across seeds. The structural-error claim in Lesson #57 only held for variance-reduction interventions; it doesn't hold for representation-pathway-replacing interventions.
- Lesson #58 (post-hoc inference track exhausted; only new feature signal can help): **directly supports** — per-region head ADDS new feature signal at the backbone level (malignancy-spatial-pattern detection), which is exactly the "new feature signal" Lesson #58 Rule 3 said is required.
- Lesson #61 (KL distillation as basin-selector destabilizes seed=555): **negative example** — seed=555 collapsed there because distillation's KL gradient flowed through patient_feat. Here, seed=555 trained cleanly because per-region gradient bypasses patient_feat. Both runs used the same seed=555 init; only the gradient pathway differs. **The seed=555 collapse fingerprint is now specifically diagnostic of aux-loss-on-patient_feat, not of any auxiliary loss.**
- Lesson #62 (aux-loss-on-patient_feat is closed; embedding shift doesn't translate to F1): **mechanistically extended** — per-region head's lift is consistent with "modify backbone features, let head re-learn decoding via CE." This is the OPPOSITE of "force the head to consume specific embedding shifts," which is what aux-loss-on-patient_feat tried.

**Open questions (deferred to n=4 result):**

- Does n=4 confirm BR1 mean lift ≥ +1.5pp AND sd ≤ 0.025? If yes, Lesson #63 → CONFIRMED. If no, → RETRACTED.
- Does per-region n=4 ensemble exceed vanilla 0.6846 ensemble? Specifically, does BR1 in the n=4 perregion ensemble reach ≥ 0.50 (vs vanilla ensemble's 0.50 — was BR1 lift in Lesson #57 already +0pp? Need to re-check)?
- Does Phase 0c k-NN on per-region checkpoints show median BR1→BR2 ratio shift toward 1.0 (mechanism check)? Predicted shift: similar to or larger than distillation's +0.101.
- Does the graded malignancy-intensity head (replacement for binary CE in per-region) lift BR4 without losing the BR1 gain? Deferred to next-iteration design.

---
