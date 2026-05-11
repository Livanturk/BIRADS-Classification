## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

# Project Technical Documentation: Mammography BI-RADS Prediction

This document is prepared to introduce Claude to the project's data structure, statistical properties, and modeling constraints. **Claude must base all recommendations on this context.**

## 1. General Structure and Image Properties

| Property | Value |
| --- | --- |
| **Image Format** | 8-bit PNG, Grayscale (mode=L, uint8) |
| **Resolution** | 1024×1024 pixels |
| **Source** | DICOM → segmentation → windowing → tight crop → CLAHE → letterbox → 8-bit PNG |
| **Number of Views** | 4 (RCC, LCC, RMLO, LMLO) |
| **Unit** | **Patient-based** (1 patient = 1 folder = 4 images) |
| **Number of Classes** | 4 (BI-RADS 1, 2, 4, 5) — **BI-RADS 3 does not exist.** |
| **Value Range** | [0, 255] (normalized: [0, 1]) |
| **Padding Fill** | **Not applied** — zero pixels are left as raw |

### Active Dataset Preprocessing Pipelines

| Dataset | Role | Preprocessing |
| --- | --- | --- |
| **Dataset_1024_8bit** | Train/Val | DICOM → MONOCHROME1 correction → Segmentation (U-Net, resnext50_32x4d, 640×640) → Largest contour mask → approxPolyDP(epsilon=2.0) → fillPoly → Zero out outside mask → Bounding box crop → Windowing (DICOM WindowCenter/Width) → Tight crop (zero border strip) → CLAHE (clipLimit=2.0, tileGrid=8×8, tissue only) → Letterbox 1024×1024 → 8-bit PNG |
| **Dataset_Test_1024_8bit** | Test (holdout) | Same pipeline: DICOM → Segmentation → Windowing → Tight crop → CLAHE → Letterbox 1024×1024 → 8-bit PNG |

> **Preprocessing detail:** U-Net performs 3-class prediction in segmentation (0=background, 1=breast tissue, 2=pectoral muscle). Only class 1 (breast) is masked. CLAHE is applied only to tissue pixels (>0); background remains zero. Tight crop removes remaining zero borders from the segmentation crop.

---

## 2. Data Distribution and Split Strategy

### Active Datasets

| Split | Dataset | Patients | Images |
| --- | --- | --- | --- |
| **Train** | Dataset_1024_8bit | 8,557 | 34,228 |
| **Test** | Dataset_Test_1024_8bit | 1,655 | 6,620 |

### Train Set: 8,557 Patients — Pixel Distribution by Class

| Class | Tissue Pixels (nonzero) | Ratio |
| --- | --- | --- |
| BI-RADS-1 | 2,314,247,381 | 18.7% |
| BI-RADS-2 | 4,013,136,317 | 32.4% |
| BI-RADS-4 | 2,738,316,060 | 22.1% |
| BI-RADS-5 | 3,301,005,784 | 26.7% |
| **Total** | **12,366,705,542** | — |

### Test Set: 1,655 Patients — Pixel Distribution by Class

| Class | Tissue Pixels (nonzero) | Ratio |
| --- | --- | --- |
| BI-RADS-1 | 236,522,596 | 9.7% |
| BI-RADS-2 | 876,891,567 | 36.0% |
| BI-RADS-4 | 421,038,405 | 17.3% |
| BI-RADS-5 | 900,864,020 | 37.0% |
| **Total** | **2,435,316,588** | — |

> BI-RADS-3 class does not exist. The test set is **imbalanced** — BI-RADS-2 and BI-RADS-5 are dominant.

### Split Details

* **Train (85%):** Stratified random split (seed=42).
* **Val (15%):** Stratified random split (seed=42).
* **Test (Fixed):** 1,655 patients, 6,620 images. Independent holdout (class distribution is imbalanced).

---

## 3. Normalization Statistics (0–1 scale)

### Dataset_1024_8bit (Train/Val)

#### Train Statistics

| Metric | Value |
| --- | --- |
| All-pixel mean / std | 0.1210 / 0.1977 |
| Nonzero (tissue) mean / std | 0.3512 / 0.1804 |
| Zero pixel ratio | 65.54% |
| Total pixels | 35,890,659,328 |
| Tissue pixels | 12,366,705,542 |

#### Class-wise Nonzero (Tissue) Mean / Std — Train

| Class | Mean | Std | Tissue Pixels |
| --- | --- | --- | --- |
| BI-RADS-1 | 0.3518 | 0.1831 | 2,314,247,381 |
| BI-RADS-2 | 0.3532 | 0.1799 | 4,013,136,317 |
| BI-RADS-4 | 0.3512 | 0.1818 | 2,738,316,060 |
| BI-RADS-5 | 0.3483 | 0.1780 | 3,301,005,784 |

#### Patient-wise Tissue Mean Distribution — Train (n=8,557)

| Percentile | Value |
| --- | --- |
| min | 0.1010 |
| p5 | 0.2910 |
| p25 | 0.3221 |
| p50 | 0.3444 |
| p75 | 0.3744 |
| p95 | 0.4267 |
| max | 0.6040 |

#### Train vs Test Validation

| Metric | Train | Test |
| --- | --- | --- |
| All-pixel mean/std | 0.1210 / 0.1977 | 0.1237 / 0.1986 |
| Nonzero mean/std | 0.3512 / 0.1804 | 0.3526 / 0.1779 |
| Zero pixel % | 65.54% | 64.92% |

> Train–Test distributions are very close — no domain shift.

#### Backbone Normalization Values

```python
# Option 1: All-pixel statistics (including zeros)
mean=[0.1210, 0.1210, 0.1210], std=[0.1977, 0.1977, 0.1977]

# Option 2: Nonzero (tissue) statistics — must be used together with key_padding_mask
mean=[0.3512, 0.3512, 0.3512], std=[0.1804, 0.1804, 0.1804]
```

> **Critical:** If nonzero statistics are used, `key_padding_mask` must be passed to CrossAttn; otherwise letterbox zero pixels will corrupt attention.

> **CLAHE effect (comparison with old 512px):** CLAHE raised tissue mean from 0.284 → 0.351 (+24%), std increased from 0.158 → 0.180. Local contrast enhancement shifted the histogram to the right as expected.

---

## 4. Training Methodology

* **Balancing:** `Sqrt-inverse frequency class weights` are used (normalized: max-freq class = 1.0).
* **Current Weights:** `[1.28, 1.00, 1.20, 1.11]` (BI-RADS [1, 2, 4, 5])
* **Patient Distribution (Train):** BR1=1678, BR2=2754, BR4=1898, BR5=2227 | Benign=4432, Malignant=4125
* **Preprocessing:** Padding fill is not applied. Raw 8-bit images are used.

> **Critical Note:** Tissue density (brightness) is higher in malignant classes. There is a risk that the model may learn brightness as a "shortcut" rather than learning morphological features.

---

## 5. Config Naming and Output Structure

### Config Convention
Format: `configs/{backbone}_{variant}_{extra_param}.yaml`

Examples:
- `configs/convnext_large_seg_v1.yaml` — ConvNeXt-Large, version 1
- `configs/swinv2_base_seg_focal.yaml` — SwinV2-Base, focal loss
- `configs/dinov2_large_noseg_lr3e5.yaml` — DINOv2-Large, without segmentation, different LR

### Output Directory
Automatically derived from the config file name:
```
python train.py --config configs/convnext_large_seg_v1.yaml
→ outputs/convnext_large_seg_v1/
    — checkpoints/
    — plots/
    — reports/
    — gradcam/
```

### Benchmark Comparison
```bash
python benchmark.py --configs configs/configname.yaml configs/configname2.yaml
```

## 6. 8-bit Image Reading Pipeline

**Normalization**: Statistics computed from the train set are used for both datasets (train–test distributions are very close).

```python
# Dataset_1024_8bit (Train/Val) — All-pixel
mean=[0.1210, 0.1210, 0.1210], std=[0.1977, 0.1977, 0.1977]

# Dataset_Test_1024_8bit (Test) — Same statistics applied
# Train: 0.1210/0.1977 | Test: 0.1237/0.1986 — difference is negligible
mean=[0.1210, 0.1210, 0.1210], std=[0.1977, 0.1977, 0.1977]
```

## 7. Operational Rules for Claude

* **Anomaly:** Metric interpretations should be made carefully due to the imbalanced nature of the test set.
* **Normalization:** Statistics were computed with `compute_norm_stats.py` and are current as of 2026-04-08.

---

# Project Architecture: Multi-View Hierarchical Mammography Classifier

> **Status:** the C6 multi-task hierarchical model is the **current champion** (test macro F1 = 0.6762). A 3-stage soft cascade (G-series) was tested and rejected on test (-4.96pp regression vs C6); see `tasks/lessons.md` Lessons #49 + #50 and `tasks/cascade_log.md`. The cascade architecture is documented at the end of this section as a frozen negative result for paper write-up; do not propose it as a forward path.

## Overview

The model takes 4 mammography images (RCC, LCC, RMLO, LMLO) and produces a patient-level BI-RADS prediction. Input tensor: `(B, 4, 3, H, W)`.

```
Input: (B, 4, 3, 1024, 1024)
    |
[Level 1] Backbone — Weight-Shared (single backbone, runs 4 times)
    → {RCC, LCC, RMLO, LMLO}: each (B, S, D)  [S = number of spatial tokens]
    |
[Level 2] Lateral Fusion — Bidirectional Spatial Cross-Attention
    Right: CrossAttn(RCC ↔ RMLO) → attention pool → (B, D)
    Left:  CrossAttn(LCC ↔ LMLO) → attention pool → (B, D)
    |
[Level 3] Bilateral Fusion — Asymmetry-Aware Self-Attention
    tokens = [F_left, F_right, F_diff, F_avg]
    2-layer TransformerEncoder → attention pool → patient_feat (B, D)
    |
[Level 4] Hierarchical Classification Heads (all 4 fed by patient_feat)
    binary_head(patient_feat)   → (B, 2)  Benign/Malignant
    benign_sub(patient_feat)    → (B, 2)  BI-RADS 1 vs 2
    malign_sub(patient_feat)    → (B, 2)  BI-RADS 4 vs 5
    full_head(patient_feat)     → (B, 4)  BI-RADS 1/2/4/5
    temperature_scaling         → confidence score (full_head only; not in loss)
```

**Doc/code drift note:** an earlier revision of this doc claimed `binary_head` consumed `global_feat` (a backbone-averaged feature that bypasses the fusion chain). The actual code (`models/full_model.py`, `models/classification_heads.py`) feeds **`patient_feat`** to all four heads. The G-series cascade (Lesson #50) tested the bypass-fusion design experimentally and it lost on test (G1 val_binary_f1 = 0.9664 → test 0.9309, **−0.81pp** below C6's binary head; val→test gap widened from 1.4pp to 3.55pp). The fusion-fed binary head is the version that ships.

---

## Level 1: Backbone (`models/backbone.py`)

| Parameter | Value |
| --- | --- |
| **Class** | `MultiViewBackbone` → `BackboneFeatureExtractor` |
| **Weight Sharing** | 1 backbone, shared across 4 views (4× fewer parameters) |
| **Global Pool** | **NONE** — spatial feature map is preserved (required for Lateral Fusion) |
| **Output** | `(B, S, projection_dim)` — S = H×W spatial token count |
| **Projection** | `Linear(backbone_dim → D) + LayerNorm + GELU + Dropout(0.2)` |

**Backbone output format normalization:**
- CNN (channels-first `B,C,H,W`) → permute → `(B, H*W, C)`
- Swin (channels-last `B,H,W,C`) → reshape → `(B, H*W, C)`
- ViT (`B,N,C`) → already in correct format

---

## Level 2: Lateral Fusion (`models/lateral_fusion.py`)

| Parameter | Value |
| --- | --- |
| **Class** | `BilateralLateralFusion` → `LateralFusion` → `CrossAttentionBlock` |
| **Weight Sharing** | Right (RCC+RMLO) and left (LCC+LMLO) **share the same weights** |
| **Positional Embed** | Learnable `(1, S, dim)` — added at **full resolution** (before pooling) |
| **Attention Direction** | Bidirectional: CC→MLO + MLO→CC (both parallel, 2 layers) |
| **Pooling** | Attention pooling (learned weights: `(B,T,dim)→(B,dim)`) |
| **Merging** | `concat([CC_pooled, MLO_pooled]) → Linear(dim*2 → dim) + LN + GELU` |

**CrossAttentionBlock (Pre-LN):**
```
h = source + MultiHeadAttn(LN(source), LN(target), LN(target))
output = h + FFN(LN(h))
```

---

## Level 3: Bilateral Fusion (`models/bilateral_fusion.py`)

| Parameter | Value |
| --- | --- |
| **Class** | `BilateralFusion` |
| **Token Count** | 4: `[F_left, F_right, F_diff, F_avg]` |
| **F_diff** | `F_left − F_right` — captures bilateral breast asymmetry |
| **F_avg** | `(F_left + F_right) / 2` — shared tissue density information |
| **Self-Attention** | 2-layer `TransformerEncoderLayer` (Pre-LN, `batch_first=True`) |
| **Pooling** | Attention pooling: `Linear→Tanh→Linear(→1)` → softmax → weighted sum |
| **Output Projection** | `Linear(dim→dim) + LN + GELU + Dropout(0.25)` |

---

## Level 4: Classification Heads (`models/classification_heads.py`)

| Head | Input | Output | Loss |
| --- | --- | --- | --- |
| **Binary** | `patient_feat` | (B, 2) | CrossEntropy |
| **Benign Sub** | `patient_feat` | (B, 2) | CrossEntropy (only on benign-mask samples) |
| **Malignant Sub** | `patient_feat` | (B, 2) | CrossEntropy (only on malign-mask samples) |
| **Full** | `patient_feat` | (B, 4) | CrossEntropy or Focal (config-driven, `loss_type`) |

All four heads are always instantiated and always run (the `ablation.use_*_head` flags only gate which heads contribute to the loss, not which heads are built — see `models/classification_heads.py:HierarchicalClassifier.__init__`).

**Temperature Scaling:**
```python
log_temperature = nn.Parameter(log(1.5))   # Learnable parameter, but ...
confidence = softmax(full_logits / exp(log_temperature)).max()
```
**Caveat (Lesson #44):** `log_temperature` is *not* in any loss term, so it never receives a gradient and stays at its config init (1.5). Temperature scaling is a post-hoc reporting hook, not a learned calibration. Tier-1 Task 1.2 was the first real temperature search.

---

## Loss Function (`utils/losses.py`)

C6 (champion config `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml`):

```
L_total = 0.10 × L_binary + 0.45 × L_subgroup + 0.45 × L_full
```

| Component | Function | C6 weight |
| --- | --- | --- |
| `L_binary` | CrossEntropy (label_smoothing=0.05) | 0.10 |
| `L_subgroup` | CE (benign sub + malign sub averaged on masked samples) | 0.45 |
| `L_full` | CrossEntropy (label_smoothing=0.05; `loss_type: ce`) | 0.45 |
| `L_asymmetry` | (off in C6 — Lesson #22) | 0.0 |

> **Loss-type drift:** earlier configs used Focal (gamma=2.0) on `L_full`. Lesson #28 + Lesson #37 found focal loss harmful for ConvNeXtV2 on this dataset; C6 ships with `loss_type: ce`. Focal is still selectable via `training.loss_type: focal` in any config that wants it.

**Class weights (sqrt-inverse frequency, Dataset_1024_8bit train):**
- 4-class: `[1.28, 1.00, 1.20, 1.11]` → BI-RADS [1, 2, 4, 5]
- Binary: `[1.00, 1.04]` → [Benign, Malignant] (hardcoded in `utils/losses.py:467`)
- Benign sub: `[1.28, 1.00]` → [BR1, BR2]
- Malignant sub: `[1.08, 1.00]` → [BR4, BR5]

---

## Ablation Support

Modules can be selectively disabled via the `ablation` section in the config:

```yaml
ablation:
  use_lateral_fusion: true      # false → simple concat + projection
  use_bilateral_fusion: true    # false → simple concat + projection
  use_binary_head: true         # false → binary loss not computed
  use_subgroup_head: true       # false → subgroup loss not computed
  use_uncertainty: true         # temperature scaling on/off
  use_ordinal_head: false       # CORAL ordinal head replaces full_head's CE
  use_flat_fusion: false        # 4-view GAP → concat → MLP (drops both fusions)
```

D-series (Lesson #37) confirmed that **every** removal from C6's hierarchy hurts test F1. Use ablation flags for diagnostics, not for proposing simpler shipping configs.

---

## Cascade Architecture (G-series — REJECTED, archived for paper write-up)

A separate, additive code path (`models/cascade_model.py`, `data/cascade_loader.py`, `train_cascade.py`, `tools/cascade_*.py`, `configs/cascade/`) implements a 3-stage soft cascade tested in April 2026.

```
                                  +-- G2a: ConvNeXtV2-L + lateral + bilateral
                                  |        + 2-class head (BR1 vs BR2)
                                  |        → P(BR1|benign), P(BR2|benign)
G1: ConvNeXtV2-L → backbone-only -+
    + 2-class head (benign / malign)
    → P(benign), P(malign)        |
                                  +-- G2b: ConvNeXtV2-L + lateral + bilateral
                                           + 2-class head (BR4 vs BR5)
                                           → P(BR4|malign), P(BR5|malign)

Soft compose:  P(BRk) = P(stage1=k.parent) × P(BRk | stage1=k.parent)
```

- **G1** trains on all 8557 train patients (relabeled binary), uses backbone + global mean-pool over (4 views × S spatial tokens) → 2-class head. **No fusion.**
- **G2a** trains only on BR1/BR2 patients (3767 train / 665 val) with full C6 fusion stack → `patient_feat` → 2-class head.
- **G2b** trains only on BR4/BR5 patients (3506 train / 619 val) with the same architecture as G2a.

**Outcome (Lesson #50):**
- Phase E val gates passed (G1=0.9664, G2a=0.7020, G2b=0.7740 — all ≥ C6 lower bounds), but **test cascade macro F1 = 0.6266, −4.96pp vs C6 (0.6762)**.
- BR2 cratered −12.2pp (BR2→BR1 drift doubled to 28% from C6's 13%), BR4 statistically tied, BR1 lost −3.5pp despite higher recall (precision crashed).
- Soft cascade ≈ hard cascade (Δ +0.0004pp): all three stages produce >95% confident outputs, so `P × P` collapses to argmax × argmax.
- G1's no-fusion design beat C6 binary on val (+1.34pp) but lost on test (−0.81pp) — fusion is a regularizer, not just a feature path.
- Root cause: train→test prior shift compounds multiplicatively across cascade stages (each stage's class weights are train-prior-calibrated; test priors differ; product of mis-calibrated stages explodes the error).

**Status:** archived. Do not propose cascade variants as a forward path. Next experiment is Tier-2 logit-adjusted training (Menon et al. 2021), which targets the prior shift directly. Cascade artifacts (`outputs/cascade/test_probs.parquet`, `evaluation_report.md`, MLflow runs `ad4526d7…`/`bae58239…`/`c5926b3c…`) are preserved for the paper's negative-result section.
