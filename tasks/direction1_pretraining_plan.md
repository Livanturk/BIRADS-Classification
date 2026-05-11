# Direction #1 — Mammography Domain Pretraining Plan

**Status:** PLANNING. Do not launch until corpus access + storage + plan approval.

**Background:** After 10 BR1 intervention classes failed (Lesson #62), backbone replacement is the highest-EV remaining BR1 path. The hypothesis: the ImageNet-22k-pretrained ConvNeXtV2-Large encodes natural-image priors (textures, shapes, scenes) that are misaligned with mammographic tissue. A backbone pretrained on a large mammography corpus would replace those priors with mammography-specific features (parenchymal pattern, fibroglandular density, calcification distributions).

**Literature support:** Shen 2019 (NYU-CBIS), Wei 2021 (RSNA-Breast), Lopez 2024 — all show ≥3pp BI-RADS lifts from mammography pretraining vs ImageNet-pretrained baselines on similar architectures.

---

## Decision Gates

Three gates need user input before any GPU is committed.

### Gate 1: Corpus Access

We need a public mammography corpus large enough for self-supervised pretraining. Candidates ranked by feasibility:

| Corpus | Size | Bit depth | Format | Access | Pre-flight checks |
| --- | --- | --- | --- | --- | --- |
| **RSNA-Breast Cancer Detection 2022** | ~54k patients (~216k DICOMs) | 12-16 bit DICOM | DICOM | Free, Kaggle account | Largest publicly available; matches our DICOM source |
| **EMBED (Emory Breast Imaging Dataset)** | ~3.4M images, ~116k patients | 12-16 bit DICOM | DICOM | Free, IRB approved data sharing agreement | Largest absolute size; access requires DUA signing |
| **CBIS-DDSM** | ~2.6k cases, ~10k images | 12-16 bit (originally 16-bit film scans) | DICOM | Free, TCIA download | Smaller; well-curated; calcification + mass labels |
| **INbreast** | 410 images, 115 patients | 14 bit | DICOM | Free with registration | Too small for standalone pretrain; can be merged |
| **MIAS / DDSM (mini)** | <500 images | 8 bit | PGM | Free | Resolution mismatch; older data |

**Recommendation:** RSNA-Breast as primary (largest readily-accessible), optionally augmented with CBIS-DDSM.

**User action required:**
- Confirm whether RSNA-Breast is already downloaded on this cluster.
- If not, confirm storage availability (~500-800 GB raw DICOMs; ~150-200 GB after our preprocessing pipeline).
- If neither, this direction is blocked at corpus access.

### Gate 2: Pretraining Method

Three viable self-supervised methods, ordered by likelihood of useful features for our downstream BI-RADS task:

| Method | Compute (per epoch) | Sample efficiency | Tested in mammography lit | Best for |
| --- | --- | --- | --- | --- |
| **MAE (Masked Autoencoder)** | Lower (decoder-only loss) | High | Yes (Wei 2021) | Reconstructs masked patches → learns local + global tissue structure. Strong for our diffuse-attention regime (Phase 0c). |
| **DINO / DINOv2** | Higher (multi-crop augmentations) | Highest | Mixed (Lesson #33: DINOv2 backbone failed at 0.638 ceiling, but that was natural-image DINOv2; mammography-DINO is different) | Self-distilled features; produces semantically-clustered embeddings — but our Phase 0c shows the embedding-space mechanism alone doesn't translate to F1 (Lesson #62). |
| **SimCLR / MoCo** | Lower | Medium | Yes (Shen 2019) | Contrastive on augmented views. Less sample-efficient than MAE; well-understood. |

**Recommendation: MAE.** Three reasons:
1. Best mammography literature support (Shen 2019 + Wei 2021).
2. The reconstruction loss naturally captures the diffuse-tissue + sub-pixel-calcification regime we need (Phase 0c showed concentration_median ≤ 0.09 — global features matter more than localized).
3. Lesson #33's DINOv2 failure was on a *natural-image* backbone — a mammography-pretrained DINO is plausibly fine, but MAE is the safer choice given prior negative result.

**Decoder choice:** MAE-Lite (small decoder) is sufficient for representation learning. The decoder is discarded after pretrain; only the encoder transfers.

**Mask ratio:** 0.75 (canonical MAE). High mask ratio + reconstruction loss in mammography is documented to learn density-aware features (Wei 2021).

### Gate 3: Compute Budget

| Phase | Compute estimate | Wall-clock on 4 H100s | Cost equivalent |
| --- | --- | --- | --- |
| **Pretrain MAE on RSNA-Breast (50k patients × ~4 views = 216k images at 1024²)** | 100-150 GPU-hours per epoch × ~50 epochs | ~5000-7500 GPU-hours total / ~50-75 days at 4 GPUs | High |
| **Pretrain MAE-Lite (smaller decoder, faster)** | 50-75 GPU-hours per epoch × ~50 epochs | ~2500-3750 GPU-hours / ~25-40 days | High |
| **Pretrain MAE-Lite at 512² (downsampled)** | 12-18 GPU-hours per epoch × ~50 epochs | ~600-900 GPU-hours / ~6-10 days | Medium |
| **Fine-tune (4 students with C6 architecture loading the pretrained encoder)** | 12-16 GPU-hours per student × 4 in parallel | ~12-16 hours wall-clock | Low |

**The dominant cost is pretraining.** Three concrete options the user must choose from:

**Option A: Full-resolution pretraining (~50-75 days wall-clock).**
- Pretrain at 1024² (matches our downstream resolution).
- Highest fidelity transfer; no resolution mismatch when fine-tuning.
- Highest-EV but blocking compute commitment.

**Option B: Downsampled-pretrain + downstream-fine-tune (~6-10 days pretrain + 16h fine-tune).**
- Pretrain at 512² (faster).
- Fine-tune at 1024² (re-interpolating positional embeddings).
- Tradeoff: features are trained on a different scale; some fine-tune adaptation needed; positional embedding re-init may lose spatial-token-level features.
- Most pragmatic if compute is constrained.

**Option C: Skip pretraining; use a published mammography-pretrained checkpoint.**
- E.g., the NYU model checkpoint (Shen 2019, ResNet-22) or RSNA-Breast leaderboard checkpoints.
- Cost: zero pretraining compute. Risk: architecture mismatch (we use ConvNeXtV2-Large; published models are typically ResNet/EfficientNet) — would require full backbone replacement, not just weight transfer.
- Preferred IF a ConvNeXt-family or transformer mammography checkpoint is accessible. Spot check: HuggingFace `microsoft/mammography-*`, `facebook/mae-*-mammography`, individual researcher releases.

**Recommendation: Option B if storage + compute available; Option C if the right checkpoint exists.**

---

## Implementation Plan (after gates approved)

### Phase 1 — Corpus prep (1-2 days)

Adapt the existing pipeline to handle the chosen pretraining corpus:
- Reuse `Dataset_1024_8bit` preprocessing pipeline (DICOM → segmentation → windowing → CLAHE → letterbox 1024²).
- Run on RSNA-Breast / EMBED / CBIS-DDSM DICOMs.
- Output: `Dataset_Pretrain_1024_8bit/` (or 512² for Option B).

**Deliverable:** `tools/preprocess_external_corpus.py` or extension of existing preprocessing scripts. Estimated 10-50 GB output depending on corpus + resolution.

### Phase 2 — MAE pretraining script (3-5 days dev + multi-day compute)

Adapt one of:
- `timm`'s built-in MAE training loop (recommended; well-tested).
- `facebookresearch/mae` upstream code.
- Custom implementation using the existing `models/backbone.py` ConvNeXtV2 trunk.

**Deliverable:** `train_mae.py` that loads pretraining corpus + runs MAE on the encoder portion of `convnextv2_large.fcmae_ft_in22k_in1k_384` and saves an encoder-only checkpoint.

**Critical detail:** the ConvNeXtV2 architecture we use is already FCMAE-pretrained (`fcmae_ft_in22k_in1k_384`), but on natural images. The fine-tuning step (`ft_in22k`) overwrites the FCMAE features with classification-task features. Our domain pretraining should re-do FCMAE on mammography data starting from the natural-image FCMAE weights, then run the standard C6 training without the `ft_in22k` step.

### Phase 3 — Fine-tune integration (1 day dev + 16h compute)

Modify the C6 config family to load the pretrained encoder:
- Add `model.backbone.checkpoint_path: outputs/mae_pretrain/encoder_only.pt`
- Add a flag in `MultiViewBackbone.__init__` to load this checkpoint after `timm.create_model` (overriding the natural-image weights).
- Otherwise reuse the C6 architecture identically.

**Deliverable:** `configs/.../convnextv2_large_8bit_ablation_c6_mammopre_seed{42,7,555,999}.yaml` + a launcher.

### Phase 4 — Evaluation

Standard 4-seed parallel training. Compare to:
- C6 vanilla (n=6 seed-mean = 0.6502)
- C6 vanilla ensemble (n=4 = 0.6846)
- Per-region head (Track B; in-flight)

Decision rules:
- **PASS:** mean macro F1 ≥ 0.69 AND BR1 F1 mean ≥ 0.51 (≥+3pp vs vanilla mean) AND sd ≤ 0.025 → publish as new champion.
- **MIXED:** macro F1 lift positive but <2pp → escalate by combining with per-region head (compatible mechanisms).
- **FAIL:** mean ≤ 0.65 OR BR1 lift ≤ 1pp → mammography pretraining alone doesn't fix BR1 in this dataset; the ceiling is data-bound (label noise + dataset size); ship the n=4 ensemble (0.6846) as the production model and document the floor.

---

## Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| **Pretraining data domain shift** (RSNA-Breast cohort vs our cohort) | Medium | Pretrain features are coarser than fine-tune features; some shift is tolerable. EMBED-instead-of-RSNA reduces this if accessible. |
| **MAE doesn't surface BR1↔BR2 features** | Medium-low | The diffuse-attention regime (Phase 0c) is exactly what MAE captures (global tissue structure). But MAE optimizes pixel reconstruction, not BI-RADS discrimination — the features may be useful but not class-aligned. |
| **Compute commitment exceeds budget** | Medium-high (Option A); Low (Option B/C) | Option B caps at ~10 days; Option C is essentially free. Decide before committing. |
| **Compatibility with existing C6 training pipeline** | Low | The change is encoder weights only; rest of stack unchanged. |
| **Pretrained encoder destabilizes the closed-class fingerprint AGAIN** | Low-medium | Lesson #62's structural finding implies the C6 fusion+heads stack is the bottleneck. Replacing the backbone touches a DIFFERENT pathway, but if the bottleneck is "any change to representation distribution," this could also fail. Test plan: stress-test seed=555 alone first (12 hours) before committing all 4 seeds. |

---

## Outstanding User Decisions

1. **Corpus:** Is RSNA-Breast / EMBED / CBIS-DDSM already on the cluster? If not, can it be downloaded? If neither, this direction is blocked.
2. **Method + resolution:** Option A (1024² full pretrain, ~50 days), Option B (512² fast pretrain, ~10 days), or Option C (use published checkpoint, ~0 compute)?
3. **Compute budget:** Is there a 50-day or 10-day window of 4-GPU dedicated compute available, or is the cluster shared with other workloads?
4. **Sequencing:** Direction #1 should run AFTER Track B (per-region head) results are in. If per-region head produces a +1.5pp BR4 lift, the macro F1 baseline has already shifted; Direction #1 should be evaluated against that new baseline, not against vanilla 0.6846.

---

## Pre-launch Checklist

When/if Direction #1 is greenlit:

- [ ] User confirms corpus availability and disk space
- [ ] Per-region head (Track B) results in
- [ ] Direction #1 method (A/B/C) chosen
- [ ] `tools/preprocess_external_corpus.py` written + tested on a 100-patient sample
- [ ] `train_mae.py` (or upstream adaptation) tested at small scale
- [ ] `models/backbone.py` modified to support `checkpoint_path` override
- [ ] `configs/.../c6_mammopre_seed{42,7,555,999}.yaml` written
- [ ] `scripts/run_mammopre_parallel.sh` written
- [ ] Stress-test seed=555 alone for 12 hours before committing 4-seed compute
- [ ] Lesson #63 (provisional) drafted with method + compute commitment for posterity

---

## Source-of-Truth References

- Shen et al. 2019, "Deep learning to improve breast cancer detection on screening mammography," Sci Rep — the NYU model, ResNet-22, BI-RADS classification.
- Wei et al. 2021, "Self-supervised pretraining for mammography classification," — RSNA-Breast MAE result.
- Lopez et al. 2024 — recent mammography-MAE benchmark.
- Lesson #62 (this repo): structural finding that aux-loss-on-patient_feat is closed; backbone replacement is highest-EV remaining path.
- Phase 0c (Lesson #60): diffuse-attention regime supports global-feature pretraining (MAE) over localized-feature methods.
