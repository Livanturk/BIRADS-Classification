---
experiment: convnextv2_large_8bit_ablation_b3
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_b3.yaml
baseline: convnextv2_large_8bit_ablation_b1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_b3

## Motivasyon / Hipotez
Regularizasyon ablation hücresi. B1 ile birebir aynı config, tek fark Mixup ve CutMix aktif.
A-serisi ve B1'de en büyük sorun val→test generalizasyon gap'i (7-8pp macro F1). Mixup/CutMix,
hasta bazlı karıştırma ile karar sınırlarını yumuşatır ve overconfident tahminleri azaltır.

Her batch'te %50 Mixup (alpha=0.2, mild interpolation), %50 CutMix (alpha=1.0, spatial cutout)
rastgele seçilir. Soft target: lambda * loss_a + (1-lambda) * loss_b.

**Beklenti:** Val-test gap < 5pp. Test F1 macro >= 0.70.

**Kill kriteri:** val_full_f1_macro < 0.60 by epoch 10.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **Mixup:** alpha=0.2 | **CutMix:** alpha=1.0

## Baseline'dan Degisiklikler (B1 → B3: tek degisim — regularization)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b1`) | Bu Deney | Gerekce |
|---|---|---|---|
| `training.use_mixup` | `false` (default) | `true` | Mixup regularizasyonu aktif |
| `training.mixup_alpha` | — | `0.2` | Mild interpolation (lambda ~0.9) |
| `training.use_cutmix` | `false` (default) | `true` | CutMix regularizasyonu aktif |
| `training.cutmix_alpha` | — | `1.0` | Uniform lambda (diverse cutout boyutlari) |

## Sonuclar

_Deney tamamlandiginda doldurulacak._

## Analiz

_Deney tamamlandiginda doldurulacak._

## Sonraki Adim

- B1 ile karsilastir: val-test gap daraldi mi?
- Per-class F1 profili: Mixup/CutMix hangi sinifi en cok etkiledi?
- Eger gap < 5pp VE test F1 > 0.70 ise, B3+B5 (Mixup+SWA) kombinasyonu dene
