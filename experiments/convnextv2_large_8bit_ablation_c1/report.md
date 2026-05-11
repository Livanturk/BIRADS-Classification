---
experiment: convnextv2_large_8bit_ablation_c1
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c1.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c1

## Motivasyon / Hipotez
Kombinasyon ablation hücresi. B5 (SWA, test F1: 0.6615) üzerine Mixup/CutMix regularizasyonu eklenir.
Tek değişken: Mixup/CutMix. SWA dahil tüm B5 parametreleri korunur.

B-serisi iki kritik bulgu ortaya koydu:
1. **Mixup/CutMix (B3):** En iyi val-test gap regularizer. Gap: 9.5pp→7.3pp, test F1 +0.72pp.
   Train F1'i 0.795→0.588'e düşürerek güçlü regularizasyon sağladı.
2. **SWA (B5):** En iyi mutlak test F1 (0.6615). Flat minima sayesinde gap=6.7pp.

Bu iki mekanizma ortogonaldir:
- Mixup, eğitim sırasında gradient sinyalini yumuşatır (soft targets, lambda blending)
- SWA, eğitim sonrası weight trajectory'yi ortalayarak düz minimaya yakınsar
- Mixup overfit'i azaltır → SWA daha temiz bir weight trajectory ortalar

**Ek hipotez — BR1 toparlanması:** B5'te BR1 -7.3pp geriledi (0.526→0.453) çünkü SWA,
BR1-BR2 sınırını BR2'ye doğru kaydırdı. B3'te Mixup BR1'i 0.513'te tuttu (B1: 0.526'dan
sadece -1.3pp). Mixup'ın soft target regularizasyonu, SWA'nın BR1 regresyonunu kısmen
telafi etmeli.

**Beklenti:** Test F1 macro >= 0.68. Val-test gap <= 6pp. BR1 test F1 >= 0.48.

**Kill kriteri:** val_full_f1_macro < 0.60 by epoch 15.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn
- **Mixup:** alpha=0.2 | **CutMix:** alpha=1.0

## Baseline'dan Değişiklikler (B5 → C1: tek değişim — Mixup/CutMix)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.use_mixup` | `false` (default) | `true` | Hasta bazlı mixing regularizasyonu |
| `training.mixup_alpha` | — | `0.2` | Mild interpolation (lambda ~0.9) |
| `training.use_cutmix` | `false` (default) | `true` | Spatial cutout regularizasyonu |
| `training.cutmix_alpha` | — | `1.0` | Uniform lambda (çeşitli cutout boyutları) |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

**Not:** Train metriklerinde NaN beklenidir (Lesson #15 — Mixup soft targets asymmetry/binary
loss logging'i bozar, eğitimi etkilemez).

## Sonraki Adım

- B5 ile karşılaştır: Mixup, SWA'nın üzerine ek kazanım sağladı mı?
- B3 ile karşılaştır: SWA, Mixup'ın üzerine ek kazanım sağladı mı?
- Per-class F1: BR1 toparlandı mı (>= 0.48)?
- Val-test gap <= 6pp ise: 8-bit'in ulaşabileceği tavan buna yakın
- Test F1 >= 0.68 ise: D-serisi ensemble (C1 + C2) planla
