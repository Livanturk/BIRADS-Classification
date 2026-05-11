---
experiment: convnextv2_large_8bit_ablation_c5
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c5.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c5

## Motivasyon / Hipotez
Backbone öğrenme hızı ablation hücresi. B5 (SWA, test F1: 0.6615) baseline üzerine
backbone_lr_scale 0.2→0.05 düşürülür.
Tek değişken: backbone fine-tuning agresifliği.

Medikal görüntülemede ImageNet-22k ön-eğitiminin düşük seviye özellikleri (kenar, doku,
kontrast detektörleri) kritik öneme sahiptir. Mevcut backbone_lr_scale=0.2, backbone'un
ilk katmanlarını eğitim sırasında önemli ölçüde değiştiriyor olabilir — bu da doğal
görüntülerde öğrenilen transferable özellik temsilerini bozuyor olabilir.

0.05'e düşürmek:
- Backbone'u neredeyse dondurur ama tamamen değil (freeze_layers=0 korunur)
- Fusion/head katmanlarını ağır işi yapmaya zorlar
- Daha stabil özellik uzayı → daha tutarlı lateral/bilateral fusion
- Backbone gradientleri 20× küçük: 5e-5 × 0.05 = 2.5e-6

**Beklenti:** Daha yavaş yakınsama ama daha iyi generalizasyon.
Test F1 >= 0.66. Per-class daha dengeli dağılım.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 20.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=**0.05** (backbone effective LR: 2.5e-6)
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn

## Baseline'dan Değişiklikler (B5 → C5: tek değişim — backbone LR)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.optimizer.backbone_lr_scale` | `0.2` | `0.05` | Ön-eğitim özelliklerini koruma |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: Düşük backbone LR generalizasyonu iyileştirdi mi?
- Yakınsama hızı: Kaç epoch'ta best val F1'e ulaştı?
- Per-class F1: Daha stabil backbone hangi sınıflarda avantaj sağlıyor?
- GradCAM karşılaştırması: Attention bölgeleri daha anatomik olarak tutarlı mı?
