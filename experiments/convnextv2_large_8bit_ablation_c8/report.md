---
experiment: convnextv2_large_8bit_ablation_c8
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c8.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c8

## Motivasyon / Hipotez
Agresif dropout ablation hücresi. B5 (SWA, test F1: 0.6615) baseline üzerine
dropout oranları önemli ölçüde artırılır.
Tek değişken: dropout seviyeleri.

Model, belirli hasta kalsifikasyon koordinatlarını veya doku pattern'larını
ezberliyorsa, agresif dropout modeli dağıtılmış özellik temsillerine
güvenmeye zorlar:

- `classification.dropout`: 0.5 → 0.7 (sınıflandırma başlığı)
- `lateral_fusion.projection_dropout`: 0.2 → 0.4 (lateral fusion çıkışı)

Bu iki katman, backbone özelliklerini nihai sınıflandırmaya dönüştüren
bottleneck noktalarıdır. Buradaki agresif dropout:
1. Eğitim sırasında farklı özellik alt-kümeleri kullanmaya zorlar
2. Her nöronun bağımsız olarak bilgilendirici olmasını gerektirir
3. SWA ile kombine: dropout → çeşitli alt-ağlar, SWA → hepsinin ortalaması

**Beklenti:** Val-test gap <= 5pp. Val F1 düşebilir ama test F1 korunmalı.
Test F1 >= 0.65.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 20 (yavaş yakınsama beklenir).

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn
- **Dropout:** classification=**0.7**, lateral_proj=**0.4**

## Baseline'dan Değişiklikler (B5 → C8: tek değişim — dropout)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.classification.dropout` | `0.5` | `0.7` | Agresif head regularizasyonu |
| `model.lateral_fusion.projection_dropout` | `0.2` | `0.4` | Agresif fusion regularizasyonu |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: Agresif dropout gap'i azalttı mı?
- B3 (Mixup) ile karşılaştır: Hangisi daha etkili regularizer?
- Per-class F1: Dropout hangi sınıfları etkiliyor?
- Eğer gap <= 5pp: D-serisi için dropout + Mixup + SWA kombinasyonu düşünülmeli
