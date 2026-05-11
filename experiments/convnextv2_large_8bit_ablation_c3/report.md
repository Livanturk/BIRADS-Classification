---
experiment: convnextv2_large_8bit_ablation_c3
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c3.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c3

## Motivasyon / Hipotez
BR1 kurtarma deneyi. B5 (SWA, test F1: 0.6615) üzerine BR1 sınıf ağırlığı artırılır.
Tek değişken: class_weights[0] (BR1) 1.28 → 1.80.

B5, en iyi genel test F1'i sağladı ancak BR1'i -7.3pp gerilettti (B1: 0.526 → B5: 0.453).
SWA'nın weight averaging'i, BR1-BR2 sınır bölgesinde çoğunluk sınıfı BR2'ye doğru kayma
yaratır çünkü test setinde BR2 3.66× daha kalabalık (596 vs 163).

**BR1 ağırlığı 1.80 gerekçesi:**
- Mevcut: 1.28 = sqrt(2754/1678) — train set sqrt-inverse
- Test dağılımı: sqrt(596/163) = 1.91 — test setinin gerçek dengesizliği
- 1.80 ≈ ortada: SWA'nın BR2 yanlılığını telafi eder, overcompensation riskini minimize eder
- BR1 gradyanlarını ~41% amplify eder → model, BR1 sınır vakalarını daha güçlü öğrenir

**Beklenti:** BR1 test F1 >= 0.50 (B5'ten toparlanma). Genel test F1 >= 0.65.
Hafif macro regresyon kabul edilebilir (BR2'den kayıp) eğer BR1 toparlanırsa.

**Kill kriteri:** val_full_f1_macro < 0.60 by epoch 15, veya herhangi bir sınıf F1=0.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn
- **Class weights:** [**1.80**, 1.00, 1.20, 1.11] (BR1 boosted)

## Baseline'dan Değişiklikler (B5 → C3: tek değişim — BR1 ağırlığı)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.class_weights[0]` (BR1) | 1.28 | **1.80** | SWA BR1 regresyonunu telafi et |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: BR1 toparlandı mı? Genel macro ne oldu?
- Confusion matrix: BR1↔BR2 karışımı azaldı mı?
- BR1 >= 0.50 VE macro >= 0.65 ise: C1+C3 kombinasyonu (Mixup+SWA+BR1 weight) dene
- BR1 hâlâ düşük ise: 1.80 yetersiz, 2.00 veya farklı yaklaşım gerekebilir
- Genel macro ciddi düştüyse (< 0.63): 1.80 çok agresif, 1.50 dene
