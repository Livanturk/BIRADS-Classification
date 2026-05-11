---
experiment: convnextv2_large_8bit_ablation_c4
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c4.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_base.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c4

## Motivasyon / Hipotez
Kapasite azaltma ablation hücresi. B5 (SWA, test F1: 0.6615) baseline üzerine
backbone ConvNeXtV2-Large → ConvNeXtV2-Base düşürülür.
Tek değişken: model kapasitesi (Large ~197M → Base ~89M param).

ConvNeXtV2-Large, 8,557 eğitim hastası için aşırı parametrelenmiş olabilir.
Daha küçük backbone yapısal regularizer görevi görecek:
- Memorization kapasitesi azalır → daha az overfit
- Her parametre daha verimli kullanılmak zorunda → daha iyi genelleme
- Feature dim 1536→1024: projeksiyon ve fusion katmanları daha kompakt

Base backbone daha önceki pre-ablation deneylerde test edildi (`convnextv2_base_original`, F1: 0.6719)
ancak o zamanki pipeline'da bug'lar vardı ve SWA yoktu.

**Beklenti:** Val F1 hafif düşebilir ama test F1 artmalı. Gap <= 5pp.
Test F1 >= 0.65.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 15.

## Config Özeti
- **Backbone:** `convnextv2_base.fcmae_ft_in22k_in1k_384` (feature_dim=1024)
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn

## Baseline'dan Değişiklikler (B5 → C4: tek değişim — backbone kapasitesi)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.backbone.name` | `convnextv2_large.fcmae_ft_in22k_in1k_384` | `convnextv2_base.fcmae_ft_in22k_in1k_384` | Kapasite azaltma |
| `model.backbone.feature_dim` | `1536` | `1024` | Base backbone output dim |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: Kapasite düşürmek gap'i azalttı mı?
- Pre-ablation convnextv2_base_original ile karşılaştır: Bug fix + SWA ne kadar ekledi?
- Per-class F1: Küçük model hangi sınıflarda kayıp yaşıyor?
- Gap <= 5pp ise: Kapasite azaltma D-serisi için değerli bir bileşen
