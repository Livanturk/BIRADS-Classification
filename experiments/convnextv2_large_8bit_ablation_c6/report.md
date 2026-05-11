---
experiment: convnextv2_large_8bit_ablation_c6
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c6

## Motivasyon / Hipotez
Asimetri kaybı ablation hücresi. B5 (SWA, test F1: 0.6615) baseline üzerine
asymmetry_loss_weight 0.10→0.0 yapılır.
Tek değişken: asimetri kaybı varlığı.

Bilateral Fusion bloğu zaten sol-sağ meme farkını öğreniyor (use_diff=true, use_avg=true).
Ek asymmetry_loss (ağırlık=0.10), sol-sağ asimetriyi explicit olarak
penalize eden bir auxiliary loss. Bug fix'ler sonrası pipeline temizlendi —
artık bu loss'un gerçek katkısını matematiksel olarak kanıtlamamız gerekiyor.

Üç olası sonuç:
1. **Test F1 artar:** Asimetri kaybı zararlı gürültü ekliyordu → bilateral fusion yeterli
2. **Test F1 aynı kalır:** Asimetri kaybı ne faydalı ne zararlı → gereksiz karmaşıklık
3. **Test F1 düşer:** Asimetri kaybı faydalı → korunmalı

Bu deney, mimarinin minimum gerekli bileşenlerini belirleme açısından kritiktir.

**Beklenti:** Test F1 >= 0.66 (B5 ile eşit veya daha iyi).

**Kill kriteri:** val_full_f1_macro < 0.60 by epoch 15.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn
- **Asymmetry loss:** DISABLED (weight=0.0)

## Baseline'dan Değişiklikler (B5 → C6: tek değişim — asimetri kaybı)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.asymmetry_loss_weight` | `0.10` | `0.0` | Asimetri kaybı katkı testi |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: Asimetri kaybı olmadan ne değişti?
- Per-class F1: Hangi sınıflar etkilendi? (Özellikle malign BR4/BR5)
- GradCAM: Bilateral fusion hala asimetrik bölgelere odaklanıyor mu?
- Sonuç pozitifse: Gelecek deneylerde asymmetry_loss_weight=0.0 varsayılan yapılır
