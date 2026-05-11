---
experiment: convnextv2_large_seg_deformable_asymmetry_malignonly_v2
date: 2026-03-17
config: configs/convnextv2_large_seg_deformable_asymmetry_malignonly_v2.yaml
baseline: convnextv2_large_seg_deformable_asymmetry_malignonly_v1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: completed
best_epoch: 100
val_f1_macro: 0.6959
test_f1_macro: 0.7018
---

# convnextv2_large_seg_deformable_asymmetry_malignonly_v2

## Motivasyon / Hipotez
v1 çok erken durdu (epoch 10). Early stopping patience ve regularizasyon
parametrelerini ayarlayarak daha uzun eğitim ile val üst sınırına ulaş.
Label smoothing ve class weight varyasyonu dene.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Asimetri kaybı:** Aktif
- **Epochs:** 100 (full convergence)
- **Fark:** Muhtemelen dropout / label_smoothing varyasyonu

## Baseline'dan Değişiklikler (asymmetry_v1 → v2)
| Alan | Baseline | Bu Deney | Gerekçe |
|---|---|---|---|
| *(Detaylar için config'e bak)* | | | |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 100)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6959 |
| F1 BR1 | 0.6850 |
| F1 BR2 | 0.6630 |
| F1 BR4 | 0.6638 |
| F1 BR5 | 0.7719 |
| AUC-ROC | 0.8956 (test'ten) |
| Cohen's Kappa | — |
| Binary F1 | — |

### Test
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.7018 |
| F1 BR1 | 0.6752 |
| F1 BR2 | 0.6617 |
| F1 BR4 | 0.6920 |
| F1 BR5 | 0.7782 |
| AUC-ROC | 0.8956 |

## Analiz
- Val F1 v1'den yüksek (0.6959 > 0.6867) ama test F1 daha düşük (0.7018 < 0.7233)
- Val → Test gap ters yönde: +0.0059 — model v1'e göre test'e daha az genelleniyor
- BR5 test F1 en yüksek (0.7782) — ama BR4 geriledi (0.6920 < 0.7376)
- AUC-ROC 0.8956 — v1'in 0.9179'undan önemli düşüş → overfitting veya regularizasyon uyumsuzluğu
- Ensemble'a dahil edildi çünkü BR2 ve BR5'te v1'den farklı hata örüntüsü var

## Sonraki Adım
→ 3-model homojen ensemble: deformable_v1 + asymmetry_v1 + **asymmetry_v2**
→ `ensemble_3model_convnextv2_large`
