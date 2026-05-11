---
experiment: ensemble_3model_convnextv2_large
date: 2026-03-18
config: "ensemble_evaluate.py (MODELS listesi)"
baseline: convnextv2_large_seg_deformable_asymmetry_malignonly_v1
backbone: convnextv2_large (×3)
status: completed
best_epoch: ~
val_f1_macro: ~
test_f1_macro: 0.7357
---

# ensemble_3model_convnextv2_large

## Motivasyon / Hipotez
3 bağımsız ConvNeXtV2-Large modeli aynı mimariyle farklı hiperparametre
ve regularizasyon ile eğitildi. Homojen ensemble her model için farklı
hata örüntüleri oluşturuyorsa test F1'i artırmalı.

## Config Özeti
- **Model 1:** `convnextv2_large_seg_deformable_v1` — test F1: 0.7029
- **Model 2:** `convnextv2_large_seg_deformable_asymmetry_malignonly_v1` — test F1: 0.7233
- **Model 3:** `convnextv2_large_seg_deformable_asymmetry_malignonly_v2` — test F1: 0.7018
- **Yöntem:** Softmax + TTA (Test-Time Augmentation)
- **Ağırlıklar:** Eşit ortalama

## Sonuçlar

### Test (Ensemble + TTA)
| Metrik | M1 | M2 | M3 | **Ensemble** |
|---|---|---|---|---|
| **F1 Macro** | 0.7029 | 0.7233 | 0.7018 | **0.7357** |
| F1 BR1 | 0.7319 | 0.7536 | 0.6752 | 0.7336 |
| F1 BR2 | 0.6311 | 0.6391 | 0.6617 | **0.6613** |
| F1 BR4 | 0.7086 | 0.7376 | 0.6920 | **0.7460** |
| F1 BR5 | 0.7401 | 0.7630 | 0.7782 | **0.8017** |
| AUC-ROC | 0.9129 | 0.9179 | 0.8956 | 0.9184 |
| Accuracy | 0.704 | 0.725 | 0.702 | 0.735 |
| Cohen's κ | 0.6053 | 0.6333 | 0.6027 | 0.6467 |
| Binary F1 | 0.9489 | 0.9526 | 0.9615 | 0.9489 |

## Analiz
- Ensemble, bireysel en iyi modeli (0.7233) 0.0124 geçti — diversity çalışıyor
- BR2 (+0.0222 vs M2): M3'ün BR2'deki yüksek performansı ensemble'ı çekiyor
- BR4 (+0.0084 vs M2): asimetri kaybının iki versiyonu birbirini tamamlıyor
- BR5 (+0.0387 vs M2): M3'ün BR5 dominasyonu ensemble'da korunuyor
- BR1 geriledi (-0.02 vs M2): M3'ün düşük BR1 performansı ortalamayı çekiyor
- **Darboğaz:** BR2 hala 0.66 → homojen ensemble bunu çözemedi

## Sonraki Adım
→ BR2 bottleneck için farklı backbone gerekiyor — **heterojen ensemble**
→ CNN + Transformer: SwinV2-Base veya DINOv2 partneri
→ Hedef: test F1 > 0.75 (ensemble'dan +0.014 kazanım gerekiyor)
→ `swinv2_base_focal_v1`
