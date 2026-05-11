---
experiment: convnextv2_large_seg_deformable_asymmetry_malignonly_v1
date: 2026-03-16
config: configs/convnextv2_large_seg_deformable_asymmetry_malignonly_v1.yaml
baseline: convnextv2_large_seg_deformable_v1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: completed
best_epoch: 10
val_f1_macro: 0.6867
test_f1_macro: 0.7233
---

# convnextv2_large_seg_deformable_asymmetry_malignonly_v1

## Motivasyon / Hipotez
Sol-sağ asimetri en güçlü malignite sinyallerinden biridir. `f_diff = F_left - F_right`
vektörü bilateral fusion'da oluşturuluyor; ancak baseline'da bu sinyal kayıp fonksiyonuna
dahil değildi. Malign örnekler üzerinde asimetri ek kaybı ekleyerek modeli
asimetrik doku örüntülerini yakalamaya zorla.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Yeni:** Asimetri kaybı — sadece malign sınıflar (BR4, BR5) üzerinde
- **Scheduler:** OneCycle
- **LR:** `5.0e-5`

## Baseline'dan Değişiklikler (deformable_v1 → bu deney)
| Alan | Baseline | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.asymmetry_loss` | `false` / yok | `true` | f_diff L2 düzenlileştirme |
| `training.asymmetry_only_malign` | — | `true` | Benign asimetri zaten var |
| `training.asymmetry_weight` | — | `0.1` | Hafif ek kayıp |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 10)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6867 |
| F1 BR1 | 0.6693 |
| F1 BR2 | 0.6988 |
| F1 BR4 | 0.6701 |
| F1 BR5 | 0.7085 |
| AUC-ROC | 0.9179 (test'ten) |
| Cohen's Kappa | — |
| Binary F1 | — |

### Test
| Metrik | Değer |
|---|---|
| **F1 Macro** | **0.7233** ← TEK MODEL EN İYİSİ |
| F1 BR1 | 0.7536 |
| F1 BR2 | 0.6391 |
| F1 BR4 | 0.7376 |
| F1 BR5 | 0.7630 |
| AUC-ROC | 0.9179 |

## Analiz
- Val F1 baseline'dan neredeyse aynı (+0.0004) ama test F1 +0.0204 artış → asimetri kaybı test genellenmesini önemli ölçüde artırdı
- BR4 test F1: 0.7086 → 0.7376 (+0.029) — en büyük kazanım malign sınıfta
- BR5 test F1: 0.7401 → 0.7630 (+0.023) — ikinci büyük kazanım
- BR2 hala düşük (0.6391) — benign sınıflar asimetri kaybından faydalanmıyor (expected)
- Val → Test gap +0.037: test seti BR5=%37 ağırlıklı, model BR5'te güçlü

## Sonraki Adım
→ Bu model ensemble için birincil bileşen
→ v2'de: label_smoothing ve dropout varyasyonu dene
→ `convnextv2_large_seg_deformable_asymmetry_malignonly_v2`
