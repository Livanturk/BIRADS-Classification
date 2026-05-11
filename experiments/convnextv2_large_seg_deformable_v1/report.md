---
experiment: convnextv2_large_seg_deformable_v1
date: 2026-03-15
config: configs/convnextv2_large_seg_deformable_v1.yaml
baseline: convnextv2_base_original
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: completed
best_epoch: 10
val_f1_macro: 0.6863
test_f1_macro: 0.7029
---

# convnextv2_large_seg_deformable_v1

## Motivasyon / Hipotez
ConvNeXtV2-Base → Large'e geçişin getirdiği kapasite artışı ile birlikte
deformable cross-attention, CC ve MLO görüntüleri arasında global yerine
seçici spatial bölgelere odaklanmayı sağlar. Segmentasyon maskesi ile
letterbox sıfır piksellerin attention'ı bozması önlenir.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Lateral Fusion:** Deformable Cross-Attention (`num_points=4`)
- **Scheduler:** OneCycle
- **LR:** `5.0e-5`, backbone_lr_scale=`0.2`

## Baseline'dan Değişiklikler (convnextv2_base_original → bu deney)
| Alan | Baseline | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.backbone.name` | `convnextv2_base.*` | `convnextv2_large.*` | Daha fazla kapasite |
| `model.backbone.feature_dim` | `1024` | `1536` | Large'ın çıkış boyutu |
| `lateral_fusion.use_deformable` | `false` | `true` | Seçici spatial dikkat |
| `data.root_dir` | `PNG-8Bit` | `BIRADS-Full-Train-8Bit-Processed` | Segmentasyonlu dataset |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 10)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6863 |
| F1 BR1 | 0.6573 |
| F1 BR2 | 0.6880 |
| F1 BR4 | 0.6713 |
| F1 BR5 | 0.7283 |
| AUC-ROC | 0.9129 (test'ten) |
| Cohen's Kappa | — |
| Binary F1 | — |

### Test
| Metrik | Değer |
|---|---|
| **F1 Macro** | **0.7029** |
| F1 BR1 | 0.7319 |
| F1 BR2 | 0.6311 |
| F1 BR4 | 0.7086 |
| F1 BR5 | 0.7401 |
| AUC-ROC | 0.9129 |

## Analiz
- Val → Test gap: +0.0166 (normal, test seti BR2/BR5 ağırlıklı)
- BR2 test F1 en düşük (0.6311) — BI-RADS 2 hala en zor sınıf
- Deformable attention standart cross-attention'a göre marginal fark, ancak
  bellek tasarrufu açısından avantajlı

## Sonraki Adım
→ Asimetri bilinci eklenmeli: F_diff (sol-sağ farkı) kaybı bilateral fusion'a ekle
→ `convnextv2_large_seg_deformable_asymmetry_malignonly_v1`
