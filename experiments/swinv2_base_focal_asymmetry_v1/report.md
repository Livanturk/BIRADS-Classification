---
experiment: swinv2_base_focal_asymmetry_v1
date: 2026-04-05
config: configs/swinv2_base_focal_asymmetry_v1.yaml
baseline: swinv2_base_focal_v1
backbone: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
status: planned
best_epoch: ~
val_f1_macro: ~
test_f1_macro: ~
---

# swinv2_base_focal_asymmetry_v1

## Motivasyon / Hipotez
*(Bu deneyin amacını ve beklentini buraya yaz.)*

## Config Özeti
- **Backbone:** `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- **Scheduler:** `cosine_warmup`
- **LR:** `3e-05`

## Baseline'dan Değişiklikler
| Alan | Baseline (`swinv2_base_focal_v1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.asymmetry_loss` | `—` | `True` | *(açıkla)* |
| `training.asymmetry_only_malign` | `—` | `True` | *(açıkla)* |
| `training.asymmetry_weight` | `—` | `0.1` | *(açıkla)* |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: ?)
| Metrik | Değer |
|---|---|
| **F1 Macro** | ? |
| F1 BR1 | ? |
| F1 BR2 | ? |
| F1 BR4 | ? |
| F1 BR5 | ? |
| AUC-ROC | ? |
| Cohen's Kappa | ? |
| Binary F1 | ? |

### Test
| Metrik | Değer |
|---|---|
| **F1 Macro** | ? |
| F1 BR1 | ? |
| F1 BR2 | ? |
| F1 BR4 | ? |
| F1 BR5 | ? |
| AUC-ROC | ? |

## Analiz
*(Eğitim tamamlandığında doldur: ne çalıştı, ne çalışmadı, dikkat çeken bulgular.)*

## Sonraki Adım
*(Bu deneyin sonucuna göre bir sonraki adım ne olmalı?)*
