---
experiment: swinv2_base_focal_v1
date: 2026-04-05
config: configs/swinv2_base_focal_v1.yaml
baseline: swinv2_base_cosine_v1
backbone: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
status: planned
best_epoch: ~
val_f1_macro: ~
test_f1_macro: ~
---

# swinv2_base_focal_v1

## Motivasyon / Hipotez
*(Bu deneyin amacını ve beklentini buraya yaz.)*

## Config Özeti
- **Backbone:** `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- **Scheduler:** `cosine_warmup`
- **LR:** `3e-05`

## Baseline'dan Değişiklikler
| Alan | Baseline (`swinv2_base_cosine_v1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `data.augmentation.brightness` | `0.1` | `0.15` | *(açıkla)* |
| `data.augmentation.contrast` | `0.1` | `0.15` | *(açıkla)* |
| `data.augmentation.random_erasing` | `0.15` | `0.1` | *(açıkla)* |
| `training.class_weights` | `[1.28, 1.0, 1.2, 1.11]` | `[1.2, 1.4, 1.35, 1.0]` | *(açıkla)* |
| `training.focal_gamma` | `—` | `2.0` | *(açıkla)* |
| `training.label_smoothing` | `0.05` | `0.08` | *(açıkla)* |
| `training.loss_type` | `cross_entropy` | `focal` | *(açıkla)* |
| `training.loss_weights.binary_head` | `0.15` | `0.1` | *(açıkla)* |
| `training.loss_weights.full_head` | `0.5` | `0.45` | *(açıkla)* |
| `training.loss_weights.subgroup_head` | `0.35` | `0.45` | *(açıkla)* |

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
