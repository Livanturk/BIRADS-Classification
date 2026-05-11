---
experiment: swinv2_base_cosine_v1
date: 2026-04-05
config: configs/swinv2_base_cosine_v1.yaml
baseline: swinv2_base_v4
backbone: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
status: planned
best_epoch: ~
val_f1_macro: ~
test_f1_macro: ~
---

# swinv2_base_cosine_v1

## Motivasyon / Hipotez
*(Bu deneyin amacını ve beklentini buraya yaz.)*

## Config Özeti
- **Backbone:** `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- **Scheduler:** `cosine_warmup`
- **LR:** `3e-05`

## Baseline'dan Değişiklikler
| Alan | Baseline (`swinv2_base_v4`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.backbone.drop_path_rate` | `—` | `0.2` | *(açıkla)* |
| `model.backbone.projection_dropout` | `0.3` | `0.2` | *(açıkla)* |
| `model.bilateral_fusion.attention_dropout` | `0.25` | `0.2` | *(açıkla)* |
| `model.bilateral_fusion.output_dropout` | `0.3` | `0.25` | *(açıkla)* |
| `model.classification.dropout` | `0.6` | `0.5` | *(açıkla)* |
| `model.lateral_fusion.attention_dropout` | `0.2` | `0.15` | *(açıkla)* |
| `model.lateral_fusion.ffn_dropout` | `0.3` | `0.2` | *(açıkla)* |
| `model.lateral_fusion.projection_dropout` | `0.25` | `0.2` | *(açıkla)* |
| `training.label_smoothing` | `—` | `0.05` | *(açıkla)* |
| `training.loss_type` | `—` | `cross_entropy` | *(açıkla)* |
| `training.optimizer.backbone_lr_scale` | `0.2` | `0.15` | *(açıkla)* |
| `training.optimizer.lr` | `5e-05` | `3e-05` | *(açıkla)* |
| `training.scheduler.anneal_strategy` | `cos` | `—` | *(açıkla)* |
| `training.scheduler.div_factor` | `10.0` | `—` | *(açıkla)* |
| `training.scheduler.final_div_factor` | `300.0` | `—` | *(açıkla)* |
| `training.scheduler.max_lr` | `0.0003` | `—` | *(açıkla)* |
| `training.scheduler.min_lr` | `—` | `1e-07` | *(açıkla)* |
| `training.scheduler.name` | `onecycle` | `cosine_warmup` | *(açıkla)* |
| `training.scheduler.pct_start` | `0.3` | `—` | *(açıkla)* |
| `training.scheduler.warmup_epochs` | `—` | `10` | *(açıkla)* |

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
