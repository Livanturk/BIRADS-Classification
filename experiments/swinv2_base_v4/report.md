---
experiment: swinv2_base_v4
date: 2026-03-20
config: configs/swinv2_base_v4.yaml
baseline: swinv2_base_v1
backbone: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
status: completed
best_epoch: 14
val_f1_macro: 0.6465
test_f1_macro: ~
---
# swinv2_base_v4

## Motivasyon / Hipotez
SwinV2-Base v1–v3 serisi yüksek val F1 üretemedi. v4: overfitting'i agresif
regularizasyon ile kontrol altına al — tüm dropout değerlerini artır, orta LR kullan.

## Config Özeti
- **Backbone:** `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- **Scheduler:** OneCycle (max_lr=3e-4)
- **Drop path rate:** 0.0 (backbone'da belirtilmedi)
- **Yüksek dropout:** projection=0.3, classification=0.6, attention=0.2, ffn=0.3

## Baseline'dan Değişiklikler (swinv2_base_v1 → v4)
| Alan | v1 (baseline) | v4 | Gerekçe |
|---|---|---|---|
| `model.backbone.projection_dropout` | `0.2` | `0.3` | Overfitting kontrolü |
| `model.classification.dropout` | `0.5` | `0.6` | Güçlü regularizasyon |
| `model.lateral_fusion.attention_dropout` | `0.15` | `0.2` | Fusion'da düzenleme |
| `model.lateral_fusion.ffn_dropout` | `0.2` | `0.3` | FFN düzenleme |
| `model.bilateral_fusion.output_dropout` | `0.25` | `0.3` | Hasta temsili düzenleme |
| `training.scheduler.max_lr` | `5.0e-4` | `3.0e-4` | Orta LR (v1-v2 arası) |
| `training.optimizer.weight_decay.backbone` | `0.05` | `0.08` | Backbone ağırlık düzeni |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 14)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6465 |
| F1 BR1 | 0.6071 |
| F1 BR2 | 0.6585 |
| F1 BR4 | 0.5814 |
| F1 BR5 | 0.7390 |
| AUC-ROC | — |
| Cohen's Kappa | — |
| Binary F1 | — |

### Test
Test değerlendirilmedi (val F1 hedefin altında, early stopped).

## Analiz
- Tüm SwinV2 deneyleri (v1–v4) epoch 10–17 civarında early stopped → scheduler problemi
- **Asıl sorun:** OneCycle scheduler SwinV2 için uygun değil — warmup olmadan başlangıçta
  yüksek LR attention ağırlıklarını bozuyor, erken convergence yerine erken durma
- BR4 en düşük (0.5814) — malign-benign ayrımı SwinV2'de ConvNeXtV2'ye göre zayıf kalmış
- SwinV2 v1-v4 serisi boyunca BR4 sistematik olarak düşük: CNN'lerin local doku tespiti
  transformer'ın global attention'ına göre bu görevde daha etkili görünüyor

## Sonraki Adım
→ SwinV2 için **cosine_warmup** scheduler zorunlu — warmup_epochs=10
→ LR: 5e-5 → 3e-5 (daha konservatif)
→ backbone_lr_scale: 0.2 → 0.15
→ Focal loss + agresif class weights ekle (focal_v1 stratejisi)
→ `swinv2_base_focal_v1`
