---
experiment: swinv2_base_8bit_ablation_a2
date: 2026-04-09
config: configs/experiment_v2_birads/swinv2_base_8bit_ablation_a2.yaml
baseline: convnextv2_large_8bit_ablation_a1
backbone: swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
status: planned
best_epoch: ~
val_f1_macro: ~
test_f1_macro: ~
---

# swinv2_base_8bit_ablation_a2

## Motivasyon / Hipotez
Ablation A2 hücresi — hibrit pencere transformer (SwinV2-Base). ConvNeXtV2'nin lokal
receptive field'ı ile karşılaştırıldığında, SwinV2'nin shifted-window self-attention
mekanizması farklı uzamsal ilişkileri yakalayabilir. Bu, özellikle bilateral asimetri
tespitinde (sol/sağ meme karşılaştırması) farklı hata profili üretmeli ve heterogeneous
ensemble'da tamamlayıcı olmalı.

Önceki SwinV2 denemeleri (v1–v4) OneCycle scheduler ile epoch 10–17'de erken durdu.
Bu config cosine_warmup (10 warmup epoch) kullanarak bu sorunu çözmeli.

**Beklenti:** val F1 0.64–0.70. SwinV2-Base, ConvNeXtV2-Large'dan daha küçük (~88M vs ~198M
param) olduğundan tek başına daha düşük performans bekleniyor, ancak ensemble'a katkısı
hata çeşitliliği üzerinden olacak.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 20 veya cosine_warmup rağmen erken
durma (epoch < 20) ise durdur.

## Config Özeti
- **Backbone:** `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- **Scheduler:** `cosine_warmup` (warmup=10 epochs, min_lr=1e-6)
- **LR:** `3e-05`, backbone_lr_scale=0.15
- **Loss:** Focal (gamma=2.0), label_smoothing=0.08
- **Image size:** 768 (window_size=24 requires feature maps ÷24; 1024→256 stage0, 256%24≠0; 768→192, all clean)
- **Effective batch:** 2 × 32 = 64
- **drop_path_rate:** 0.2 (zorunlu — IN-22k supervised transformer overfit riski)

## Baseline'dan Değişiklikler (A1 → A2: sadece backbone-specific)
| Alan | Baseline (`convnextv2_large_8bit_ablation_a1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.backbone.name` | `convnextv2_large.fcmae_ft_in22k_in1k_384` | `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` | Ablation değişkeni: hibrit window transformer |
| `model.backbone.feature_dim` | `1536` | `1024` | SwinV2-Base son stage çıkışı |
| `model.backbone.drop_path_rate` | `—` | `0.2` | IN-22k supervised transformer overfit önlemi (zorunlu) |
| `data.image_size` | `1024` | `768` | window_size=24: 1024/4=256, 256%24≠0 → RuntimeError. 768/4=192, tüm stage'ler ÷24 |
| `training.optimizer.lr` | `5e-05` | `3e-05` | Transformer backbone daha düşük LR gerektirir |
| `training.optimizer.backbone_lr_scale` | `0.2` | `0.15` | Daha muhafazakar backbone fine-tuning |
| `training.scheduler.name` | `onecycle` | `cosine_warmup` | OneCycle Swin/ViT'te erken durma (pitfall #1) |
| `training.scheduler.warmup_epochs` | `—` | `10` | Transformer warmup standardı |
| `training.scheduler.min_lr` | `—` | `1e-6` | Cosine annealing alt sınır |
| `training.scheduler.max_lr` | `0.0005` | `—` | OneCycle parametresi (cosine'da yok) |
| `training.scheduler.pct_start` | `0.3` | `—` | OneCycle parametresi (cosine'da yok) |
| `training.scheduler.div_factor` | `10.0` | `—` | OneCycle parametresi |
| `training.scheduler.final_div_factor` | `100.0` | `—` | OneCycle parametresi |
| `training.scheduler.anneal_strategy` | `cos` | `—` | OneCycle parametresi |

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
