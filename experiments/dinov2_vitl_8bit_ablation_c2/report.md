---
experiment: dinov2_vitl_8bit_ablation_c2
date: 2026-04-11
config: configs/experiment_v2_birads/dinov2_vitl_8bit_ablation_c2.yaml
baseline: dinov2_vitl_8bit_ablation_a3
backbone: vit_large_patch14_dinov2.lvd142m
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# dinov2_vitl_8bit_ablation_c2

## Motivasyon / Hipotez
DINOv2 kurtarma deneyi. A3'ün (DINOv2 + focal, test F1: 0.6325) B-serisi bug fix paketi
uygulanmış versiyonu. Kritik fark: **focal loss KORUNUR** (B2'deki CE geçişi yapılmaz).

B2 (DINOv2 + CE + bug fixes) test F1'i 0.6136'ya geriletmişti (-1.9pp vs A3).
Lesson #14: DINOv2'nin self-supervised özellikleri domain-agnostic olduğundan, focal loss'un
hard-example mining'i (gamma=2.0) zor BI-RADS sınırlarına odaklanmak için gereklidir.
CE loss bu avantajı ortadan kaldırıp DINOv2'nun kapasitesini kolay örneklere harcar.

Bug fix paketi (A3'ten değişiklikler):
1. **class_weights:** [1.90,1.00,1.44,1.00] → [1.28,1.00,1.20,1.11] (test→train set)
2. **label_smoothing:** 0.08 → 0.05 (16-bit baseline)
3. **asymmetry_benign_weight:** 0.10 → 0.0 (BR1'e penalty uygulanmaz)
4. **patience:** 15 → 20 (daha fazla convergence süresi)

**KORUNAN:** loss_type=focal, focal_gamma=2.0, tüm DINOv2-spesifik parametreler
(image_size=518, lr=2e-5, backbone_lr_scale=0.10, cosine_warmup, weight_decay.backbone=0.10).

**Beklenti:** Test F1 macro >= 0.65. Hem A3'ü (0.6325, buggy) hem B2'yi (0.6136, yanlış loss)
geçmeli. BR5 gücünü (0.857) koruyarak ensemble'da complementary katkı sağlayacak.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 20.

## Config Özeti
- **Backbone:** `vit_large_patch14_dinov2.lvd142m` (LVD-142M self-supervised)
- **Scheduler:** `cosine_warmup` (warmup=10 epochs, min_lr=1e-6)
- **LR:** `2e-05`, backbone_lr_scale=0.10
- **Loss:** Focal (gamma=2.0), label_smoothing=0.05
- **Image size:** 518 (37x14 patch grid, 1369 spatial tokens)
- **Effective batch:** 2 x 32 = 64
- **Normalizasyon:** mean=0.1210, std=0.1977 (Dataset_1024_8bit, düzeltilmiş)

## Baseline'dan Değişiklikler (A3 → C2: bug fix paketi, focal KORUNUR)
| Alan | Baseline (`dinov2_vitl_8bit_ablation_a3`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.class_weights` | [1.90, 1.00, 1.44, 1.00] | **[1.28, 1.00, 1.20, 1.11]** | Bug fix: test → train set sqrt-inverse |
| `training.label_smoothing` | 0.08 | **0.05** | 16-bit baseline değeri |
| `training.asymmetry_benign_weight` | 0.10 | **0.0** | BR1'e asimetri penalty uygulanmaz |
| `training.early_stopping.patience` | 15 | **20** | A3 ep17'de peak — daha fazla süre |
| `training.loss_type` | focal | **focal** (KORUNDU) | Lesson #14: DINOv2 focal gerektirir |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- A3 ile karşılaştır: Bug fix'ler focal loss ile birlikte çalıştı mı?
- B2 ile karşılaştır: Focal (C2) vs CE (B2) farkı doğrulandı mı?
- BR5 hâlâ güçlü mü (>= 0.85)?
- Test F1 >= 0.65 ise: ensemble adayı (C1 + C2 complementary error profili)
- Gelecek: C2 + SWA kombinasyonu düşünülebilir
