---
experiment: dinov2_vitl_8bit_ablation_b2
date: 2026-04-09
config: configs/experiment_v2_birads/dinov2_vitl_8bit_ablation_b2.yaml
baseline: dinov2_vitl_8bit_ablation_a3
backbone: vit_large_patch14_dinov2.lvd142m
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---
# dinov2_vitl_8bit_ablation_b2

## Motivasyon / Hipotez
A3 (DINOv2 ViT-L/14, focal loss, test F1: 0.6325) A-serisinin en iyi generalizasyon
profiline sahipti (val-test gap=6.2pp vs ConvNeXtV2'nin 7.7-8.4pp). Self-supervised
ön-eğitim domain-agnostic özellikler ürettiğinden overfit riski düşük.

B2, A3'ün düzeltilmiş versiyonu. B1 ile aynı bug-fix'ler uygulanır, ayrıca A-serisi
ablation sonucu olan CE loss'a geçiş yapılır:

1. **Normalizasyon düzeltmesi** (transforms.py: 0.0990/0.1644 → 0.1210/0.1977)
2. **Class weights düzeltmesi** ([1.90, 1.00, 1.44, 1.00] → [1.28, 1.00, 1.20, 1.11])
3. **CE loss** (focal → ce, ablation kazanani)
4. **Config drift reversal** (label_smoothing, asymmetry_benign_weight, patience)

**Beklenti:** Test F1 macro >= 0.66. DINOv2'nin BR5 gücü (0.857) ve farklı hata
profili ensemble için complementary katkı sağlayacak.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 20.

## Config Ozeti
- **Backbone:** `vit_large_patch14_dinov2.lvd142m` (LVD-142M self-supervised)
- **Scheduler:** `cosine_warmup` (warmup=10 epochs, min_lr=1e-6)
- **LR:** `2e-05`, backbone_lr_scale=0.10
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 518 (37x14 patch grid, 1369 spatial tokens)
- **Effective batch:** 2 x 32 = 64
- **Normalizasyon:** mean=0.1210, std=0.1977 (Dataset_1024_8bit, duzeltilmis)

## Baseline'dan Degisiklikler (A3 -> B2)
| Alan | Baseline (`dinov2_vitl_8bit_ablation_a3`) | Bu Deney | Gerekce |
|---|---|---|---|
| Normalizasyon (transforms.py) | mean=0.0990, std=0.1644 | **mean=0.1210, std=0.1977** | Bug fix #1 |
| `training.class_weights` | [1.90, 1.00, 1.44, 1.00] | **[1.28, 1.00, 1.20, 1.11]** | Bug fix #2: train set sqrt-inverse |
| `training.loss_type` | focal | **ce** | Ablation sonucu: CE > focal (+1.0pp macro F1) |
| `training.focal_gamma` | 2.0 | — | CE loss'ta gamma yok |
| `training.label_smoothing` | 0.08 | **0.05** | 16-bit baseline degeri |
| `training.asymmetry_benign_weight` | 0.10 | **0.0** | BR1'e asimetri penalty uygulanmaz |
| `training.early_stopping.patience` | 15 | **20** | A3 ep17'de peak — daha fazla sure |

## Sonuclar

### Val — En Iyi Checkpoint (Epoch: —)
_Deney henuz calistirilmadi._

### Test (non-TTA, best checkpoint)
_Deney henuz calistirilmadi._

## Analiz
_Deney tamamlandiktan sonra doldurulacak._

## Sonraki Adim
- Ensemble'da B2'nin rolu: BR5 gucu ve ConvNeXtV2'den farkli hata profili
- TTA ile test F1 iyilesmesini olc — DINOv2'nin rotation invariance ozelligi
  TTA'dan daha fazla fayda saglayabilir
