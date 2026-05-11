---
experiment: convnextv2_large_8bit_ablation_b1
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_b1.yaml
baseline: convnextv2_large_8bit_ablation_a1_ce
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---
# convnextv2_large_8bit_ablation_b1

## Motivasyon / Hipotez
A-serisi 8-bit deneyleri 16-bit baseline'dan (test F1: 0.7233) ~9pp geride kaldı.
Kök neden analizi iki kritik hata ve config drift tespit etti:

1. **Normalizasyon hatası:** `transforms.py`'daki `DATASET_STATS_8BIT` eski
   `BIRADS-Full-Train-8Bit-Processed` datasetten kalma (mean=0.0990, std=0.1644).
   Gerçek `Dataset_1024_8bit` istatistikleri: mean=0.1210, std=0.1977. Mean'de %22,
   std'de %20 sapma — backbone'a yanlış dağılımda girdi gidiyor.

2. **Class weights hatası:** [1.90, 1.00, 1.44, 1.00] TEST set frekanslarından
   hesaplandı (B1=163, B2=596, B4=288, B5=608). Doğrusu TRAIN set sqrt-inverse:
   [1.28, 1.00, 1.20, 1.11]. Test dağılımı sızdırması + BR1'e %48 fazla ağırlık.

3. **Config drift:** label_smoothing 0.05→0.08, asymmetry_benign_weight 0.0→0.10,
   patience 20→15 — hepsi kanıtlanmış baseline'dan sapma.

Bu deney A1-CE'nin (en iyi A-serisi, test F1: 0.6370) düzeltilmiş versiyonu.
Tüm hatalar giderilip, 16-bit baseline parametrelerine geri dönülüyor.

**Beklenti:** Test F1 macro >= 0.70. Eğer sağlanırsa, hatalar doğrulanmış olur.

**Kill kriteri:** val_full_f1_macro < 0.65 by epoch 15.

## Config Ozeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **Normalizasyon:** mean=0.1210, std=0.1977 (Dataset_1024_8bit, duzeltilmis)

## Baseline'dan Degisiklikler (A1-CE -> B1)
| Alan | Baseline (`convnextv2_large_8bit_ablation_a1_ce`) | Bu Deney | Gerekce |
|---|---|---|---|
| Normalizasyon (transforms.py) | mean=0.0990, std=0.1644 | **mean=0.1210, std=0.1977** | Bug fix #1: eski dataset istatistikleri |
| `training.class_weights` | [1.90, 1.00, 1.44, 1.00] | **[1.28, 1.00, 1.20, 1.11]** | Bug fix #2: test → train set sqrt-inverse |
| `training.label_smoothing` | 0.08 | **0.05** | 16-bit baseline degeri |
| `training.asymmetry_benign_weight` | 0.10 | **0.0** | 16-bit baseline: BR1'e asimetri penalty uygulanmaz |
| `training.early_stopping.patience` | 15 | **20** | Daha fazla convergence suresi |

## Sonuclar

### Val — En Iyi Checkpoint (Epoch: —)
_Deney henuz calistirilmadi._

### Test (non-TTA, best checkpoint)
_Deney henuz calistirilmadi._

## Analiz
_Deney tamamlandiktan sonra doldurulacak._

## Sonraki Adim
- B1 test F1 >= 0.70 ise: Hatalar dogrulanmis. Ensemble icin B1 + B2 kullan.
- B1 test F1 < 0.70 ise: 8-bit pipeline veya 1024px cozunurluk sorunu.
  B3 ile 512px test et.
