---
experiment: convnextv2_large_8bit_ablation_b5
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_b5.yaml
baseline: convnextv2_large_8bit_ablation_b1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_b5

## Motivasyon / Hipotez
Weight averaging ablation hucresi. B1 ile birebir ayni config, tek fark SWA aktif.

A1-CE (B1'in atasi) epoch 6'da best val F1'e ulasti ve epoch 21'de early stop oldu.
Bu, modelin keskin (sharp) bir minimuma hizla ulastigini gosterir. SWA, epoch 5'ten
itibaren model agirliklarini ortalayarak daha duz (flat) bir minimuma yakinsar.
Flat minima'lar generalizasyonda daha iyi performans gosterir (Izmailov et al., 2018).

SWA ek egitim maliyeti gerektirmez — ayni epoch'lar boyunca model parametreleri ortalalanir.
Egitim sonunda BatchNorm istatistikleri train set uzerinde guncellenir (update_bn).
Eger SWA F1 > best checkpoint F1 ise, SWA modeli otomatik olarak secilir.

**Beklenti:** SWA test F1, best checkpoint'tan +0.5-2pp daha iyi olmali.

**Kill kriteri:** B1 ile ayni (val_full_f1_macro < 0.65 by epoch 15).
SWA post-training degerlendirme oldugu icin best checkpoint'a zarar veremez.

## Config Ozeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn

## Baseline'dan Degisiklikler (B1 → B5: tek degisim — SWA)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b1`) | Bu Deney | Gerekce |
|---|---|---|---|
| `training.use_swa` | `false` (default) | `true` | Weight averaging aktif |
| `training.swa_start_epoch` | — | `5` | A1-CE best epoch=6, oncesindem basla |

## Sonuclar

_Deney tamamlandiginda doldurulacak._

### Beklenen Cikti Formati
Bu deney iki sonuc uretir:
- **Best Checkpoint F1:** Standard early stopping ile secilen en iyi model (B1 ile ayni olmali)
- **SWA F1:** Epoch 5'ten itibaren ortalalanmis model

## Analiz

_Deney tamamlandiginda doldurulacak._

## Sonraki Adim

- Best checkpoint vs SWA: SWA ne kadar iyilestirdi?
- Per-class F1 karsilastirmasi: SWA hangi siniflari iyilestirdi?
- Eger SWA >= +1pp ise, B3+B5 (Mixup+SWA) veya B4+B5 (CORAL+SWA) kombinasyonu dene
- SWA her zaman "free lunch" oldugu icin, final model seciminde SWA versiyonu tercih edilebilir
