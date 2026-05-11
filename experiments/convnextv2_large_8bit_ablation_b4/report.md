---
experiment: convnextv2_large_8bit_ablation_b4
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_b4.yaml
baseline: convnextv2_large_8bit_ablation_b1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_b4

## Motivasyon / Hipotez
Loss fonksiyonu ablation hucresi. B1'deki standard 4-class CE loss yerine CORAL ordinal
regression loss kullanilir. BI-RADS siniflari dogal ordinal yapiya sahiptir (1 < 2 < 4 < 5).

CORAL, K-1=3 kumulatif binary siniflandirici ile uzak sinif hatalarini (BR1→BR5) komsu
hatalardan (BR1→BR2) daha agir cezalandirir. Bu, en zayif sinif BR1'i (test F1: 0.42-0.50)
iyilestirmeli — model, komsu sinif karisiklarinda daha yumusak gradyanlar alir.

**KRITIK:** Subgroup head'ler KAPATILDI. Onceki ordinal_v1 deneyinde CORAL P(>=2) esigi ile
benign subgroup head (BR1 vs BR2) cakisti ve F1=0 cokusu yasandi. Bu config'de
`use_subgroup_head: false` ve loss_weights yeniden dengelendi.

**Beklenti:** Test F1 macro >= 0.68. BR1 test F1'de B1'e gore +3pp iyilesme.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 15, veya herhangi bir sinif F1=0.

## Config Ozeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CORAL Ordinal (CE-based binary thresholds), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **Head yapisi:** Binary (0.25) + CORAL Ordinal (0.75), subgroup kapatildi

## Baseline'dan Degisiklikler (B1 → B4: loss yapisi degisimi)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b1`) | Bu Deney | Gerekce |
|---|---|---|---|
| `ablation.use_ordinal_head` | `false` | `true` | CORAL ordinal loss aktif |
| `ablation.use_subgroup_head` | `true` | `false` | Gradient cakismasini onle |
| `training.loss_weights.binary_head` | `0.10` | `0.25` | Subgroup payini karsilama |
| `training.loss_weights.subgroup_head` | `0.45` | `0.00` | Kapatildi |
| `training.loss_weights.full_head` | `0.45` | `0.75` | CORAL'a guclu sinyal |

## Sonuclar

_Deney tamamlandiginda doldurulacak._

## Analiz

_Deney tamamlandiginda doldurulacak._

## Sonraki Adim

- B1 ile karsilastir: BR1 F1 iyilesti mi? Uzak sinif hatalari (BR1→BR5) azaldi mi?
- Confusion matrix off-diagonal analizi: CORAL ile komsu vs uzak hatalar nasil degisti?
- Eger BR1 +3pp VE macro F1 >= 0.68 ise, B4+B5 (CORAL+SWA) kombinasyonu dene
- Paper'da ordinal loss ablation satiri olarak raporlanabilir
