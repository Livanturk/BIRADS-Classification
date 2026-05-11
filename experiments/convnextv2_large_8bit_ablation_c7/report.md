---
experiment: convnextv2_large_8bit_ablation_c7
date: 2026-04-11
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c7.yaml
baseline: convnextv2_large_8bit_ablation_b5
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: planned
best_epoch: —
val_f1_macro: —
test_f1_macro: —
mlflow_run_id: —
---

# convnextv2_large_8bit_ablation_c7

## Motivasyon / Hipotez
Focal loss ablation hücresi. B5 (SWA, test F1: 0.6615) baseline üzerine
loss_type CE→Focal değiştirilir.
Tek değişken: loss fonksiyonu türü.

A-serisinde ConvNeXtV2 için CE > Focal bulunmuştu (Lesson #11):
- A1 (Focal): test 0.6270, BR1 F1 düşük
- A1-CE (CE): test 0.6370, BR1 F1 daha yüksek

Ancak bu testler:
1. **Buglu pipeline** ile yapıldı (yanlış norm, yanlış class weights)
2. **SWA olmadan** yapıldı

Temiz pipeline + SWA ile focal loss'un davranışı farklı olabilir:
- Focal (gamma=2.0): Yüksek güvenirlikli kolay örneklerin gradient katkısını azaltır
- SWA: Karar sınırlarını düzleştirir
- Kombine etki: Focal → zor örneklere (BR1, BR4) odaklan, SWA → sınırları stabilize et

Özellikle SWA'nın BR1 regresyonu (Lesson #17, -7.3pp) focal loss'un
hard-example mining'i ile telafi edilebilir.

**Beklenti:** Test F1 >= 0.66, BR1 F1 >= 0.50.

**Kill kriteri:** val_full_f1_macro < 0.55 by epoch 15.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** **Focal** (gamma=2.0), label_smoothing=0.05
- **Image size:** 1024 (native)
- **Effective batch:** 2 x 32 = 64
- **SWA:** start_epoch=5, AveragedModel + update_bn

## Baseline'dan Değişiklikler (B5 → C7: tek değişim — focal loss)
| Alan | Baseline (`convnextv2_large_8bit_ablation_b5`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.loss_type` | `"ce"` | `"focal"` | Hard-example mining |
| `training.focal_gamma` | — | `2.0` | Standart focal gamma |

## Sonuçlar

_Deney tamamlandığında doldurulacak._

## Analiz

_Deney tamamlandığında doldurulacak._

## Sonraki Adım

- B5 ile karşılaştır: Focal loss + SWA kombine etkisi nedir?
- A1 (Focal, no SWA) ile karşılaştır: SWA, focal'ın BR1 sorununu çözdü mü?
- Per-class F1: BR1 ve BR4 (zor sınıflar) kazanç sağladı mı?
- Lesson #11 revize edilmeli mi? (Focal zarar verir → koşullu olarak faydalı)
