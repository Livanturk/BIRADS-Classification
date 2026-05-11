---
experiment: convnextv2_large_8bit_ablation_a1_ce
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_a1_ce.yaml
baseline: convnextv2_large_8bit_ablation_a1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: completed
best_epoch: 6
val_f1_macro: 0.7141
test_f1_macro: 0.6370
mlflow_run_id: 1b6f7f063f8349e5aded6c2ba3693526
---

# convnextv2_large_8bit_ablation_a1_ce

## Motivasyon / Hipotez
Focal loss ablation kontrolü. A1 ile birebir aynı config, tek fark loss_type=ce (focal yerine).
Bu deney, focal loss'un BR2 darboğazı üzerindeki etkisini izole etmek için gerekli.
Eğer A1 (focal) A1-CE'den (CE) anlamlı şekilde üstünse, focal loss paper'da ayrı bir
ablation satırı olarak raporlanabilir.

**Beklenti:** val F1 0.66–0.71 (A1'den ~0.02 düşük). CE loss, kolay örneklere eşit ağırlık
verdiğinden BR2 gibi zor sınıflarda daha zayıf kalmalı.

**Kill kriteri:** A1 ile aynı — val_full_f1_macro < 0.60 by epoch 15.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** CE (CrossEntropy), label_smoothing=0.08
- **Image size:** 1024 (native)
- **Effective batch:** 2 × 32 = 64

## Baseline'dan Değişiklikler (A1-focal → A1-CE: tek değişim)
| Alan | Baseline (`convnextv2_large_8bit_ablation_a1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `training.loss_type` | `focal` | `ce` | Ablation kontrolü: focal loss etkisini izole et |
| `training.focal_gamma` | `2.0` | `—` | CE loss'ta gamma yok |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 6)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.7141 |
| F1 BR1 | 0.6366 |
| F1 BR2 | 0.7323 |
| F1 BR4 | 0.6908 |
| F1 BR5 | 0.7969 |
| AUC-ROC | 0.9159 |
| Cohen's Kappa | 0.6227 |
| Binary F1 | 0.9518 |

### Test (non-TTA, best checkpoint)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6370 |
| F1 BR1 | 0.5033 |
| F1 BR2 | 0.6891 |
| F1 BR4 | 0.5166 |
| F1 BR5 | 0.8390 |
| AUC-ROC | 0.8984 |
| Cohen's Kappa | 0.5572 |
| Binary F1 | 0.8950 |
| Accuracy | 0.6816 |

### Val → Test Gap
| Metrik | Val | Test | Δ |
|---|---|---|---|
| F1 Macro | 0.7141 | 0.6370 | **-0.0771** |
| F1 BR1 | 0.6366 | 0.5033 | **-0.1333** |
| F1 BR2 | 0.7323 | 0.6891 | -0.0432 |
| F1 BR4 | 0.6908 | 0.5166 | **-0.1742** |
| F1 BR5 | 0.7969 | 0.8390 | +0.0421 |
| AUC-ROC | 0.9159 | 0.8984 | -0.0175 |

### Focal vs CE Karşılaştırma (A1 vs A1-CE)
| Metrik | A1 (Focal) | A1-CE | Δ (Focal−CE) | Yorum |
|---|---|---|---|---|
| **Test F1 Macro** | 0.6270 | **0.6370** | -0.0100 | CE +1.0pp |
| Test F1 BR1 | 0.4271 | **0.5033** | -0.0762 | CE **+7.6pp** |
| Test F1 BR2 | **0.7120** | 0.6891 | +0.0229 | Focal +2.3pp |
| Test F1 BR4 | 0.5163 | **0.5166** | -0.0003 | ~eşit |
| Test F1 BR5 | **0.8524** | 0.8390 | +0.0134 | Focal +1.3pp |
| Test AUC-ROC | 0.8981 | **0.8984** | -0.0003 | ~eşit |
| Val F1 Macro | 0.7108 | **0.7141** | -0.0033 | CE +0.3pp |

## Analiz

**Eğitim dinamikleri:** A1 ile benzer convergence profili — epoch 6'da best val F1'e ulaştı
(A1: epoch 4). CE loss'un daha smooth gradient landscape'i 2 epoch daha fazla iyileşme
sağlamış olabilir. Toplam 21 epoch, patience=15 ile early stop.

**Ana bulgu — Focal loss hipotezi RED:**
Bu deney, focal loss ablation kontrolü olarak tasarlanmıştı. Sonuç **beklenenden farklı**:
CE loss, focal loss'tan hem val'de (+0.3pp) hem test'te (+1.0pp) **daha iyi** macro F1
üretiyor. Hipotez ("focal loss BR2 darboğazında iyileşme sağlar") kısmen doğru — focal,
BR2'de +2.3pp daha iyi — ancak bu kazanım, BR1'deki **-7.6pp** kayıpla fazlasıyla siliniyor.

**Focal loss mekanizması analizi:**
Focal loss gamma=2.0, yüksek confidence örneklerin loss katkısını $(1-p)^\gamma$ ile
azaltır. BR1 (normal meme dokusu) örneklerinin çoğu yüksek confidence ile sınıflanıyor
→ focal loss bunları down-weight ediyor → model, BR1'in sınır vakalarını (density
varyasyonları, benign kalsifikasyonlar) öğrenemiyor. Sonuç: test'te BR1 precision düşüyor,
BR1→BR2 karışımı artıyor.

**Generalizasyon:** Val→test gap 7.7pp — A1'den (8.4pp) biraz daha iyi. BR1 test gap'i
-13.3pp — A1'den (-21.6pp) **çok daha iyi**. CE loss, sınıf ayrımını daha dengeli öğretiyor.

**Binary F1:** Val 0.9518 → test 0.8950. Benign/malign ayrımında CE loss val'de focal'dan
iyi (0.9518 vs 0.9473), ama test'te focal daha iyi (0.9125 vs 0.8950). Bu da focal loss'un
malign tespitte daha güçlü olduğunu doğruluyor — ama 4-sınıf macro F1'de bu avantaj
kayboluyor.

**Paper implikasyonu:** Focal loss bu dataset/mimari için net kazanım sağlamıyor. Paper'da
"focal loss does not significantly improve macro F1 on this 4-class BI-RADS task; it
trades BR1 recall for BR2 precision" şeklinde raporlanabilir. McNemar testi ile
istatistiksel anlamlılık doğrulanmalı.

## Sonraki Adım
- McNemar testi: A1 vs A1-CE farkının istatistiksel anlamlılığını doğrula (p < 0.05?)
- TTA ile her iki modeli de değerlendir — focal vs CE farkı TTA altında değişiyor mu?
- Paper'da focal loss ablation satırı: CE'nin varsayılan olarak kullanılmasını öner
- Eğer ensemble'a sadece bir ConvNeXtV2 girecekse, A1-CE tercih edilmeli (daha dengeli
  per-class profil, daha iyi BR1)
