---
experiment: EXPERIMENT_NAME
date: YYYY-MM-DD
config: configs/EXPERIMENT_NAME.yaml
baseline: BASELINE_NAME
backbone: BACKBONE_TIMM_NAME
status: planned
best_epoch: ~
val_f1_macro: ~
test_f1_macro: ~
---

# EXPERIMENT_NAME

## Motivasyon / Hipotez
*(Bu deneyin amacını 2-3 cümleyle açıkla. Ne test ediyorsun, neden bu parametre?)*

## Config Özeti
- **Backbone:** `BACKBONE_TIMM_NAME`
- **Scheduler:** `SCHEDULER_NAME`
- **LR:** `LR_VALUE` — backbone_lr_scale=`SCALE`

## Baseline'dan Değişiklikler
| Alan | Baseline (`BASELINE_NAME`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `path.to.param` | `eski_değer` | `yeni_değer` | Neden değiştirdi? |

*(Otomatik doldurmak için: `python scripts/report.py new EXPERIMENT_NAME --baseline BASELINE_NAME`)*

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

*(Otomatik doldurmak için: `python scripts/report.py fill EXPERIMENT_NAME`)*

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
*(Eğitim tamamlandığında doldur.)*

- Val → Test gap ne kadar? (normal: +0.02 – +0.04)
- Hangi sınıf en çok iyileşti / geriledi?
- Training curve'de anomali var mı (plateau, spike)?
- Hipotez doğrulandı mı?

## Sonraki Adım
*(Bu deneyin sonucuna göre bir sonraki adım ne olmalı?)*
