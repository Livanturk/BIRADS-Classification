---
experiment: convnextv2_large_8bit_ablation_a1
date: 2026-04-09
config: configs/experiment_v2_birads/convnextv2_large_8bit_ablation_a1.yaml
baseline: convnextv2_large_seg_deformable_asymmetry_malignonly_v1
backbone: convnextv2_large.fcmae_ft_in22k_in1k_384
status: completed
best_epoch: 4
val_f1_macro: 0.7108
test_f1_macro: 0.6270
mlflow_run_id: 74947f5c869f49cb9889675b338dab04
---
# convnextv2_large_8bit_ablation_a1

## Motivasyon / Hipotez
Ablation study A1 referans hücresi. En iyi 16-bit ConvNeXtV2-Large modelini (test F1: 0.7233)
yeni 8-bit 1024² datasete taşıyarak, focal loss (gamma=2.0) ve düzeltilmiş class weight'lerle
birlikte test F1 >= 0.72 bekleniyor. 1024 çözünürlük daha ince doku detaylarını yakalamalı;
focal loss ise BR2 (en zor sınıf, ensemble darboğazı ~0.66) için zor örneklere odaklanarak
ayrımı güçlendirmeli. Bu config, A2 (SwinV2) ve A3 (DINOv2) hücreleri için sabit tutulan
tüm parametrelerin kanonik kaynağı olacak.

**Kill kriteri:** val_full_f1_macro < 0.60 by epoch 15 ise durdur; ilk 5 epoch'ta loss
diverge ederse focal gamma'yı gözden geçir.

## Config Özeti
- **Backbone:** `convnextv2_large.fcmae_ft_in22k_in1k_384`
- **Scheduler:** `onecycle` (max_lr=5e-4, pct_start=0.3)
- **LR:** `5e-05`, backbone_lr_scale=0.20
- **Loss:** Focal (gamma=2.0), label_smoothing=0.08
- **Image size:** 1024 (native)
- **Effective batch:** 2 × 32 = 64
- **Dataset:** Dataset_1024_8bit / Dataset_Test_1024_8bit (8-bit, noseg)

## Baseline'dan Değişiklikler
| Alan | Baseline (`convnextv2_large_seg_deformable_asymmetry_malignonly_v1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `data.bit_depth` | `16` | `8` | 8-bit pipeline: CLAHE + letterbox + 8-bit PNG (yeni standart) |
| `data.dataset_variant` | `seg` | `noseg` | 8-bit dataset segmentasyon maskesi içermiyor |
| `data.image_size` | `512` | `1024` | Orijinal çözünürlük — daha ince doku detayı |
| `data.root_dir` | `Dataset_512` | `Dataset_1024_8bit` | Yeni 8-bit dataset dizini |
| `data.test_dir` | `Dataset_512_Test` | `Dataset_Test_1024_8bit` | Yeni 8-bit test dizini |
| `training.asymmetry_benign_weight` | `0.0` | `0.1` | BR1→BR2 sızıntısını önleyen benign simetri sinyali |
| `training.batch_size` | `4` | `2` | 1024² VRAM gereksinimi (H100 98GB) |
| `training.class_weights` | `[1.28, 1.0, 1.2, 1.11]` | `[1.9, 1.0, 1.44, 1.0]` | Train set sqrt-inverse (BR2=1.0 baseline) — doğru hesaplama |
| `training.early_stopping.patience` | `20` | `15` | Ablation tablosu standardı |
| `training.focal_gamma` | `—` | `2.0` | Focal loss: zor örneklere odaklanma (BR2 darboğazı) |
| `training.gradient_accumulation_steps` | `16` | `32` | Efektif batch = 64 korunuyor (2×32) |
| `training.label_smoothing` | `0.05` | `0.08` | Ablation tablosu standardı — daha güçlü regularizasyon |
| `training.loss_type` | `ce` | `focal` | Focal loss ablation standardı |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 4)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.7108 |
| F1 BR1 | 0.6429 |
| F1 BR2 | 0.7266 |
| F1 BR4 | 0.6703 |
| F1 BR5 | 0.8036 |
| AUC-ROC | 0.9171 |
| Cohen's Kappa | 0.6188 |
| Binary F1 | 0.9473 |

### Test (non-TTA, best checkpoint)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6270 |
| F1 BR1 | 0.4271 |
| F1 BR2 | 0.7120 |
| F1 BR4 | 0.5163 |
| F1 BR5 | 0.8524 |
| AUC-ROC | 0.8981 |
| Cohen's Kappa | 0.5689 |
| Binary F1 | 0.9125 |
| Accuracy | 0.6961 |

### Val → Test Gap
| Metrik | Val | Test | Δ |
|---|---|---|---|
| F1 Macro | 0.7108 | 0.6270 | **-0.0838** |
| F1 BR1 | 0.6429 | 0.4271 | **-0.2158** |
| F1 BR2 | 0.7266 | 0.7120 | -0.0146 |
| F1 BR4 | 0.6703 | 0.5163 | **-0.1540** |
| F1 BR5 | 0.8036 | 0.8524 | +0.0488 |
| AUC-ROC | 0.9171 | 0.8981 | -0.0190 |

## Analiz

**Eğitim dinamikleri:** Çok hızlı convergence — epoch 4'te best val F1'e ulaştı, ardından
15 epoch boyunca bir daha geçemedi (ep5-19 arası 0.66–0.71 bandında ossilasyon). OneCycle
scheduler'ın agresif LR decay'i erken convergence'ı açıklıyor. Toplam 19 epoch, patience=15
ile early stop.

**Generalizasyon sorunu:** Val→test F1 macro düşüşü **8.4pp** — bu, normal aralık olan
2–4pp'nin çok üzerinde. En kritik düşüş BR1'de: val 0.6429 → test 0.4271 (**-21.6pp**).
BR4 de ciddi: val 0.6703 → test 0.5163 (-15.4pp). Buna karşın BR5 test'te iyileşiyor
(+4.9pp), bu da modelin malign sınıfları öğrenirken benign alt-sınıf ayrımında overfitting
yaptığını gösteriyor.

**Focal loss etkisi:** Focal gamma=2.0, kolay örnekleri down-weight ederek zor BR2 örneklerine
odaklanmayı hedefliyordu. BR2 test F1 0.7120 ile makul, ancak A1-CE (CE loss) ile
karşılaştırıldığında focal loss BR1'i **ciddi şekilde zayıflattı** (0.4271 vs 0.5033). Focal
loss, BR1 örneklerini "kolay" olarak algılayıp down-weight etmiş olabilir — oysa BR1'in
test'teki zorluğu farklı bir distribüsyon yapısından kaynaklanıyor.

**BR5 gücü:** Test BR5 F1 = 0.8524 — ablation tablosunun en güçlü tekil sınıf performansı.
ConvNeXtV2'nin FCMAE ön-eğitimi, malign doku paternlerini yakalamakta başarılı.

**Epoch time:** ~2662s (~44 dk/epoch) — 1024² çözünürlük, H100'de batch=2 ile beklenen düzey.

## Sonraki Adım
- TTA değerlendirmesi ile test F1'in ne kadar iyileştiğini ölç (beklenti: +1–3pp)
- A1-CE ile McNemar testi: focal vs CE farkının istatistiksel anlamlılığını doğrula
- BR1 hata analizi: confusion matrix'te BR1→BR2 karışımı dominant mı? Bu, focal loss'un
  down-weight ettiği "kolay" BR1 örneklerinin aslında test'te sınır vakalar olduğunu gösterir
- Ensemble'da A1'in rolü: BR5 gücü ve A3'ün farklı hata profili ile complementary olmalı
