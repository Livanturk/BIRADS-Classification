---
experiment: dinov2_vitl_8bit_ablation_a3
date: 2026-04-09
config: configs/experiment_v2_birads/dinov2_vitl_8bit_ablation_a3.yaml
baseline: convnextv2_large_8bit_ablation_a1
backbone: vit_large_patch14_dinov2.lvd142m
status: completed
best_epoch: 17
val_f1_macro: 0.6940
test_f1_macro: 0.6325
mlflow_run_id: 4262c2eee0e944639f6c7b52b947830a
---

# dinov2_vitl_8bit_ablation_a3

## Motivasyon / Hipotez
Ablation A3 hücresi — self-supervised ViT (DINOv2 ViT-L/14, LVD-142M). DINOv2,
ImageNet etiketleri olmadan 142M görüntüde eğitilmiş olup, doku ve yapısal
özellikler için güçlü genel-amaçlı temsiller öğrenmiştir. Bu self-supervised
ön-eğitim, mamografi gibi etiket-kısıtlı tıbbi görüntüleme alanlarında supervised
IN-22k backbone'lardan daha iyi genellenebilir.

**Image size:** 518 (37×14 = 1369 spatial token). patch_size=14, 1024'ü bölmediğinden
en yakın uyumlu boyut kullanılıyor. 1022 (5329 token) VRAM açısından riskli.

**Beklenti:** val F1 0.63–0.69. DINOv2 self-supervised özellikleri farklı hata profili
üretmeli — özellikle doku bazlı sınıflandırmada (BR4 vs BR5) güçlü olabilir.
Ensemble'a en büyük katkı, ConvNeXtV2 ve SwinV2'den farklı yanılma paternleri üzerinden.

**Kill kriteri:** val_full_f1_macro < 0.50 by epoch 20 veya warmup sonrası loss patlaması.

## Config Özeti
- **Backbone:** `vit_large_patch14_dinov2.lvd142m` (LVD-142M self-supervised)
- **Scheduler:** `cosine_warmup` (warmup=10 epochs, min_lr=1e-6)
- **LR:** `2e-05`, backbone_lr_scale=0.10
- **Loss:** Focal (gamma=2.0), label_smoothing=0.08
- **Image size:** 518 (37×14 patch grid, 1369 spatial tokens)
- **Effective batch:** 2 × 32 = 64
- **weight_decay backbone:** 0.10 (self-supervised backbone için güçlü regularizasyon)

## Baseline'dan Değişiklikler (A1 → A3: sadece backbone-specific)
| Alan | Baseline (`convnextv2_large_8bit_ablation_a1`) | Bu Deney | Gerekçe |
|---|---|---|---|
| `model.backbone.name` | `convnextv2_large.fcmae_ft_in22k_in1k_384` | `vit_large_patch14_dinov2.lvd142m` | Ablation değişkeni: self-supervised ViT |
| `model.backbone.feature_dim` | `1536` | `1024` | ViT-L/14 çıkış boyutu |
| `model.backbone.drop_path_rate` | `—` | `0.0` | DINOv2 self-supervised: drop_path gereksiz (overfit riski düşük) |
| `data.image_size` | `1024` | `518` | patch_size=14 → 518=37×14 (1369 token, VRAM uyumlu) |
| `training.optimizer.lr` | `5e-05` | `2e-05` | Self-supervised backbone çok düşük LR gerektirir |
| `training.optimizer.backbone_lr_scale` | `0.2` | `0.1` | Self-supervised özelikleri korumak için çok muhafazakar |
| `training.optimizer.weight_decay.backbone` | `0.05` | `0.1` | Self-supervised backbone güçlü decay (overfit önlemi) |
| `training.scheduler.name` | `onecycle` | `cosine_warmup` | ViT backbone için stabil warmup (pitfall #1) |
| `training.scheduler.warmup_epochs` | `—` | `10` | Transformer warmup standardı |
| `training.scheduler.min_lr` | `—` | `1e-6` | Cosine annealing alt sınır |
| `training.scheduler.max_lr` | `0.0005` | `—` | OneCycle parametresi (cosine'da yok) |
| `training.scheduler.pct_start` | `0.3` | `—` | OneCycle parametresi |
| `training.scheduler.div_factor` | `10.0` | `—` | OneCycle parametresi |
| `training.scheduler.final_div_factor` | `100.0` | `—` | OneCycle parametresi |
| `training.scheduler.anneal_strategy` | `cos` | `—` | OneCycle parametresi |

## Sonuçlar

### Val — En İyi Checkpoint (Epoch: 17)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6940 |
| F1 BR1 | 0.6655 |
| F1 BR2 | 0.6885 |
| F1 BR4 | 0.6283 |
| F1 BR5 | 0.7937 |
| AUC-ROC | 0.9020 |
| Cohen's Kappa | 0.5927 |
| Binary F1 | 0.9237 |

### Test (non-TTA, best checkpoint)
| Metrik | Değer |
|---|---|
| **F1 Macro** | 0.6325 |
| F1 BR1 | 0.4817 |
| F1 BR2 | 0.6812 |
| F1 BR4 | 0.5106 |
| F1 BR5 | 0.8567 |
| AUC-ROC | 0.8976 |
| Cohen's Kappa | 0.5596 |
| Binary F1 | 0.8936 |
| Accuracy | 0.6846 |

### Val → Test Gap
| Metrik | Val | Test | Δ |
|---|---|---|---|
| F1 Macro | 0.6940 | 0.6325 | **-0.0615** |
| F1 BR1 | 0.6655 | 0.4817 | **-0.1838** |
| F1 BR2 | 0.6885 | 0.6812 | -0.0073 |
| F1 BR4 | 0.6283 | 0.5106 | **-0.1177** |
| F1 BR5 | 0.7937 | 0.8567 | +0.0630 |
| AUC-ROC | 0.9020 | 0.8976 | -0.0044 |

## Analiz

**Eğitim dinamikleri:** Beklenen şekilde yavaş convergence — 10 epoch cosine warmup boyunca
kademeli artış (ep1: 0.09 → ep5: 0.54 → ep10: ~0.59). Best val F1'e ancak epoch 17'de
ulaştı (0.6940), ardından 15 epoch daha ossilasyon (0.66–0.68 bandı). Toplam 32 epoch,
patience=15 ile early stop. Self-supervised backbone'un fine-tuning'i beklendiği gibi sabır
gerektiriyor.

**Epoch hızı:** ~729s (~12 dk/epoch) — 518² çözünürlük sayesinde A1'den (1024²) **3.7x daha
hızlı**. Toplam eğitim süresi, A1'den daha fazla epoch'a rağmen karşılaştırılabilir düzeyde.

**Generalizasyon:** Val→test F1 macro düşüşü **6.2pp** — A1'den (8.4pp) daha iyi generalize
ediyor. Self-supervised DINOv2 ön-eğitimi, domain-agnostic feature'lar ürettiğinden daha
düşük overfit riski taşıyor. AUC-ROC gap sadece 0.4pp — ranking performansı çok stabil.

**Per-class analiz:**
- **BR1:** Val 0.6655 → test 0.4817 (-18.4pp). A1'den (0.4271) daha iyi ama hâlâ ciddi
  generalizasyon gap'i var. DINOv2'nin doku bazlı özelikleri, normal meme dokusunun
  heterojen yapısını kısmen yakalıyor.
- **BR2:** Test 0.6812 — ablation tablosunda orta düzey. Val-test gap sadece 0.7pp ile en
  stabil sınıf.
- **BR4:** Test 0.5106 — A1 (0.5163) ile hemen hemen eşit. Şüpheli lezyonlar hem CNN hem
  ViT için zor.
- **BR5:** Test **0.8567** — ablation tablosunun **en iyi tekil BR5 performansı**. DINOv2'nin
  self-supervised özelikleri malign doku paternlerini güçlü yakalıyor.

**Hipotez değerlendirmesi:** "Val F1 0.63–0.69" beklentisi karşılandı (0.6940).
"Farklı hata profili" hipotezi kısmen doğrulandı: A1'e göre BR1'de +5.5pp, BR5'te +0.4pp
daha iyi; BR2'de -3.1pp daha zayıf. Bu farklılıklar ensemble için complementary katkı
potansiyeli taşıyor.

**Train performance:** Train F1 macro 0.7505 — val 0.6940 ile gap sadece 5.7pp. A1'in
train-val gap'i (0.7175-0.7108=0.7pp) çok daha düşük, ama A1'in test gap'i daha büyük.
Bu, A1'in val sete de overfit ettiğini düşündürüyor.

## Sonraki Adım
- TTA ile test F1 iyileşmesini ölç — DINOv2'nin rotation invariance özelliği TTA'dan daha
  fazla fayda sağlayabilir
- Ensemble'da A3'ün rolü: BR5 gücü ve A1'den farklı BR1 hata profili ile complementary katkı
- BR2 zayıflığını ensemble'ın telafi edip edemeyeceğini kontrol et (A1 BR2=0.7120 taşıyıcı)
- 518² çözünürlük kısıtı: patch_size=14 daha yüksek çözünürlüğe izin vermiyor; daha ince
  doku detayları kaçırılıyor olabilir
