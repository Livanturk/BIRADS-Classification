# BI-RADS Mammography Classification: Lesson #51-#60 Deney Sunumu

**Hazırlanan dosya:** Hoca sunumu / teknik rapor taslağı  
**Kapsam tarihi:** 2026-05-02 - 2026-05-05  
**Kaynaklar:** `CLAUDE.md`, `tasks/lessons.md` Lesson #51 ve #54-#60, ilgili `artifacts/`, `tools/`, `configs/` ve mevcut checkpoint dosyaları  
**Güncel baseline disiplini:** Yeni deneyler artık tek-seed C6 `0.6762` ile değil, n=4 C6 logit-mean seed ensemble `0.6846` ile karşılaştırılmalıdır.

> Kapsam notu: `tasks/lessons.md` içinde daha eski bir numaralandırma bloğunda Lesson #52-#53 başlıkları da var. Bu rapor, IDE'de seçili olan yeni Mayıs 2026 analiz bloğunu esas alır: Lesson #51, ardından Lesson #54-#60.

---

## 1. Kısa Yönetici Özeti

Lesson #51-#60 arasındaki çalışma, projenin temel varsayımını değiştirdi. Önceki anlatıda C6 tek-seed sonucunun, yani seed=42 ile elde edilen `0.6762` test macro F1'in, modelin doğal performans tavanı olduğu düşünülüyordu. Lesson #51 ile bunun istatistiksel olarak şanslı bir seed olduğu gösterildi: altı C6 seed'inin ortalaması `0.6502`, standart sapması `0.0137`; seed=42 ise yaklaşık `+1.9 sigma` üst kuyrukta.

Bu nedenle önceki birçok "regresyon" iddiası yeniden yorumlandı. Tek-seed farkların çoğu seed gürültüsü içinde kaldı. Asıl üretim seviyesinde pozitif sonuç Lesson #57'de geldi: seed {42, 7, 555, 999} ile n=4 logit-mean ensemble, test macro F1'i `0.6846`'ya çıkardı. Bu, seed ortalamasına göre yaklaşık `+3.4 pp` kazanç ve artık resmi production inference baseline'dir.

Buna karşılık post-hoc inference track büyük ölçüde kapandı. TTA + ensemble, Saerens-EM, oracle prior correction, pairwise threshold/gate ve cascade gibi yaklaşımlar ensemble baseline'i geçemedi. En sert negatif bulgu Saerens-EM'de görüldü: F2 üzerinde BR1 F1 `0.000`'a indi; vanilla C6 seed'lerinde de EM her seed'de başarısız oldu.

BR1 ve BR4 için tablo ayrıldı. BR1, ensemble ile seed=42'nin şanslı BR1 sonucunu sadece geri kazanabildi; ensemble, seed=42'nin üzerine ek BR1 kazancı üretmedi. Buna rağmen Lesson #60'taki k-NN + Grad-CAM programmatic audit, BR1 hatalarının label-noise gibi dağınık değil, embedding uzayında geometrik olarak yapılı olduğunu gösterdi. Bu, BR1 için ucuz augmentation/inference çözümlerinin değil, yeni training-time sinyalin gerekli olduğunu söylüyor: hard-negative SupCon, density-conditioned head veya benzeri ayrımcı embedding kayıpları.

BR4 daha umutlu sınıf olarak ayrıldı. Ensemble BR4'u `0.555`'e çıkardı; no-hflip ablation BR4'te yönlü pozitif sinyal verdi; Phase 0c audit BR4->BR5 hatalarının da geometrik overlap olduğunu gösterdi. Bu nedenle BR4 için per-region / mass-localization veya yüksek çözünürlük gibi mimari track mantıklı.

---

## 2. Dataset ve Deney Zemini

Bu bölüm doğrudan `CLAUDE.md` içindeki güncel dataset istatistiklerine dayanır.

| Özellik | Değer |
| --- | --- |
| Görüntü formatı | 8-bit PNG, grayscale, `uint8` |
| Çözünürlük | 1024 x 1024 |
| Hasta bazlı giriş | 1 hasta = 4 görüntü: RCC, LCC, RMLO, LMLO |
| Sınıflar | BI-RADS 1, 2, 4, 5; BI-RADS 3 yok |
| Train/Val dataset | `Dataset_1024_8bit` |
| Test dataset | `Dataset_Test_1024_8bit`, fixed holdout |
| Train hasta / görüntü | 8,557 hasta / 34,228 görüntü |
| Test hasta / görüntü | 1,655 hasta / 6,620 görüntü |

Preprocessing hattında DICOM verisi MONOCHROME1 correction, U-Net segmentasyon, tissue mask, windowing, tight crop, CLAHE ve 1024 x 1024 letterbox adımlarından geçiyor. CLAHE sadece tissue pixel'lara uygulanmış; background zero olarak bırakılmış.

### 2.1. Train-Test Sınıf Dağılımı

Pixel dağılımı train ve test arasında aynı değil. Bu rapordaki prior-shift ve sınıf-conditional boundary yorumlarının ana nedeni bu.

| Sınıf | Train tissue pixel oranı | Test tissue pixel oranı | Yorum |
| --- | ---: | ---: | --- |
| BI-RADS-1 | 18.7% | 9.7% | Testte ciddi azalma; BR1 en kırılgan sınıf |
| BI-RADS-2 | 32.4% | 36.0% | Testte daha baskın benign sınıf |
| BI-RADS-4 | 22.1% | 17.3% | Testte azalma; BR4 boundary sınıfı |
| BI-RADS-5 | 26.7% | 37.0% | Testte ciddi artış; malignant taraf baskınlaşıyor |

Hasta dağılımı train tarafında `BR1=1678`, `BR2=2754`, `BR4=1898`, `BR5=2227`; benign toplam `4432`, malignant toplam `4125`. C6 class weight'leri sqrt-inverse frequency ile `[1.28, 1.00, 1.20, 1.11]`.

### 2.2. Normalizasyon ve Domain Shift Yorumu

Train-test görüntü istatistikleri yakın:

| Metrik | Train | Test |
| --- | ---: | ---: |
| All-pixel mean/std | 0.1210 / 0.1977 | 0.1237 / 0.1986 |
| Nonzero tissue mean/std | 0.3512 / 0.1804 | 0.3526 / 0.1779 |
| Zero pixel ratio | 65.54% | 64.92% |

Bu nedenle problem kaba image-domain shift değil. Ana problem, sınıf priors ve class-conditional karar sınırları: özellikle BR1<->BR2 ve BR4<->BR5.

---

## 3. C6 Baseline Model ve Eğitim Reçetesi

C6, 1024 x 1024 8-bit pipeline için aktif training baseline'dir. Üretim inference baseline'i ise Lesson #57'deki n=4 C6 ensemble'dir.

### 3.1. Mimari

`CLAUDE.md` ve `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml` temelinde:

| Bileşen | Kullanılan yapı |
| --- | --- |
| Backbone | `convnextv2_large.fcmae_ft_in22k_in1k_384`, pretrained |
| Giriş | `(B, 4, 3, 1024, 1024)` |
| View yapısı | RCC, LCC, RMLO, LMLO |
| Feature dim | Backbone 1536, projection 512 |
| Lateral fusion | CC-MLO bidirectional spatial cross-attention, deformable attention aktif |
| Bilateral fusion | `[F_left, F_right, F_diff, F_avg]` tokenları üzerinde self-attention |
| Hasta temsili | `patient_feat` |
| Head'ler | binary, benign_sub, malign_sub, full_head; hepsi `patient_feat` kullanır |

Modelin ana loss'u:

```text
L_total = 0.10 * L_binary + 0.45 * L_subgroup + 0.45 * L_full
```

C6'da `loss_type=ce`, `label_smoothing=0.05`, SWA aktif, `swa_start_epoch=5`, asymmetry loss kapalı (`asymmetry_loss_weight=0.0`). Augmentation tarafında horizontal flip `0.5`, rotation `10 derece`, brightness/contrast `0.1`, random erasing `0.1`.

### 3.2. Deneylerde Kullanılan Ana Dosyalar

| Amac | Dosya / artifact |
| --- | --- |
| C6 ana config | `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6.yaml` |
| Ek C6 seed configleri | `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6_seed{7,555,999}.yaml` |
| F2 logit adjustment | `configs/experiment_v2_birads/convnextv2_large_8bit_F2_la_tau{05,10,15}.yaml` |
| Logit extraction | `tools/extract_logits.py` |
| Saerens-EM | `tools/saerens_em_logit_adjust.py` |
| Pairwise boundary diagnostic | `tools/pairwise_boundary_diagnostics.py` |
| Seed ensemble | `tools/seed_ensemble.py`, `artifacts/seed_ensemble_n4.json` |
| Ensemble + TTA | `tools/seed_ensemble_tta.py`, `artifacts/seed_ensemble_n4_tta.json` |
| No-hflip configs | `configs/experiment_v2_birads/convnextv2_large_8bit_ablation_c6_nohflip_seed{42,7,555,999}.yaml` |
| Phase 0c features | `tools/extract_patient_feat.py` |
| Phase 0c k-NN | `tools/phase0c_knn_analysis.py`, `artifacts/phase0c_knn_analysis.{json,csv}` |
| Phase 0c Grad-CAM | `scripts/generate_gradcam_targeted.py`, `tools/phase0c_gradcam_analysis.py` |
| Phase 0c synthesis | `tools/phase0c_synthesize.py`, `artifacts/phase0c_audit_substitute.{md,json}` |

---

## 4. Lesson #51: Seed Variance ve Yeni Istatistiksel Baseline

### Ne yapildi?

Byte-identical C6 config ile 6 bağımsız seed karşılaştırildi: `{42, 123, 2024, 7, 555, 999}`. Amac, tek-seed C6 champion sonucunun gercek model kapasitesi mi yoksa şanslı seed mi olduğunu test etmekti.

### Ana sonuç

| Seed | Test macro F1 | BR1 | BR2 | BR4 | BR5 | SWA |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.6762 | 0.531 | 0.798 | 0.518 | 0.857 | WON |
| 123 | 0.6403 | - | - | - | - | - |
| 2024 | 0.6467 | - | - | - | - | - |
| 7 | 0.6390 | 0.436 | 0.755 | 0.499 | 0.866 | WON |
| 555 | 0.6537 | 0.490 | 0.721 | 0.560 | 0.844 | WON |
| 999 | 0.6456 | 0.439 | 0.772 | 0.513 | 0.857 | LOST |

Test macro F1 dağılımı:

| Metrik | Değer |
| --- | ---: |
| n | 6 |
| Ortalama | 0.6502 |
| Std | 0.0137 |
| Range | 0.6390 - 0.6762 |
| 95% t-CI | 0.6358 - 0.6647 |
| seed=42 z-score | +1.89 sigma |

### Yorum

Eski "C6 = 0.6762" anlatimi yanlış baseline kuruyordu. `0.6762`, modelin beklenen performansi değil, şanslı üst kuyruk seed'i. Bundan sonra tek-seed farkların anlamlı sayilmasi için yaklaşık `2 sigma`, yani `~2.7 pp` üzerinde olması gerekiyor.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | Proje istatistiksel olarak sağlam hale geldi; tek-seed yorum hatasi düzeltildi |
| Pozitif | BR1 ve BR4 varyansinin seed kaynaklı olduğu görüldü; hangi siniflarin kırılgan olduğu netlesti |
| Negatif | Önceki birçok "regression" claim'i zayifladi; deneylerin bir kismi seed noise içinde kaldı |
| Negatif | C6'nin tek model tek seed beklenen performansi `0.6502 +/- 0.0137`, yani dusunulenden daha düşük |

---

## 5. Lesson #54: F2 Logit-Adjusted Training

### Ne yapildi?

Menon-style logit-adjusted training, C6 recetesinin üzerine full 4-class head için denendi. Tau degerleri: `0.5`, `1.0`, `1.5`.

### Sonuclar

| Run | Best val F1 | Test macro F1 | BR1 | BR2 | BR4 | BR5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C6 reference seed=42 | 0.7183 | 0.6762 | 0.531 | 0.798 | 0.518 | 0.857 |
| F2 tau=0.5 | 0.7300 | 0.6606 | 0.503 | 0.769 | 0.523 | 0.848 |
| F2 tau=1.0 | 0.7084 | 0.6443 | 0.521 | 0.692 | 0.509 | 0.856 |
| F2 tau=1.5 | 0.7163 | 0.6335 | 0.514 | 0.677 | 0.491 | 0.853 |

### Yorum

Tau=0.5 validation F1'i yukseltti ama test'e transfer olmadı. Tau arttikca BR4 recall tarafi kazanir gibi oldu, fakat BR2 ciddi bedel odedi. Bu, modelin yeni morfolojik temsil ogrenmedigini; sadece karar sinirini baska operating point'e tasidigini gösterdi.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | Prior-shift hassasiyeti için iyi diagnostic oldu |
| Pozitif | BR2-BR4 trade-off mekanizmasi net gösterildi |
| Negatif | Üretim iyileştirmesi değil; en iyi F2 bile seed=42 C6'dan `-1.56 pp`, ensemble baseline'dan `-2.40 pp` düşük |
| Negatif | Tau sweep devam etmeye değer değil |

---

## 6. Lesson #55: Saerens-EM Post-Hoc Prior Correction

### Ne yapildi?

`F2_la_tau05_best` logits üzerinde Saerens-EM ile test-time prior correction denendi. Ayrica oracle true-test-prior correction da diagnostic olarak hesaplandi.

### Sonuclar

| Setting | Test macro F1 | Accuracy | BR1 | BR2 | BR4 | BR5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 tau=0.5 baseline | 0.6606 | 0.7251 | 0.503 | 0.769 | 0.523 | 0.848 |
| Saerens-EM adjusted | 0.5190 | 0.7281 | 0.000 | 0.806 | 0.415 | 0.855 |
| Oracle true-prior shift | 0.6111 | 0.7293 | 0.371 | 0.793 | 0.425 | 0.855 |

EM-estimated prior, true test prior'a göre BR1'i çok düşük tahmin etti:

| Prior | BR1 | BR2 | BR4 | BR5 |
| --- | ---: | ---: | ---: | ---: |
| True test prior | 0.0985 | 0.3601 | 0.1740 | 0.3674 |
| EM test prior | 0.0455 | 0.3663 | 0.1765 | 0.4117 |

### Yorum

BR1 F1'in `0.000`'a dusmesi kritik. Bu, sorunun sadece global label-prior shift olmadigini gosteriyor. Eger sorun yalnizca prior olsaydi, oracle correction en azindan baseline'i gecmeliydi. Tam tersine oracle bile `0.6606 -> 0.6111` geriledi.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | Label-shift varsayımını temiz şekilde test etti |
| Negatif | Saerens-EM deploy edilmemeli |
| Negatif | Global prior correction BR1'i yok ediyor; class-conditional boundary bozuk |

---

## 7. Lesson #56: Pairwise Boundary Tuning

### Ne yapildi?

F2 tau=0.5 logits üzerinde sadece komsu sinirlar ayarlandi:

- BR1 vs BR2 için BR1 logit offset'i: `d12_add_to_BR1`
- BR4 vs BR5 için BR4 logit offset'i: `d45_add_to_BR4`

Offset'ler validation üzerinde secildi, test'e frozen olarak uygulandi.

### Sonuclar

| Method | Test macro F1 | BR1 | BR4 | Yorum |
| --- | ---: | ---: | ---: | --- |
| Baseline F2 tau=0.5 | 0.6606 | 0.5029 | 0.5229 | Baslangic |
| full_pair_gate, macro-safe | 0.6684 | 0.5064 | 0.5399 | +0.78 pp ama C6 altında |
| binary_subhead_pair_gate, weak-class push | 0.6590 | 0.5093 | 0.5733 | BR4 artar, BR5 zarar gorur |

BR4 agresif offset ile toparlanabildi ama bunun bedeli BR5->BR4 hatasiydi:

| Hata | Baseline | Aggressive pair gate |
| --- | ---: | ---: |
| BR4 -> BR5 | 35.8% | 15.6% |
| BR5 -> BR4 | 13.0% | 27.3% |

### Yorum

Pairwise tuning, çözüm değil diagnostic. BR4 siniri threshold ile hareket edebiliyor, ama representation iyilesmiyor. BR1 tarafında validation-test transfer zayif; val BR1 margin pozitifken test BR1 margin sifira/negatife yaklasiyor.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | Hangi sinirin hareket edebilir, hangisinin transfer etmez olduğu netlesti |
| Pozitif | BR4 için tunable operating point var |
| Negatif | Yeni champion değil; ensemble baseline'in çok altında |
| Negatif | BR1 için threshold çözüm değil |

---

## 8. Lesson #57: n=4 Seed Ensemble

### Ne yapildi?

Dört C6 seed'inin `{42, 7, 555, 999}` test logits'i birlestirildi. Iki ensemble tipi denendi:

- logit-mean: logits ortalaması, geometric mean etkisi
- softmax-mean: softmax probability ortalaması

### Sonuclar

| Model | Macro F1 | Accuracy | BR1 | BR2 | BR4 | BR5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=4 seed mean | 0.6536 | 0.7237 | 0.474 | 0.762 | 0.522 | 0.856 |
| Ensemble logit-mean | 0.6846 | 0.7462 | 0.531 | 0.788 | 0.555 | 0.865 |
| Ensemble softmax-mean | 0.6821 | 0.7432 | 0.529 | 0.781 | 0.553 | 0.866 |

Decision gate sonucu: PASS. `0.6846`, Lesson #51'deki +2 sigma threshold'u olan `0.6776`'yi gecti.

### Unanimous-wrong bulgusu

| Metrik | Değer |
| --- | ---: |
| Test hasta sayisi | 1655 |
| 4 seed unanimous prediction | 1147 |
| Unanimous correct | 940 |
| Unanimous wrong | 207 |

207 hasta, dört bağımsız seed tarafindan aynı şekilde yanlış siniflandi. Bu, hatalarin sadece seed variance olmadigini; bir kismi yapisal olduğunu gösterdi.

### Yorum

Ensemble genel performansi ciddi iyileştirdi. BR2, BR4 ve BR5 tarafında gercek variance-reduction faydasi var. Ancak BR1 kazancı aldatmaci: ensemble BR1 F1 `0.531`, seed=42'nin şanslı BR1 degerini geçmedi. Yani ensemble BR1 için yeni bilgi yaratmadi, sadece seed=42'nin iyi BR1 davranisini geri kazandi.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | En büyük production kazancı: macro F1 `0.6846` |
| Pozitif | BR4 `0.522 -> 0.555`, BR2 `0.762 -> 0.788`; ensemble gercekten faydalı |
| Pozitif | Yeni production inference baseline belirlendi |
| Negatif | BR1 için ensemble ek kazanç üretmedi |
| Negatif | 207 unanimous-wrong hasta yapisal hata sinyali verdi |

---

## 9. Lesson #58: TTA + Ensemble, Vanilla Saerens-EM ve Error Cell Dağılımı

### Ne yapildi?

Uc post-hoc track test edildi:

1. n=4 ensemble üzerine rotation TTA eklendi.
2. Saerens-EM, dört vanilla C6 seed'inin her birinde denendi.
3. 207 unanimous-wrong hastanin sınıf-cell dağılımı incelendi.

### 9.1. TTA + Ensemble Sonucu

| Setting | Macro F1 | BR1 | BR2 | BR4 | BR5 | Delta vs ensemble |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n=4 ensemble | 0.6846 | 0.531 | 0.788 | 0.555 | 0.865 | - |
| Ensemble x TTA logit | 0.6741 | 0.510 | 0.771 | 0.548 | 0.867 | -1.05 pp |
| Ensemble x TTA softmax | 0.6714 | 0.513 | 0.765 | 0.542 | 0.866 | -1.32 pp |

TTA tekil seed'lerde ortalama hafif pozitifti (`+0.21 pp`), fakat ensemble ustune eklendiginde negatif oldu. Bu, smoothing operator'larinin üst uste geldiginde BR1<->BR2 margin'ini bozdugunu gösterdi.

### 9.2. Vanilla C6 Saerens-EM

| Seed | Baseline F1 | EM F1 | Oracle F1 | EM delta |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6762 | 0.4152 | 0.6515 | -26.10 pp |
| 7 | 0.6390 | 0.5248 | 0.5906 | -11.42 pp |
| 555 | 0.6537 | 0.5278 | 0.6386 | -12.58 pp |
| 999 | 0.6456 | 0.5180 | 0.6074 | -12.76 pp |

EM her vanilla seed'de başarısız oldu. Uc seed'de BR1 sifirlandi; seed=42'de BR2 sifirlandi.

### 9.3. Unanimous-Wrong Error Cells

| True -> Pred | Count | Sınıf destek içindeki oran |
| --- | ---: | ---: |
| BR1 -> BR2 | 35 | 21.5% of BR1 |
| BR1 -> BR4 | 9 | 5.5% of BR1 |
| BR2 -> BR1 | 31 | 5.2% of BR2 |
| BR2 -> BR4 | 38 | 6.4% of BR2 |
| BR4 -> BR5 | 46 | 16.0% of BR4 |
| BR5 -> BR4 | 38 | 6.3% of BR5 |

Per-class unanimous-wrong oranlari:

| Sınıf | Oran |
| --- | ---: |
| BR1 | 27.0% |
| BR4 | 18.8% |
| BR2 | 11.7% |
| BR5 | 6.4% |

### Yorum

BR1 problemi asimetrik: BR1 -> BR2 ciddi, ama BR2 -> BR1 ters oran düşük. BR4 ise çift yönlü confusion merkezi; unanimous-wrong hastalarin yaklaşık %59'u BR4'u true veya predicted class olarak içeriyor.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | n=4 ensemble'in inference-time ceiling olduğu netlesti |
| Pozitif | BR1 ve BR4 hata anatomisi ayrıldı |
| Negatif | TTA ensemble ustune eklenmemeli |
| Negatif | Saerens-EM ve oracle prior correction tamamen reddedildi |
| Negatif | Post-hoc inference çözümleri tükenmiş gorundu |

---

## 10. Lesson #59: Drop-Horizontal-Flip Ablation

### Ne yapildi?

C6'da tek değişken olarak `augmentation.horizontal_flip: 0.5 -> 0.0` yapildi. Dört paired seed `{42, 7, 555, 999}` ile test edildi.

### Sonuclar

| Metrik | hflip mean | no-hflip mean | Ortalama delta | Verdict |
| --- | ---: | ---: | ---: | --- |
| Test macro F1 | 0.6536 | 0.6539 | +0.0002 pp | Null |
| BR1 F1 | 0.474 | 0.490 | +1.6 pp | Yonlu, NS |
| BR1 recall | 0.442 | 0.489 | +4.8 pp | Yonlu, NS |
| BR2 F1 | 0.762 | 0.748 | -1.4 pp | NS |
| BR4 F1 | 0.522 | 0.543 | +2.0 pp | Borderline, NS |
| BR4 recall | 0.603 | 0.691 | +8.9 pp | Yonlu, NS |
| BR5 F1 | 0.856 | 0.835 | -2.1 pp | Significant loss |

### Yorum

Lesson #45'teki L/R semantic corruption hipotezi mekanik olarak doğruydu: hflip, view-swap yapilmazsa bilateral `F_diff = F_left - F_right` sinyalini gürültülü hale getirebilir. Fakat deney net gösterdi ki bu gürültü aynı zamanda implicit dropout gibi regularize ediyor. Hflip'i kaldirmak macro F1'i iyilestirmedi, BR5'i anlamlı şekilde bozdu ve SWA stabilitesini zayiflatti.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | BR4 için yönlü bir sinyal yakalandi; dedicated BR4 architecture track desteklendi |
| Pozitif | hflip'in faydalı regularizer olduğu anlasildi |
| Negatif | no-hflip production'a alınmamalı |
| Negatif | BR5 her seed'de zarar gordu |
| Negatif | BR1 kazancı kucuk ve istatistiksel olarak anlamlı değil |

---

## 11. Lesson #60: Programmatic Audit Substitute

### Ne yapildi?

Planlanan radyolog audit'i klinik erişim olmadığı için yapilamadi. Bunun yerine 207 unanimous-wrong hasta üzerinde iki sinyal hesaplandi:

1. `patient_feat` embedding uzayında k-NN distance ratio:
   - `r = mean distance to confident true-predicted-class / mean distance to confident true-class`
   - `r << 1` ise hasta embedding olarak yanlış class cluster'inin içinde.
2. Grad-CAM:
   - concentration,
   - cross-seed cosine agreement,
   - within-cell k-means silhouette.

### k-NN Sonuclari

| Cell | n | Median ratio | frac < 0.9 | Verdict |
| --- | ---: | ---: | ---: | --- |
| BR1 -> BR2 | 42 | 0.078 | 97.6% | geometric_overlap |
| BR4 -> BR5 | 48 | 0.312 | 87.5% | geometric_overlap |
| BR2 -> BR1 | 35 | 0.068 | 100% | geometric_overlap |
| BR2 -> BR4 | 32 | 0.067 | 100% | geometric_overlap |
| BR5 -> BR4 | 34 | 0.040 | 100% | geometric_overlap |

### Grad-CAM Sonuclari

| Cell | Concentration median | Cross-seed cosine | Best k | Silhouette |
| --- | ---: | ---: | ---: | ---: |
| BR1 -> BR2 | 0.064 | 0.411 | 2 | 0.42 |
| BR4 -> BR5 | 0.059 | 0.224 | 4 | 0.34 |
| BR2 -> BR1 | 0.065 | 0.315 | 2 | 0.52 |
| BR4 -> BR2 | 0.065 | 0.354 | 3 | 0.77 |

### Yorum

Otomatik synthesis JSON'i, concentration threshold'unu lokal lezyon gorevlerine göre çok yüksek tuttugu için `MIXED` sonuç verdi. Fakat mammography density/parenchymal-pattern kararlarinda attention'in diffuse olması doğal. Bu nedenle k-NN baskın yorum daha doğru: hatalar label-noise gibi rastgele değil, embedding uzayında yapılı ve representation-side.

### Pozitif / negatif etki

| Etki | Yorum |
| --- | --- |
| Pozitif | "BR1 label-noise floor olabilir" hipotezi zayifladi; temsil problemi olduğu gösterildi |
| Pozitif | SupCon hard-negative mining ve yeni training-time signal tekrar geçerli next step oldu |
| Pozitif | BR4 için per-region / mass-localization head daha iyi gerekçelendirildi |
| Negatif | Mevcut C6 representation'i BR1 ve BR4 error cell'lerinde yanlış cluster'a yakın |
| Negatif | Synthesis rule domain'e uygun kalibre edilmemis; k-NN-dominant scoring gerekli |

---

## 12. Deneylerin Toplu Pozitif-Negatif Etki Matrisi

| Deney / Müdahale | Kullanılan sey | Ana metrik | Etki | Karar |
| --- | --- | ---: | --- | --- |
| C6 multi-seed audit | 6 seed C6 | mean 0.6502 +/- 0.0137 | Pozitif diagnostic | Yeni istatistiksel baseline |
| seed=42 C6 | Tek seed | 0.6762 | Yanıltıcı pozitif | Lucky outlier, baseline değil |
| F2 tau=0.5 | Logit-adjusted training | 0.6606 | Negatif/diagnostic | Devam etme |
| F2 tau=1.0 | Daha yüksek tau | 0.6443 | Negatif | BR2 kaybı |
| F2 tau=1.5 | Daha yüksek tau | 0.6335 | Negatif | High tau zararli |
| Saerens-EM F2 | Post-hoc prior correction | 0.5190 | Çok negatif | Deploy etme |
| Oracle prior F2 | True test prior | 0.6111 | Negatif | Prior-only çözüm değil |
| Pairwise macro-safe gate | BR1/BR4 offset | 0.6684 | Kucuk pozitif ama yetersiz | Diagnostic |
| Pairwise weak push | Aggressive offset | 0.6590 | BR4 pozitif, macro/BR5 negatif | Diagnostic |
| n=4 logit ensemble | 4 C6 seed | 0.6846 | En güçlü pozitif | Production inference baseline |
| n=4 softmax ensemble | 4 C6 seed | 0.6821 | Pozitif ama logit altında | Secondary |
| Ensemble + TTA logit | 4 seed x 5 rotations | 0.6741 | Negatif | Ship etme |
| Ensemble + TTA softmax | 4 seed x 5 rotations | 0.6714 | Negatif | Ship etme |
| Vanilla C6 Saerens-EM | 4 seed | -11 to -26 pp | Çok negatif | Tamamen reddedildi |
| Drop hflip | 4 paired seed | macro +0.0002 pp | Neutral/negatif | Production değil |
| Drop hflip BR5 etkisi | hflip 0.0 | BR5 -2.1 pp | Negatif | hflip kalmalı |
| Phase 0c k-NN audit | patient_feat k-NN | BR1->BR2 median 0.078 | Pozitif diagnostic | Representation-side next step |
| Phase 0c Grad-CAM | targeted Grad-CAM | silhouettes 0.34-0.77 | Pozitif diagnostic | Error archetype var |

---

## 13. Sınıf Bazli Bilimsel Sonuc

### BR1

BR1, projenin en zor sınıfı. Testte pixel oranı sadece `9.7%`, seed variance yüksek, BR1 F1 expected value yaklaşık `0.47`. Ensemble BR1'i seed=42 seviyesine geri getirdi ama onun üzerine çıkarmadı. TTA, Saerens-EM, threshold, hflip removal gibi ucuz yollar çalışmadı.

Lesson #60 sonrasinda en güncel yorum: BR1 sadece label-noise floor değil; `patient_feat` uzayında BR1->BR2 unanimous-wrong hastalarin %97.6'si yanlış class cluster'ina daha yakın. Bu nedenle BR1 için artık post-hoc değil, training-time representation sinyali gerekir.

Önerilen BR1 yolu:

- Ensemble self-distillation zaten repo'da Direction #5 olarak uygulanmaya baslamis durumda.
- Eger distillation BR1'i anlamlı kaldirmazsa, hard-negative SupCon BR1<->BR2 geçmeli.
- Density-conditioned head veya global tissue heterogeneity sinyali, BR1 için yeni feature prior sağlayabilir.

### BR2

BR2 testte baskın siniflardan biri ve birçok trade-off'un bedelini oduyor. F2 tau artinca BR4 recall yukselirken BR2 zarar gordu. Pairwise tuning ve BR1/BR4 hedefli mudahaleler BR2 mass'i korumazsa macro F1 artmiyor.

BR2 için ana strateji:

- BR1/BR4 iyileştirmeleri BR2 precision/recall kaybina karsi mutlaka paired olarak raporlanmali.
- Her yeni loss veya head, BR2 false transfer'ini ana safety metric olarak izlemeli.

### BR4

BR4 en umutlu hedef. Ensemble BR4'u `0.555`'e çıkardı. No-hflip ablation BR4 F1'de `+2.0 pp`, recall'da `+8.9 pp` yönlü sinyal verdi. Phase 0c k-NN, BR4->BR5 median ratio `0.312` ile geometrik overlap gösterdi.

BR4 için ana strateji:

- Per-region / mass-localization head.
- Top-k token aggregation veya token-level malignancy logits.
- 1024 cozunurlukta kaybolabilecek microcalcification / mass morphology için 2048 input deneyleri.

### BR5

BR5 en stabil sınıf. Seed CV düşük, baseline F1 yüksek. No-hflip BR5'i her seed'de bozdu; bu nedenle hflip regularization korunmali. BR5 tarafında amac F1'i artırmaktan çok, BR4 iyileştirmesi yaparken BR5'i BR4'e itmemek.

---

## 14. Güncel Repo State Notu

Bu rapor hazırlanırken mevcut dosya durumunda su noktalar doğrulandi:

| Durum | Kanit |
| --- | --- |
| `CLAUDE.md` baseline olarak n=4 C6 ensemble `0.6846` diyor | `CLAUDE.md` operational rules ve architecture status |
| `artifacts/seed_ensemble_n4.json` mevcut | n=4 ensemble metrics ve confusion matrix |
| `artifacts/seed_ensemble_n4_tta.json` mevcut | TTA regression metrics |
| `artifacts/phase0c_audit_substitute.{md,json}` mevcut | Phase 0c programmatic audit |
| `tools/build_ensemble_soft_targets.py` mevcut | Direction #5 soft target generation |
| `utils/losses.py` distillation loss kodu içeriyor | `distill_alpha`, `temperature`, KL term |
| `configs/...c6_distill_seed{42,7,555,999}.yaml` mevcut | Self-distillation seed configleri |
| `outputs/...c6_distill_seed{42,7,555,999}/checkpoints/best_model.pt` mevcut | Distillation checkpointleri May 6 itibariyle uretilmis |
| Distillation için final test ensemble report artifact'i bulunmadi | `outputs/*distill*/reports` ve `artifacts/*distill*` bos gorundu |

Bu nedenle Lesson #60 sonrasi Direction #5 uygulanmaya baslamis/egitilmis gorunuyor; fakat bu raporda distillation performansi yeni sonuç olarak sunulmadi, çünkü final test logits/ensemble artifact veya classification report mevcut degildi.

---

## 15. Hocalara Sunulacak Ana Mesaj

Bu deney serisinin ana bilimsel katkisi, "modeli rastgele kurcalayarak C6'yi gecemiyoruz" sonucundan daha olgun bir noktaya tasinmasidir. Artik elimizde uc katmanli kanit var:

1. Seed istatistigi: Tek-seed C6 sonucuna güvenmek yanlış; beklenen tek model F1 `0.6502 +/- 0.0137`.
2. Inference aggregation: n=4 logit ensemble gercek ve deploy edilebilir kazanç veriyor; production baseline `0.6846`.
3. Hata mekanizmasi: BR1 ve BR4 hatalari rastgele değil, embedding uzayında geometrik olarak yapılı; bu nedenle gelecekteki çalışma yeni representation sinyali eklemeli.

Hocalara önerilecek net karar:

- Production için n=4 C6 logit-mean ensemble kullanılsın.
- TTA, Saerens-EM, threshold/gating ve no-hflip production'a alinmasin.
- Yeni deneyler ensemble baseline `0.6846`'yi gecmek zorunda.
- BR1 için ucuz inference ve augmentation track kapandı; hard-negative SupCon / density-conditioned representation çalışması gerekiyor.
- BR4 için per-region mass-localization veya yüksek çözünürlük mimari track acilmali.

---

## 16. Tek Slaytlik Özet

| Soru | Cevap |
| --- | --- |
| Eski C6 champion güvenilir miydi? | Hayir. seed=42 `0.6762`, +1.9 sigma lucky outlier |
| Tek C6 modelin beklenen F1'i nedir? | `0.6502 +/- 0.0137` |
| En iyi mevcut production baseline nedir? | n=4 C6 logit-mean ensemble, macro F1 `0.6846` |
| Ensemble neyi iyileştirdi? | BR2, BR4, BR5; özellikle BR4 `0.555` |
| Ensemble BR1'i cozdü mu? | Hayir. BR1 seed=42 seviyesine geldi ama üzerine cikmadi |
| TTA faydalı mi? | Tek seed'de az fayda, ensemble ustunde negatif |
| Prior correction faydalı mi? | Hayir. Saerens-EM BR1/BR2 collapse üretti |
| Pairwise threshold faydalı mi? | Diagnostic; production çözüm değil |
| No-hflip faydalı mi? | Hayir. Macro aynı, BR5 anlamlı zarar gordu |
| Hatalar label noise mu? | Phase 0c'ye göre cogunlukla representation-side geometric overlap |
| Sonraki en mantıklı is | Ensemble self-distillation sonucu değerlendir; sonra BR1 hard-negative SupCon ve BR4 per-region head |

