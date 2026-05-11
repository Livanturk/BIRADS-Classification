# Deney Raporlama Adımları

## 1. Yeni deney başlamadan önce

Config dosyasını oluşturduktan sonra raporu oluştur:

```bash
python scripts/report.py new <deney_adı> --baseline <baseline_adı>
```

Örnek:
```bash
python scripts/report.py new swinv2_base_focal_v1 --baseline swinv2_base_v4
```

Sonra `experiments/<deney_adı>/report.md` dosyasını aç ve **Motivasyon / Hipotez** bölümünü yaz.

---

## 2. Eğitim başlarken

`report.md` frontmatter'ında `status` alanını güncelle:

```yaml
status: running
```

---

## 3. Eğitim bittikten sonra

Val metriklerini checkpoint'ten otomatik doldur:

```bash
python scripts/report.py fill <deney_adı>
```

---

## 4. Test değerlendirmesinden sonra

`experiments/<deney_adı>/report.md` dosyasını manuel aç ve şunları doldur:

- **Test tablosu** — `benchmark.py` çıktısından kopyala
- **Analiz** — ne çalıştı, ne çalışmadı, dikkat çeken bulgular
- **Sonraki Adım** — bu deneyin sonucuna göre ne yapılacak

Frontmatter'daki `test_f1_macro` alanını da güncelle:

```yaml
test_f1_macro: 0.7233
```

---

## 5. Index'i güncelle

```bash
python scripts/report.py index
```

`experiments/EXPERIMENTS.md` tüm deneylerin güncel özetini gösterir.

---

## Özet

| Adım | Ne zaman | Komut / Aksiyon |
|---|---|---|
| 1 | Config oluşturulunca | `report.py new <ad> --baseline <ad>` + Motivasyon yaz |
| 2 | Eğitim başlarken | `status: running` yaz |
| 3 | Eğitim bitince | `report.py fill <ad>` |
| 4 | Test bitince | Test tablosu + Analiz + Sonraki Adım yaz |
| 5 | Her zaman | `report.py index` |
