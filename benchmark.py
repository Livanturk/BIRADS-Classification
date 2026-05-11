"""
Benchmark Karşılaştırma Scripti
=================================
İki veya daha fazla deneyin test sonuçlarını karşılaştırır.

Kullanım:
    # İki deneyi karşılaştır (checkpoint'lar mevcut olmalı):
    python benchmark.py --configs configs/convnext_large_seg_v1.yaml configs/convnext_large_noseg_v1.yaml

    # Sadece sonuçları karşılaştır (eğitim tamamlanmış, checkpoint'lar mevcut):
    python benchmark.py --configs configs/convnext_large_seg_v1.yaml configs/convnext_large_noseg_v1.yaml --eval-only

    # Sıralı eğitim + karşılaştırma:
    python benchmark.py --configs configs/convnext_large_seg_v1.yaml configs/convnext_large_noseg_v1.yaml --train-first

Çıktı:
    - Karşılaştırmalı metrik tablosu (konsol)
    - McNemar testi sonuçları (istatistiksel anlamlılık)
    - outputs/benchmark_report.txt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from tabulate import tabulate

from data.dataset import create_dataloaders
from models.full_model import build_model
from utils.losses import build_loss_function
from utils.metrics import MetricTracker, mcnemar_test


def tta_forward(model: torch.nn.Module, images: torch.Tensor) -> dict:
    """
    Test Time Augmentation (TTA) ile inference.

    4 augmentation versiyonunun softmax olasılıklarını ortalar:
        1. Orijinal
        2. Yatay flip
        3. +5° rotasyon
        4. -5° rotasyon

    Args:
        model: Eval modunda model.
        images: (B, 4, 3, H, W) — 4 mammografi görüntüsü.

    Returns:
        dict: full_logits ortalamasıyla güncellenmiş outputs.
    """
    B, V, C, H, W = images.shape

    def apply_aug(imgs, aug_fn):
        # imgs: (B, V, C, H, W) → her view'a aynı aug
        flat = imgs.view(B * V, C, H, W)
        aug = aug_fn(flat)
        return aug.view(B, V, C, H, W)

    augmented = [
        images,                                                         # orijinal
        apply_aug(images, lambda x: torch.flip(x, dims=[-1])),         # h-flip
        apply_aug(images, lambda x: TF.rotate(x, angle=5)),            # +5°
        apply_aug(images, lambda x: TF.rotate(x, angle=-5)),           # -5°
    ]

    all_full_probs = []
    base_outputs = None

    for aug_images in augmented:
        with torch.no_grad():
            out = model(aug_images)
        probs = F.softmax(out["full_logits"], dim=-1)   # (B, 4)
        all_full_probs.append(probs)
        if base_outputs is None:
            base_outputs = out

    # Ortalama olasılıkları log-prob'a çevir (MetricTracker argmax'a bakıyor)
    avg_probs = torch.stack(all_full_probs, dim=0).mean(dim=0)  # (B, 4)
    base_outputs = dict(base_outputs)
    base_outputs["full_logits"] = avg_probs   # tracker argmax(full_logits) kullanır
    base_outputs["confidence"] = avg_probs.max(dim=-1).values

    return base_outputs


def load_config(config_path: str) -> dict:
    """YAML config dosyasını yükler."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_experiment_name(config_path: str) -> str:
    """Config dosya adından deney adını türetir."""
    return Path(config_path).stem


def evaluate_ensemble(
    config_paths: List[str],
    device: torch.device,
    use_tta: bool = False,
    weights: List[float] = None,
) -> Dict:
    """
    Birden fazla modelin softmax çıktılarını ağırlıklı ortalar (model ensemble).

    Her model ayrı ayrı forward pass yapar, softmax olasılıkları ağırlıklı ortalaması alınır.
    TTA ile birlikte kullanılırsa her model önce TTA'ya girer, sonra ensemble yapılır.

    Args:
        config_paths: Birleştirilecek config dosyaları.
        device: CUDA/CPU cihazı.
        use_tta: Her model için TTA uygula.
        weights: Her model için ağırlık (None → eşit ağırlık). Otomatik normalize edilir.

    Returns:
        dict: {"metrics": dict, "tracker": MetricTracker, "experiment_name": str}
    """
    n = len(config_paths)

    # Ağırlıkları normalize et
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    print(f"\n{'='*60}")
    print(f"Ensemble: {n} model birleştiriliyor")
    print(f"  Ağırlıklar: {[f'{w:.3f}' for w in weights]}")
    if use_tta:
        print(f"  [TTA] Her model için 4-augmentation inference aktif")
    print(f"{'='*60}")

    # Tüm modelleri yükle
    loaded = []
    for config_path in config_paths:
        config = load_config(config_path)
        exp_name = get_experiment_name(config_path)
        config["checkpoint"]["save_dir"] = os.path.join("outputs", exp_name, "checkpoints")
        checkpoint_path = os.path.join(config["checkpoint"]["save_dir"], "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {checkpoint_path}\n"
                f"Önce eğitimi tamamlayın: python train.py --config {config_path}"
            )
        model = build_model(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaded.append((model, config, exp_name))
        print(f"  Yüklendi: {exp_name}")

    # Test veri seti ve loss fonksiyonu için ilk config'i kullan
    first_config = loaded[0][1]
    dataloaders = create_dataloaders(first_config)
    criterion = build_loss_function(first_config, device)

    tracker = MetricTracker()
    tracker.reset()

    with torch.no_grad():
        for batch in dataloaders["test"]:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            all_probs = []
            base_outputs = None

            for (model, config, _), w in zip(loaded, weights):
                if use_tta:
                    out = tta_forward(model, images)
                    # tta_forward full_logits'i zaten avg prob olarak döner
                    probs = out["full_logits"]
                else:
                    with torch.no_grad():
                        out = model(images)
                    probs = F.softmax(out["full_logits"], dim=-1)

                all_probs.append(probs * w)
                if base_outputs is None:
                    base_outputs = dict(out)

            # Ağırlıklı toplam (ağırlıklar zaten normalize, toplamı 1.0)
            avg_probs = torch.stack(all_probs, dim=0).sum(dim=0)    # (B, 4)
            base_outputs = dict(base_outputs)
            base_outputs["full_logits"] = avg_probs
            base_outputs["confidence"] = avg_probs.max(dim=-1).values

            loss_dict = criterion(base_outputs, labels)
            tracker.update(base_outputs, labels, loss_dict)

    metrics = tracker.compute()
    ensemble_name = f"Ensemble({n})"

    print(f"  F1 (Macro):      {metrics.get('full_f1_macro', 0):.4f}")
    print(f"  AUC-ROC:         {metrics.get('full_auc_roc', 0):.4f}")
    print(f"  Cohen's Kappa:   {metrics.get('full_cohens_kappa', 0):.4f}")
    print(f"  Accuracy:        {metrics.get('full_accuracy', 0):.4f}")

    return {
        "metrics": metrics,
        "tracker": tracker,
        "experiment_name": ensemble_name,
        "config_path": "ensemble",
    }


def evaluate_model(config_path: str, device: torch.device, use_tta: bool = False) -> Dict:
    """
    Belirtilen config ile eğitilmiş modeli test setinde değerlendirir.

    Args:
        config_path: Config dosya yolu.
        device: CUDA/CPU cihazı.

    Returns:
        dict: {"metrics": dict, "tracker": MetricTracker, "experiment_name": str}
    """
    if use_tta:
        print(f"  [TTA] 4-augmentation inference aktif (orig + hflip + rot±5°)")
    config = load_config(config_path)
    experiment_name = get_experiment_name(config_path)

    # Output dizinlerini güncelle
    base_dir = os.path.join("outputs", experiment_name)
    config["checkpoint"]["save_dir"] = os.path.join(base_dir, "checkpoints")

    # Checkpoint yolunu kontrol et
    checkpoint_path = os.path.join(config["checkpoint"]["save_dir"], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {checkpoint_path}\n"
            f"Önce eğitimi tamamlayın: python train.py --config {config_path}"
        )

    print(f"\n{'='*60}")
    print(f"Değerlendirme: {experiment_name}")
    print(f"{'='*60}")

    # Veri yükle
    dataloaders = create_dataloaders(config)

    # Model oluştur ve checkpoint yükle
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Loss fonksiyonu
    criterion = build_loss_function(config, device)

    # Test değerlendirmesi
    tracker = MetricTracker()
    tracker.reset()

    with torch.no_grad():
        for batch in dataloaders["test"]:
            images = batch["images"].to(device)
            labels = batch["label"].to(device)

            if use_tta:
                outputs = tta_forward(model, images)
            else:
                outputs = model(images)
            loss_dict = criterion(outputs, labels)
            tracker.update(outputs, labels, loss_dict)

    metrics = tracker.compute()

    print(f"  F1 (Macro):      {metrics.get('full_f1_macro', 0):.4f}")
    print(f"  AUC-ROC:         {metrics.get('full_auc_roc', 0):.4f}")
    print(f"  Cohen's Kappa:   {metrics.get('full_cohens_kappa', 0):.4f}")
    print(f"  Accuracy:        {metrics.get('full_accuracy', 0):.4f}")

    return {
        "metrics": metrics,
        "tracker": tracker,
        "experiment_name": experiment_name,
        "config_path": config_path,
    }


def compare_experiments(results: List[Dict], save_path: str = "outputs/benchmark_report.txt"):
    """
    İki veya daha fazla deneyin sonuçlarını karşılaştırır.

    Args:
        results: evaluate_model çıktılarının listesi.
        save_path: Rapor kayıt yolu.
    """
    print(f"\n{'='*80}")
    print("BENCHMARK KARŞILAŞTIRMA RAPORU")
    print(f"{'='*80}")

    # Karşılaştırma metrikleri
    compare_metrics = [
        ("F1 (Macro)", "full_f1_macro"),
        ("AUC-ROC (Macro)", "full_auc_roc"),
        ("Accuracy", "full_accuracy"),
        ("Cohen's Kappa", "full_cohens_kappa"),
        ("Binary F1", "binary_f1"),
        ("Mean Confidence", "mean_confidence"),
        ("F1 BIRADS-1", "full_f1_BIRADS_1"),
        ("F1 BIRADS-2", "full_f1_BIRADS_2"),
        ("F1 BIRADS-4", "full_f1_BIRADS_4"),
        ("F1 BIRADS-5", "full_f1_BIRADS_5"),
        ("Total Loss", "total_loss"),
    ]

    # Tablo oluştur
    headers = ["Metrik"] + [r["experiment_name"] for r in results]
    if len(results) == 2:
        headers.append("Δ (A-B)")

    rows = []
    for display_name, metric_key in compare_metrics:
        row = [display_name]
        values = []
        for r in results:
            val = r["metrics"].get(metric_key, None)
            if val is not None:
                row.append(f"{val:.4f}")
                values.append(val)
            else:
                row.append("N/A")
                values.append(None)

        # Delta hesapla (2 deney varsa)
        if len(results) == 2 and all(v is not None for v in values):
            delta = values[0] - values[1]
            sign = "+" if delta > 0 else ""
            row.append(f"{sign}{delta:.4f}")

        rows.append(row)

    table = tabulate(rows, headers=headers, tablefmt="grid", stralign="center")
    print(f"\n{table}")

    # McNemar testi (2 deney varsa)
    mcnemar_result = None
    if len(results) == 2:
        preds_a = results[0]["tracker"].get_predictions()
        preds_b = results[1]["tracker"].get_predictions()

        # Etiketlerin aynı olduğunu doğrula (aynı test seti)
        if np.array_equal(preds_a["labels"], preds_b["labels"]):
            mcnemar_result = mcnemar_test(
                preds_a["preds"], preds_b["preds"], preds_a["labels"]
            )

            print(f"\n--- McNemar Testi ---")
            print(f"  Sadece {results[0]['experiment_name']} doğru: {mcnemar_result['b']} örnek")
            print(f"  Sadece {results[1]['experiment_name']} doğru: {mcnemar_result['c']} örnek")
            print(f"  Chi-squared: {mcnemar_result['chi2']:.4f}")
            print(f"  p-value: {mcnemar_result['p_value']:.6f}")

            if mcnemar_result["p_value"] < 0.05:
                print(f"  → İki model istatistiksel olarak FARKLI performans gösteriyor (p < 0.05)")
            else:
                print(f"  → İki model arasında anlamlı fark YOK (p ≥ 0.05)")
        else:
            print("\n[UYARI] Test etiketleri eşleşmiyor — McNemar testi uygulanamaz.")

    # Raporu dosyaya kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("BENCHMARK KARŞILAŞTIRMA RAPORU\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Deneyler:\n")
        for r in results:
            f.write(f"  - {r['experiment_name']} ({r['config_path']})\n")
        f.write(f"\n{table}\n")

        if mcnemar_result:
            f.write(f"\nMcNemar Testi:\n")
            f.write(f"  Sadece {results[0]['experiment_name']} doğru: {mcnemar_result['b']}\n")
            f.write(f"  Sadece {results[1]['experiment_name']} doğru: {mcnemar_result['c']}\n")
            f.write(f"  Chi-squared: {mcnemar_result['chi2']:.4f}\n")
            f.write(f"  p-value: {mcnemar_result['p_value']:.6f}\n")

    print(f"\n[BİLGİ] Rapor kaydedildi: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: Birden fazla deneyin test sonuçlarını karşılaştır"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Karşılaştırılacak config dosyaları (örn: configs/a.yaml configs/b.yaml)",
    )
    parser.add_argument(
        "--train-first",
        action="store_true",
        help="Önce eğitimi çalıştır, sonra karşılaştır",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=True,
        help="Sadece değerlendirme yap (varsayılan)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU indeksi",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Rapor kayıt yolu (belirtilmezse config adlarından otomatik üretilir)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Test Time Augmentation aktif et (orig + hflip + rot±5° ortalaması)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="Verilen tüm config'leri birleştirerek ensemble değerlendirmesi de yap",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        default=None,
        metavar="W",
        help="Ensemble model ağırlıkları (örn: 0.4 0.6). Otomatik normalize edilir. Verilmezse eşit ağırlık.",
    )

    args = parser.parse_args()

    # Cihaz
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # Eğitimi çalıştır (opsiyonel)
    if args.train_first:
        import subprocess
        for config_path in args.configs:
            print(f"\n[EĞİTİM] {config_path} çalıştırılıyor...")
            result = subprocess.run(
                [sys.executable, "train.py", "--config", config_path, "--device", str(args.device)],
                check=True,
            )
            if result.returncode != 0:
                print(f"[HATA] Eğitim başarısız: {config_path}")
                sys.exit(1)

    # Değerlendirme
    results = []
    for config_path in args.configs:
        result = evaluate_model(config_path, device, use_tta=args.tta)
        results.append(result)

    # Ensemble değerlendirmesi (isteğe bağlı, en az 2 config gerekli)
    if args.ensemble:
        if len(args.configs) < 2:
            print("\n[UYARI] Ensemble için en az 2 config gereklidir, atlanıyor.")
        else:
            ew = args.ensemble_weights
            if ew is not None and len(ew) != len(args.configs):
                print(f"\n[UYARI] --ensemble-weights sayısı ({len(ew)}) config sayısıyla ({len(args.configs)}) eşleşmiyor, eşit ağırlık kullanılıyor.")
                ew = None
            ensemble_result = evaluate_ensemble(args.configs, device, use_tta=args.tta, weights=ew)
            results.append(ensemble_result)

    # Otomatik çıktı yolu üret
    if args.output is None:
        exp_names = [get_experiment_name(c) for c in args.configs]
        name_part = "__vs__".join(exp_names)
        suffix_parts = []
        if args.tta:
            suffix_parts.append("tta")
        if args.ensemble:
            ew = args.ensemble_weights
            if ew is not None:
                w_str = "-".join(f"{w:.2g}" for w in ew)
                suffix_parts.append(f"ensemble_w{w_str}")
            else:
                suffix_parts.append("ensemble")
        suffix = ("__" + "__".join(suffix_parts)) if suffix_parts else ""
        save_path = f"outputs/benchmark__{name_part}{suffix}.txt"
    else:
        save_path = args.output

    # Karşılaştırma
    compare_experiments(results, save_path=save_path)


if __name__ == "__main__":
    main()
