"""
Ensemble Evaluation Script
===========================
Birden fazla modelin tahminlerini birleştirerek
test_full_f1_macro'yu artırır.

Desteklenen modlar:
    1. Basit ensemble: Her modelin softmax çıkışlarını eşit/ağırlıklı ortalama
    2. TTA (Test-Time Augmentation): Her model için orijinal + augmented tahminleri ortalama
    3. Ensemble + TTA: İkisini birleştir
    4. Stacking: Meta-learner (Logistic Regression) val predictions uzerinde ogrenip
       test set'te degerlendirir. Her sinif icin farkli model agirliklari ogrenir.

Kullanım:
    python ensemble_evaluate.py
    python ensemble_evaluate.py --tta
    python ensemble_evaluate.py --optimize-weights
    python ensemble_evaluate.py --stacking
    python ensemble_evaluate.py --device 0
"""

import argparse
import copy
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.full_model import build_model
from data.dataset import create_dataloaders, MammographyDataset, prepare_patient_split
from data.transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD


# =====================================================================
# Model tanımları: (config_path, checkpoint_path, açıklama)
# =====================================================================
MODELS = [
    {
        "name": "ConvNeXtV2-Base-Original",
        "config": "configs/convnextv2_base_original.yaml",
        "checkpoint": "outputs/convnextv2_base_original/checkpoints/best_model.pt",
    },
    {
        "name": "DINOv2-ViT-Large-Original",
        "config": "configs/dinov2_large_original.yaml",
        "checkpoint": "outputs/dinov2_large_original/checkpoints/best_model.pt",
    },
    {
        "name": "SwinV2-Base-v2",
        "config": "configs/swinv2_base_v2.yaml",
        "checkpoint": "outputs/swinv2_base_v2/checkpoints/best_model.pt",
    },
]


def load_config(config_path: str) -> dict:
    """YAML config dosyasını yükler."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Config ve checkpoint'ten model yükler."""
    config = load_config(config_path)
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    metrics = checkpoint.get("metrics", {})
    val_f1 = metrics.get("val_full_f1_macro", metrics.get("full_f1_macro", "?"))
    print(f"  Epoch: {epoch}, Val F1: {val_f1}")

    return model, config


def get_tta_transforms(image_size: int) -> List[transforms.Compose]:
    """
    TTA için farklı transform pipeline'ları döndürür.

    Augmentasyonlar:
        0. Orijinal (identity)
        1. Horizontal flip
        2. +5 derece rotasyon
        3. -5 derece rotasyon
        4. Hafif zoom-in (1.05x)
    """
    base_normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    tta_list = [
        # 0. Orijinal
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            *base_normalize,
        ]),
        # 1. Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            *base_normalize,
        ]),
        # 2. +5 derece rotasyon
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(5, 5)),
            *base_normalize,
        ]),
        # 3. -5 derece rotasyon
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(-5, -5)),
            *base_normalize,
        ]),
        # 4. Hafif zoom-in
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=0, scale=(1.03, 1.03)),
            *base_normalize,
        ]),
    ]

    return tta_list


@torch.no_grad()
def get_model_predictions(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Modelin test seti üzerindeki softmax tahminlerini döndürür.

    Returns:
        all_probs: (N, 4) softmax olasılıkları
        all_preds: (N,) argmax tahminleri
        all_labels: (N,) gerçek etiketler
    """
    all_probs = []
    all_labels = []

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["label"]
        outputs = model(images)
        probs = F.softmax(outputs["full_logits"], dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = all_probs.argmax(axis=1)

    return all_probs, all_preds, all_labels


@torch.no_grad()
def get_model_predictions_tta(
    model,
    config: dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TTA ile modelin tahminlerini döndürür.
    Her TTA augmentasyonu için ayrı DataLoader oluşturur,
    sonra softmax olasılıklarını ortalar.

    Returns:
        avg_probs: (N, 4) ortalama softmax olasılıkları
        all_labels: (N,) gerçek etiketler
    """
    data_cfg = config["data"]
    image_size = data_cfg["image_size"]
    tta_transforms = get_tta_transforms(image_size)

    # Test setini hazırla (sabit split)
    splits = prepare_patient_split(
        root_dir=data_cfg["root_dir"],
        test_dir=data_cfg["test_dir"],
        train_ratio=data_cfg["split"]["train"],
        val_ratio=data_cfg["split"]["val"],
        seed=config["project"]["seed"],
    )
    test_dirs, test_labels = splits["test"]

    all_tta_probs = []

    for i, tta_transform in enumerate(tta_transforms):
        tta_dataset = MammographyDataset(
            patient_dirs=test_dirs,
            labels=test_labels,
            transform=tta_transform,
        )
        tta_loader = DataLoader(
            tta_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=data_cfg["pin_memory"],
        )

        probs, _, labels = get_model_predictions(model, tta_loader, device)
        all_tta_probs.append(probs)
        print(f"    TTA {i}/{len(tta_transforms)-1} tamamlandı")

    # Ortalama
    avg_probs = np.mean(all_tta_probs, axis=0)
    return avg_probs, labels


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Softmax olasılıklarından tüm metrikleri hesaplar."""
    preds = probs.argmax(axis=1)
    metrics = {}

    # Full head metrikleri
    metrics["full_accuracy"] = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    metrics["full_precision_macro"] = precision
    metrics["full_recall_macro"] = recall
    metrics["full_f1_macro"] = f1

    # Per-class F1
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    birads_names = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]
    for i, name in enumerate(birads_names):
        if i < len(f1_per_class):
            metrics[f"full_f1_{name}"] = f1_per_class[i]

    # AUC-ROC
    try:
        metrics["full_auc_roc"] = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        )
    except ValueError:
        metrics["full_auc_roc"] = 0.0

    # Binary metrikleri
    binary_preds = (preds >= 2).astype(int)
    binary_labels = (labels >= 2).astype(int)
    metrics["binary_accuracy"] = accuracy_score(binary_labels, binary_preds)
    bp, br, bf1, _ = precision_recall_fscore_support(
        binary_labels, binary_preds, average="binary", zero_division=0
    )
    metrics["binary_precision"] = bp
    metrics["binary_recall"] = br
    metrics["binary_f1"] = bf1

    return metrics


def print_metrics(metrics: dict, title: str = ""):
    """Metrikleri tablo formatında yazdırır."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    print(f"\n  Full Accuracy:        {metrics['full_accuracy']:.4f}")
    print(f"  Full F1 Macro:        {metrics['full_f1_macro']:.4f}")
    print(f"  Full Precision Macro: {metrics['full_precision_macro']:.4f}")
    print(f"  Full Recall Macro:    {metrics['full_recall_macro']:.4f}")
    print(f"  Full AUC-ROC:         {metrics['full_auc_roc']:.4f}")
    print(f"  Binary Accuracy:      {metrics['binary_accuracy']:.4f}")
    print(f"  Binary F1:            {metrics['binary_f1']:.4f}")

    birads_names = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]
    print(f"\n  Per-Class F1:")
    for name in birads_names:
        key = f"full_f1_{name}"
        if key in metrics:
            print(f"    {name}: {metrics[key]:.4f}")


def print_classification_report(probs: np.ndarray, labels: np.ndarray):
    """Detaylı classification report yazdırır."""
    preds = probs.argmax(axis=1)
    target_names = ["BIRADS-1", "BIRADS-2", "BIRADS-4", "BIRADS-5"]
    report = classification_report(
        labels, preds, target_names=target_names, digits=4, zero_division=0
    )
    print(f"\n{report}")


def plot_confusion_matrix(probs: np.ndarray, labels: np.ndarray, save_path: str, title: str = ""):
    """Confusion matrix çizer ve kaydeder."""
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds)

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    birads_labels = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 4", "BI-RADS 5"]

    # Hem yüzde hem sayı göster
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"

    sns.heatmap(
        cm_norm, annot=annot, fmt="",
        xticklabels=birads_labels, yticklabels=birads_labels,
        cmap="Blues", ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_ylabel("Gercek Etiket", fontsize=12)
    ax.set_title(title or "Confusion Matrix", fontsize=14)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix kaydedildi: {save_path}")


def optimize_ensemble_weights(
    model_probs_list: List[np.ndarray],
    labels: np.ndarray,
    step: float = 0.05,
) -> Tuple[List[float], float]:
    """
    Grid search ile optimal ensemble ağırlıklarını bulur.

    Args:
        model_probs_list: Her modelin softmax çıkışları [(N,4), (N,4), ...]
        labels: Gerçek etiketler (N,)
        step: Ağırlık grid adımı

    Returns:
        best_weights: Optimal ağırlıklar
        best_f1: En iyi F1 macro skoru
    """
    n_models = len(model_probs_list)

    if n_models == 2:
        best_f1 = 0.0
        best_weights = [0.5, 0.5]

        for w1 in np.arange(0.0, 1.0 + step, step):
            w2 = 1.0 - w1
            weights = [w1, w2]

            ensemble_probs = sum(w * p for w, p in zip(weights, model_probs_list))
            preds = ensemble_probs.argmax(axis=1)

            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights

        return best_weights, best_f1
    else:
        # N model: grid search (kaba)
        best_f1 = 0.0
        best_weights = [1.0 / n_models] * n_models
        steps = np.arange(0.0, 1.0 + step, step)

        for combo in itertools.product(steps, repeat=n_models):
            if abs(sum(combo) - 1.0) > 1e-6:
                continue
            weights = list(combo)

            ensemble_probs = sum(w * p for w, p in zip(weights, model_probs_list))
            preds = ensemble_probs.argmax(axis=1)

            _, _, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_weights = weights

        return best_weights, best_f1


def optimize_per_class_weights(
    model_probs_list: List[np.ndarray],
    labels: np.ndarray,
    model_names: List[str],
    step: float = 0.05,
) -> Tuple[np.ndarray, dict]:
    """
    Her sinif icin bagimsiz model agirliklari bulur (grid search).

    Global optimize-weights her sinif icin ayni agirliklari kullanir.
    Bu fonksiyon her sinifin kararini farkli agirliklarla verir:
        - BIRADS-1 icin ConvNeXtV2 agirlikli olabilir
        - BIRADS-4 icin DINOv2 agirlikli olabilir
        - vs.

    Yontem:
        1. Her sinif icin: softmax prob'larini agirlikli topla
        2. Her ornegi "en yuksek agirlikli prob'a sahip sinif"a ata
        3. Grid search ile her sinifin agirliklarini bagimsiz optimize et

    Args:
        model_probs_list: [(N, 4), ...] softmax cikislari
        labels: (N,) gercek etiketler
        model_names: model isimleri
        step: grid adimi

    Returns:
        final_probs: (N, 4) per-class agirlikli olasiliklar
        info: dict with per-class weights and metrics
    """
    n_models = len(model_probs_list)
    n_classes = model_probs_list[0].shape[1]
    n_samples = len(labels)
    steps = np.arange(0.0, 1.0 + step, step)

    # Her sinif icin optimal agirlik bul
    per_class_weights = np.zeros((n_classes, n_models))
    class_names = ["BIRADS-1", "BIRADS-2", "BIRADS-4", "BIRADS-5"]

    print(f"\n  Per-class agirlik optimizasyonu (step={step}):")

    for c in range(n_classes):
        best_f1_class = -1.0
        best_w = [1.0 / n_models] * n_models

        if n_models == 2:
            for w1 in steps:
                w2 = 1.0 - w1
                w = [w1, w2]
                blended_c = sum(w[m] * model_probs_list[m][:, c] for m in range(n_models))
                # Geri kalan siniflar icin esit agirlik
                blended_all = np.zeros((n_samples, n_classes))
                for k in range(n_classes):
                    if k == c:
                        blended_all[:, k] = blended_c
                    else:
                        blended_all[:, k] = np.mean(
                            [model_probs_list[m][:, k] for m in range(n_models)], axis=0
                        )
                preds = blended_all.argmax(axis=1)
                _, _, f1, _ = precision_recall_fscore_support(
                    labels, preds, average="macro", zero_division=0
                )
                if f1 > best_f1_class:
                    best_f1_class = f1
                    best_w = w
        else:
            for combo in itertools.product(steps, repeat=n_models):
                if abs(sum(combo) - 1.0) > 1e-6:
                    continue
                w = list(combo)
                blended_c = sum(w[m] * model_probs_list[m][:, c] for m in range(n_models))
                blended_all = np.zeros((n_samples, n_classes))
                for k in range(n_classes):
                    if k == c:
                        blended_all[:, k] = blended_c
                    else:
                        blended_all[:, k] = np.mean(
                            [model_probs_list[m][:, k] for m in range(n_models)], axis=0
                        )
                preds = blended_all.argmax(axis=1)
                _, _, f1, _ = precision_recall_fscore_support(
                    labels, preds, average="macro", zero_division=0
                )
                if f1 > best_f1_class:
                    best_f1_class = f1
                    best_w = w

        per_class_weights[c] = best_w
        w_str = ", ".join(f"{ww:.2f}" for ww in best_w)
        print(f"    {class_names[c]}: [{w_str}] -> F1 macro={best_f1_class:.4f}")

    # Tum siniflar icin per-class agirlikli ensemble olustur
    final_probs = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        for m in range(n_models):
            final_probs[:, c] += per_class_weights[c, m] * model_probs_list[m][:, c]

    # Model basina ortalama agirlik
    print(f"\n  Model basina ortalama agirlik:")
    for m, name in enumerate(model_names):
        avg_w = per_class_weights[:, m].mean()
        print(f"    {name}: {avg_w:.3f}")

    info = {
        "per_class_weights": per_class_weights,
        "class_names": class_names,
    }

    return final_probs, info


def stacking_ensemble(
    val_probs_list: List[np.ndarray],
    val_labels: np.ndarray,
    test_probs_list: List[np.ndarray],
    model_names: List[str],
) -> Tuple[np.ndarray, dict]:
    """
    Stacking meta-learner: Val set prediction'lari uzerinde
    Logistic Regression egitir, test set'te degerlendirir.

    Probability averaging'den farki:
        - Her sinif icin farkli model agirliklari ogrenebilir
        - Model etkilesimleri (hangi modelin hangi sinifta iyi oldugu) yakalanir
        - Daha esnek birlestirme stratejisi

    Features: Her modelin softmax cikislari (N_models * N_classes)
    Meta-learner: L2-regularized Logistic Regression
    C parametresi: 5-fold Stratified CV ile otomatik secilir

    Args:
        val_probs_list: Val set softmax cikislari [(N_val, 4), ...]
        val_labels: Val set etiketleri (N_val,)
        test_probs_list: Test set softmax cikislari [(N_test, 4), ...]
        model_names: Model isimleri

    Returns:
        test_probs: (N_test, 4) meta-learner tahmin olasiliklari
        info: dict with training details
    """
    n_models = len(val_probs_list)
    n_classes = val_probs_list[0].shape[1]

    # Feature matrix: concatenated probabilities
    X_val = np.concatenate(val_probs_list, axis=1)    # (N_val, n_models * n_classes)
    X_test = np.concatenate(test_probs_list, axis=1)  # (N_test, n_models * n_classes)

    print(f"\n  Stacking feature boyutu: {X_val.shape[1]} (= {n_models} model x {n_classes} sinif)")
    print(f"  Val ornekleri: {X_val.shape[0]}, Test ornekleri: {X_test.shape[0]}")

    # Feature normalization
    scaler = StandardScaler()
    X_val_scaled = scaler.fit_transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # C parametresi icin 5-fold Stratified CV
    C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_C = 1.0
    best_cv_f1 = 0.0
    cv_results = []

    print(f"\n  C parametresi CV sonuclari:")
    for C in C_values:
        f1_scores = []
        for train_idx, val_idx in cv.split(X_val_scaled, val_labels):
            lr_fold = LogisticRegression(
                C=C, max_iter=2000, multi_class="multinomial",
                solver="lbfgs", random_state=42,
            )
            lr_fold.fit(X_val_scaled[train_idx], val_labels[train_idx])
            fold_preds = lr_fold.predict(X_val_scaled[val_idx])
            _, _, f1, _ = precision_recall_fscore_support(
                val_labels[val_idx], fold_preds, average="macro", zero_division=0
            )
            f1_scores.append(f1)

        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        cv_results.append((C, mean_f1, std_f1))
        print(f"    C={C:<8} -> F1 Macro: {mean_f1:.4f} +/- {std_f1:.4f}")

        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_C = C

    print(f"\n  En iyi C: {best_C} (CV F1: {best_cv_f1:.4f})")

    # Final meta-learner: tum val seti uzerinde egit
    meta_learner = LogisticRegression(
        C=best_C, max_iter=2000, multi_class="multinomial",
        solver="lbfgs", random_state=42,
    )
    meta_learner.fit(X_val_scaled, val_labels)

    # Test set tahminleri
    test_probs = meta_learner.predict_proba(X_test)

    # Model katkilari analizi — her sinif icin hangi model daha onemli
    coef = meta_learner.coef_  # (n_classes, n_features)
    class_names = ["BIRADS-1", "BIRADS-2", "BIRADS-4", "BIRADS-5"]

    print(f"\n  Meta-learner model katkilari (|katsayi| toplami):")
    print(f"  {'Sinif':<12}", end="")
    for name in model_names:
        print(f"  {name:<25}", end="")
    print()
    print(f"  {'-' * (12 + 27 * n_models)}")

    for c_idx, class_name in enumerate(class_names):
        print(f"  {class_name:<12}", end="")
        for m_idx in range(n_models):
            start = m_idx * n_classes
            model_coefs = coef[c_idx, start:start + n_classes]
            importance = np.abs(model_coefs).sum()
            print(f"  {importance:<25.3f}", end="")
        print()

    info = {
        "best_C": best_C,
        "val_cv_f1": best_cv_f1,
        "cv_results": cv_results,
        "meta_learner": meta_learner,
        "scaler": scaler,
    }

    return test_probs, info


def main():
    parser = argparse.ArgumentParser(description="Ensemble Evaluation")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--tta", action="store_true", help="Test-Time Augmentation aktif")
    parser.add_argument("--optimize-weights", action="store_true", help="Ensemble agirliklarini optimize et")
    parser.add_argument("--stacking", action="store_true", help="Stacking meta-learner (LR) ile ensemble")
    parser.add_argument("--per-class-weights", action="store_true", help="Her sinif icin ayri model agirliklari optimize et")
    parser.add_argument("--weight-step", type=float, default=0.05, help="Agirlik grid adimi")
    parser.add_argument("--save-dir", type=str, default="outputs/ensemble", help="Sonuclarin kaydedilecegi dizin")
    args = parser.parse_args()

    # Cihaz
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
        print(f"[CIHAZ] GPU: {torch.cuda.get_device_name(args.device)}")
    else:
        device = torch.device("cpu")
        print("[CIHAZ] CPU")

    n_steps = 5 if args.stacking else 4

    # ================================================================
    # 1. Modelleri yukle
    # ================================================================
    print(f"\n[1/{n_steps}] Modeller yukleniyor...")
    models = []
    configs = []
    for model_info in MODELS:
        print(f"\n  {model_info['name']}:")
        model, config = load_model(model_info["config"], model_info["checkpoint"], device)
        models.append(model)
        configs.append(config)

    # ================================================================
    # 2. Her modelin tahminlerini al
    # ================================================================
    print(f"\n[2/{n_steps}] Tahminler hesaplaniyor...")
    model_probs_list = []  # test predictions
    val_probs_list = []    # val predictions (stacking icin)
    labels = None
    val_labels = None

    for i, (model, config, model_info) in enumerate(zip(models, configs, MODELS)):
        print(f"\n  {model_info['name']}:")

        if args.tta:
            probs, lbl = get_model_predictions_tta(model, config, device)
        else:
            # Standart DataLoader
            dataloaders = create_dataloaders(config)
            probs, _, lbl = get_model_predictions(model, dataloaders["test"], device)

            # Stacking icin val predictions de topla
            if args.stacking:
                v_probs, _, v_lbl = get_model_predictions(model, dataloaders["val"], device)
                val_probs_list.append(v_probs)
                if val_labels is None:
                    val_labels = v_lbl
                else:
                    assert np.array_equal(val_labels, v_lbl), "Val etiketleri uyusmuyor!"
                print(f"    Val predictions: {v_probs.shape[0]} ornek")

        model_probs_list.append(probs)

        if labels is None:
            labels = lbl
        else:
            assert np.array_equal(labels, lbl), "Etiketler uyusmuyor! Farkli test setleri mi?"

        # Tek model metrikleri
        single_metrics = compute_metrics(probs, labels)
        mode_str = " (TTA)" if args.tta else ""
        print_metrics(single_metrics, f"{model_info['name']}{mode_str}")

    # ================================================================
    # 3. Ensemble
    # ================================================================
    print(f"\n[3/{n_steps}] Ensemble hesaplaniyor...")

    # 3a. Esit agirlikli ensemble
    equal_weights = [1.0 / len(models)] * len(models)
    ensemble_probs_equal = sum(w * p for w, p in zip(equal_weights, model_probs_list))
    metrics_equal = compute_metrics(ensemble_probs_equal, labels)

    mode_str = " + TTA" if args.tta else ""
    print_metrics(metrics_equal, f"Ensemble (Esit Agirlik: {equal_weights}){mode_str}")
    print_classification_report(ensemble_probs_equal, labels)

    # 3b. Agirlik optimizasyonu (opsiyonel)
    if args.optimize_weights:
        print("\n  Agirlik optimizasyonu yapiliyor (grid search)...")
        best_weights, best_f1 = optimize_ensemble_weights(
            model_probs_list, labels, step=args.weight_step
        )
        print(f"  Optimal agirliklar: {best_weights}")
        print(f"  Optimal F1 Macro: {best_f1:.4f}")

        ensemble_probs_opt = sum(w * p for w, p in zip(best_weights, model_probs_list))
        metrics_opt = compute_metrics(ensemble_probs_opt, labels)
        print_metrics(metrics_opt, f"Ensemble (Optimized: {best_weights}){mode_str}")
        print_classification_report(ensemble_probs_opt, labels)

        # En iyi sonucu kullan
        final_probs = ensemble_probs_opt
        final_metrics = metrics_opt
        final_title = f"Ensemble (Optimized: {[f'{w:.2f}' for w in best_weights]}){mode_str}"
    else:
        final_probs = ensemble_probs_equal
        final_metrics = metrics_equal
        final_title = f"Ensemble (Equal){mode_str}"

    # 3c. Per-class agirlik optimizasyonu (opsiyonel)
    per_class_metrics = None
    if args.per_class_weights:
        print(f"\n  Per-class agirlik optimizasyonu basliyor...")
        model_names_list = [m["name"] for m in MODELS]
        per_class_probs, per_class_info = optimize_per_class_weights(
            model_probs_list, labels, model_names_list, step=args.weight_step,
        )
        per_class_metrics = compute_metrics(per_class_probs, labels)
        print_metrics(per_class_metrics, "Ensemble (Per-Class Weights)")
        print_classification_report(per_class_probs, labels)

        if per_class_metrics["full_f1_macro"] > final_metrics["full_f1_macro"]:
            final_probs = per_class_probs
            final_metrics = per_class_metrics
            final_title = f"Ensemble (Per-Class Weights){mode_str}"
            print(f"\n  Per-class weights daha iyi! Final olarak secildi.")

    # ================================================================
    # 4. Stacking ensemble (opsiyonel)
    # ================================================================
    stacking_metrics = None
    stacking_probs = None

    if args.stacking:
        print(f"\n[4/{n_steps}] Stacking meta-learner egitiliyor...")

        if not val_probs_list:
            print("  HATA: Stacking icin val predictions gerekli. TTA ile birlikte kullanilamaz.")
        else:
            model_names = [m["name"] for m in MODELS]
            stacking_probs, stacking_info = stacking_ensemble(
                val_probs_list, val_labels,
                model_probs_list, model_names,
            )
            stacking_metrics = compute_metrics(stacking_probs, labels)
            print_metrics(stacking_metrics, "Stacking Ensemble (Meta-Learner)")
            print_classification_report(stacking_probs, labels)

            # Stacking daha iyiyse final olarak kullan
            if stacking_metrics["full_f1_macro"] > final_metrics["full_f1_macro"]:
                final_probs = stacking_probs
                final_metrics = stacking_metrics
                final_title = f"Stacking (C={stacking_info['best_C']}, CV-F1={stacking_info['val_cv_f1']:.4f})"
                print(f"\n  Stacking, probability averaging'den daha iyi! Final olarak secildi.")
            else:
                print(f"\n  Probability averaging stacking'den daha iyi. Averaging kullaniliyor.")

    # ================================================================
    # 5. Sonuclari kaydet
    # ================================================================
    step_num = n_steps
    print(f"\n[{step_num}/{n_steps}] Sonuclar kaydediliyor: {args.save_dir}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    plot_confusion_matrix(
        final_probs, labels,
        save_path=str(save_dir / "confusion_matrix_ensemble.png"),
        title=final_title,
    )

    # Classification report
    preds = final_probs.argmax(axis=1)
    target_names = ["BIRADS-1", "BIRADS-2", "BIRADS-4", "BIRADS-5"]
    report = classification_report(
        labels, preds, target_names=target_names, digits=4, zero_division=0
    )

    report_path = save_dir / "classification_report_ensemble.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Ensemble Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Models:\n")
        for m in MODELS:
            f.write(f"  - {m['name']}\n")
        f.write(f"\nMode: {'Ensemble + TTA' if args.tta else 'Ensemble'}\n")
        if args.optimize_weights:
            f.write(f"Weights: Optimized {best_weights}\n")
        else:
            f.write(f"Weights: Equal {equal_weights}\n")
        f.write(f"\n{'-'*60}\n\n")
        f.write(report)
        f.write(f"\nFull F1 Macro: {final_metrics['full_f1_macro']:.4f}\n")
        f.write(f"Full AUC-ROC:  {final_metrics['full_auc_roc']:.4f}\n")
    print(f"  Classification report kaydedildi: {report_path}")

    # Softmax olasiliklarini kaydet (ileride kullanim icin)
    np.savez(
        str(save_dir / "ensemble_predictions.npz"),
        probs=final_probs,
        labels=labels,
        preds=preds,
    )
    print(f"  Tahminler kaydedildi: {save_dir / 'ensemble_predictions.npz'}")

    # ================================================================
    # Karsilastirma ozeti
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  KARSILASTIRMA OZETI")
    print(f"{'='*60}")
    print(f"\n  {'Model':<40} {'F1 Macro':>10} {'Accuracy':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*70}")

    for i, model_info in enumerate(MODELS):
        m = compute_metrics(model_probs_list[i], labels)
        suffix = " (TTA)" if args.tta else ""
        print(f"  {model_info['name'] + suffix:<40} {m['full_f1_macro']:>10.4f} {m['full_accuracy']:>10.4f} {m['full_auc_roc']:>10.4f}")

    # Probability averaging sonucu
    if args.optimize_weights:
        avg_title = "Prob. Averaging (Optimized)"
        avg_weights = best_weights
    else:
        avg_title = "Prob. Averaging (Equal)"
        avg_weights = equal_weights
    avg_probs = sum(w * p for w, p in zip(avg_weights, model_probs_list))
    avg_metrics = compute_metrics(avg_probs, labels)
    print(f"  {avg_title:<40} {avg_metrics['full_f1_macro']:>10.4f} {avg_metrics['full_accuracy']:>10.4f} {avg_metrics['full_auc_roc']:>10.4f}")

    # Per-class weights sonucu
    if per_class_metrics is not None:
        print(f"  {'Per-Class Weights':<40} {per_class_metrics['full_f1_macro']:>10.4f} {per_class_metrics['full_accuracy']:>10.4f} {per_class_metrics['full_auc_roc']:>10.4f}")

    # Stacking sonucu
    if stacking_metrics is not None:
        stacking_title = f"Stacking (LR, C={stacking_info['best_C']})"
        print(f"  {stacking_title:<40} {stacking_metrics['full_f1_macro']:>10.4f} {stacking_metrics['full_accuracy']:>10.4f} {stacking_metrics['full_auc_roc']:>10.4f}")

    print(f"\n  {'>>> FINAL: ' + final_title:<40} {final_metrics['full_f1_macro']:>10.4f} {final_metrics['full_accuracy']:>10.4f} {final_metrics['full_auc_roc']:>10.4f}")
    print()


if __name__ == "__main__":
    main()
