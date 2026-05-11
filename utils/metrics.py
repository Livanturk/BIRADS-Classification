"""
Metrik Hesaplama Modülü
========================
Eğitim ve değerlendirme sırasında hesaplanan tüm metrikler.

Her head için ayrı ayrı metrikler üretilir:
    - Accuracy (doğruluk)
    - Precision, Recall, F1-Score (sınıf bazlı ve makro ortalama)
    - AUC-ROC (çoklu sınıf)
    - Confusion Matrix
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


class MetricTracker:
    """
    Epoch boyunca tahminleri biriktirir ve epoch sonunda metrikleri hesaplar.

    Kullanım:
        tracker = MetricTracker()
        for batch in dataloader:
            tracker.update(predictions, labels)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Yeni epoch için sıfırla."""
        self.full_preds = []
        self.full_labels = []
        self.full_probs = []
        self.binary_preds = []
        self.binary_labels = []
        self.confidences = []
        self.losses = []

    def update(
        self,
        outputs: dict,
        labels: torch.Tensor,
        loss_dict: Optional[dict] = None,
    ):
        """
        Bir batch'in sonuçlarını biriktirir.

        Args:
            outputs: Model çıkışları (logitler).
            labels: (B,) gerçek etiketler.
            loss_dict: Kayıp değerleri.
        """
        with torch.no_grad():
            # Full head tahminleri
            full_probs = torch.softmax(outputs["full_logits"], dim=-1)
            full_preds = full_probs.argmax(dim=-1)

            self.full_preds.extend(full_preds.cpu().numpy())
            self.full_labels.extend(labels.cpu().numpy())
            self.full_probs.extend(full_probs.cpu().numpy())

            # Binary head tahminleri
            binary_preds = outputs["binary_logits"].argmax(dim=-1)
            binary_labels = (labels >= 2).long()
            self.binary_preds.extend(binary_preds.cpu().numpy())
            self.binary_labels.extend(binary_labels.cpu().numpy())

            # Confidence
            if "confidence" in outputs:
                self.confidences.extend(outputs["confidence"].cpu().numpy())

            # Loss
            if loss_dict:
                self.losses.append(
                    {k: v.item() for k, v in loss_dict.items()}
                )

    def compute(self) -> dict:
        """
        Birikmiş tahminlerden metrikleri hesaplar.

        Returns:
            dict: Tüm metrikler.
        """
        metrics = {}

        full_preds = np.array(self.full_preds)
        full_labels = np.array(self.full_labels)
        full_probs = np.array(self.full_probs)

        # --- Full Head Metrikleri ---
        metrics["full_accuracy"] = accuracy_score(full_labels, full_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            full_labels, full_preds, average="macro", zero_division=0
        )
        metrics["full_precision_macro"] = precision
        metrics["full_recall_macro"] = recall
        metrics["full_f1_macro"] = f1

        # Sınıf bazlı F1 skorları
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            full_labels, full_preds, average=None, zero_division=0
        )
        birads_names = ["BIRADS_1", "BIRADS_2", "BIRADS_4", "BIRADS_5"]
        for i, name in enumerate(birads_names):
            if i < len(f1_per_class):
                metrics[f"full_f1_{name}"] = f1_per_class[i]

        # AUC-ROC (One-vs-Rest)
        try:
            metrics["full_auc_roc"] = roc_auc_score(
                full_labels, full_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["full_auc_roc"] = 0.0

        # Cohen's Kappa — şans-düzeltilmiş uyum metriği
        metrics["full_cohens_kappa"] = cohen_kappa_score(full_labels, full_preds)

        # --- Binary Head Metrikleri ---
        binary_preds = np.array(self.binary_preds)
        binary_labels = np.array(self.binary_labels)

        metrics["binary_accuracy"] = accuracy_score(binary_labels, binary_preds)
        bp, br, bf1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average="binary", zero_division=0
        )
        metrics["binary_precision"] = bp
        metrics["binary_recall"] = br
        metrics["binary_f1"] = bf1

        # --- Ortalama Confidence ---
        if self.confidences:
            metrics["mean_confidence"] = float(np.mean(self.confidences))

        # --- Ortalama Loss ---
        if self.losses:
            avg_losses = {}
            for key in self.losses[0].keys():
                avg_losses[key] = np.mean([l[key] for l in self.losses])
            metrics.update(avg_losses)

        return metrics

    def get_classification_report(self) -> str:
        """Detaylı sınıflandırma raporu (metin formatında)."""
        target_names = ["BIRADS-1", "BIRADS-2", "BIRADS-4", "BIRADS-5"]
        return classification_report(
            np.array(self.full_labels),
            np.array(self.full_preds),
            target_names=target_names,
            digits=4,
            zero_division=0,
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """Confusion matrix (4x4)."""
        return confusion_matrix(
            np.array(self.full_labels),
            np.array(self.full_preds),
        )

    def get_predictions(self) -> Dict[str, np.ndarray]:
        """
        Tahminleri ve etiketleri döndürür (benchmark karşılaştırması için).

        Returns:
            dict: {"labels": np.ndarray, "preds": np.ndarray, "probs": np.ndarray}
        """
        return {
            "labels": np.array(self.full_labels),
            "preds": np.array(self.full_preds),
            "probs": np.array(self.full_probs),
        }


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    McNemar testi: İki modelin tahminlerinin istatistiksel olarak farklı olup olmadığını test eder.

    Aynı test setinde iki modelin farklı hata yaptığı örnekleri karşılaştırır:
    - b: A doğru, B yanlış (sadece A'nın doğru bildiği)
    - c: A yanlış, B doğru (sadece B'nin doğru bildiği)

    H0: İki model aynı performansta (b == c)
    p < 0.05 ise modeller istatistiksel olarak farklı performans gösteriyor.

    Args:
        preds_a: Model A'nın tahminleri (np.ndarray).
        preds_b: Model B'nin tahminleri (np.ndarray).
        labels: Gerçek etiketler (np.ndarray).

    Returns:
        dict: {"b": int, "c": int, "chi2": float, "p_value": float}
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b: A doğru, B yanlış
    b = np.sum(correct_a & ~correct_b)
    # c: A yanlış, B doğru
    c = np.sum(~correct_a & correct_b)

    # McNemar test istatistiği (süreklilik düzeltmeli)
    if b + c == 0:
        return {"b": int(b), "c": int(c), "chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-değeri (chi-squared dağılımı, df=1)
    from scipy import stats
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)

    return {
        "b": int(b),
        "c": int(c),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }
