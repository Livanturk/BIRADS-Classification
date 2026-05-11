"""
Seviye 4: Hierarchical Multi-Head Classification
==================================================
Birden fazla sınıflandırma kafası (head) ile hiyerarşik tahmin yapar.

Hiyerarşik Yaklaşımın Mantığı:
    BI-RADS sınıfları doğal bir hiyerarşiye sahiptir:
        - Üst düzey: Benign (1, 2) vs Malign (4, 5)
        - Alt düzey: 1 vs 2 (benign alt gruplar), 4 vs 5 (malign alt gruplar)

    Bu hiyerarşiyi modele öğretmek:
    1. Binary Head: Önce benign/malign ayrımını yapar (kolay görev).
    2. Subgroup Head: Alt grup ayrımını yapar (zor görev).
    3. Full Head: 4-sınıf direkt tahmin (en zor görev).

    Multi-task learning: 3 görev birlikte eğitilir, birbirlerine yardımcı olur.

Uncertainty Estimation:
    Temperature Scaling ile confidence score üretilir.
    Yüksek temperature → düşük güven → daha dengeli olasılıklar.
    Düşük temperature → yüksek güven → keskin olasılıklar.

    Klinik uygulamada, düşük güvenli tahminler radyologa yönlendirilebilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Tek bir sınıflandırma kafası.

    Gizli katman + sınıflandırma katmanından oluşan basit bir MLP.

    Args:
        input_dim: Giriş öznitelik boyutu.
        hidden_dim: Gizli katman boyutu.
        num_classes: Çıkış sınıf sayısı.
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) öznitelik vektörü.
        Returns:
            (B, num_classes) logit değerleri (softmax öncesi ham skorlar).
        """
        return self.head(x)


class OrdinalHead(nn.Module):
    """
    CORAL Ordinal Sınıflandırma Kafası.

    K sınıf için K-1 kümülatif binary sınıflandırıcı.
    Paylaşılan ağırlık vektörü + per-rank bias (CORAL kısıtı):
    aynı özellik uzayından her rank eşiği ayrı bir bias ile öğrenilir,
    böylece P(rank>=1) >= P(rank>=2) >= ... tutarlılığı sağlanır.

    Args:
        input_dim: Giriş öznitelik boyutu.
        hidden_dim: Gizli katman boyutu.
        num_classes: Toplam sınıf sayısı (K). Default: 4
        dropout: Dropout oranı.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks = num_classes - 1  # K-1

        # Paylaşılan öznitelik çıkarıcı (tüm rank'ler için ortak)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CORAL: tek paylaşılan ağırlık vektörü (bias yok)
        self.fc = nn.Linear(hidden_dim, 1, bias=False)

        # Her rank eşiği için ayrı bias → sıralamayı encode eder
        self.rank_biases = nn.Parameter(torch.zeros(self.num_ranks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) öznitelik vektörü.
        Returns:
            (B, K-1) ordinal logit değerleri.
        """
        h = self.shared_layers(x)                               # (B, hidden_dim)
        shared_score = self.fc(h)                               # (B, 1)
        logits = shared_score + self.rank_biases.unsqueeze(0)   # (B, K-1)
        return logits

    @staticmethod
    def to_class_probs_static(logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal logitlerden sınıf olasılıkları hesapla.

        P(class=0) = 1 - P(rank>=1)
        P(class=k) = P(rank>=k) - P(rank>=k+1)   k=1,...,K-2
        P(class=K-1) = P(rank>=K-1)

        Args:
            logits: (B, K-1) ordinal logitler.
        Returns:
            (B, K) normalize sınıf olasılıkları.
        """
        B = logits.size(0)
        probs_cumul = torch.sigmoid(logits)                                 # (B, K-1)
        ones  = torch.ones(B, 1, device=logits.device)
        zeros = torch.zeros(B, 1, device=logits.device)
        probs_ext = torch.cat([ones, probs_cumul, zeros], dim=-1)           # (B, K+1)
        class_probs = (probs_ext[:, :-1] - probs_ext[:, 1:]).clamp(min=0)  # (B, K)
        class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return class_probs


class HierarchicalClassifier(nn.Module):
    """
    Hiyerarşik çoklu-kafa sınıflandırıcı.

    Head'ler:
        1. Binary Head: Benign (0) vs Malign (1) → 2 sınıf
           - BIRADS 1, 2 → Benign (0)
           - BIRADS 4, 5 → Malign (1)

        2. Benign Subgroup Head: BIRADS 1 (0) vs BIRADS 2 (1) → 2 sınıf
           - Sadece benign tahmin edilen örnekler için

        3. Malign Subgroup Head: BIRADS 4 (0) vs BIRADS 5 (1) → 2 sınıf
           - Sadece malign tahmin edilen örnekler için

        4. Full Head: 4-sınıf direkt tahmin → 4 sınıf
           - BIRADS 1, 2, 4, 5 → 0, 1, 2, 3

    Uncertainty Estimation:
        Temperature scaling ile softmax olasılıklarını kalibre eder.
        T > 1 → daha yumuşak dağılım (düşük güven)
        T < 1 → daha keskin dağılım (yüksek güven)
        T = 1 → standart softmax

    Args:
        input_dim: Patient-level öznitelik boyutu.
        hidden_dim: Head'lerdeki gizli katman boyutu.
        dropout: Dropout oranı.
        temperature: Başlangıç temperature değeri.
    """

    # Etiket dönüşüm haritaları (sınıf_indeksi → binary/subgroup etiket)
    # BIRADS indeksler: 0=BIRADS1, 1=BIRADS2, 2=BIRADS4, 3=BIRADS5
    FULL_TO_BINARY = {0: 0, 1: 0, 2: 1, 3: 1}   # Benign=0, Malign=1
    FULL_TO_BENIGN_SUB = {0: 0, 1: 1}             # BIRADS1=0, BIRADS2=1
    FULL_TO_MALIGN_SUB = {2: 0, 3: 1}             # BIRADS4=0, BIRADS5=1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        temperature: float = 1.5,
        use_ordinal: bool = False,
    ):
        super().__init__()

        self.use_ordinal = use_ordinal

        # Binary head: Benign vs Malign (2 sınıf)
        self.binary_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

        # Benign subgroup head: BI-RADS 1 vs BI-RADS 2
        self.benign_sub_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

        # Malign subgroup head: BIRADS 4 vs BIRADS 5
        self.malign_sub_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            dropout=dropout,
        )

        if use_ordinal:
            # Ordinal Head: CORAL rank-tabanlı (3 kümülatif binary sınıflandırıcı)
            self.ordinal_head = OrdinalHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=4,
                dropout=dropout,
            )
        else:
            # Standard Full 4-class head: BIRADS 1, 2, 4, 5
            self.full_head = ClassificationHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=4,
                dropout=dropout,
            )

        # Temperature: Öğrenilebilir parametre (eğitim sırasında optimum değere yakınsar)
        # log_temperature kullanılır çünkü temperature her zaman pozitif olmalıdır
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(temperature))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Mevcut temperature değeri (her zaman pozitif)."""
        return torch.exp(self.log_temperature)

    def forward(self, patient_feat: torch.Tensor) -> dict:
        """
        Tüm head'lerden tahmin üretir.

        Args:
            patient_feat: (B, input_dim) hasta düzeyinde öznitelik.

        Returns:
            dict:
                - "binary_logits": (B, 2) Benign/Malign logitleri.
                - "benign_sub_logits": (B, 2) BIRADS 1/2 logitleri.
                - "malign_sub_logits": (B, 2) BIRADS 4/5 logitleri.
                - "full_logits": (B, 4) 4-sınıf logitleri.
                - "confidence": (B,) Güven skoru [0, 1].
                - "temperature": Skaler temperature değeri.
        """
        # Her head'den logit al
        binary_logits = self.binary_head(patient_feat)         # (B, 2)
        benign_sub_logits = self.benign_sub_head(patient_feat) # (B, 2)
        malign_sub_logits = self.malign_sub_head(patient_feat) # (B, 2)

        T = self.temperature

        if self.use_ordinal:
            # --- Ordinal Head ---
            ordinal_logits = self.ordinal_head(patient_feat)   # (B, K-1=3)

            # Sınıf olasılıkları ve tahmin (metrik + confidence için)
            class_probs = OrdinalHead.to_class_probs_static(ordinal_logits)  # (B, 4)
            # full_logits olarak sınıf olasılıklarını kullan (argmax aynı davranır)
            full_logits = class_probs

            # Confidence: max olasılık (temperature scaling gerek yok, zaten prob)
            confidence = class_probs.max(dim=-1).values        # (B,)

            return {
                "binary_logits": binary_logits,
                "benign_sub_logits": benign_sub_logits,
                "malign_sub_logits": malign_sub_logits,
                "full_logits": full_logits,
                "ordinal_logits": ordinal_logits,
                "confidence": confidence,
                "temperature": T.item(),
            }
        else:
            # --- Standard Full Head ---
            full_logits = self.full_head(patient_feat)         # (B, 4)

            # --- Uncertainty Estimation ---
            scaled_probs = F.softmax(full_logits / T, dim=-1)  # (B, 4)
            confidence = scaled_probs.max(dim=-1).values        # (B,)

            return {
                "binary_logits": binary_logits,
                "benign_sub_logits": benign_sub_logits,
                "malign_sub_logits": malign_sub_logits,
                "full_logits": full_logits,
                "confidence": confidence,
                "temperature": T.item(),
            }

    @staticmethod
    def convert_labels(labels: torch.Tensor) -> dict:
        """
        4-sınıf etiketleri her head için uygun formata çevirir.

        Args:
            labels: (B,) 4-sınıf etiketleri (0=BIRADS1, 1=BIRADS2, 2=BIRADS4, 3=BIRADS5).

        Returns:
            dict:
                - "full": (B,) Orijinal 4-sınıf etiketleri.
                - "binary": (B,) Binary etiketler (0=Benign, 1=Malign).
                - "benign_sub": (B_benign,) Benign alt grup etiketleri.
                - "malign_sub": (B_malign,) Malign alt grup etiketleri.
                - "benign_mask": (B,) Benign örneklerin boolean maskesi.
                - "malign_mask": (B,) Malign örneklerin boolean maskesi.
        """
        device = labels.device

        # Binary etiketler: 0,1 → Benign(0), 2,3 → Malign(1)
        binary_labels = (labels >= 2).long()

        # Benign alt grup (sadece BIRADS 1 ve 2 örnekleri)
        benign_mask = labels < 2            # BIRADS 1 veya 2
        benign_sub_labels = labels[benign_mask]  # 0 veya 1

        # Malign alt grup (sadece BIRADS 4 ve 5 örnekleri)
        malign_mask = labels >= 2           # BIRADS 4 veya 5
        malign_sub_labels = labels[malign_mask] - 2  # 2→0, 3→1

        return {
            "full": labels,
            "binary": binary_labels,
            "benign_sub": benign_sub_labels,
            "malign_sub": malign_sub_labels,
            "benign_mask": benign_mask,
            "malign_mask": malign_mask,
        }
