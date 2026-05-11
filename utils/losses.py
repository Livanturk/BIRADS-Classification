"""
Multi-Head Loss Function
=========================
Hiyerarşik çoklu-kafa sınıflandırıcı için bileşik kayıp fonksiyonu.

Kayıp Bileşenleri:
    L_total = w1 * L_binary + w2 * L_subgroup + w3 * L_full

    L_binary:   CrossEntropy(binary_logits, binary_labels)
                Benign vs Malign ayrımı için.

    L_subgroup: CrossEntropy(benign_sub_logits, benign_labels) +
                CrossEntropy(malign_sub_logits, malign_labels)
                Alt grup ayrımları için. Sadece ilgili örneklere uygulanır.

    L_full:     CrossEntropy(full_logits, full_labels)
                4-sınıf direkt tahmin için.

Class Weights:
    Sınıf dengesizliğini telafi etmek için ağırlıklar kullanılır.
    Az temsil edilen sınıflar daha yüksek ağırlık alır.

Loss Türleri:
    - "ce": Standard CrossEntropyLoss (varsayılan)
    - "focal": Focal Loss — zor örneklere daha fazla odaklanır.
      Kolay örneklerin (yüksek olasılıklı doğru tahminler) kaybı
      (1-p)^gamma ile azaltılır. gamma=2.0 önerilir.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification_heads import HierarchicalClassifier
from utils.logit_adjustment import train_log_prior_4class


class OrdinalLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) tabanlı Ordinal Regression Kaybı.

    BI-RADS sınıfları arasındaki doğal sıralamayı (1 < 2 < 4 < 5) yakalamak için
    K-1 bağımsız binary sınıflandırıcı kullanır.

    Her sınıflandırıcı P(rank >= k) kümülatif olasılığını öğrenir:
        - label=0 (BIRADS 1): P(≥1)=0, P(≥2)=0, P(≥3)=0
        - label=1 (BIRADS 2): P(≥1)=1, P(≥2)=0, P(≥3)=0
        - label=2 (BIRADS 4): P(≥1)=1, P(≥2)=1, P(≥3)=0
        - label=3 (BIRADS 5): P(≥1)=1, P(≥2)=1, P(≥3)=1

    Standard CE'den farkı: Komşu sınıf hataları uzak sınıf hatalarından
    daha az cezalandırılır — bu BI-RADS klinik yapısına uygundur.

    Args:
        num_classes: Toplam sınıf sayısı (K). Default: 4
        weight: (K,) sınıf ağırlıkları. None ise eşit ağırlık.
    """

    def __init__(
        self,
        num_classes: int = 4,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ranks = num_classes - 1  # K-1 binary threshold
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, K-1) ordinal logit değerleri.
            targets: (B,) sınıf indeksleri (0 to K-1).

        Returns:
            Skaler ordinal kayıp.
        """
        B = logits.size(0)

        # Kümülatif binary hedefler: I(label >= k) her k=1,...,K-1 için
        rank_targets = torch.zeros(B, self.num_ranks, device=logits.device)
        for k in range(1, self.num_ranks + 1):
            rank_targets[:, k - 1] = (targets >= k).float()

        if self.weight is not None:
            # Her örnek için sınıf ağırlığı uygula
            sample_weights = self.weight[targets]   # (B,)
            loss = F.binary_cross_entropy_with_logits(
                logits, rank_targets, reduction="none"
            ).mean(dim=-1)                          # (B,)
            loss = (loss * sample_weights).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(logits, rank_targets)

        return loss

    @staticmethod
    def decode(logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal logitlerden sınıf tahminleri üret.

        Tahmin = kaç rank eşiğinin sigmoid > 0.5 olduğu sayısı.

        Args:
            logits: (B, K-1) ordinal logitler.

        Returns:
            (B,) sınıf indeksleri (0 to K-1).
        """
        return (torch.sigmoid(logits) > 0.5).sum(dim=-1).long()

    @staticmethod
    def to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Ordinal logitlerden sınıf olasılıkları hesapla (metrik ve confidence için).

        P(class=0) = 1 - P(rank>=1)
        P(class=k) = P(rank>=k) - P(rank>=k+1)  k=1,...,K-2
        P(class=K-1) = P(rank>=K-1)

        Args:
            logits: (B, K-1) ordinal logitler.

        Returns:
            (B, K) normalize sınıf olasılıkları.
        """
        B = logits.size(0)
        probs_cumul = torch.sigmoid(logits)         # (B, K-1)

        ones  = torch.ones(B, 1, device=logits.device)
        zeros = torch.zeros(B, 1, device=logits.device)
        probs_ext = torch.cat([ones, probs_cumul, zeros], dim=-1)  # (B, K+1)

        class_probs = (probs_ext[:, :-1] - probs_ext[:, 1:]).clamp(min=0.0)  # (B, K)
        class_probs = class_probs / (class_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return class_probs


class AsymmetryContrastiveLoss(nn.Module):
    """
    Bilateral Asimetri Tutarlılık Kaybı.

    Sol-sağ meme fark vektörü (F_diff = F_left - F_right) üzerinde
    sınıfa bağımlı kısıtlar uygular:

        Benign: Asimetri KÜÇÜK olmalı (simetrik memeler normal)
            L_benign = mean( (||F_diff|| / margin)^2 )

        Malign: Asimetri BÜYÜK olmalı (kitleli taraf öne çıkmalı)
            L_malign = mean( ReLU(1 - ||F_diff|| / margin)^2 )

    Args:
        margin: Hedef F_diff norm büyüklüğü (malign için referans). Default: 1.0
        benign_weight: Benign kaybı ağırlığı.
        malign_weight: Malign kaybı ağırlığı.
    """

    def __init__(
        self,
        margin: float = 1.0,
        benign_weight: float = 1.0,
        malign_weight: float = 1.0,
    ):
        super().__init__()
        self.margin = margin
        self.benign_weight = benign_weight
        self.malign_weight = malign_weight

    def forward(
        self, f_diff: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_diff: (B, dim) sol-sağ fark vektörü (F_left - F_right).
            labels: (B,) 4-sınıf etiketleri (0=BIRADS1, 1=BIRADS2, 2=BIRADS4, 3=BIRADS5).

        Returns:
            Skaler asimetri kaybı.
        """
        binary = (labels >= 2).float()              # 0=benign, 1=malign
        diff_norm = torch.norm(f_diff, p=2, dim=-1) # (B,)

        loss = torch.tensor(0.0, device=f_diff.device)
        n_terms = 0

        benign_mask = binary < 0.5
        if benign_mask.any():
            # Benign: küçük asimetriyi ödüllendir → büyük diff'i cezalandır
            benign_loss = (diff_norm[benign_mask] / self.margin).pow(2).mean()
            loss = loss + self.benign_weight * benign_loss
            n_terms += 1

        malign_mask = binary > 0.5
        if malign_mask.any():
            # Malign: büyük asimetriyi ödüllendir → küçük diff'i cezalandır
            malign_loss = F.relu(1.0 - diff_norm[malign_mask] / self.margin).pow(2).mean()
            loss = loss + self.malign_weight * malign_loss
            n_terms += 1

        return loss / max(n_terms, 1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss with hard-negative mining + memory bank
    (Direction #3; Lesson #61 follow-up).

    Mechanism (Khosla 2020 + MoCo-v2 + Lesson #60 hard-negative variant):
        For each anchor i with class y_i:
          P(i) = bank entries with same class y_i  (positives)
          N(i) = top-k closest bank entries with class != y_i  (hard negatives)
          L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p/τ) / Z_i )
          Z_i = sum over P(i) ∪ N(i) of exp(z_i·z_a/τ)

    Why a memory bank: C6's effective batch is 64 (B=2 × grad_accum=32) but
    each forward step only sees B=2. Within-batch SupCon is degenerate at
    B=2 (≤1 same-class pair). The bank stores recent embeddings (detached,
    so no gradient flows through past steps) and provides ~256 patients of
    positives + negatives for each anchor.

    Why hard-negative mining (top-k): Phase 0c proved BR1↔BR2 (and BR4↔BR5)
    geometric overlap is concentrated on a small number of boundary patients
    per anchor. Including ALL negatives in the denominator flattens the
    gradient over easy patients (BR4/BR5 vs BR1) where the model is already
    correct. Top-k focuses the gradient on the BR2 patients closest to each
    BR1 anchor (the ones we need to push away).

    Why low weight (β ≈ 0.03–0.05): Lesson #61 established sd > 0.025 across
    n=4 C6 students as a "wrong intervention class" fingerprint. SupCon's
    embedding-space gradient could destabilize ill-conditioned seeds (e.g.,
    seed=555). Start with β=0.03 to keep SupCon below the CE/sub gradients
    in magnitude; sweep upward only if BR1 lift is positive but small.
    """

    def __init__(
        self,
        feat_dim: int = 512,
        temperature: float = 0.1,
        top_k_neg: int = 20,
        queue_size: int = 256,
        warmup_min: int = 64,
    ):
        super().__init__()
        self.tau = float(temperature)
        self.top_k_neg = int(top_k_neg)
        self.queue_size = int(queue_size)
        self.feat_dim = int(feat_dim)
        self.warmup_min = int(warmup_min)
        # Bank stored normalized; refreshed by enqueue every forward.
        self.register_buffer("queue_feat", torch.zeros(self.queue_size, self.feat_dim))
        self.register_buffer("queue_label", -torch.ones(self.queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, feat: torch.Tensor, labels: torch.Tensor) -> None:
        feat = feat.detach()
        # Cast to bank dtype to avoid fp16/fp32 mismatch under autocast.
        feat = feat.to(self.queue_feat.dtype)
        labels = labels.to(self.queue_label.dtype)
        B = feat.shape[0]
        ptr = int(self.queue_ptr.item())
        if ptr + B <= self.queue_size:
            self.queue_feat[ptr:ptr + B] = feat
            self.queue_label[ptr:ptr + B] = labels
        else:
            n1 = self.queue_size - ptr
            self.queue_feat[ptr:] = feat[:n1]
            self.queue_label[ptr:] = labels[:n1]
            self.queue_feat[:B - n1] = feat[n1:]
            self.queue_label[:B - n1] = labels[n1:]
        self.queue_ptr[0] = (ptr + B) % self.queue_size

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(feat, dim=-1)
        valid_mask = self.queue_label >= 0
        n_valid = int(valid_mask.sum().item())
        if n_valid < self.warmup_min:
            self._enqueue(feat, labels)
            return feat.new_zeros((), requires_grad=True)

        bank = self.queue_feat[valid_mask].to(feat.dtype)            # (n_valid, D)
        bank = F.normalize(bank, dim=-1)
        bank_lbl = self.queue_label[valid_mask]                       # (n_valid,)
        sim = (feat @ bank.T) / self.tau                              # (B, n_valid)

        loss = feat.new_zeros(())
        n_anchors = 0
        for i in range(feat.shape[0]):
            y = int(labels[i].item())
            pos_mask = bank_lbl == y
            neg_mask = bank_lbl != y
            n_pos = int(pos_mask.sum().item())
            n_neg = int(neg_mask.sum().item())
            if n_pos == 0 or n_neg < self.top_k_neg:
                continue

            pos_sim = sim[i][pos_mask]
            neg_sim = sim[i][neg_mask]
            k = min(self.top_k_neg, n_neg)
            top_neg = torch.topk(neg_sim, k=k).values

            denom_terms = torch.cat([pos_sim, top_neg])               # (n_pos + k,)
            log_denom = torch.logsumexp(denom_terms, dim=0)
            # SupCon: average over positives. -1/|P| * sum_p (z_p - log_denom).
            loss = loss + (log_denom * n_pos - pos_sim.sum()) / float(n_pos)
            n_anchors += 1

        self._enqueue(feat, labels)
        if n_anchors == 0:
            return feat.new_zeros((), requires_grad=True)
        return loss / float(n_anchors)


class FocalLoss(nn.Module):
    """
    Focal Loss: Zor örneklere odaklanan kayıp fonksiyonu.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        weight: Sınıf ağırlıkları (alpha). None ise eşit ağırlık.
        gamma: Focusing parametresi. gamma=0 → standard CE.
               gamma arttıkça kolay örneklerin etkisi azalır.
        label_smoothing: Etiket yumuşatma faktörü.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) ham logit değerleri.
            targets: (B,) sınıf indeksleri.

        Returns:
            Skaler focal loss değeri.
        """
        num_classes = logits.size(-1)

        # Label smoothing uygula
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Log-softmax ve softmax hesapla
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - probs) ** self.gamma

        # Class weight uygula
        if self.weight is not None:
            class_weight = self.weight.unsqueeze(0)  # (1, C)
            focal_weight = focal_weight * class_weight

        # Focal loss = -alpha * (1-p)^gamma * log(p) * smooth_target
        loss = -focal_weight * log_probs * smooth_targets
        loss = loss.sum(dim=-1).mean()

        return loss


class MultiHeadLoss(nn.Module):
    """
    Hiyerarşik çoklu-kafa kayıp fonksiyonu.

    Args:
        loss_weights: Her head'in toplam kayba katkı ağırlığı.
            {"binary_head": 0.3, "subgroup_head": 0.3, "full_head": 0.4}
        class_weights_4: 4-sınıf için sınıf ağırlıkları (tensor).
        class_weights_binary: Binary için sınıf ağırlıkları (tensor).
        class_weights_benign_sub: Benign subgroup (BIRADS 1 vs 2) ağırlıkları.
        class_weights_malign_sub: Malign subgroup (BIRADS 4 vs 5) ağırlıkları.
        use_binary: Binary head kaybını hesapla.
        use_subgroup: Subgroup head kaybını hesapla.
        label_smoothing: Etiket yumuşatma (overfitting'i azaltır).
    """

    def __init__(
        self,
        loss_weights: dict,
        class_weights_4: Optional[torch.Tensor] = None,
        class_weights_binary: Optional[torch.Tensor] = None,
        class_weights_benign_sub: Optional[torch.Tensor] = None,
        class_weights_malign_sub: Optional[torch.Tensor] = None,
        use_binary: bool = True,
        use_subgroup: bool = True,
        label_smoothing: float = 0.05,
        loss_type: str = "ce",
        focal_gamma: float = 2.0,
        use_ordinal: bool = False,
        asymmetry_loss_weight: float = 0.0,
        asymmetry_margin: float = 1.0,
        asymmetry_benign_weight: float = 1.0,
        asymmetry_malign_weight: float = 1.0,
        logit_adjustment: Optional[dict] = None,
        distill: Optional[dict] = None,
        supcon: Optional[dict] = None,
        per_region_malign: Optional[dict] = None,
    ):
        super().__init__()

        self.w_binary = loss_weights.get("binary_head", 0.3)
        self.w_subgroup = loss_weights.get("subgroup_head", 0.3)
        self.w_full = loss_weights.get("full_head", 0.4)
        self.use_binary = use_binary
        self.use_subgroup = use_subgroup
        self.loss_type = loss_type
        self.use_ordinal = use_ordinal
        self.w_asymmetry = asymmetry_loss_weight

        # --- Ensemble self-distillation (Direction #5; Lesson #60) ---
        # When enabled, the FULL HEAD's loss becomes:
        #   L_full = (1 - alpha) * CE(z_student, hard_label)
        #         + alpha * T^2 * KL( softmax(z_student / T) || softmax(z_teacher / T) )
        # The temperature scaling factor T^2 (Hinton 2015) keeps the relative
        # gradient magnitude comparable to CE as T varies. Soft targets enter
        # the forward via an extra `soft_targets` kwarg (raw mean teacher
        # logits, shape (B, 4)) — NOT mutated by augmentation, NOT class-weighted
        # (the teacher's relative class ranking already encodes the right prior
        # scaling for this dataset).
        d_cfg = distill or {}
        self.distill_enabled = bool(d_cfg.get("enabled", False))
        self.distill_alpha = float(d_cfg.get("alpha", 0.0))
        self.distill_T = float(d_cfg.get("temperature", 1.0))
        if self.distill_enabled:
            if not (0.0 <= self.distill_alpha <= 1.0):
                raise ValueError(f"distill.alpha must be in [0, 1], got {self.distill_alpha}")
            if self.distill_T <= 0:
                raise ValueError(f"distill.temperature must be > 0, got {self.distill_T}")
            print(
                f"[LOSS] Distillation ENABLED on full head "
                f"(alpha={self.distill_alpha}, T={self.distill_T})"
            )

        # --- Per-Region Malign Head (Track B; Lesson #62 follow-up) ---
        # Aux loss on outputs["per_region_malign_logits"] (B, 2). Trained as
        # binary CE on (BR1+BR2)→0, (BR4+BR5)→1. Gradient flows ONLY into the
        # backbone — NOT into bilateral_fusion or patient_feat. Structurally
        # outside the closed aux-loss-on-patient_feat class.
        pr_cfg = per_region_malign or {}
        self.pr_enabled = bool(pr_cfg.get("enabled", False))
        self.pr_weight = float(pr_cfg.get("weight", 0.0))
        if self.pr_enabled and self.pr_weight > 0:
            # Same binary class weights as the main binary head (sqrt-inv freq).
            self.pr_criterion = nn.CrossEntropyLoss(
                weight=class_weights_binary,
                label_smoothing=label_smoothing,
            )
            print(
                f"[LOSS] Per-Region Malign Head loss ENABLED "
                f"(weight={self.pr_weight}, target=binary BR12-vs-BR45)"
            )

        # --- SupCon with hard-negative mining (Direction #3; Lesson #61 follow-up) ---
        # Auxiliary embedding-space loss on patient_feat. Adds a discriminative-margin
        # signal that pushes BR1 anchors away from their nearest BR2 negatives in
        # the bank. Distinct from KL distillation (Lesson #61): no centroid pull,
        # no basin-selector dynamic — gradient is per-pair and embedding-direct.
        s_cfg = supcon or {}
        self.supcon_enabled = bool(s_cfg.get("enabled", False))
        self.supcon_weight = float(s_cfg.get("weight", 0.0))
        if self.supcon_enabled and self.supcon_weight > 0:
            self.supcon_criterion = SupConLoss(
                feat_dim=int(s_cfg.get("feat_dim", 512)),
                temperature=float(s_cfg.get("temperature", 0.1)),
                top_k_neg=int(s_cfg.get("top_k_neg", 20)),
                queue_size=int(s_cfg.get("queue_size", 256)),
                warmup_min=int(s_cfg.get("warmup_min", 64)),
            )
            print(
                f"[LOSS] SupCon ENABLED on patient_feat "
                f"(weight={self.supcon_weight}, tau={s_cfg.get('temperature', 0.1)}, "
                f"top_k_neg={s_cfg.get('top_k_neg', 20)}, "
                f"queue_size={s_cfg.get('queue_size', 256)})"
            )

        # --- Logit Adjustment (Menon et al. 2021, Tier-2 F2) ---
        # Regime A: train-prior only. Shift applied INSIDE the criterion call;
        # outputs["full_logits"] is NOT mutated, so inference (raw argmax) and
        # metrics paths are unchanged. By default, applies to full head only.
        la_cfg = logit_adjustment or {}
        self.la_enabled = bool(la_cfg.get("enabled", False))
        self.la_tau = float(la_cfg.get("tau", 1.0))
        la_apply = la_cfg.get("apply_to", ["full"]) or []
        self.la_apply_full = self.la_enabled and ("full" in la_apply)
        self.la_drop_class_weights = bool(la_cfg.get("drop_class_weights", True))
        if self.la_apply_full:
            # Buffer auto-moves with .to(device); broadcasts across batch.
            self.register_buffer("log_prior_full", train_log_prior_4class())
            print(
                f"[LOSS] Logit Adjustment ENABLED on full head "
                f"(tau={self.la_tau}, drop_class_weights={self.la_drop_class_weights}, "
                f"log_prior={self.log_prior_full.tolist()})"
            )

        # Ordinal loss (full head yerine kullanılır)
        if use_ordinal:
            print(f"[LOSS] Ordinal Loss (CORAL) aktif — full_head CE yerine")
            self.ordinal_criterion = OrdinalLoss(
                num_classes=4,
                weight=class_weights_4,
            )

        # Asymmetry contrastive loss
        if asymmetry_loss_weight > 0:
            print(f"[LOSS] Asymmetry Contrastive Loss aktif "
                  f"(w={asymmetry_loss_weight}, margin={asymmetry_margin}, "
                  f"benign_w={asymmetry_benign_weight}, malign_w={asymmetry_malign_weight})")
            self.asymmetry_criterion = AsymmetryContrastiveLoss(
                margin=asymmetry_margin,
                benign_weight=asymmetry_benign_weight,
                malign_weight=asymmetry_malign_weight,
            )

        # Menon recipe: when LA is on for full head, drop class re-weighting
        # for that head — LA replaces it. Other heads keep their class weights.
        full_weight_for_loss = (
            None if (self.la_apply_full and self.la_drop_class_weights)
            else class_weights_4
        )

        if loss_type == "focal":
            print(f"[LOSS] Focal Loss aktif (gamma={focal_gamma}, label_smoothing={label_smoothing})")

            self.full_criterion = FocalLoss(
                weight=full_weight_for_loss,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.binary_criterion = FocalLoss(
                weight=class_weights_binary,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.benign_sub_criterion = FocalLoss(
                weight=class_weights_benign_sub,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            self.malign_sub_criterion = FocalLoss(
                weight=class_weights_malign_sub,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
        else:
            # Standard CrossEntropy (varsayılan)
            self.full_criterion = nn.CrossEntropyLoss(
                weight=full_weight_for_loss,
                label_smoothing=label_smoothing,
            )
            self.binary_criterion = nn.CrossEntropyLoss(
                weight=class_weights_binary,
                label_smoothing=label_smoothing,
            )
            self.benign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_benign_sub,
                label_smoothing=label_smoothing,
            )
            self.malign_sub_criterion = nn.CrossEntropyLoss(
                weight=class_weights_malign_sub,
                label_smoothing=label_smoothing,
            )

    def forward(
        self, outputs: dict, labels: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        soft_target_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Toplam kaybı hesaplar.

        Args:
            outputs: Model çıkışları (binary_logits, subgroup logits, full_logits).
            labels: (B,) 4-sınıf etiketleri.
            soft_targets: (B, 4) raw teacher logits for distillation. If None
                or distill disabled, the full head uses standard CE.
            soft_target_mask: (B,) bool. True if the row's soft_targets is
                valid (patient was in the precomputed teacher set). False
                rows fall back to pure CE for that sample.

        Returns:
            dict:
                - "total_loss": Toplam ağırlıklı kayıp.
                - "binary_loss": Binary head kaybı.
                - "benign_sub_loss": Benign subgroup kaybı.
                - "malign_sub_loss": Malign subgroup kaybı.
                - "full_loss": Full head kaybı.
                - "distill_loss": (only if distill_enabled and any sample masked-in)
        """
        # Etiketleri her head için dönüştür
        label_dict = HierarchicalClassifier.convert_labels(labels)

        losses = {}
        total_loss = torch.tensor(0.0, device=labels.device)

        # --- Full Head Loss (her zaman aktif) ---
        if self.use_ordinal and "ordinal_logits" in outputs:
            # CORAL Ordinal Loss: full CE yerine
            full_loss = self.ordinal_criterion(outputs["ordinal_logits"], label_dict["full"])
        else:
            # Logit Adjustment: train-time shift z + tau*log(pi). Non-in-place
            # add → outputs["full_logits"] stays raw for metrics/inference.
            if self.la_apply_full:
                # Belt-and-braces device alignment in case the criterion was
                # not moved with .to(device). Cheap: no-op when devices match.
                log_prior = self.log_prior_full.to(outputs["full_logits"].device)
                full_logits_for_loss = outputs["full_logits"] + self.la_tau * log_prior
            else:
                full_logits_for_loss = outputs["full_logits"]
            ce_full = self.full_criterion(full_logits_for_loss, label_dict["full"])

            # Distillation: hybrid (1-alpha)*CE_hard + alpha*T^2*KL_soft on full head.
            # Distillation only kicks in if (a) enabled in config, (b) soft_targets
            # provided this batch, AND (c) at least one row has a valid teacher
            # target (soft_target_mask). Otherwise the full head falls back to
            # CE-only, preserving val-set behavior when no soft targets exist.
            if (
                self.distill_enabled
                and self.distill_alpha > 0
                and soft_targets is not None
                and soft_target_mask is not None
                and soft_target_mask.any()
            ):
                T = self.distill_T
                # Operate only on masked-in rows so the per-row loss has the right denominator.
                z_s = outputs["full_logits"][soft_target_mask] / T
                z_t = soft_targets[soft_target_mask].to(z_s.device) / T
                # KL(student || teacher): F.kl_div expects log-probs as input,
                # probs as target by default. Use batchmean reduction so the
                # scale is comparable to CE.
                log_p_s = F.log_softmax(z_s, dim=-1)
                p_t = F.softmax(z_t, dim=-1)
                kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
                losses["distill_loss"] = kl
                full_loss = (1.0 - self.distill_alpha) * ce_full + self.distill_alpha * kl
            else:
                full_loss = ce_full
        losses["full_loss"] = full_loss
        total_loss = total_loss + self.w_full * full_loss

        # --- Binary Head Loss ---
        if self.use_binary:
            binary_loss = self.binary_criterion(
                outputs["binary_logits"], label_dict["binary"]
            )
            losses["binary_loss"] = binary_loss
            total_loss = total_loss + self.w_binary * binary_loss

        # --- Subgroup Head Loss ---
        # Sadece ilgili örneklere uygulanır (benign → benign head, malign → malign head)
        if self.use_subgroup:
            benign_sub_loss = torch.tensor(0.0, device=labels.device)
            malign_sub_loss = torch.tensor(0.0, device=labels.device)

            # Benign alt grubu (BIRADS 1 ve 2 olan örnekler)
            benign_mask = label_dict["benign_mask"]
            if benign_mask.any():
                benign_logits = outputs["benign_sub_logits"][benign_mask]
                benign_labels = label_dict["benign_sub"]
                benign_sub_loss = self.benign_sub_criterion(benign_logits, benign_labels)

            # Malign alt grubu (BIRADS 4 ve 5 olan örnekler)
            malign_mask = label_dict["malign_mask"]
            if malign_mask.any():
                malign_logits = outputs["malign_sub_logits"][malign_mask]
                malign_labels = label_dict["malign_sub"]
                malign_sub_loss = self.malign_sub_criterion(malign_logits, malign_labels)

            subgroup_loss = (benign_sub_loss + malign_sub_loss) / 2.0
            losses["benign_sub_loss"] = benign_sub_loss
            losses["malign_sub_loss"] = malign_sub_loss
            total_loss = total_loss + self.w_subgroup * subgroup_loss

        # --- Asymmetry Contrastive Loss (bilateral f_diff üzerinde) ---
        if self.w_asymmetry > 0 and "f_diff" in outputs and outputs["f_diff"] is not None:
            asym_loss = self.asymmetry_criterion(outputs["f_diff"], labels)
            losses["asymmetry_loss"] = asym_loss
            total_loss = total_loss + self.w_asymmetry * asym_loss

        # --- SupCon on patient_feat (Direction #3; Lesson #61 follow-up) ---
        # Skipped under Mixup/CutMix (mixed-image labels are linear combinations,
        # SupCon's per-anchor class-membership assumption breaks). The student
        # logic in train.py is responsible for not calling this branch on mixed
        # batches — but here we also defensively check that labels are 1-D long
        # tensors (Mixup substitutes a 2-D float matrix in some implementations).
        if (
            self.supcon_enabled
            and self.supcon_weight > 0
            and "patient_features" in outputs
            and outputs["patient_features"] is not None
            and labels.dim() == 1
        ):
            sup_loss = self.supcon_criterion(outputs["patient_features"], labels)
            losses["supcon_loss"] = sup_loss
            total_loss = total_loss + self.supcon_weight * sup_loss

        # --- Per-Region Malign Head loss (Track B; Lesson #62 follow-up) ---
        # Binary CE: (BR1+BR2)→0, (BR4+BR5)→1. The label is reused from
        # label_dict["binary"] (already computed by HierarchicalClassifier.convert_labels).
        if (
            self.pr_enabled
            and self.pr_weight > 0
            and "per_region_malign_logits" in outputs
            and outputs["per_region_malign_logits"] is not None
        ):
            pr_loss = self.pr_criterion(
                outputs["per_region_malign_logits"], label_dict["binary"]
            )
            losses["per_region_malign_loss"] = pr_loss
            total_loss = total_loss + self.pr_weight * pr_loss

        losses["total_loss"] = total_loss
        return losses


def build_loss_function(config: dict, device: torch.device) -> MultiHeadLoss:
    """
    Config'den loss fonksiyonu oluşturur.

    Args:
        config: YAML konfigürasyonu.
        device: CUDA/CPU cihazı.

    Returns:
        MultiHeadLoss instance.
    """
    train_cfg = config["training"]
    ablation_cfg = config.get("ablation", {})

    # 4-sınıf ağırlıkları
    cw = train_cfg.get("class_weights", [1.0, 1.0, 1.0, 1.0])
    class_weights_4 = torch.tensor(cw, dtype=torch.float32).to(device)

    # Binary ağırlıklar (sqrt-inverse frequency)
    # BIRADS-Full-Train-8Bit-Processed: Benign(BR1+BR2)=4432, Malign(BR4+BR5)=4125
    # sqrt(4432/4125)=1.037 → Benign:1.00, Malign:1.04
    class_weights_binary = torch.tensor([1.00, 1.04], dtype=torch.float32).to(device)

    # Subgroup ağırlıkları (sqrt-inverse frequency)
    # Benign: BIRADS 1 (1678) vs BIRADS 2 (2754) → sqrt(2754/1678)=1.281 → [1.28, 1.00]
    class_weights_benign_sub = torch.tensor([1.28, 1.00], dtype=torch.float32).to(device)
    # Malign: BIRADS 4 (1898) vs BIRADS 5 (2227) → sqrt(2227/1898)=1.084 → [1.08, 1.00]
    class_weights_malign_sub = torch.tensor([1.08, 1.00], dtype=torch.float32).to(device)

    loss_fn = MultiHeadLoss(
        loss_weights=train_cfg["loss_weights"],
        class_weights_4=class_weights_4,
        class_weights_binary=class_weights_binary,
        class_weights_benign_sub=class_weights_benign_sub,
        class_weights_malign_sub=class_weights_malign_sub,
        use_binary=ablation_cfg.get("use_binary_head", True),
        use_subgroup=ablation_cfg.get("use_subgroup_head", True),
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
        loss_type=train_cfg.get("loss_type", "ce"),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        use_ordinal=ablation_cfg.get("use_ordinal_head", False),
        asymmetry_loss_weight=train_cfg.get("asymmetry_loss_weight", 0.0),
        asymmetry_margin=train_cfg.get("asymmetry_margin", 1.0),
        asymmetry_benign_weight=train_cfg.get("asymmetry_benign_weight", 1.0),
        asymmetry_malign_weight=train_cfg.get("asymmetry_malign_weight", 1.0),
        logit_adjustment=train_cfg.get("logit_adjustment", None),
        distill=train_cfg.get("distill", None),
        supcon=train_cfg.get("supcon", None),
        per_region_malign=train_cfg.get("per_region_malign", None),
    )
    # Move registered buffers (e.g. log_prior_full) onto the same device as
    # the model. nn.CrossEntropyLoss class weights are already on `device`
    # (passed in as device tensors); buffers registered inside the module
    # need an explicit .to() call.
    return loss_fn.to(device)
