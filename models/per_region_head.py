"""
Per-Region Malign Head (Track B for BR4; Lesson #62 follow-up)
=================================================================

Auxiliary head that operates on the BACKBONE's per-view spatial tokens
(BEFORE lateral + bilateral fusion). The gradient from this head's loss
flows ONLY into the backbone — NOT into the lateral_fusion or
bilateral_fusion modules, and NOT into patient_feat.

Why this is structurally different from the closed aux-loss-on-patient_feat
class (Lesson #62):

    Closed class (asymmetry, distill, SupCon): aux loss on patient_feat
      or its derivatives (full_logits, f_diff). Gradient enters the
      bilateral-fusion-to-classifier stack, fights the CE gradient,
      destabilizes seed=555, regresses macro F1 by ~2pp.

    Per-region head (this module): aux loss on per-view spatial tokens.
      Gradient enters ONLY the backbone. The bilateral-fusion-to-
      classifier stack is unaffected — its CE gradient flow is the same
      as in vanilla C6. The backbone gets one additional learning signal
      (predict patient-level binary label from local patches), which is
      a representation-level regularizer rather than a representation-
      level CONSTRAINT.

Why this targets BR4↔BR5 specifically:

    Phase 0c (Lesson #60) measured cluster_silhouette = 0.34 on the
    BR4→BR5 unanimous-wrong cell with k=4 sub-clusters. This means BR4
    and BR5 errors fall into a small number of consistent visual
    sub-patterns. Lesson #48 showed BR4 has a weak malign-sub margin
    (0.26 vs BR5's 1.76) — BR4 sits ON the malign decision surface.

    BR5 cases typically have multiple/strong locally-malignant features;
    BR4 cases have fewer/weaker. A per-region head trained to detect
    "malignant tissue patches" surfaces this density-of-malign-regions
    signal at the backbone feature level, where it can inform the
    fusion stack.

Architecture:

    view_features: dict {RCC, LCC, RMLO, LMLO} → each (B, S, D)
        D = projection_dim (512), S = num_spatial_tokens (1024).
                       ↓ stack
                   (B, V*S, D) = (B, 4096, 512)
                       ↓ per-token MLP
                   (B, V*S, 2) — per-patch (benign/malign) logits
                       ↓ softmax row-wise → take p(malign)
                   (B, V*S) — per-patch malign score
                       ↓ top-k by score (descending)
                   (B, k, 2) — gather original logits at top-k indices
                       ↓ mean over k
                   (B, 2) — patient-level binary logit from spatial
                            top-k aggregation

Training:

    per_region_loss = CrossEntropyLoss(per_region_logits, binary_label)
    L_total += per_region_weight * per_region_loss

    The binary_label is (BR1+BR2)→0, (BR4+BR5)→1. Every patient is in
    one of these two classes; no masking required.

Inference:

    The head's output is exposed in `outputs["per_region_malign_logits"]`
    but does NOT mix into the argmax decision. The full_head's argmax
    remains the prediction. The per-region head's role is purely a
    backbone regularizer at training time.

    (Future work: a learned mixture of full_head and per_region_head
    could produce an additional logit channel for inference, but that
    re-introduces the head-fighting risk Lesson #62 documented.)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerRegionMalignHead(nn.Module):
    """
    Top-k pooled per-region malignancy head.

    Args:
        projection_dim: Backbone projection dim (D). Must equal
            `model.projection_dim` in the C6 config (default 512).
        hidden_dim: Per-token MLP hidden width.
        top_k: Number of highest-malign-score tokens to aggregate per patient.
            Phase 0c silhouette analysis suggested 2–4 sub-patterns per cell;
            with V*S = 4096 tokens, k=30 covers ~0.7% of tokens — focused
            without being overly sparse.
        token_dropout: Stochastic dropout on per-token features before
            scoring. Prevents the head from depending on a single hot
            token; matches Lesson #62's "head dropout=0.5 caps stability"
            observation, but at the per-token level (lower).
    """

    def __init__(
        self,
        projection_dim: int = 512,
        hidden_dim: int = 64,
        top_k: int = 30,
        token_dropout: float = 0.1,
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.scorer = nn.Sequential(
            nn.Dropout(token_dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Bias the head toward "benign" at init so the early-epoch top-k
        # selection is uniformly random rather than dominated by spurious
        # large-bias-init tokens.
        with torch.no_grad():
            for m in self.scorer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)
            # Last layer: small negative malign bias.
            self.scorer[-1].bias.data[1] -= 0.5

    def forward(self, view_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            view_features: {RCC, LCC, RMLO, LMLO} each (B, S, D).
                Order is fixed by the backbone's view_names.

        Returns:
            per_region_logits: (B, 2) — top-k aggregated malignancy logits.
        """
        # Stack views — order matters only for reproducibility; the
        # aggregation is invariant to ordering since we top-k across all.
        feats = torch.stack(
            [view_features[v] for v in ("RCC", "LCC", "RMLO", "LMLO")],
            dim=1,
        )  # (B, V=4, S, D)
        B, V, S, D = feats.shape
        flat = feats.reshape(B, V * S, D)                   # (B, V*S, D)

        per_token_logits = self.scorer(flat)                # (B, V*S, 2)
        per_token_score = F.softmax(per_token_logits, dim=-1)[..., 1]  # (B, V*S) p(malign|patch)

        k = min(self.top_k, V * S)
        topk_score, topk_idx = torch.topk(per_token_score, k=k, dim=1)  # (B, k)
        # Gather the per-token logits at the top-k indices
        topk_logits = torch.gather(
            per_token_logits, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, 2),
        )  # (B, k, 2)
        # Mean over top-k → patient-level (B, 2) logit
        return topk_logits.mean(dim=1)
