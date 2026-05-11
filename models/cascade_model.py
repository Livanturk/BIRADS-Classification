"""
Cascade-stage model wrapper for the G-series soft cascade experiment.

Three specialists, each a single 2-class head:

    Stage-1 (G1)   binary       benign vs malign  — backbone -> global_feat -> 2-class
                                                    (claude.md doctrine: bypass fusion)
    Stage-2a (G2a) benign sub   BR1   vs BR2     — backbone + lateral + bilateral
                                                    -> patient_feat -> 2-class
    Stage-2b (G2b) malign sub   BR4   vs BR5     — same pipeline as G2a

Reuses C6's MultiViewBackbone, BilateralLateralFusion, BilateralFusion verbatim.
Replaces the multi-head HierarchicalClassifier with a single 2-class head.

Config block (additive):

    cascade:
      stage: "stage1" | "stage2a" | "stage2b"
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import MultiViewBackbone
from models.lateral_fusion import BilateralLateralFusion
from models.bilateral_fusion import BilateralFusion
from models.classification_heads import ClassificationHead


VALID_STAGES = ("stage1", "stage2a", "stage2b")


class CascadeStageModel(nn.Module):
    """
    Single-head specialist for one cascade stage.

    Stage-1 uses backbone only with a global-mean-pooled feature. This matches
    claude.md Section 4 (binary head feeds from global_feat to bypass the fusion
    chain and deliver gradients directly to the backbone).

    Stage-2a / Stage-2b reuse the full C6 fusion pipeline.

    Forward returns: {"logits": (B, 2), "patient_features": (B, dim)}.
    The "patient_features" entry is provided for parity with C6's interface
    so that gradcam.py can be reused without modification.
    """

    def __init__(self, config: dict):
        super().__init__()

        cascade_cfg = config.get("cascade", {})
        self.stage = cascade_cfg.get("stage")
        if self.stage not in VALID_STAGES:
            raise ValueError(
                f"cascade.stage must be one of {VALID_STAGES}, got {self.stage!r}"
            )

        model_cfg = config["model"]
        data_cfg = config.get("data", {})
        self.projection_dim = model_cfg["projection_dim"]
        image_size = data_cfg.get("image_size", 1024)

        # ---- Backbone (always used) ----
        bb = model_cfg["backbone"]
        self.backbone = MultiViewBackbone(
            backbone_name=bb["name"],
            pretrained=bb["pretrained"],
            projection_dim=self.projection_dim,
            freeze_layers=bb.get("freeze_layers", 0),
            projection_dropout=bb.get("projection_dropout", 0.2),
            image_size=image_size,
            drop_path_rate=bb.get("drop_path_rate", 0.0),
        )

        # ---- Stage-1: skip fusion (claude.md doctrine) ----
        # ---- Stage-2a/2b: full fusion pipeline ----
        self.use_fusion = self.stage != "stage1"

        if self.use_fusion:
            num_spatial_tokens = self.backbone.num_spatial_tokens
            lat = model_cfg["lateral_fusion"]
            self.lateral_fusion = BilateralLateralFusion(
                dim=self.projection_dim,
                num_spatial_tokens=num_spatial_tokens,
                num_heads=lat["num_heads"],
                attention_dropout=lat.get("attention_dropout", 0.15),
                ffn_dropout=lat.get("ffn_dropout", 0.2),
                projection_dropout=lat.get("projection_dropout", 0.2),
                num_layers=lat.get("num_layers", 2),
                use_deformable=lat.get("use_deformable", False),
                num_deformable_points=lat.get("num_deformable_points", 4),
            )
            bil = model_cfg["bilateral_fusion"]
            self.bilateral_fusion = BilateralFusion(
                dim=self.projection_dim,
                num_heads=bil["num_heads"],
                attention_dropout=bil.get("attention_dropout", 0.2),
                output_dropout=bil.get("output_dropout", 0.25),
                use_diff=bil.get("use_diff", True),
                use_avg=bil.get("use_avg", True),
            )

        # ---- 2-class head ----
        cls = model_cfg["classification"]
        self.head = ClassificationHead(
            input_dim=self.projection_dim,
            hidden_dim=cls["hidden_dim"],
            num_classes=2,
            dropout=cls["dropout"],
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[CASCADE] stage={self.stage} | use_fusion={self.use_fusion}")
        print(f"[CASCADE] total params: {total:,} | trainable: {trainable:,}")

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: (B, 4, 3, H, W)

        Returns:
            dict with "logits" (B, 2) and "patient_features" (B, projection_dim).
        """
        view_features = self.backbone(images)
        # view_features: {"RCC": (B, S, dim), "LCC": ..., "RMLO": ..., "LMLO": ...}

        if not self.use_fusion:
            # Stage-1: global_feat = mean over (4 views, S spatial tokens)
            stacked = torch.stack(
                [view_features[k] for k in ("RCC", "LCC", "RMLO", "LMLO")], dim=1
            )                                  # (B, 4, S, dim)
            patient_feat = stacked.mean(dim=(1, 2))  # (B, dim)
        else:
            lat = self.lateral_fusion(view_features)         # {"left", "right"}
            bil = self.bilateral_fusion(
                left_feat=lat["left"], right_feat=lat["right"]
            )
            patient_feat = bil["patient_feat"]   # (B, dim)

        logits = self.head(patient_feat)        # (B, 2)
        return {"logits": logits, "patient_features": patient_feat}

    def get_backbone_extractor(self):
        """Grad-CAM compatibility shim."""
        return self.backbone.backbone


def build_cascade_model(config: dict) -> CascadeStageModel:
    return CascadeStageModel(config)
