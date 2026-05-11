"""
Tam Model: Multi-View Hierarchical BI-RADS Classifier
=======================================================
4 seviyeyi birleştiren ana model sınıfı.

Model Akışı:
    ┌─────────────────────────────────────────────────────────────┐
    │  Girdi: 4 Görüntü (RCC, LCC, RMLO, LMLO)                  │
    │                                                              │
    │  Seviye 1 — Backbone (Weight-Shared, Spatial)               │
    │  ├── RCC  → (B, S, dim)   S = H×W spatial token            │
    │  ├── LCC  → (B, S, dim)                                     │
    │  ├── RMLO → (B, S, dim)                                     │
    │  └── LMLO → (B, S, dim)                                     │
    │                                                              │
    │  Seviye 2 — Lateral Spatial Cross-Attention                  │
    │  ├── Right: CrossAttn(RCC↔RMLO) → pool → f_right (dim)     │
    │  └── Left:  CrossAttn(LCC↔LMLO) → pool → f_left  (dim)     │
    │                                                              │
    │  Seviye 3 — Bilateral Fusion                                 │
    │  ├── f_diff = f_left - f_right                               │
    │  ├── f_avg  = (f_left + f_right) / 2                         │
    │  └── SelfAttn([f_L, f_R, f_diff, f_avg]) → f_pat            │
    │                                                              │
    │  Seviye 4 — Multi-Head Classification                        │
    │  ├── Binary:  f_pat → Benign/Malign                          │
    │  ├── Benign:  f_pat → BIRADS 1/2                             │
    │  ├── Malign:  f_pat → BIRADS 4/5                             │
    │  └── Full:    f_pat → BIRADS 1/2/4/5                         │
    │  └── Uncertainty: Temperature-scaled confidence              │
    └─────────────────────────────────────────────────────────────┘

Ablation Desteği:
    Config'deki ablation ayarlarına göre modüller devre dışı bırakılabilir.
    Baseline: Sadece backbone + mean pooling + full_head.
"""

import torch
import torch.nn as nn

from models.backbone import MultiViewBackbone
from models.lateral_fusion import BilateralLateralFusion
from models.bilateral_fusion import BilateralFusion
from models.classification_heads import HierarchicalClassifier
from models.per_region_head import PerRegionMalignHead


class MammographyClassifier(nn.Module):
    """
    Multi-view hierarchical mammografi sınıflandırıcı.

    Config dosyasından tüm ayarları okuyarak modeli oluşturur.
    Ablation çalışmaları için modüller seçici olarak etkinleştirilebilir.

    Args:
        config: Parsed YAML konfigürasyon sözlüğü.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_cfg = config["model"]
        data_cfg = config.get("data", {})
        ablation_cfg = config.get("ablation", {})

        self.projection_dim = model_cfg["projection_dim"]
        image_size = data_cfg.get("image_size", 384)

        # Ablation bayrakları
        self.use_flat_fusion = ablation_cfg.get("use_flat_fusion", False)
        self.use_lateral = ablation_cfg.get("use_lateral_fusion", True)
        self.use_bilateral = ablation_cfg.get("use_bilateral_fusion", True)
        self.use_binary_head = ablation_cfg.get("use_binary_head", True)
        self.use_subgroup_head = ablation_cfg.get("use_subgroup_head", True)
        self.use_uncertainty = ablation_cfg.get("use_uncertainty", True)
        self.use_ordinal = ablation_cfg.get("use_ordinal_head", False)

        # Flat fusion aktifse lateral ve bilateral devre dışı
        if self.use_flat_fusion:
            self.use_lateral = False
            self.use_bilateral = False

        # ============================================================
        # Seviye 1: Backbone (her zaman aktif, spatial çıkış)
        # ============================================================
        backbone_cfg = model_cfg["backbone"]
        self.backbone = MultiViewBackbone(
            backbone_name=backbone_cfg["name"],
            pretrained=backbone_cfg["pretrained"],
            projection_dim=self.projection_dim,
            freeze_layers=backbone_cfg.get("freeze_layers", 0),
            projection_dropout=backbone_cfg.get("projection_dropout", 0.2),
            image_size=image_size,
            drop_path_rate=backbone_cfg.get("drop_path_rate", 0.0),
            # Direction #1: optional domain-pretrained encoder checkpoint
            pretrained_checkpoint=backbone_cfg.get("pretrained_checkpoint", None),
        )

        num_spatial_tokens = self.backbone.num_spatial_tokens

        # ============================================================
        # Flat Fusion: 4 view GAP → concat → MLP (en basit baseline)
        # ============================================================
        if self.use_flat_fusion:
            flat_dropout = ablation_cfg.get("flat_fusion_dropout", 0.3)
            self.flat_fusion = nn.Sequential(
                nn.Linear(self.projection_dim * 4, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
                nn.Dropout(flat_dropout),
            )

        # ============================================================
        # Seviye 2: Lateral Fusion (opsiyonel)
        # ============================================================
        elif self.use_lateral:
            lat_cfg = model_cfg["lateral_fusion"]
            self.lateral_fusion = BilateralLateralFusion(
                dim=self.projection_dim,
                num_spatial_tokens=num_spatial_tokens,
                num_heads=lat_cfg["num_heads"],
                attention_dropout=lat_cfg.get("attention_dropout", 0.15),
                ffn_dropout=lat_cfg.get("ffn_dropout", 0.2),
                projection_dropout=lat_cfg.get("projection_dropout", 0.2),
                num_layers=lat_cfg.get("num_layers", 2),
                use_deformable=lat_cfg.get("use_deformable", False),
                num_deformable_points=lat_cfg.get("num_deformable_points", 4),
            )
        else:
            # Lateral fusion yoksa: spatial token'ları pool et, CC ve MLO'yu basitçe birleştir
            self.simple_lateral_proj = nn.Sequential(
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
            )

        # ============================================================
        # Seviye 3: Bilateral Fusion (opsiyonel, flat fusion'da atlanır)
        # ============================================================
        if self.use_flat_fusion:
            pass  # Flat fusion'da bilateral yok
        elif self.use_bilateral:
            bil_cfg = model_cfg["bilateral_fusion"]
            self.bilateral_fusion = BilateralFusion(
                dim=self.projection_dim,
                num_heads=bil_cfg["num_heads"],
                attention_dropout=bil_cfg.get("attention_dropout", 0.2),
                output_dropout=bil_cfg.get("output_dropout", 0.25),
                use_diff=bil_cfg.get("use_diff", True),
                use_avg=bil_cfg.get("use_avg", True),
            )
        else:
            # Bilateral fusion yoksa: left ve right'ı basitçe topla
            self.simple_bilateral_proj = nn.Sequential(
                nn.Linear(self.projection_dim * 2, self.projection_dim),
                nn.LayerNorm(self.projection_dim),
                nn.GELU(),
            )

        # ============================================================
        # Seviye 4: Classification Heads
        # ============================================================
        cls_cfg = model_cfg["classification"]
        self.classifier = HierarchicalClassifier(
            input_dim=self.projection_dim,
            hidden_dim=cls_cfg["hidden_dim"],
            dropout=cls_cfg["dropout"],
            temperature=cls_cfg.get("temperature", 1.5),
            use_ordinal=self.use_ordinal,
        )

        # ============================================================
        # Optional: Per-Region Malign Head (Track B, Lesson #62)
        # ============================================================
        # Reads per-view spatial tokens BEFORE fusion; gradient flows
        # only into backbone (NOT into bilateral_fusion or patient_feat).
        # Structurally outside the closed aux-loss-on-patient_feat class.
        per_region_cfg = ablation_cfg.get("per_region_malign", {}) or {}
        self.use_per_region = bool(per_region_cfg.get("enabled", False))
        if self.use_per_region:
            self.per_region_head = PerRegionMalignHead(
                projection_dim=self.projection_dim,
                hidden_dim=int(per_region_cfg.get("hidden_dim", 64)),
                top_k=int(per_region_cfg.get("top_k", 30)),
                token_dropout=float(per_region_cfg.get("token_dropout", 0.1)),
            )
            print(
                f"[MODEL] Per-Region Malign Head ENABLED "
                f"(top_k={per_region_cfg.get('top_k', 30)}, "
                f"token_dropout={per_region_cfg.get('token_dropout', 0.1)})"
            )

        # Toplam parametre sayısını yazdır
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[MODEL] Toplam parametre: {total_params:,}")
        print(f"[MODEL] Eğitilebilir parametre: {trainable_params:,}")
        if self.use_flat_fusion:
            print(f"[MODEL] Fusion modu: FLAT (4-view GAP → concat → MLP)")
        else:
            print(f"[MODEL] Lateral fusion: {'AÇIK' if self.use_lateral else 'KAPALI'}")
            print(f"[MODEL] Bilateral fusion: {'AÇIK' if self.use_bilateral else 'KAPALI'}")
        print(f"[MODEL] Spatial token sayısı: {num_spatial_tokens}")

    def forward(self, images: torch.Tensor) -> dict:
        """
        Tam ileri geçiş (forward pass).

        Args:
            images: (B, 4, 3, H, W) — 4 mammografi görüntüsü.

        Returns:
            dict: Sınıflandırma sonuçları.
                - "binary_logits": (B, 2)
                - "benign_sub_logits": (B, 2)
                - "malign_sub_logits": (B, 2)
                - "full_logits": (B, 4)
                - "confidence": (B,)
                - "patient_features": (B, projection_dim) — Grad-CAM için
        """
        # --- Seviye 1: Her görüntüden spatial öznitelik çıkar ---
        view_features = self.backbone(images)
        # view_features = {"RCC": (B, S, dim), "LCC": ..., "RMLO": ..., "LMLO": ...}

        # --- Flat Fusion: Tüm seviyeler tek adımda ---
        if self.use_flat_fusion:
            # GAP ile spatial token'ları tek vektöre indirge
            rcc_pooled = view_features["RCC"].mean(dim=1)      # (B, dim)
            lcc_pooled = view_features["LCC"].mean(dim=1)      # (B, dim)
            rmlo_pooled = view_features["RMLO"].mean(dim=1)    # (B, dim)
            lmlo_pooled = view_features["LMLO"].mean(dim=1)    # (B, dim)

            # 4 view'ı doğrudan concat → MLP
            flat_concat = torch.cat(
                [rcc_pooled, lcc_pooled, rmlo_pooled, lmlo_pooled], dim=-1
            )  # (B, dim*4)
            patient_feat = self.flat_fusion(flat_concat)  # (B, dim)

        else:
            # --- Seviye 2: Lateral Fusion ---
            if self.use_lateral:
                # Spatial cross-attention + attention pooling → (B, dim) per side
                lateral_features = self.lateral_fusion(view_features)
                # {"right": (B, dim), "left": (B, dim)}
            else:
                # Basit birleştirme: spatial pool → concat → projection
                right_pooled = view_features["RCC"].mean(dim=1)     # (B, dim)
                rmlo_pooled = view_features["RMLO"].mean(dim=1)     # (B, dim)
                left_pooled = view_features["LCC"].mean(dim=1)      # (B, dim)
                lmlo_pooled = view_features["LMLO"].mean(dim=1)     # (B, dim)

                right_concat = torch.cat([right_pooled, rmlo_pooled], dim=-1)
                left_concat = torch.cat([left_pooled, lmlo_pooled], dim=-1)
                lateral_features = {
                    "right": self.simple_lateral_proj(right_concat),
                    "left": self.simple_lateral_proj(left_concat),
                }

            # --- Seviye 3: Bilateral Fusion ---
            if self.use_bilateral:
                bilateral_out = self.bilateral_fusion(
                    left_feat=lateral_features["left"],
                    right_feat=lateral_features["right"],
                )
                patient_feat = bilateral_out["patient_feat"]
                f_diff = bilateral_out["f_diff"]
            else:
                # Basit birleştirme
                bilateral_concat = torch.cat(
                    [lateral_features["left"], lateral_features["right"]], dim=-1
                )
                patient_feat = self.simple_bilateral_proj(bilateral_concat)
                f_diff = None

        # --- Seviye 4: Sınıflandırma ---
        outputs = self.classifier(patient_feat)
        outputs["patient_features"] = patient_feat

        # Asimetri kaybı için f_diff (bilateral fusion aktifse dolu, değilse None)
        outputs["f_diff"] = f_diff if (not self.use_flat_fusion) else None

        # --- Optional: Per-Region Malign Head (Track B) ---
        # Runs only if enabled in ablation config AND view_features were
        # produced (i.e., not flat_fusion). The head's gradient flows
        # through view_features into the backbone but NOT into the
        # lateral/bilateral fusion modules.
        if self.use_per_region and not self.use_flat_fusion:
            outputs["per_region_malign_logits"] = self.per_region_head(view_features)

        return outputs

    def get_backbone_extractor(self):
        """Grad-CAM için backbone erişimi sağlar."""
        return self.backbone.backbone


def build_model(config: dict) -> MammographyClassifier:
    """
    Config'den model oluşturur.

    Args:
        config: YAML konfigürasyon sözlüğü.

    Returns:
        MammographyClassifier instance.
    """
    model = MammographyClassifier(config)
    return model


def build_baseline_config(config: dict) -> dict:
    """
    Baseline deney için ablation ayarlarını düzenler.

    Baseline = Sadece backbone + ortalama pooling + full_head.
    Lateral ve bilateral fusion kapatılır.

    Args:
        config: Orijinal konfigürasyon.

    Returns:
        Baseline konfigürasyonu (orijinal değişmez, kopya döner).
    """
    import copy
    baseline_cfg = copy.deepcopy(config)
    baseline_cfg["ablation"] = {
        "use_lateral_fusion": False,
        "use_bilateral_fusion": False,
        "use_binary_head": False,
        "use_subgroup_head": False,
        "use_uncertainty": False,
    }
    return baseline_cfg
