"""
Seviye 2: Lateral-Level Fusion (Spatial Cross-Attention)
=========================================================
Aynı taraftaki CC ve MLO görüntülerinin spatial öznitelik haritaları
arasında cross-attention uygular.

Neden Spatial Cross-Attention?
    CC (üstten) ve MLO (yandan) görüntüleri aynı memeyi farklı açılardan gösterir.
    Radyolog CC'de gördüğü bir kitlenin MLO'daki karşılığını arar.

    Spatial cross-attention bu süreci modeller:
    CC'deki HER spatial bölge (token), MLO'daki TÜM spatial bölgelere
    dikkat ederek en ilgili bölgeleri bulur. Bu sayede:
    - CC'de bir kitle varsa, MLO'daki karşılık gelen bölge vurgulanır
    - Multi-head yapı farklı ilişki türlerini paralel öğrenir
    - Birden fazla katman ile bilgi giderek daha fazla paylaşılır

Matematiksel Detay:
    Spatial tokenlar: CC = (B, S, dim), MLO = (B, S, dim)
    S = H × W (örn: 12×12 = 144 token, 384×384 girdi, stride=32)

    Cross-Attention (Pre-LN):
        Q = LayerNorm(CC),  K = V = LayerNorm(MLO)
        CC' = CC + MultiHeadAttn(Q, K, V)    (residual)
        CC'' = CC' + FFN(LayerNorm(CC'))       (residual)

    Bi-directional: Hem CC→MLO hem MLO→CC yönünde.

    Attention Pooling:
        S adet spatial token → tek lateral vektör (dim boyutlu)
        Her token için öğrenilen önem skoru hesaplanır.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Pre-LN Spatial Cross-Attention bloğu.

    Source dizisindeki her token, target dizisindeki tüm token'lara
    dikkat eder. Pre-LN (norm-first) kullanılır — daha stabil eğitim sağlar.

    Akış (Pre-LN):
        h = source + MultiHeadAttn(LN(source), LN(target), LN(target))
        output = h + FFN(LN(h))

    Args:
        dim: Öznitelik boyutu.
        num_heads: Dikkat başlığı sayısı (dim'in tam böleni olmalı).
        attention_dropout: Attention weight dropout oranı.
        ffn_dropout: Feed-forward network dropout oranı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
    ):
        super().__init__()

        assert dim % num_heads == 0, (
            f"dim ({dim}), num_heads ({num_heads}) ile tam bölünmeli."
        )

        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Pre-LN: Normalizasyon attention/FFN ÖNCESİNDE uygulanır
        self.norm_q = nn.LayerNorm(dim)      # Query (source) normalizasyonu
        self.norm_kv = nn.LayerNorm(dim)     # Key/Value (target) normalizasyonu
        self.norm_ffn = nn.LayerNorm(dim)    # FFN öncesi normalizasyon

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Spatial cross-attention uygular.

        Args:
            source: (B, S_q, dim) — Sorgu kaynağı (spatial token dizisi).
            target: (B, S_kv, dim) — Anahtar/değer kaynağı (spatial token dizisi).
                    Genelde S_q == S_kv (aynı backbone, aynı spatial boyut).

        Returns:
            (B, S_q, dim) — Cross-attention ile zenginleştirilmiş source dizisi.
        """
        # Pre-LN Cross-Attention
        src_norm = self.norm_q(source)
        tgt_norm = self.norm_kv(target)

        attn_output, _ = self.cross_attn(
            query=src_norm,
            key=tgt_norm,
            value=tgt_norm,
        )
        source = source + attn_output       # Residual bağlantı

        # Pre-LN Feed-Forward
        ffn_output = self.ffn(self.norm_ffn(source))
        source = source + ffn_output        # Residual bağlantı

        return source


class DeformableCrossAttentionBlock(nn.Module):
    """
    Deformable Cross-View Attention Bloğu.

    Standart global cross-attention yerine, her query token için
    öğrenilen K adet spatial offset ile target feature map'inden
    seçici örnekleme yapılır.

    Klinik Motivasyon:
        Radyolog CC görüntüsündeki bir lezyonu MLO'da ararken, tam karşılık
        gelen bölgeye bakar — tüm görüntüyü taramaz. Deformable attention
        bu seçici bölge odaklanmasını öğrenir.

    Matematiksel Akış (her head için):
        1. Query token q_i → offset ağı → K adet (Δx, Δy) offset
        2. Sampling noktaları: p_k = p_ref_i + Δ_k  (normalize [-1,1] coords)
        3. Bilinear sampling: V_k = grid_sample(target_2d, p_k)
        4. Attention weights: w_k = softmax(MLP(q_i))
        5. Çıkış: Σ_k w_k * V_k

    Bu sayede O(S²) global attention yerine O(S·K) sparse attention.

    Args:
        dim: Öznitelik boyutu.
        num_heads: Dikkat başlığı sayısı.
        num_points: Her query için örnekleme noktası sayısı (K). Default: 4
        attention_dropout: Attention dropout.
        ffn_dropout: FFN dropout.
        spatial_size: Spatial grid kenar uzunluğu (H=W). Örn: 16 (16×16=256 token).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_points: int = 4,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        spatial_size: int = 16,
    ):
        super().__init__()

        assert dim % num_heads == 0, f"dim ({dim}) num_heads ({num_heads}) ile bölünmeli."

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.spatial_size = spatial_size

        # Query projeksiyon
        self.query_proj = nn.Linear(dim, dim)

        # Offset tahmin ağı: her head-query için K adet 2D offset
        # Giriş: (B*S*num_heads, head_dim) → (B*S*num_heads, K*2)
        self.offset_net = nn.Sequential(
            nn.Linear(self.head_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_points * 2),
        )
        # Offset'leri küçük başlat: referans noktası etrafında başlasın
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Attention weight tahmin ağı: K nokta için softmax ağırlıkları
        self.attn_weight_net = nn.Linear(self.head_dim, num_points)

        # Value projeksiyon
        self.value_proj = nn.Linear(dim, dim)

        # Output projeksiyon
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attention_dropout)

        # Pre-LN normalizasyon
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_kv  = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ffn_dropout),
        )

        # Referans grid: normalize [-1, 1] koordinatları (bir kez oluşturulur)
        H = W = spatial_size
        ref_y = torch.linspace(-1.0, 1.0, H)
        ref_x = torch.linspace(-1.0, 1.0, W)
        # (H, W, 2) → (S, 2)
        grid_y, grid_x = torch.meshgrid(ref_y, ref_x, indexing="ij")
        ref_grid = torch.stack([ref_x.unsqueeze(0).expand(H, -1),
                                ref_y.unsqueeze(1).expand(-1, W)], dim=-1)  # (H, W, 2) [x,y]
        self.register_buffer("ref_grid", ref_grid.view(H * W, 2))          # (S, 2)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            source: (B, S, dim) — Query kaynağı.
            target: (B, S, dim) — Değer kaynağı.

        Returns:
            (B, S, dim) — Deformable cross-attention çıkışı.
        """
        B, S, _ = source.shape
        H = W = self.spatial_size

        # Pre-LN
        q_norm = self.norm_q(source)    # (B, S, dim)
        v_norm = self.norm_kv(target)   # (B, S, dim)

        # Query projeksiyon → multi-head shape
        q = self.query_proj(q_norm)                             # (B, S, dim)
        q = q.view(B, S, self.num_heads, self.head_dim)         # (B, S, Nh, D)

        # Value projeksiyon → 2D spatial map (tüm head'ler için)
        v = self.value_proj(v_norm)                             # (B, S, dim)
        # (B, dim, H, W) — her head kendi dilimini kullanır
        v_2d = v.view(B, H, W, self.dim).permute(0, 3, 1, 2)   # (B, dim, H, W)
        # Çok başlı format: (B*Nh, head_dim, H, W)
        v_2d_mh = v_2d.view(B, self.num_heads, self.head_dim, H, W)
        v_2d_mh = v_2d_mh.reshape(B * self.num_heads, self.head_dim, H, W)

        # Offset ve attention weight hesaplama — tüm head'ler vektörize
        # q: (B, S, Nh, D) → (B*S*Nh, D)
        q_flat = q.reshape(B * S * self.num_heads, self.head_dim)

        # Offset: (B*S*Nh, K*2) → (B*S*Nh, K, 2) → ölçekle → (B, Nh, S, K, 2)
        offsets_flat = self.offset_net(q_flat)                  # (B*S*Nh, K*2)
        offsets = offsets_flat.view(B, S, self.num_heads, self.num_points, 2)
        offsets = torch.tanh(offsets) * 0.5  # [-0.5, 0.5] aralığında tut

        # Referans grid: (S, 2) → (1, S, 1, 1, 2)
        ref = self.ref_grid.view(1, S, 1, 1, 2)                 # (1, S, 1, 1, 2)
        sample_pts = ref + offsets                               # (B, S, Nh, K, 2)

        # grid_sample için: (B*Nh, S*K, 1, 2)
        # permute: (B, Nh, S, K, 2)
        sample_pts = sample_pts.permute(0, 2, 1, 3, 4).reshape(
            B * self.num_heads, S * self.num_points, 1, 2
        )

        # Bilinear örnekleme: (B*Nh, head_dim, S*K, 1)
        sampled = F.grid_sample(
            v_2d_mh, sample_pts,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )                                                        # (B*Nh, D, S*K, 1)
        sampled = sampled.squeeze(-1)                            # (B*Nh, D, S*K)
        sampled = sampled.view(B, self.num_heads, self.head_dim, S, self.num_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)               # (B, S, Nh, K, D)

        # Attention weights: (B*S*Nh, K) → (B, S, Nh, K)
        attn_w = self.attn_weight_net(q_flat)                   # (B*S*Nh, K)
        attn_w = attn_w.view(B, S, self.num_heads, self.num_points)
        attn_w = torch.softmax(attn_w, dim=-1)                  # (B, S, Nh, K)
        attn_w = self.attn_dropout(attn_w)

        # Ağırlıklı toplam: (B, S, Nh, K, 1) * (B, S, Nh, K, D) → (B, S, Nh, D)
        head_out = (attn_w.unsqueeze(-1) * sampled).sum(dim=3)  # (B, S, Nh, D)
        concat = head_out.reshape(B, S, self.dim)               # (B, S, dim)

        # Output projeksiyon + residual
        attn_output = self.out_proj(concat)                     # (B, S, dim)
        source = source + attn_output

        # Pre-LN FFN + residual
        source = source + self.ffn(self.norm_ffn(source))

        return source


class LateralFusion(nn.Module):
    """
    Bir taraftaki CC ve MLO spatial özniteliklerini bi-directional
    cross-attention ile birleştirip tek bir lateral vektör üretir.

    Akış:
        1. Positional embedding ekle (spatial konum bilgisi)
        2. N katman bi-directional cross-attention:
           CC' = CrossAttn(CC → MLO)   — CC, MLO'ya dikkat eder
           MLO' = CrossAttn(MLO → CC)  — MLO, CC'ye dikkat eder
        3. Attention pooling: spatial token'ları tek vektöre indirge
        4. Fusion: concat([CC_pooled, MLO_pooled]) → projeksiyon

    Args:
        dim: Öznitelik boyutu.
        num_spatial_tokens: Spatial token sayısı (H × W).
        num_heads: Cross-attention başlık sayısı.
        attention_dropout: Attention dropout oranı.
        ffn_dropout: FFN dropout oranı.
        projection_dropout: Fusion projeksiyon dropout oranı.
        num_layers: Cross-attention katman sayısı.
    """

    def __init__(
        self,
        dim: int,
        num_spatial_tokens: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        num_layers: int = 2,
        use_deformable: bool = False,
        num_deformable_points: int = 4,
    ):
        super().__init__()

        # Spatial boyutları hesapla (deformable için 2D grid gerekli)
        spatial_size = int(math.isqrt(num_spatial_tokens))
        is_square = (spatial_size * spatial_size == num_spatial_tokens)
        # Kare olmayan token sayısı varsa deformable devre dışı (fallback)
        self.use_deformable = use_deformable and is_square
        if use_deformable and not is_square:
            print(f"[UYARI] Kare olmayan spatial grid ({num_spatial_tokens} token) "
                  f"→ Deformable attention devre dışı, standart cross-attention kullanılıyor.")

        # Learnable positional embedding — spatial konum bilgisi
        # CC ve MLO aynı backbone'dan geçtiği için aynı spatial grid'e sahip,
        # dolayısıyla aynı positional embedding paylaşılır
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_spatial_tokens, dim) * 0.02
        )

        def _make_block(direction):
            if self.use_deformable:
                return DeformableCrossAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_points=num_deformable_points,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    spatial_size=spatial_size,
                )
            else:
                return CrossAttentionBlock(dim, num_heads, attention_dropout, ffn_dropout)

        # Bi-directional cross-attention katmanları
        self.cc_to_mlo_layers = nn.ModuleList([
            _make_block("cc2mlo") for _ in range(num_layers)
        ])
        self.mlo_to_cc_layers = nn.ModuleList([
            _make_block("mlo2cc") for _ in range(num_layers)
        ])

        # Cross-attention sonrası normalizasyon
        self.final_norm = nn.LayerNorm(dim)

        # Attention pooling: spatial token'ları tek vektöre indirger
        # Her token için öğrenilen önem skoru hesaplar
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

        # Son birleştirme projeksiyon katmanı
        self.fusion_projection = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )

    def _pool_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention pooling: spatial token dizisini tek vektöre indirger.

        Args:
            x: (B, S, dim) — Spatial token dizisi.

        Returns:
            (B, dim) — Ağırlıklı toplam ile elde edilen tek vektör.
        """
        scores = self.attention_pool(x)             # (B, S, 1)
        weights = torch.softmax(scores, dim=1)      # (B, S, 1) — normalize
        pooled = (weights * x).sum(dim=1)           # (B, dim)
        return pooled

    def forward(
        self, cc_feat: torch.Tensor, mlo_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        CC ve MLO spatial özniteliklerini cross-attention ile birleştirir.

        Args:
            cc_feat:  (B, S, dim) — CC görüntüsünün spatial öznitelik dizisi.
            mlo_feat: (B, S, dim) — MLO görüntüsünün spatial öznitelik dizisi.

        Returns:
            (B, dim) — Birleştirilmiş lateral öznitelik vektörü.
        """
        # Positional embedding ekle — spatial konum bilgisi
        cc_enhanced = cc_feat + self.pos_embed
        mlo_enhanced = mlo_feat + self.pos_embed

        # Her katmanda çift yönlü spatial cross-attention
        for cc2mlo, mlo2cc in zip(self.cc_to_mlo_layers, self.mlo_to_cc_layers):
            # CC'deki her token, MLO'daki tüm token'lara dikkat eder
            cc_new = cc2mlo(cc_enhanced, mlo_enhanced)
            # MLO'daki her token, CC'deki tüm token'lara dikkat eder
            mlo_new = mlo2cc(mlo_enhanced, cc_enhanced)

            cc_enhanced = cc_new
            mlo_enhanced = mlo_new

        # Final normalizasyon
        cc_enhanced = self.final_norm(cc_enhanced)
        mlo_enhanced = self.final_norm(mlo_enhanced)

        # Attention pooling: (B, S, dim) → (B, dim)
        cc_pooled = self._pool_spatial(cc_enhanced)
        mlo_pooled = self._pool_spatial(mlo_enhanced)

        # İki pooled vektörü birleştir ve projeksiyon uygula
        fused = torch.cat([cc_pooled, mlo_pooled], dim=-1)   # (B, dim*2)
        lateral = self.fusion_projection(fused)               # (B, dim)

        return lateral


class BilateralLateralFusion(nn.Module):
    """
    Sağ ve sol meme için ayrı ayrı lateral fusion uygular.

    Sağ meme: RCC + RMLO → Right Lateral Feature
    Sol meme: LCC + LMLO → Left Lateral Feature

    Not: İki taraf için AYNI ağırlıklar paylaşılır (weight sharing).
    Çünkü sağ ve sol meme anatomik olarak simetrik yapılardır,
    dolayısıyla aynı öznitelik çıkarma stratejisi uygulanabilir.

    Args:
        dim: Öznitelik boyutu.
        num_spatial_tokens: Spatial token sayısı (H × W).
        num_heads: Cross-attention başlık sayısı.
        attention_dropout: Attention dropout oranı.
        ffn_dropout: FFN dropout oranı.
        projection_dropout: Fusion projeksiyon dropout oranı.
        num_layers: Cross-attention katman sayısı.
    """

    def __init__(
        self,
        dim: int,
        num_spatial_tokens: int,
        num_heads: int = 8,
        attention_dropout: float = 0.15,
        ffn_dropout: float = 0.2,
        projection_dropout: float = 0.2,
        num_layers: int = 2,
        use_deformable: bool = False,
        num_deformable_points: int = 4,
    ):
        super().__init__()

        # Weight-shared lateral fusion (sağ ve sol aynı ağırlıkları paylaşır)
        self.lateral_fusion = LateralFusion(
            dim=dim,
            num_spatial_tokens=num_spatial_tokens,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            projection_dropout=projection_dropout,
            num_layers=num_layers,
            use_deformable=use_deformable,
            num_deformable_points=num_deformable_points,
        )

    def forward(self, view_features: dict) -> dict:
        """
        Args:
            view_features: Backbone çıkışları (spatial).
                {
                    "RCC":  (B, S, dim),
                    "LCC":  (B, S, dim),
                    "RMLO": (B, S, dim),
                    "LMLO": (B, S, dim),
                }

        Returns:
            dict:
                {
                    "right": (B, dim) — Sağ meme lateral özniteliği.
                    "left":  (B, dim) — Sol meme lateral özniteliği.
                }
        """
        # Sağ meme: RCC + RMLO → spatial cross-attention → pooled vektör
        right_lateral = self.lateral_fusion(
            cc_feat=view_features["RCC"],
            mlo_feat=view_features["RMLO"],
        )

        # Sol meme: LCC + LMLO → spatial cross-attention → pooled vektör
        left_lateral = self.lateral_fusion(
            cc_feat=view_features["LCC"],
            mlo_feat=view_features["LMLO"],
        )

        return {"right": right_lateral, "left": left_lateral}
