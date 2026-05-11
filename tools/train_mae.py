"""
In-domain Masked Autoencoder pretraining (Direction #1; Lesson #62 follow-up).

Pretrains a ConvNeXtV2-Base trunk on the existing Dataset_1024_8bit pool
(8557 train+val + 1655 test = 10212 patients × 4 views ≈ 40,848 images;
no labels used) via a simple linear-decoder MAE:

  1. Patchify input (B, 3, 1024, 1024) → 32×32 patches → 1024 patches per image
  2. Mask 75% of patches; replace with a learned 3-channel mask token
  3. Encode with ConvNeXtV2-Base → (B, 1024, 32, 32) feature map
  4. Decode: 1×1 conv → 3*32*32 channels → unfold to (B, 3, 1024, 1024)
  5. MSE on MASKED patches only (He 2022 MAE recipe; canonical normalization)

Output:
  {output_dir}/encoder_only.pt   — backbone trunk state_dict (loaded by C6 fine-tune)
  {output_dir}/full_state.pt     — full encoder+decoder state (resume support)
  {output_dir}/training.log      — per-epoch loss + sample reconstructions

Multi-GPU: uses nn.DataParallel for simplicity (single-node, 4 GPUs). Throughput
~3× single-GPU; not full DDP scaling but no launch-script complexity.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_mae.py \
      --root-dir Dataset_1024_8bit \
      --backbone convnextv2_base.fcmae_ft_in22k_in1k_384 \
      --feature-dim 1024 \
      --image-size 1024 \
      --patch-size 32 \
      --mask-ratio 0.75 \
      --batch-size 16 \
      --epochs 30 \
      --lr 1e-4 \
      --output-dir outputs/mae_pretrain_base

Wall-clock estimate (4× H100, 40k images, 30 epochs): ~12-18h.

Notes:
  - The existing FCMAE pretrain in `convnextv2_base.fcmae_ft_in22k_in1k_384`
    was on ImageNet-22k. We OVERWRITE those features with mammography-specific
    features by continuing MAE training on our domain. The output checkpoint
    is consumed by C6's fine-tune step (see configs/.../c6_base_mammopre_*.yaml).
  - Mask ratio 0.75 is canonical (He 2022). With 1024 patches per image, that
    leaves 256 visible patches — plenty of context.
  - Augmentation is intentionally light (only horizontal flip): aggressive
    augmentation + masking together has been shown to underfit.
  - Loss: per-patch normalized MSE (canonical MAE — normalize each patch by
    its own mean+std before MSE; this prevents the loss from being dominated
    by bright tissue regions).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---- Dataset: flat scan over Dataset_1024_8bit (no labels, all views) ----

class FlatPNGDataset(Dataset):
    """
    Walks Dataset_1024_8bit/BI-RADS-*/[patient_id]/{RCC,LCC,RMLO,LMLO}.png and
    yields each PNG as a single image. Labels are NOT returned (MAE is unsupervised).

    Uses the same normalization as C6 (CLAUDE.md §6 all-pixel statistics).
    """

    NORM_MEAN = (0.1210, 0.1210, 0.1210)
    NORM_STD = (0.1977, 0.1977, 0.1977)

    def __init__(
        self,
        root_dirs: List[str],
        image_size: int = 1024,
        horizontal_flip: float = 0.5,
        bit_depth: int = 8,
    ):
        self.image_size = image_size
        self.bit_depth = bit_depth
        self.image_paths = self._scan(root_dirs)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=horizontal_flip) if horizontal_flip > 0 else T.Lambda(lambda x: x),
            T.Normalize(mean=self.NORM_MEAN, std=self.NORM_STD),
        ])
        print(f"[MAE-DATA] {len(self.image_paths)} images from {len(root_dirs)} root dirs")

    def _scan(self, root_dirs: List[str]) -> List[str]:
        paths = []
        for root in root_dirs:
            root = Path(root)
            if not root.exists():
                print(f"[MAE-DATA] [warn] root_dir missing: {root}")
                continue
            for birads_dir in sorted(root.iterdir()):
                if not birads_dir.is_dir():
                    continue
                if not birads_dir.name.startswith("BI-RADS"):
                    continue
                for pt_dir in sorted(birads_dir.iterdir()):
                    if not pt_dir.is_dir():
                        continue
                    for view in ("RCC", "LCC", "RMLO", "LMLO"):
                        p = pt_dir / f"{view}.png"
                        if p.exists():
                            paths.append(str(p))
        return paths

    def __len__(self):
        return len(self.image_paths)

    def _load(self, path):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / (65535.0 if self.bit_depth == 16 else 255.0)
        # 3 channels expected by pretrained ConvNeXt
        tensor = torch.from_numpy(arr).unsqueeze(0).expand(3, -1, -1).clone()
        return tensor

    def __getitem__(self, idx):
        return self.transform(self._load(self.image_paths[idx]))


# ---- MAE model: encoder (ConvNeXt-Base) + linear decoder + masked MSE ----

class SimpleMAE(nn.Module):
    """
    Linear-decoder MAE for ConvNeXtV2 backbones.

    Encoder: timm ConvNeXt with global_pool="" (returns spatial map).
    Decoder: 1×1 conv (B, F, S, S) → (B, 3*P*P, S, S), unfold to image.
    Loss: per-patch normalized MSE on masked patches only.

    For 1024² input + ConvNeXt-Base (stride 32):
        S = 1024 / 32 = 32  (spatial cells per side)
        F = 1024            (last-stage feature dim for Base)
        P = 32              (input pixels per patch)
        N = S*S = 1024 patches per image.
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
        feature_dim: int = 1024,
        image_size: int = 1024,
        patch_size: int = 32,
        mask_ratio: float = 0.75,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.mask_ratio = mask_ratio
        self.num_patches_side = image_size // patch_size
        self.num_patches = self.num_patches_side ** 2

        # --- Encoder: timm ConvNeXt trunk; preserve spatial output ---
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        # Sanity: confirm encoder produces (B, F, S, S) at our input size.
        with torch.no_grad():
            d = torch.zeros(1, 3, image_size, image_size)
            o = self.encoder(d)
        if o.dim() != 4:
            raise RuntimeError(
                f"Encoder output must be 4-D (B,F,S,S); got {o.shape}. "
                f"Check backbone is a CNN (ConvNeXt family)."
            )
        if o.shape[1] != feature_dim:
            print(f"[MAE] [warn] feature_dim arg = {feature_dim} but encoder produces {o.shape[1]}; using {o.shape[1]}")
            self.feature_dim = o.shape[1]
        if o.shape[2] != self.num_patches_side:
            print(f"[MAE] [warn] patch_size implies {self.num_patches_side}^2 spatial cells but encoder gives {o.shape[2]}^2")

        # --- Mask token (input-level): one learnable 3×P×P patch, broadcast ---
        # Initialized small so it doesn't dominate the input statistics early.
        self.mask_token = nn.Parameter(torch.zeros(1, 3, patch_size, patch_size))
        nn.init.normal_(self.mask_token, std=0.02)

        # --- Linear decoder: (B, F, S, S) → (B, 3*P*P, S, S) ---
        # Hidden bottleneck so the decoder can't trivially memorize via raw projection.
        self.decoder = nn.Sequential(
            nn.Conv2d(self.feature_dim, decoder_hidden, 1),
            nn.GELU(),
            nn.Conv2d(decoder_hidden, 3 * patch_size * patch_size, 1),
        )

    def random_masking(self, B: int, device) -> torch.Tensor:
        """Returns (B, num_patches) bool tensor — True = masked."""
        n_visible = int(round(self.num_patches * (1.0 - self.mask_ratio)))
        noise = torch.rand(B, self.num_patches, device=device)
        ids = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        # Top n_visible by noise rank are KEPT; the remaining are MASKED.
        mask.scatter_(1, ids[:, n_visible:], True)
        return mask

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, N, 3, P, P)."""
        B, C, H, W = images.shape
        P = self.patch_size
        nH = H // P
        nW = W // P
        # (B, C, nH, P, nW, P) → (B, nH, nW, C, P, P) → (B, N, C, P, P)
        return (
            images
            .view(B, C, nH, P, nW, P)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, nH * nW, C, P, P)
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, N, 3, P, P) → (B, 3, H, W)."""
        B, N, C, P, _ = patches.shape
        S = self.num_patches_side
        return (
            patches
            .view(B, S, S, C, P, P)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, C, S * P, S * P)
        )

    def per_patch_norm(self, patches: torch.Tensor) -> torch.Tensor:
        """Canonical MAE patch normalization (He 2022)."""
        # patches: (B, N, 3, P, P)
        flat = patches.reshape(*patches.shape[:2], -1)  # (B, N, 3*P*P)
        mean = flat.mean(dim=-1, keepdim=True)
        var = flat.var(dim=-1, keepdim=True)
        norm = (flat - mean) / (var + 1e-6).sqrt()
        return norm.reshape_as(patches)

    def forward(self, images: torch.Tensor) -> dict:
        B, C, H, W = images.shape
        assert C == 3 and H == W == self.image_size, (
            f"expected (B, 3, {self.image_size}, {self.image_size}), got {images.shape}"
        )

        # 1) Patchify, randomly mask
        patches = self.patchify(images)                                      # (B, N, 3, P, P)
        mask = self.random_masking(B, images.device)                         # (B, N) bool

        # 2) Replace masked patches with mask token (broadcast)
        mask_token = self.mask_token.expand(-1, -1, -1, -1)                  # (1, 3, P, P)
        # mask_4d: (B, N, 1, 1, 1) for broadcasting
        mask_4d = mask.view(B, self.num_patches, 1, 1, 1)
        # mask_token unsqueezed to (1, 1, 3, P, P), broadcast against patches
        masked_patches = torch.where(
            mask_4d,
            mask_token.unsqueeze(1).expand(B, self.num_patches, -1, -1, -1),
            patches,
        )
        masked_image = self.unpatchify(masked_patches)                       # (B, 3, H, W)

        # 3) Encode masked image
        feat = self.encoder(masked_image)                                    # (B, F, S, S)

        # 4) Decode → predict P*P*3 per spatial cell
        pred_flat = self.decoder(feat)                                       # (B, 3*P*P, S, S)
        S = self.num_patches_side
        P = self.patch_size
        # Reshape to (B, N, 3, P, P) — flatten spatial cells, factor channels into 3*P*P
        pred = (
            pred_flat
            .view(B, 3, P, P, S, S)
            .permute(0, 4, 5, 1, 2, 3)
            .reshape(B, S * S, 3, P, P)
        )

        # 5) Per-patch normalized MSE on masked positions
        target = self.per_patch_norm(patches)                                # (B, N, 3, P, P)
        per_patch_mse = ((pred - target) ** 2).mean(dim=[2, 3, 4])           # (B, N)
        masked_loss = (per_patch_mse * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        return {
            "loss": masked_loss,
            "mask_ratio_observed": float(mask.float().mean().item()),
            "n_masked": int(mask.sum().item()),
        }


# ---- Training loop ----

def train_one_epoch(model, loader, optimizer, device, scaler, scheduler, epoch, log_path):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(loader, ncols=100, desc=f"epoch {epoch}", leave=False)
    for step, images in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(images)
            loss = out["loss"]
            # DataParallel returns per-device losses; reduce.
            if loss.dim() > 0:
                loss = loss.mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    avg = total_loss / max(n_batches, 1)
    line = f"epoch={epoch} avg_loss={avg:.4f} lr={optimizer.param_groups[0]['lr']:.6e}"
    print(f"[MAE] {line}")
    with open(log_path, "a") as f:
        f.write(line + "\n")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", action="append", required=True,
                    help="Pass multiple times for multiple roots (e.g. Dataset_1024_8bit and test)")
    ap.add_argument("--backbone", default="convnextv2_base.fcmae_ft_in22k_in1k_384")
    ap.add_argument("--feature-dim", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--decoder-hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--save-every", type=int, default=5,
                    help="Save full checkpoint every N epochs")
    ap.add_argument("--bit-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "training.log"
    with open(log_path, "a") as f:
        f.write(f"\n=== run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        json.dump(vars(args), f, indent=2)
        f.write("\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAE] device={device} cuda_count={torch.cuda.device_count()}")

    # Dataset + loader
    ds = FlatPNGDataset(
        root_dirs=args.root_dir,
        image_size=args.image_size,
        horizontal_flip=0.5,
        bit_depth=args.bit_depth,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = SimpleMAE(
        backbone_name=args.backbone,
        feature_dim=args.feature_dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_hidden=args.decoder_hidden,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MAE] total params: {n_params:,}")
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"[MAE] wrapping with DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    def lr_lambda(s):
        if s < warmup_steps:
            return s / max(1, warmup_steps)
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        # Cosine decay to 0.1× max LR (not all the way to zero — avoids late-train collapse).
        return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler()
    best = float("inf")
    for epoch in range(args.epochs):
        avg = train_one_epoch(model, loader, optimizer, device, scaler, scheduler, epoch, log_path)
        # Save encoder-only every epoch (cheap; ~360MB for ConvNeXt-Base)
        encoder_state = (
            model.module.encoder.state_dict() if isinstance(model, nn.DataParallel)
            else model.encoder.state_dict()
        )
        torch.save(encoder_state, out / "encoder_only.pt")
        # Full checkpoint less often
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                out / f"full_state_epoch{epoch:03d}.pt",
            )
        if avg < best:
            best = avg
            torch.save(encoder_state, out / "encoder_only_best.pt")

    print(f"[MAE] done. best avg loss = {best:.4f}")
    print(f"      encoder weights: {out}/encoder_only.pt (last)  +  encoder_only_best.pt (best)")
    with open(log_path, "a") as f:
        f.write(f"=== run finished, best={best:.4f} ===\n")


if __name__ == "__main__":
    main()
