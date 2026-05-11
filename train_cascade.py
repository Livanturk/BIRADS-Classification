"""
Cascade-stage trainer for the G-series soft cascade.

Mirrors train.py's loop structure (gradient accumulation, mixed precision,
OneCycleLR, SWA, early stopping, MLflow + WandB logging) but trains a single
2-class head per stage. No multi-head loss, no asymmetry, no Mixup/CutMix.

Usage:
    python train_cascade.py --config configs/cascade/G1_stage1_binary.yaml
    python train_cascade.py --config configs/cascade/G2a_stage2_benign.yaml
    python train_cascade.py --config configs/cascade/G2b_stage2_malign.yaml
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from data.cascade_loader import (
    create_cascade_dataloaders,
    inverse_freq_class_weights,
    read_manifest,
)
from models.cascade_model import build_cascade_model

# Lazy-imported below in main() to avoid eagerly loading utils/__init__.py
# (which pulls seaborn). We import the logger modules directly:
#   from utils.mlflow_logger import ExperimentLogger
#   from utils.wandb_logger  import WandbLogger
# Importing these still loads utils/__init__.py because Python loads parent
# packages on submodule import, but we only need the loggers at runtime so
# we keep the import local to main().


# ============================================================================
# Inlined helpers (mirrors train.py — avoids importing train.py's eager deps)
# ============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_output_dirs(config: dict, config_path: str) -> str:
    experiment_name = Path(config_path).stem
    base_dir = os.path.join("outputs", experiment_name)
    config.setdefault("checkpoint", {})["save_dir"] = os.path.join(base_dir, "checkpoints")
    vis = config.get("visualization", {})
    if "gradcam" in vis:
        vis["gradcam"]["save_dir"] = os.path.join(base_dir, "gradcam")
    if "confusion_matrix" in vis:
        vis["confusion_matrix"]["save_dir"] = os.path.join(base_dir, "plots")
    if "classification_report" in vis:
        vis["classification_report"]["save_dir"] = os.path.join(base_dir, "reports")
    print(f"[INFO] experiment: {experiment_name}")
    print(f"[INFO] outputs:    {base_dir}/")
    return experiment_name


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        },
        save_path,
    )


class EarlyStopping:
    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
            else (score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def build_scheduler(optimizer, config: dict, steps_per_epoch: Optional[int] = None):
    sched_cfg = config["training"]["scheduler"]
    name = sched_cfg["name"].lower()
    epochs = config["training"]["epochs"]

    if name == "onecycle":
        assert steps_per_epoch is not None
        max_lr = sched_cfg.get("max_lr", 5e-4)
        backbone_lr_scale = config["training"]["optimizer"].get("backbone_lr_scale", 0.2)
        max_lrs = []
        for group in optimizer.param_groups:
            gn = group.get("group_name", "")
            max_lrs.append(max_lr * backbone_lr_scale if "backbone" in gn else max_lr)
        grad_accum = config["training"].get("gradient_accumulation_steps", 1)
        effective_steps = (steps_per_epoch + grad_accum - 1) // grad_accum
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lrs, epochs=epochs, steps_per_epoch=effective_steps,
            pct_start=sched_cfg.get("pct_start", 0.3),
            anneal_strategy=sched_cfg.get("anneal_strategy", "cos"),
            div_factor=sched_cfg.get("div_factor", 10.0),
            final_div_factor=sched_cfg.get("final_div_factor", 100.0),
        )
    if name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=sched_cfg.get("T_0", 20),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("eta_min", 1e-7),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched_cfg.get("step_size", 30),
            gamma=sched_cfg.get("gamma", 0.1),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


# ============================================================================
# Cascade-specific binary metric tracker
# ============================================================================
class BinaryMetricTracker:
    def __init__(self, class_names=("class0", "class1")):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self._preds = []
        self._labels = []
        self._probs = []
        self._losses = []

    def update(self, outputs: dict, labels: torch.Tensor, loss: Optional[torch.Tensor] = None):
        with torch.no_grad():
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = probs.argmax(dim=-1)
            self._preds.extend(preds.cpu().numpy())
            self._labels.extend(labels.cpu().numpy())
            self._probs.extend(probs[:, 1].cpu().numpy())   # P(class=1)
            if loss is not None:
                self._losses.append(float(loss.item()))

    def compute(self) -> dict:
        labels = np.array(self._labels)
        preds = np.array(self._preds)
        probs1 = np.array(self._probs)

        m = {}
        m["accuracy"] = float(accuracy_score(labels, preds))
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        m["precision_macro"] = float(prec_macro)
        m["recall_macro"] = float(rec_macro)
        m["f1_macro"] = float(f1_macro)

        prec_pos, rec_pos, f1_pos, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0, pos_label=1
        )
        m["precision_pos"] = float(prec_pos)
        m["recall_pos"] = float(rec_pos)
        m["f1_pos"] = float(f1_pos)

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0, labels=[0, 1]
        )
        for i, name in enumerate(self.class_names):
            if i < len(f1_per_class):
                m[f"f1_{name}"] = float(f1_per_class[i])

        try:
            m["auc_roc"] = float(roc_auc_score(labels, probs1))
        except ValueError:
            m["auc_roc"] = 0.0

        if self._losses:
            m["total_loss"] = float(np.mean(self._losses))
        return m

    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(np.array(self._labels), np.array(self._preds), labels=[0, 1])


# ============================================================================
# Cascade-specific param groups (no "classifier." prefix; use "head.")
# ============================================================================
def _cascade_param_groups(model: nn.Module, config: dict) -> list:
    opt_cfg = config["training"]["optimizer"]
    base_lr = opt_cfg["lr"]
    backbone_lr = base_lr * opt_cfg.get("backbone_lr_scale", 0.2)

    wd_cfg = opt_cfg["weight_decay"]
    if isinstance(wd_cfg, (int, float)):
        wd_backbone = wd_fusion = wd_head = float(wd_cfg)
    else:
        wd_backbone = wd_cfg.get("backbone", 0.05)
        wd_fusion = wd_cfg.get("fusion", 0.05)
        wd_head = wd_cfg.get("head", 0.01)

    no_decay_kw = {"bias", "LayerNorm", "layernorm", "layer_norm", "norm", "ln"}

    def is_no_decay(name: str) -> bool:
        return any(kw in name for kw in no_decay_kw)

    groups = {
        "backbone_decay": [],   "backbone_no_decay": [],
        "fusion_decay":   [],   "fusion_no_decay":   [],
        "head_decay":     [],   "head_no_decay":     [],
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone.backbone.backbone"):
            bucket = "backbone_no_decay" if is_no_decay(name) else "backbone_decay"
        elif name.startswith("backbone.backbone.projection"):
            bucket = "fusion_no_decay" if is_no_decay(name) else "fusion_decay"
        elif "lateral_fusion" in name or "bilateral_fusion" in name:
            bucket = "fusion_no_decay" if is_no_decay(name) else "fusion_decay"
        elif name.startswith("head."):
            bucket = "head_no_decay" if is_no_decay(name) else "head_decay"
        else:
            bucket = "fusion_no_decay" if is_no_decay(name) else "fusion_decay"
        groups[bucket].append(p)

    out = []
    lr_map = {"backbone": backbone_lr, "fusion": base_lr, "head": base_lr}
    wd_map = {"backbone": wd_backbone, "fusion": wd_fusion, "head": wd_head}
    for bucket, params in groups.items():
        if not params:
            continue
        kind = bucket.split("_")[0]   # backbone | fusion | head
        decay = "no_decay" not in bucket
        out.append({
            "params": params,
            "lr": lr_map[kind],
            "weight_decay": wd_map[kind] if decay else 0.0,
            "group_name": bucket,
        })

    total = sum(p.numel() for g in out for p in g["params"])
    print("\n[CASCADE-OPT] param groups:")
    for g in out:
        n = sum(p.numel() for p in g["params"])
        print(f"  {g['group_name']:22s} {n:>10,} params  lr={g['lr']:.2e}  wd={g['weight_decay']:.3f}")
    print(f"  {'TOTAL':22s} {total:>10,} params")
    return out


def _build_optimizer(model, config):
    """Drop-in clone of train.build_optimizer using cascade param groups."""
    opt_cfg = config["training"]["optimizer"]
    name = opt_cfg["name"].lower()
    pg = _cascade_param_groups(model, config)
    if name == "adamw":
        return torch.optim.AdamW(pg, betas=tuple(opt_cfg.get("betas", [0.9, 0.999])))
    if name == "adam":
        return torch.optim.Adam(pg)
    if name == "sgd":
        return torch.optim.SGD(pg, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


# ============================================================================
# Train / eval loops
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    config, tracker, scheduler=None):
    model.train()
    tracker.reset()
    grad_accum = config["training"].get("gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  Train", leave=False, ncols=100)
    for step, batch in enumerate(pbar):
        images = batch["images"].to(device)
        labels = batch["label"].to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs["logits"], labels) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        tracker.update(outputs, labels, loss=loss.detach() * grad_accum)
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
    pbar.close()
    return tracker.compute()


@torch.no_grad()
def evaluate(model, loader, criterion, device, tracker):
    model.eval()
    tracker.reset()
    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=100)
    for batch in pbar:
        images = batch["images"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        loss = criterion(outputs["logits"], labels)
        tracker.update(outputs, labels, loss=loss)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    pbar.close()
    return tracker.compute()


# ============================================================================
# Main
# ============================================================================
def main(config_path: str, device_id: int = 0):
    config = load_config(config_path)
    experiment_name = apply_output_dirs(config, config_path)
    seed = config["project"]["seed"]
    set_seed(seed)

    if "cascade" not in config:
        raise SystemExit("[ERR] config missing top-level 'cascade' block")
    stage = config["cascade"]["stage"]
    print(f"\n[CASCADE] training stage = {stage}")

    if torch.cuda.is_available() and device_id >= 0:
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(device_id)}")

    # --- Data ---
    print("\n[1/5] data...")
    dataloaders = create_cascade_dataloaders(config)

    # --- Model ---
    print("\n[2/5] model...")
    model = build_cascade_model(config).to(device)

    # --- Loss (single 2-class CE with class weights from config or manifest) ---
    print("\n[3/5] loss / optim / sched...")
    cw_cfg = config["training"].get("class_weights")
    if cw_cfg in (None, "auto"):
        cw = inverse_freq_class_weights(
            read_manifest(config["cascade"]["train_manifest"])[1], num_classes=2
        )
    else:
        cw = torch.tensor(cw_cfg, dtype=torch.float32)
    cw = cw.to(device)
    print(f"[CASCADE-LOSS] class_weights = {cw.tolist()} "
          f"(label_smoothing={config['training'].get('label_smoothing', 0.0)})")
    criterion = nn.CrossEntropyLoss(
        weight=cw,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    )

    optimizer = _build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(dataloaders["train"]))
    scaler = GradScaler()

    # --- SWA ---
    use_swa = config["training"].get("use_swa", False)
    swa_start_epoch = config["training"].get("swa_start_epoch", 5)
    swa_model = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model, device=device)
        print(f"[INFO] SWA enabled, start_epoch={swa_start_epoch}")

    # --- Early stopping ---
    es_cfg = config["training"].get("early_stopping", {})
    early_stopping = None
    if es_cfg.get("enabled", True):
        early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 20),
            mode=es_cfg.get("mode", "max"),
        )

    # --- Loggers (lazy import: avoids loading utils/__init__.py at module load) ---
    from utils.mlflow_logger import ExperimentLogger
    from utils.wandb_logger import WandbLogger
    logger = ExperimentLogger(config)
    run_name = f"cascade_{stage}_{experiment_name}"
    logger.start_run(run_name=run_name)
    logger.log_params_flat(config)
    wandb_logger = WandbLogger(config)
    wandb_logger.start_run(run_name=run_name)
    wandb_logger.log_params_flat(config)

    # Class names for tracker
    if stage == "stage1":
        class_names = ("benign", "malign")
    elif stage == "stage2a":
        class_names = ("BR1", "BR2")
    elif stage == "stage2b":
        class_names = ("BR4", "BR5")
    else:
        class_names = ("c0", "c1")

    train_tracker = BinaryMetricTracker(class_names)
    val_tracker = BinaryMetricTracker(class_names)

    print("\n[4/5] training...")
    history = {}
    best_val_f1 = 0.0
    best_model_state = None
    epochs = config["training"]["epochs"]
    checkpoint_dir = config["checkpoint"]["save_dir"]

    is_step_scheduler = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, scaler,
            config, train_tracker, scheduler=scheduler if is_step_scheduler else None,
        )
        val_metrics = evaluate(model, dataloaders["val"], criterion, device, val_tracker)

        if not is_step_scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get("f1_macro", 0))
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        log_metrics = {}
        for k, v in train_metrics.items():
            log_metrics[f"train_{k}"] = v
        for k, v in val_metrics.items():
            log_metrics[f"val_{k}"] = v
        log_metrics["lr"] = lr
        log_metrics["epoch_time_s"] = dt
        logger.log_metrics(log_metrics, step=epoch)
        wandb_logger.log_metrics(log_metrics, step=epoch)

        for k, v in log_metrics.items():
            if isinstance(v, (int, float)):
                history.setdefault(k, []).append(v)

        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        val_f1 = val_metrics.get("f1_macro", 0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics.get('total_loss', 0):.4f} | "
            f"Val Loss: {val_metrics.get('total_loss', 0):.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f} | "
            f"LR: {lr:.2e} | dt: {dt:.1f}s"
        )

        if early_stopping and early_stopping.step(val_f1):
            print(f"\n[INFO] early stopping after {early_stopping.patience} flat epochs")
            break

    # --- SWA BN update + final checkpoint selection ---
    if swa_model is not None and swa_model.n_averaged > 0:
        print(f"\n[INFO] SWA averaged {swa_model.n_averaged} epochs; updating BN…")
        from torch.optim.swa_utils import update_bn
        update_bn(dataloaders["train"], swa_model, device=device)

    # --- Final val eval (best model + optional SWA model), pick winner ---
    print("\n[5/5] final val eval...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    final_val = evaluate(model, dataloaders["val"], criterion, device, val_tracker)
    print(f"[FINAL] best-model val F1 macro = {final_val['f1_macro']:.4f}")

    final_metrics = final_val
    using_swa = False
    if swa_model is not None and swa_model.n_averaged > 0:
        swa_val = evaluate(swa_model, dataloaders["val"], criterion, device, val_tracker)
        print(f"[FINAL] swa-model  val F1 macro = {swa_val['f1_macro']:.4f}")
        if swa_val["f1_macro"] >= final_val["f1_macro"]:
            print("[FINAL] SWA model wins; using it as best_model.pt")
            save_checkpoint(
                swa_model.module, optimizer, scheduler, epochs, swa_val,
                os.path.join(checkpoint_dir, "best_model.pt"),
            )
            final_metrics = swa_val
            using_swa = True
        else:
            save_checkpoint(
                swa_model.module, optimizer, scheduler, epochs, swa_val,
                os.path.join(checkpoint_dir, "swa_model.pt"),
            )

    # Also write a stable-named copy at checkpoints/cascade/<stage>_best.pt
    cascade_alias = Path("checkpoints/cascade") / f"{stage_alias(stage)}_best.pt"
    cascade_alias.parent.mkdir(parents=True, exist_ok=True)
    src_state = (swa_model.module if using_swa else model).state_dict()
    torch.save(
        {
            "stage": stage,
            "config_path": config_path,
            "model_state_dict": src_state,
            "val_metrics": final_metrics,
            "using_swa": using_swa,
            "class_names": class_names,
        },
        cascade_alias,
    )
    print(f"[FINAL] cascade alias written -> {cascade_alias}")

    # Log final test-style metrics under "val_*" prefix (no test set in cascade train)
    final_log = {f"final_{k}": v for k, v in final_metrics.items()}
    final_log["final_using_swa"] = float(using_swa)
    logger.log_metrics(final_log)
    wandb_logger.log_metrics(final_log)

    # Confusion matrix
    cm = val_tracker.get_confusion_matrix()
    print(f"\n[FINAL] val confusion matrix ({class_names}):\n{cm}")

    logger.end_run()
    wandb_logger.end_run()
    print("\n[CASCADE] training done.")


def stage_alias(stage: str) -> str:
    return {"stage1": "G1", "stage2a": "G2a", "stage2b": "G2b"}[stage]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade-stage trainer (G-series)")
    parser.add_argument("--config", type=str, required=True, help="cascade config path")
    parser.add_argument("--device", type=int, default=0, help="GPU index")
    args = parser.parse_args()
    main(config_path=args.config, device_id=args.device)
