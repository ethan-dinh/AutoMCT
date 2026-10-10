"""
Training pipeline for 3D mandible segmentation.

Supports:
- K-fold cross-validation (default: leave-one-out with 9 folds)
- Iterative retraining from a checkpoint (--resume)
- Per-class Dice evaluation
- MPS / CUDA / CPU

Usage:
    # Run full leave-one-out CV
    python train.py

    # Run a single fold
    python train.py --fold 0

    # Resume training from a checkpoint
    python train.py --fold 0 --resume runs/fold_0/best_model.pt

    # Point to a different data directory
    python train.py --data_dir /path/to/segmentation_results
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from config import Config
from dataset import LoadAndCombineMasksd, build_datasets, discover_samples
from model import build_model, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
    epoch: int,
) -> float:
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_samples: list[dict],
    device: str,
    cfg: Config,
    dice_metric: DiceMetric,
    post_pred: AsDiscrete,
    post_label: AsDiscrete,
) -> tuple[float, list[float]]:
    """
    Run sliding-window inference on each full validation volume.

    Returns (mean_dice, per_class_dice) where per_class_dice has one entry
    per foreground class (incisor, bone, molar).
    """
    model.eval()
    dice_metric.reset()

    loader = LoadAndCombineMasksd()

    for sample in val_samples:
        d = loader(sample)
        image = torch.from_numpy(d["image"]).unsqueeze(0).float().to(device)
        label = torch.from_numpy(d["label"]).unsqueeze(0).long().to(device)

        output = sliding_window_inference(
            image,
            roi_size=cfg.patch_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.sw_overlap,
        )
        # Post-process
        pred = [post_pred(i) for i in decollate_batch(output)]
        lbl = [post_label(i) for i in decollate_batch(label)]
        dice_metric(y_pred=pred, y=lbl)

    # Aggregate: shape (n_val, n_classes)
    per_sample = dice_metric.aggregate()
    # Mean across validation samples, per class
    per_class = per_sample.nanmean(dim=0).cpu().tolist()
    mean_dice = float(np.nanmean(per_class))
    return mean_dice, per_class


def train_fold(
    fold_idx: int,
    train_samples: list[dict],
    val_samples: list[dict],
    cfg: Config,
) -> dict:
    """Train one CV fold and return metrics."""
    device = cfg.resolve_device()
    logger.info(
        "=== Fold %d/%d  |  train=%d  val=%d  |  device=%s ===",
        fold_idx, cfg.n_folds, len(train_samples), len(val_samples), device,
    )

    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets
    train_ds, val_ds = build_datasets(
        train_samples=train_samples,
        val_samples=val_samples,
        patch_size=cfg.patch_size,
        samples_per_volume=cfg.samples_per_volume,
        pos_ratio=cfg.pos_sample_ratio,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device != "cpu",
    )

    # Build model
    model = build_model(
        model_name=cfg.model,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        strides=cfg.strides,
        dropout=cfg.dropout,
    ).to(device)

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Scheduler
    if cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=20, factor=0.5
        )

    # Dice metric (one value per foreground class)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.num_classes)
    post_label = AsDiscrete(to_onehot=cfg.num_classes)

    start_epoch = 0
    best_dice = -1.0
    epochs_no_improve = 0

    # Resume from checkpoint
    if cfg.resume_checkpoint:
        logger.info("Resuming from %s", cfg.resume_checkpoint)
        ckpt = load_checkpoint(model, cfg.resume_checkpoint, device=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", -1.0)
        logger.info("Resumed at epoch %d (best_dice=%.4f)", start_epoch, best_dice)

    # Training loop
    history: dict[str, list] = {
        "epoch": [], "train_loss": [], "val_dice": [], "per_class_dice": [],
    }
    last_epoch = start_epoch
    for epoch in range(start_epoch, cfg.max_epochs):
        last_epoch = epoch
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        dt_train = time.time() - t0

        # Validate every 5 epochs (sliding window on full volumes is slow)
        val_dice, per_class = 0.0, [0.0, 0.0, 0.0]
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            t1 = time.time()
            val_dice, per_class = validate(
                model, val_samples, device, cfg, dice_metric, post_pred, post_label
            )
            dt_val = time.time() - t1
            logger.info(
                "Epoch %3d/%d  loss=%.4f  dice=%.4f  "
                "[incisor=%.3f bone=%.3f molar=%.3f]  "
                "train=%.0fs val=%.0fs  lr=%.2e",
                epoch + 1, cfg.max_epochs, train_loss, val_dice,
                per_class[0], per_class[1], per_class[2],
                dt_train, dt_val, optimizer.param_groups[0]["lr"],
            )

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "per_class_dice": per_class,
                        "fold": fold_idx,
                    },
                    fold_dir / "best_model.pt",
                )
                logger.info("  -> New best model saved (dice=%.4f)", best_dice)
            else:
                epochs_no_improve += 5
        else:
            logger.info(
                "Epoch %3d/%d  loss=%.4f  train=%.0fs  lr=%.2e",
                epoch + 1, cfg.max_epochs, train_loss, dt_train,
                optimizer.param_groups[0]["lr"],
            )

        # Update scheduler
        if cfg.scheduler == "cosine":
            scheduler.step()
        elif (epoch + 1) % 5 == 0:
            scheduler.step(val_dice)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        history["per_class_dice"].append(per_class)

        # Early stopping
        if epochs_no_improve >= cfg.early_stop_patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, cfg.early_stop_patience)
            break

    # Save last checkpoint (for resuming later)
    torch.save(
        {
            "epoch": last_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "fold": fold_idx,
        },
        fold_dir / "last_checkpoint.pt",
    )

    # Save training history
    with open(fold_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    cfg.save(fold_dir / "config.json")

    logger.info("Fold %d complete — best dice=%.4f", fold_idx, best_dice)
    return {"fold": fold_idx, "best_dice": best_dice, "per_class_dice": per_class}


def run_cross_validation(cfg: Config) -> None:
    """Run k-fold cross-validation (or a single fold)."""
    samples = discover_samples(cfg.data_dir)
    if len(samples) < 2:
        raise RuntimeError(
            f"Need at least 2 samples for cross-validation, found {len(samples)}"
        )

    n_folds = min(cfg.n_folds, len(samples))
    logger.info(
        "Starting %d-fold cross-validation on %d samples", n_folds, len(samples)
    )

    fold_results = []

    folds_to_run = [cfg.fold] if cfg.fold is not None else range(n_folds)

    for fold_idx in folds_to_run:
        # Leave-one-out split: val = fold_idx, train = everything else
        val_indices = [fold_idx % len(samples)]
        train_indices = [i for i in range(len(samples)) if i not in val_indices]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        logger.info(
            "Fold %d: val=%s, train=%s",
            fold_idx,
            [s["name"] for s in val_samples],
            [s["name"] for s in train_samples],
        )

        result = train_fold(fold_idx, train_samples, val_samples, cfg)
        fold_results.append(result)

    # Aggregate results
    if len(fold_results) > 1:
        all_dice = [r["best_dice"] for r in fold_results]
        all_per_class = np.array([r["per_class_dice"] for r in fold_results])
        logger.info("=" * 60)
        logger.info("Cross-validation results (%d folds):", len(fold_results))
        logger.info(
            "  Mean Dice:     %.4f +/- %.4f", np.mean(all_dice), np.std(all_dice)
        )
        for i, name in enumerate(["incisor", "bone", "molar"]):
            logger.info(
                "  %-12s   %.4f +/- %.4f",
                name, np.nanmean(all_per_class[:, i]), np.nanstd(all_per_class[:, i]),
            )
        logger.info("=" * 60)

        summary_path = Path(cfg.output_dir) / "cv_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "n_folds": len(fold_results),
                    "mean_dice": float(np.mean(all_dice)),
                    "std_dice": float(np.std(all_dice)),
                    "per_fold": fold_results,
                },
                f,
                indent=2,
            )
        logger.info("Summary saved to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D mandible segmentation model")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to segmentation_results")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for runs")
    parser.add_argument("--fold", type=int, default=None, help="Run a single fold (0-indexed)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--model", type=str, default=None, choices=["unet", "segresnet"])
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--patch_size", type=int, nargs=3, default=None, help="Patch size (D H W)")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.fold is not None:
        cfg.fold = args.fold
    if args.resume:
        cfg.resume_checkpoint = args.resume
    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.model:
        cfg.model = args.model
    if args.device:
        cfg.device = args.device
    if args.patch_size:
        cfg.patch_size = tuple(args.patch_size)

    run_cross_validation(cfg)


if __name__ == "__main__":
    main()
