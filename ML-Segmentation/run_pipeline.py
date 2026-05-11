#!/usr/bin/env python3
"""
Mandible segmentation training pipeline with Rich TUI.

Usage
-----
    python run_pipeline.py                     # all 9 folds, default settings
    python run_pipeline.py --folds 0,1,2       # specific folds only
    python run_pipeline.py --epochs 200        # override max epochs
    python run_pipeline.py --output ./runs2    # different output directory

Skips folds that already have a completed history.json (no pending checkpoint).
Resumes folds that have a last_checkpoint.pt (interrupted mid-run).
Writes runs/pipeline_status.json each epoch so the notebook can monitor live.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from monai.data import ThreadDataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from dataset import build_datasets, discover_samples, LoadAndCombineMasksd, warm_disk_cache
from model import build_model

# ──────────────────────────────────────────────────────────────────────────────

console = Console()
CLASS_NAMES = ["incisor", "bone", "molar"]


def _fmt_time(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h  = td.days * 24 + td.seconds // 3600
    m  = (td.seconds % 3600) // 60
    s  = td.seconds % 60
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mandible segmentation training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       default="./training", metavar="DIR")
    p.add_argument("--output",     default="./runs",                  metavar="DIR")
    p.add_argument("--folds",      default=None,  metavar="0,1,2",
                   help="Comma-separated fold indices to run (default: all)")
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch",      type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=50)
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader worker threads per fold")
    p.add_argument("--val-every",  type=int,   default=5,  dest="val_every",
                   help="Validate every N epochs")
    p.add_argument("--ckpt-every", type=int,   default=5,  dest="ckpt_every",
                   help="Save checkpoint every N epochs")
    p.add_argument("--cache-dir",  default=None, metavar="DIR", dest="cache_dir",
                   help="Disk cache for preprocessed samples (default: <output>/.det_cache)")
    p.add_argument("--extend", action="store_true",
                   help="Resume completed folds from best_model.pt at LR/10 for more epochs")
    return p.parse_args()


def _build_cfg(args: argparse.Namespace) -> Config:
    return Config(
        data_dir=args.data, output_dir=args.output,
        model="unet", channels=(32, 64, 128, 256), dropout=0.2,
        max_epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, weight_decay=1e-5,
        scheduler="cosine", early_stop_patience=args.patience,
        patch_size=(96, 96, 96), samples_per_volume=24,
        pos_sample_ratio=0.7, n_folds=9, device="auto",
        sw_batch_size=32, sw_overlap=0.25,
    )


# ── Status file (read by notebook monitor) ─────────────────────────────────────

def _write_status(path: Path, data: dict) -> None:
    """Atomic write so the notebook never reads a partially-written file."""
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    data["updated_at"] = datetime.now().isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── Rich helpers ───────────────────────────────────────────────────────────────

def _print_val_row(
    progress: Progress, fold_idx: int, epoch: int,
    loss: float, val_dice: float, per_class: list,
    best_dice: float, epoch_s: float, val_s: float,
) -> None:
    star = "[bold yellow]★[/] " if val_dice >= best_dice else "  "
    progress.console.print(
        f"  {star}"
        f"[dim]fold {fold_idx}  ep {epoch:>3d}[/]  "
        f"loss=[yellow]{loss:.4f}[/]  "
        f"dice=[bold green]{val_dice:.4f}[/]  "
        f"inc=[#ff6b6b]{per_class[0]:.3f}[/]  "
        f"bone=[#4da6ff]{per_class[1]:.3f}[/]  "
        f"molar=[#69db7c]{per_class[2]:.3f}[/]  "
        f"[dim]ep={epoch_s:.1f}s  val={val_s:.1f}s[/]"
    )


def _print_fold_summary(
    progress: Progress, fold_idx: int, val_name: str,
    best_dice: float, per_class: list, elapsed_s: float,
    epochs_run: int,
) -> None:
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold dim", no_wrap=True)
    t.add_column()
    t.add_row("Fold",    f"[cyan]{fold_idx}[/]  ·  val=[yellow]{val_name}[/]")
    t.add_row("Epochs",  str(epochs_run))
    t.add_row("Best",    f"[bold green]{best_dice:.4f}[/]")
    t.add_row("Classes", (
        f"incisor=[#ff6b6b]{per_class[0]:.3f}[/]  "
        f"bone=[#4da6ff]{per_class[1]:.3f}[/]  "
        f"molar=[#69db7c]{per_class[2]:.3f}[/]"
    ))
    t.add_row("Time",    _fmt_time(elapsed_s))
    progress.console.print(
        Panel(t, title=f"[bold green]Fold {fold_idx} complete",
              border_style="green", expand=False)
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx:      int,
    train_samples: list,
    val_samples:   list,
    cfg:           Config,
    cache_dir:     Path,
    progress:      Progress,
    status_path:   Path,
    status:        dict,
    log:           logging.Logger,
    val_every:     int = 5,
    ckpt_every:    int = 5,
    num_workers:   int = 4,
    extend:        bool = False,
) -> dict:
    """Train one fold. Returns {fold, best_dice, per_class, time_s, epochs_run}."""
    torch.backends.cudnn.benchmark = True
    device   = cfg.resolve_device()
    fold_dir = Path(cfg.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    val_name = val_samples[0]["name"]

    # ── Dataset / loader ─────────────────────────────────────────────────────
    log.info("Fold %d: loading %d training samples into RAM...", fold_idx, len(train_samples))
    ds_t0    = time.perf_counter()
    train_ds, _ = build_datasets(
        train_samples, val_samples,
        patch_size=cfg.patch_size,
        samples_per_volume=cfg.samples_per_volume,
        pos_ratio=cfg.pos_sample_ratio,
        cache_dir=cache_dir,
    )
    train_loader = ThreadDataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, buffer_size=num_workers * 2,
        pin_memory=True,
    )
    log.info("Fold %d: dataset ready  items=%d  load=%.1fs",
             fold_idx, len(train_ds), time.perf_counter() - ds_t0)

    # ── Model / optimiser ─────────────────────────────────────────────────────
    model     = build_model(
        model_name=cfg.model, in_channels=cfg.in_channels,
        num_classes=cfg.num_classes, channels=cfg.channels,
        strides=cfg.strides, dropout=cfg.dropout,
    ).to(device)
    ce_weight = torch.tensor([0.5, 1.0, 1.0, 2.0], device=device)  # bg, incisor, bone, molar
    dice_fn   = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    ce_fn     = torch.nn.CrossEntropyLoss(weight=ce_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    scaler    = torch.amp.GradScaler("cuda")
    metric    = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred  = AsDiscrete(argmax=True, to_onehot=cfg.num_classes)
    post_label = AsDiscrete(to_onehot=cfg.num_classes)
    loader_fn  = LoadAndCombineMasksd()

    best_dice         = -1.0
    epochs_no_improve = 0
    start_epoch       = 0
    per_class         = [0.0, 0.0, 0.0]
    history           = {"epochs": [], "train_loss": [], "val_dice": [], "per_class": []}

    resume_path     = fold_dir / "last_checkpoint.pt"
    best_model_path = fold_dir / "best_model.pt"
    hist_path       = fold_dir / "history.json"

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch       = ckpt["epoch"] + 1
        best_dice         = ckpt["best_dice"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        if hist_path.exists():
            with open(hist_path) as fh:
                history = json.load(fh)
        progress.console.print(
            f"  [yellow]↺[/]  Fold {fold_idx}: resuming from epoch [bold]{start_epoch}[/]  "
            f"best dice so far=[bold green]{best_dice:.4f}[/]"
        )
        log.info("Fold %d: resuming from ep %d  best=%.4f", fold_idx, start_epoch, best_dice)
    elif extend and best_model_path.exists():
        # Extend a completed fold: load weights only, fresh optimizer at LR/10
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch       = ckpt["epoch"] + 1
        best_dice         = ckpt["best_dice"]
        epochs_no_improve = 0
        extend_lr = cfg.learning_rate * 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = extend_lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(cfg.max_epochs - start_epoch, 1)
        )
        if hist_path.exists():
            with open(hist_path) as fh:
                history = json.load(fh)
        progress.console.print(
            f"  [magenta]↗[/]  Fold {fold_idx}: extending from epoch [bold]{start_epoch}[/]  "
            f"best=[bold green]{best_dice:.4f}[/]  lr={extend_lr:.1e}"
        )
        log.info("Fold %d: extending from ep %d  best=%.4f  lr=%.1e",
                 fold_idx, start_epoch, best_dice, extend_lr)
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        progress.console.print(
            f"  [bold cyan]▶[/]  Fold {fold_idx}  "
            f"train={len(train_samples)} samples  val=[yellow]{val_name}[/]  "
            f"params={n_params:,}"
        )
        log.info("Fold %d: starting fresh | val=%s | params=%s",
                 fold_idx, val_name, f"{n_params:,}")

    # ── Epoch progress task ───────────────────────────────────────────────────
    epoch_task = progress.add_task(
        f"  [yellow]fold {fold_idx}[/] epochs",
        total=cfg.max_epochs,
        completed=start_epoch,
    )

    fold_t0    = time.perf_counter()
    last_epoch = start_epoch
    log.info("Fold %d: entering training loop  start_epoch=%d", fold_idx, start_epoch)

    for epoch in range(start_epoch, cfg.max_epochs):
        epoch_t0        = time.perf_counter()
        model.train()
        epoch_loss      = 0.0
        _batch_wait_s   = 0.0  # cumulative time waiting for data
        _batch_gpu_s    = 0.0  # cumulative time on GPU
        _batch_end      = epoch_t0  # timestamp after previous batch finished

        for batch_idx, batch in enumerate(train_loader):
            _t_got_batch = time.perf_counter()
            _batch_wait_s += _t_got_batch - _batch_end

            if epoch == start_epoch and batch_idx == 0:
                log.info("Fold %d ep %d: first batch  wait=%.1fs  shape=%s",
                         fold_idx, epoch + 1,
                         _t_got_batch - epoch_t0,
                         list(batch["image"].shape))

            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out  = model(imgs)
                loss = dice_fn(out, lbls) + ce_fn(out, lbls.squeeze(1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

            _batch_end   = time.perf_counter()
            _batch_gpu_s += _batch_end - _t_got_batch

        avg_loss = epoch_loss / max(len(train_loader), 1)
        epoch_s  = time.perf_counter() - epoch_t0
        lr_now   = optimizer.param_groups[0]["lr"]
        val_dice = 0.0
        log.info("Fold %d ep %d: loss=%.4f  epoch=%.1fs  data_wait=%.1fs  gpu=%.1fs",
                 fold_idx, epoch + 1, avg_loss, epoch_s, _batch_wait_s, _batch_gpu_s)

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0:
            val_t0 = time.perf_counter()
            model.eval()
            metric.reset()
            with torch.no_grad():
                for sample in val_samples:
                    d   = loader_fn(sample)
                    img = torch.from_numpy(d["image"]).unsqueeze(0).float().to(device)
                    lbl = torch.from_numpy(d["label"]).unsqueeze(0).long().to(device)
                    with torch.amp.autocast("cuda"):
                        out = sliding_window_inference(
                            img, roi_size=cfg.patch_size,
                            sw_batch_size=cfg.sw_batch_size,
                            predictor=model, overlap=cfg.sw_overlap,
                        )
                    metric(
                        y_pred=[post_pred(i) for i in decollate_batch(out)],
                        y=[post_label(i) for i in decollate_batch(lbl)],
                    )
            per_sample = metric.aggregate()
            per_class  = (per_sample.nanmean(dim=0) if per_sample.dim() > 1
                          else per_sample).cpu().tolist()
            val_dice   = float(np.nanmean(per_class))
            val_s      = time.perf_counter() - val_t0

            if val_dice > best_dice:
                best_dice         = val_dice
                epochs_no_improve = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "best_dice": best_dice, "per_class_dice": per_class,
                     "fold": fold_idx},
                    fold_dir / "best_model.pt",
                )
            else:
                epochs_no_improve += val_every

            _print_val_row(progress, fold_idx, epoch + 1, avg_loss,
                           val_dice, per_class, best_dice, epoch_s, val_s)
            log.info(
                "fold=%d ep=%d loss=%.4f dice=%.4f inc=%.3f bone=%.3f molar=%.3f "
                "ep_s=%.1f val_s=%.1f",
                fold_idx, epoch + 1, avg_loss, val_dice,
                per_class[0], per_class[1], per_class[2], epoch_s, val_s,
            )

        history["epochs"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)
        history["per_class"].append(per_class)

        scheduler.step()
        progress.advance(epoch_task)
        last_epoch = epoch

        # ── Periodic checkpoint + status ─────────────────────────────────────
        if (epoch + 1) % ckpt_every == 0:
            fold_elapsed = time.perf_counter() - fold_t0
            done_eps     = epoch - start_epoch + 1
            avg_ep_s     = fold_elapsed / max(done_eps, 1)
            remaining    = cfg.max_epochs - epoch - 1

            status.update({
                "current_fold":   fold_idx,
                "current_epoch":  epoch + 1,
                "total_epochs":   cfg.max_epochs,
                "best_dice_fold": round(best_dice, 4),
                "last_val_dice":  round(val_dice, 4) if val_dice else None,
                "last_per_class": {n: round(v, 4) for n, v in zip(CLASS_NAMES, per_class)},
                "timing": {
                    **status.get("timing", {}),
                    "fold_elapsed_s":  round(fold_elapsed, 1),
                    "avg_epoch_s":     round(avg_ep_s, 2),
                    "eta_fold_s":      round(remaining * avg_ep_s, 0),
                    "elapsed_total_s": round(time.perf_counter() - status["_t0"], 1),
                },
            })
            _write_status(status_path, status)

            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict(),
                 "scaler_state_dict": scaler.state_dict(),
                 "best_dice": best_dice, "epochs_no_improve": epochs_no_improve,
                 "fold": fold_idx},
                resume_path,
            )
            with open(hist_path, "w") as fh:
                json.dump(history, fh)

        if epochs_no_improve >= cfg.early_stop_patience:
            progress.console.print(
                f"  [dim]Fold {fold_idx}: early stop at epoch {epoch + 1}  "
                f"(no improvement for {cfg.early_stop_patience} epochs)[/]"
            )
            log.info("Fold %d: early stop at ep %d  best=%.4f",
                     fold_idx, epoch + 1, best_dice)
            break

    # ── Final save ────────────────────────────────────────────────────────────
    with open(hist_path, "w") as fh:
        json.dump(history, fh)
    resume_path.unlink(missing_ok=True)

    fold_s     = time.perf_counter() - fold_t0
    epochs_run = last_epoch - start_epoch + 1
    _print_fold_summary(progress, fold_idx, val_name, best_dice, per_class,
                        fold_s, epochs_run)
    progress.remove_task(epoch_task)

    del model, train_ds, train_loader, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fold":       fold_idx,
        "best_dice":  best_dice,
        "per_class":  per_class,
        "time_s":     fold_s,
        "epochs_run": epochs_run,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = _build_cfg(args)
    out  = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_path = out / "pipeline.log"
    file_h   = logging.FileHandler(log_path, encoding="utf-8")
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S"
    ))
    rich_h = RichHandler(console=console, show_path=False, rich_tracebacks=True,
                         markup=True)
    logging.basicConfig(level=logging.INFO, handlers=[file_h, rich_h])
    log = logging.getLogger("pipeline")
    log.info("Pipeline starting  |  output=%s", out)
    log.info("Log file: %s", log_path)

    # ── Discover samples ──────────────────────────────────────────────────────
    samples = discover_samples(cfg.data_dir)
    if not samples:
        log.error("No samples found in %s — check --data path.", cfg.data_dir)
        raise SystemExit(1)
    log.info("Found %d samples: %s", len(samples), [s["name"] for s in samples])

    folds_to_run = (
        [int(f.strip()) for f in args.folds.split(",")]
        if args.folds else list(range(len(samples)))
    )

    # ── Skip completed folds ──────────────────────────────────────────────────
    to_run, to_skip = [], []
    for fi in folds_to_run:
        fold_dir   = out / f"fold_{fi}"
        has_hist   = (fold_dir / "history.json").exists()
        has_resume = (fold_dir / "last_checkpoint.pt").exists()
        has_best   = (fold_dir / "best_model.pt").exists()
        if has_hist and not has_resume:
            if args.extend and has_best:
                to_run.append(fi)   # extend from best_model.pt
            else:
                to_skip.append(fi)
        else:
            to_run.append(fi)

    if to_skip:
        log.info("Skipping already-complete folds: %s", to_skip)
    log.info("Folds to train: %s  (%d total)", to_run, len(to_run))

    if not to_run:
        console.print("\n[bold green]All requested folds are already complete.[/]")
        console.print(f"Results are in [cyan]{out}[/]. Load them in the notebook.")
        return

    # ── Status init ───────────────────────────────────────────────────────────
    status_path = out / "pipeline_status.json"
    t0          = time.perf_counter()
    status: dict = {
        "_t0":            t0,
        "status":         "running",
        "pid":            os.getpid(),
        "started_at":     datetime.now().isoformat(),
        "folds_planned":  folds_to_run,
        "folds_complete": list(to_skip),
        "folds_skipped":  list(to_skip),
        "current_fold":   None,
        "current_epoch":  0,
        "total_epochs":   cfg.max_epochs,
        "timing":         {},
    }
    _write_status(status_path, status)

    # ── Disk cache warmup ─────────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else out / ".det_cache"
    console.rule("[bold cyan]Warming disk cache")
    load_t0 = time.perf_counter()

    progress_load = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress_load:
        load_task = progress_load.add_task("Caching samples", total=len(samples))
        warm_disk_cache(
            samples, cfg.patch_size, cache_dir,
            num_workers=args.workers,
            on_done=lambda _: progress_load.advance(load_task),
        )

    load_s = time.perf_counter() - load_t0
    status["timing"]["preload_s"] = round(load_s, 1)
    log.info("Disk cache ready for %d samples  |  time=%s", len(samples), _fmt_time(load_s))
    console.rule()

    # ── Training ──────────────────────────────────────────────────────────────
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=35),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    fold_results: dict[int, dict] = {}

    with progress:
        fold_task = progress.add_task(
            "[bold blue]Overall — folds", total=len(to_run)
        )

        for fold_idx in to_run:
            val_idx   = [fold_idx % len(samples)]
            train_idx = [i for i in range(len(samples)) if i not in val_idx]

            progress.console.rule(
                f"[bold cyan]Fold {fold_idx}  ·  val=[yellow]{samples[val_idx[0]]['name']}[/yellow][/bold cyan]"
            )

            status["current_fold"]   = fold_idx
            status["folds_complete"] = list(to_skip) + [f for f in to_run if f < fold_idx]
            _write_status(status_path, status)

            result = train_fold(
                fold_idx      = fold_idx,
                train_samples = [samples[i] for i in train_idx],
                val_samples   = [samples[i] for i in val_idx],
                cfg           = cfg,
                cache_dir     = cache_dir,
                progress      = progress,
                status_path   = status_path,
                status        = status,
                log           = log,
                val_every     = args.val_every,
                ckpt_every    = args.ckpt_every,
                num_workers   = args.workers,
                extend        = args.extend,
            )
            fold_results[fold_idx] = result
            progress.advance(fold_task)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_s = time.perf_counter() - t0
    console.rule("[bold green]Training Complete")

    tbl = Table(
        title="Cross-Validation Summary",
        box=box.ROUNDED, header_style="bold cyan", show_lines=False,
    )
    tbl.add_column("Fold",    justify="center", style="bold")
    tbl.add_column("Epochs",  justify="right",  style="dim")
    tbl.add_column("Mean",    justify="right",  style="bold green")
    tbl.add_column("Incisor", justify="right",  style="#ff6b6b")
    tbl.add_column("Bone",    justify="right",  style="#4da6ff")
    tbl.add_column("Molar",   justify="right",  style="#69db7c")
    tbl.add_column("Time",    justify="right",  style="dim")

    all_dices, all_pc = [], []
    for fi, res in sorted(fold_results.items()):
        pc = res["per_class"]
        all_dices.append(res["best_dice"])
        all_pc.append(pc)
        tbl.add_row(
            str(fi), str(res["epochs_run"]),
            f"{res['best_dice']:.4f}",
            f"{pc[0]:.3f}", f"{pc[1]:.3f}", f"{pc[2]:.3f}",
            _fmt_time(res["time_s"]),
        )

    pc_arr = np.array(all_pc) if all_pc else np.zeros((1, 3))
    tbl.add_section()
    tbl.add_row(
        "mean", "—",
        f"{np.mean(all_dices):.4f}",
        f"{pc_arr[:, 0].mean():.3f}",
        f"{pc_arr[:, 1].mean():.3f}",
        f"{pc_arr[:, 2].mean():.3f}",
        _fmt_time(total_s),
        style="bold",
    )
    console.print(tbl)
    console.print(f"\n[dim]Log: {log_path}[/]")

    status["status"]             = "complete"
    status["folds_complete"]     = sorted(set(list(fold_results.keys()) + list(to_skip)))
    status["timing"]["total_s"]  = round(total_s, 1)
    _write_status(status_path, status)
    log.info("Pipeline complete  total=%s  mean_dice=%.4f",
             _fmt_time(total_s), np.mean(all_dices) if all_dices else 0.0)


if __name__ == "__main__":
    main()
