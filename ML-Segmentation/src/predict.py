"""
Inference script for trained mandible segmentation models.

Loads a checkpoint, runs sliding-window inference on one or more input
volumes, and saves the predicted masks as TIFF files.

Usage:
    # Single volume
    python predict.py --checkpoint runs/fold_0/best_model.pt \
                      --input /path/to/volume.tif \
                      --output ./predictions/sample_01

    # Directory of volumes (each .tif or .nrrd)
    python predict.py --checkpoint runs/fold_0/best_model.pt \
                      --input /path/to/volumes/ \
                      --output ./predictions

    # Ensemble across all CV folds (averages softmax probabilities)
    python predict.py --checkpoint runs/ \
                      --input /path/to/volume.tif \
                      --output ./predictions/sample_01 \
                      --ensemble

    # Two-stage prediction: background gate first, then foreground-only classification
    python predict.py --checkpoint runs/fold_0/best_model.pt \
                      --input /path/to/volume.tif \
                      --output ./predictions/sample_01 \
                      --staged

    # Staged + bone gate: assign bone first, then split incisor/molar by argmax
    python predict.py --checkpoint runs/fold_0/best_model.pt \
                      --input /path/to/volume.tif \
                      --output ./predictions/sample_01 \
                      --staged --bone-threshold 0.5

    # --staged composes with --ensemble and --tta
    python predict.py --checkpoint runs/ --ensemble --staged \
                      --input /path/to/volume.tif --output ./predictions/sample_01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
import torch
from monai.inferers import sliding_window_inference

from config import Config
from model import build_model, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "background", 1: "incisor", 2: "bone", 3: "molar"}


def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D volume from TIFF or NRRD."""
    path = Path(path)
    if path.suffix == ".nrrd":
        import nrrd
        data, header = nrrd.read(str(path))
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        return data
    return tifffile.imread(str(path))


def predict_volume(
    model: torch.nn.Module,
    volume: np.ndarray,
    cfg: Config,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run sliding-window inference on a single volume.

    Returns:
        (label_map, probabilities) where label_map is int32 (D, H, W)
        and probabilities is float32 (C, D, H, W).
    """
    img = volume.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    img = (img - p1) / max(float(p99 - p1), 1e-6)

    # Shape: (1, 1, D, H, W)
    tensor = torch.from_numpy(img[np.newaxis, np.newaxis]).float().to(device)

    model.eval()
    with torch.no_grad():
        output = sliding_window_inference(
            tensor,
            roi_size=cfg.patch_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=model,
            overlap=cfg.sw_overlap,
        )

    probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # (C, D, H, W)
    label_map = probs.argmax(axis=0).astype(np.int32)  # (D, H, W)
    return label_map, probs


def predict_ensemble(
    checkpoint_dir: Path,
    volume: np.ndarray,
    cfg: Config,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average softmax probabilities across all fold checkpoints in a directory.
    """
    fold_dirs = sorted(checkpoint_dir.glob("fold_*"))
    checkpoints = [d / "best_model.pt" for d in fold_dirs if (d / "best_model.pt").exists()]

    if not checkpoints:
        raise FileNotFoundError(
            f"No best_model.pt found under {checkpoint_dir}/fold_*/"
        )

    logger.info("Ensemble: found %d fold checkpoint(s)", len(checkpoints))
    accumulated_probs = None

    for ckpt_path in checkpoints:
        model = build_model(
            model_name=cfg.model,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            channels=cfg.channels,
            strides=cfg.strides,
            dropout=cfg.dropout,
        ).to(device)
        load_checkpoint(model, str(ckpt_path), device=device)

        _, probs = predict_volume(model, volume, cfg, device)
        if accumulated_probs is None:
            accumulated_probs = probs
        else:
            accumulated_probs += probs

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    accumulated_probs /= len(checkpoints)
    label_map = accumulated_probs.argmax(axis=0).astype(np.int32)
    return label_map, accumulated_probs


def predict_with_tta(
    model: torch.nn.Module,
    volume: np.ndarray,
    cfg: Config,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    8-flip test-time augmentation: predict with every axis-flip combination,
    flip the probability maps back, and return the average.
    """
    flip_combos = [
        (),
        (0,), (1,), (2,),
        (0, 1), (0, 2), (1, 2),
        (0, 1, 2),
    ]
    accumulated: np.ndarray | None = None
    for axes in flip_combos:
        vol_aug = volume
        for ax in axes:
            vol_aug = np.flip(vol_aug, axis=ax)
        vol_aug = np.ascontiguousarray(vol_aug)

        _, probs = predict_volume(model, vol_aug, cfg, device)

        # Undo the flip on the probability map (spatial dims are 1,2,3 in C,D,H,W)
        for ax in axes:
            probs = np.flip(probs, axis=ax + 1)
        probs = np.ascontiguousarray(probs)

        accumulated = probs if accumulated is None else accumulated + probs

    accumulated /= len(flip_combos)
    label_map = accumulated.argmax(axis=0).astype(np.int32)
    return label_map, accumulated


def _staged_from_probs(
    probs: np.ndarray,
    bg_threshold: float,
    bone_threshold: float | None,
) -> np.ndarray:
    """
    Apply two-stage decision logic to a (C, D, H, W) softmax probability array.

    Stage 1 — background gate:
        Voxels where bg_prob >= bg_threshold are labelled 0.  This is a hard
        veto: no foreground class can win here, so false-foreground predictions
        are suppressed without touching the interior of the anatomy.

    Stage 2 — foreground-only classification:
        Within the confirmed foreground (bg_prob < bg_threshold):

        Without bone_threshold (default):
            argmax over [incisor, bone, molar] only.  Background is no longer
            a competing hypothesis so the three foreground classes share the
            full softmax budget.

        With bone_threshold set:
            Bone is assigned first wherever bone_prob >= bone_threshold.
            The remaining non-bone foreground voxels are split between incisor
            and molar by argmax on just those two channels.  Because incisor
            and molar occupy anatomically disjoint regions of the mandible this
            binary decision is much easier than a four-way competition.

    This function is probability-source agnostic: call it on single-model,
    ensemble-averaged, or TTA-averaged probabilities.
    """
    foreground = probs[0] < bg_threshold  # (D, H, W) bool

    if bone_threshold is None:
        # Argmax over the three foreground channels — no bone gate
        fg_label = probs[1:].argmax(axis=0) + 1  # → 1, 2, or 3
        return np.where(foreground, fg_label, 0).astype(np.int32)

    # Sequential: bone first, then incisor vs molar
    is_bone     = foreground & (probs[2] >= bone_threshold)
    is_non_bone = foreground & ~is_bone

    staged = np.zeros(probs.shape[1:], dtype=np.int32)
    staged[is_bone] = 2
    # Within non-bone foreground: molar wins where molar_prob > incisor_prob
    staged[is_non_bone] = np.where(
        probs[3][is_non_bone] > probs[1][is_non_bone], 3, 1
    )
    return staged


def predict_staged(
    model: torch.nn.Module,
    volume: np.ndarray,
    cfg: Config,
    device: str,
    bg_threshold: float = 0.05,
    bone_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-model staged prediction.  Returns (staged_label_map, probs)."""
    _, probs = predict_volume(model, volume, cfg, device)
    return _staged_from_probs(probs, bg_threshold, bone_threshold), probs


def save_predictions(
    label_map: np.ndarray,
    output_dir: Path,
    save_individual: bool = True,
) -> None:
    """Save the combined label map and individual binary masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined label map
    tifffile.imwrite(
        str(output_dir / "label_map.tif"),
        label_map.astype(np.int16),
        compression="zlib",
        bigtiff=True,
    )
    logger.info("Saved label map to %s", output_dir / "label_map.tif")

    if save_individual:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        for label_val, name in LABEL_NAMES.items():
            if label_val == 0:
                continue
            mask = (label_map == label_val).astype(np.uint8)
            tifffile.imwrite(
                str(masks_dir / f"{name}_mask.tif"),
                mask,
                compression="zlib",
                bigtiff=True,
            )
            n_voxels = int(mask.sum())
            logger.info("  %s: %d voxels", name, n_voxels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt file or runs/ dir for ensemble",
    )
    parser.add_argument(
        "--input", required=True,
        help="Input volume (.tif/.nrrd) or directory",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Average predictions across all folds",
    )
    parser.add_argument(
        "--tta", action="store_true",
        help="Test-time augmentation: average over 8 axis-flip combinations",
    )
    parser.add_argument(
        "--staged", action="store_true",
        help="Two-stage prediction: background gate then foreground-only classification. "
             "Composes with --ensemble and --tta.",
    )
    parser.add_argument(
        "--bg-threshold", type=float, default=0.05, dest="bg_threshold",
        help="Staged: voxels with bg_prob >= this are background (default: 0.05)",
    )
    parser.add_argument(
        "--bone-threshold", type=float, default=None, dest="bone_threshold",
        help="Staged: assign bone first where bone_prob >= this, then split "
             "incisor/molar by argmax. Omit to use argmax over all three foreground "
             "classes instead.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument(
        "--patch_size", type=int, nargs=3, default=None,
        help="Override patch size",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.patch_size:
        cfg.patch_size = tuple(args.patch_size)
    device = cfg.resolve_device()

    # Collect input volumes
    input_path = Path(args.input)
    if input_path.is_file():
        volumes = [(input_path, Path(args.output))]
    elif input_path.is_dir():
        exts = {".tif", ".tiff", ".nrrd"}
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)
        volumes = [(f, Path(args.output) / f.stem) for f in files]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    logger.info("Processing %d volume(s) on %s", len(volumes), device)

    checkpoint_path = Path(args.checkpoint)

    # Load config from checkpoint if available
    if checkpoint_path.is_file():
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            cfg = Config.load(config_path)
            cfg.device = args.device or "auto"
            if args.patch_size:
                cfg.patch_size = tuple(args.patch_size)
            device = cfg.resolve_device()

    # Build model (for single-checkpoint mode)
    model = None
    if not args.ensemble:
        model = build_model(
            model_name=cfg.model,
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            channels=cfg.channels,
            strides=cfg.strides,
            dropout=cfg.dropout,
        ).to(device)
        load_checkpoint(model, str(checkpoint_path), device=device)

    for vol_path, out_dir in volumes:
        logger.info("Predicting %s ...", vol_path.name)
        volume = load_volume(vol_path)
        logger.info("  Volume shape: %s, dtype: %s", volume.shape, volume.dtype)

        # Step 1: compute probabilities via chosen method
        if args.ensemble:
            label_map, probs = predict_ensemble(checkpoint_path, volume, cfg, device)
        elif args.tta:
            label_map, probs = predict_with_tta(model, volume, cfg, device)
        else:
            label_map, probs = predict_volume(model, volume, cfg, device)

        # Step 2: optionally replace argmax label map with staged decision
        if args.staged:
            label_map = _staged_from_probs(probs, args.bg_threshold, args.bone_threshold)
            logger.info(
                "  Staged (bg_thr=%.2f%s)",
                args.bg_threshold,
                f"  bone_thr={args.bone_threshold}" if args.bone_threshold else "",
            )

        save_predictions(label_map, out_dir)
        logger.info("  Done -> %s", out_dir)


if __name__ == "__main__":
    main()
