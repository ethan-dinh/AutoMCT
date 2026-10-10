"""
Command line: ``python -m super_resolution {check,train,upscale,segment}``.

Run from the repository root (the Docker image's entrypoint does this for you).
"""

from __future__ import annotations

import argparse
import logging
import os

from .classical import add_segment_mandible_to_path
from .config import TrainConfig

add_segment_mandible_to_path()
from logging_setup import setup_logging  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order

logger = logging.getLogger(__name__)


def _triple(value: str, what: str, cast):
    parts = [p.strip() for p in value.replace(" ", ",").split(",") if p.strip()]
    try:
        numbers = [cast(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{what}: cannot parse {value!r}") from exc
    if len(numbers) == 1:
        numbers *= 3
    if len(numbers) != 3 or any(n <= 0 for n in numbers):
        raise argparse.ArgumentTypeError(f"{what}: expected 1 or 3 positive values, got {value!r}")
    return tuple(numbers)


def parse_scale(value: str) -> tuple[int, int, int]:
    """``2`` -> (2, 2, 2); ``1,2,2`` -> (1, 2, 2) as (z, y, x)."""
    return _triple(value, "--scale", int)


def parse_voxel_size(value: str) -> tuple[float, float, float]:
    """``0.008`` -> isotropic; ``dz,dy,dx`` otherwise (mm), like segment_mandible."""
    return _triple(value, "--voxel-size", float)


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--device", default="auto",
        help="auto (CUDA, then Apple MPS, then CPU), or cuda, cuda:1, mps, cpu.",
    )
    common.add_argument("--verbose", action="store_true", help="Log at DEBUG.")
    common.add_argument("--quiet", "-q", action="store_true", help="Log warnings and errors only.")
    common.add_argument("--no-color", dest="no_color", action="store_true", help="Plain console output.")
    common.add_argument(
        "--log", "-l", action="store_true",
        help="Also write a timestamped log file into the output directory.",
    )
    return common


def _input_parser() -> argparse.ArgumentParser:
    inputs = argparse.ArgumentParser(add_help=False)
    group = inputs.add_argument_group("input")
    group.add_argument(
        "--voxel-size", dest="voxel_size", type=parse_voxel_size, default=None,
        help="Override the voxel size in mm: one value, or dz,dy,dx. Needed for BMP "
             "stacks, which carry none.",
    )
    group.add_argument(
        "--apply-dicom-rescale", dest="apply_dicom_rescale", action="store_true",
        help="DICOM only: apply RescaleSlope/Intercept (as segment_mandible's flag).",
    )
    return inputs


def _training_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    training = argparse.ArgumentParser(add_help=False)
    group = training.add_argument_group(
        "model and training",
        "Architecture and schedule. For `segment` these apply only when no --model is given.",
    )
    group.add_argument("--scale", type=parse_scale, default=defaults.scale,
                       help="Upscaling factor: one integer, or z,y,x (default 2). Each "
                            "factor multiplies the voxel count, so 2 means 8x the voxels.")
    group.add_argument("--features", type=int, default=defaults.features, help="Generator width (default %(default)s).")
    group.add_argument("--growth", type=int, default=defaults.growth, help="Dense-block growth (default %(default)s).")
    group.add_argument("--blocks", type=int, default=defaults.blocks, help="RRDB blocks (default %(default)s).")
    group.add_argument("--no-consistency", dest="consistency", action="store_false",
                       help="Disable the data-consistency projection (not recommended).")
    group.add_argument("--lr-patch", type=int, default=defaults.lr_patch,
                       help="Training patch edge at the input resolution (default %(default)s).")
    group.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Default %(default)s.")
    group.add_argument("--pretrain-iters", type=int, default=defaults.pretrain_iters,
                       help="Fidelity (L1) iterations (default %(default)s).")
    group.add_argument("--gan-iters", type=int, default=defaults.gan_iters,
                       help="Adversarial iterations after that; 0 for none (default %(default)s).")
    group.add_argument("--lr", dest="lr_generator", type=float, default=defaults.lr_generator,
                       help="Generator learning rate (default %(default)s).")
    group.add_argument("--adv-weight", dest="adversarial_weight", type=float,
                       default=defaults.adversarial_weight, help="Adversarial loss weight (default %(default)s).")
    group.add_argument("--grad-weight", dest="gradient_weight", type=float,
                       default=defaults.gradient_weight, help="Edge (gradient) loss weight (default %(default)s).")
    group.add_argument("--val-every", type=int, default=defaults.val_every, help="Default %(default)s.")
    group.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every, help="Default %(default)s.")
    group.add_argument("--seed", type=int, default=defaults.seed, help="Default %(default)s.")
    return training


def _tiling_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("inference")
    group.add_argument("--tile", type=int, default=64,
                       help="Tile edge in input voxels; lower it if the GPU runs out of memory (default 64).")
    group.add_argument("--halo", type=int, default=10,
                       help="Context voxels around each tile, discarded after (default 10).")
    group.add_argument("--no-skip-air", dest="skip_air", action="store_false",
                       help="Run the network on all-air tiles too, instead of replicating them.")
    group.add_argument("--crop", action="store_true",
                       help="Only upscale the specimen's bounding box. Saves time, disk and RAM, "
                            "but changes the intensity histogram segment_mandible thresholds from.")
    group.add_argument("--crop-margin", type=int, default=24,
                       help="Voxels of air kept around the specimen with --crop (default 24).")


def build_parser() -> argparse.ArgumentParser:
    common, inputs, training = _common_parser(), _input_parser(), _training_parser()
    parser = argparse.ArgumentParser(
        prog="python -m super_resolution",
        description="Generative 3-D super-resolution for segmenting micro-CT scans on a finer grid.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", parents=[common], help="Verify the install and the GPU.")
    check.add_argument("--work-dir", default=".", help="Report free disk space here.")

    train = sub.add_parser(
        "train", parents=[common, inputs, training],
        help="Train a model on one or more scans (self-supervised).",
    )
    train.add_argument("--input", "-i", nargs="+", required=True,
                       help="Scans to learn from (.nrrd/.tif file, or a DICOM/BMP folder). "
                            "Several scans make one shared model; all are held in RAM.")
    train.add_argument("--out", "-o", required=True, help="Directory for model.pt and training records.")
    train.add_argument("--resume", action="store_true", help="Continue from --out/last_state.pt.")

    upscale = sub.add_parser(
        "upscale", parents=[common, inputs],
        help="Super-resolve one scan to an NRRD (for inspection).",
    )
    upscale.add_argument("--input", "-i", required=True, help="Scan to upscale.")
    upscale.add_argument("--model", "-m", required=True, help="model.pt from `train`.")
    upscale.add_argument("--out", "-o", required=True, help="Output .nrrd path (uncompressed; scale^3 x input).")
    _tiling_arguments(upscale)

    segment = sub.add_parser(
        "segment", parents=[common, inputs, training],
        help="Upscale, segment with segment_mandible, and apply the masks to the original.",
    )
    segment.add_argument("--input", "-i", required=True,
                         help="A scan, or a folder of samples (same rules as segment_mandible).")
    segment.add_argument("--model", "-m", default=None,
                         help="model.pt from `train`. Omitted: train one on each sample first.")
    segment.add_argument("--out", "-o", default="segmentation_results_sr",
                         help="Output root (default ./segmentation_results_sr).")
    segment.add_argument("--work-dir", default=os.environ.get("SR_WORK_DIR"),
                         help="Where the temporary upscaled volume goes (default <out>/_sr_work, "
                              "or $SR_WORK_DIR). Needs scale^3 x the scan's size free.")
    _tiling_arguments(segment)
    group = segment.add_argument_group("masks and outputs")
    group.add_argument("--min-fraction", type=float, default=0.5,
                       help="Share of an original voxel that must be tissue on the fine grid "
                            "for it to be kept (default 0.5).")
    group.add_argument("--format", "-f", dest="output_format", choices=("tif", "nrrd"), default="tif",
                       help="Container for outputs, as segment_mandible (default tif).")
    group.add_argument("--save-fractions", action="store_true",
                       help="Also save, per structure, how many fine voxels of each original voxel it covers.")
    group.add_argument("--keep-upscaled", action="store_true", help="Keep the upscaled volume.")
    group.add_argument("--overwrite", action="store_true",
                       help="Redo samples that already have results (and retrain their models).")
    group.add_argument("--resume-training", action="store_true",
                       help="Continue an interrupted per-sample training run.")
    group = segment.add_argument_group("segment_mandible (passed through)")
    group.add_argument("--incisor-margin-shrink", dest="incisor_margin_shrink", type=float, default=0.0,
                       help="Trim this many mm off the incisor surface (default 0).")
    group.add_argument("--workers", type=int, default=None, help="Threads for the margin shrink.")
    group.add_argument("--strict", action="store_true", help="Abort a sample when its checks fail.")
    group.add_argument("--cache-dir", default=None, help="segment_mandible's filter cache directory.")
    group.add_argument("--validation-report", default=None,
                       help="Path for the JSON check report (default <out>/validation_report.json).")
    return parser


def _train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        scale=args.scale, features=args.features, growth=args.growth, blocks=args.blocks,
        consistency=args.consistency, lr_patch=args.lr_patch, batch_size=args.batch_size,
        pretrain_iters=args.pretrain_iters, gan_iters=args.gan_iters,
        lr_generator=args.lr_generator, adversarial_weight=args.adversarial_weight,
        gradient_weight=args.gradient_weight, val_every=args.val_every,
        checkpoint_every=args.checkpoint_every, seed=args.seed,
    )


def _run_train(args: argparse.Namespace) -> int:
    from .model import select_device  # pylint: disable=import-outside-toplevel
    from .train import train_model  # pylint: disable=import-outside-toplevel
    from .volume_io import load_scan  # pylint: disable=import-outside-toplevel

    config = _train_config(args)
    device = select_device(args.device)
    scans = [
        load_scan(path, voxel_size=args.voxel_size, apply_dicom_rescale=args.apply_dicom_rescale)
        for path in args.input
    ]
    model_path = train_model(scans, args.out, config, device, resume=args.resume)
    print(model_path)
    return 0


def _run_upscale(args: argparse.Namespace) -> int:
    # pylint: disable=import-outside-toplevel
    from .model import describe_device, load_generator, select_device
    from .upscale import check_training_resolution, upscale_to_nrrd
    from .volume_io import load_scan, specimen_bounding_box, tissue_threshold

    device = select_device(args.device)
    scan = load_scan(args.input, voxel_size=args.voxel_size, apply_dicom_rescale=args.apply_dicom_rescale)
    data, geometry = scan.data, scan.geometry
    if args.crop:
        box = specimen_bounding_box(data, tissue_threshold(data), args.crop_margin)
        start = tuple(s.start or 0 for s in box)
        data, geometry = data[box], geometry.shifted(start)
        logger.info("Cropped to the specimen: %s", tuple((s.start, s.stop) for s in box))
    generator, checkpoint = load_generator(args.model, device)
    if scan.spacing_from_file:
        check_training_resolution(checkpoint, geometry.spacing)
    logger.info("Upscaling %s on %s", scan.name, describe_device(device))
    upscale_to_nrrd(
        data, geometry if scan.spacing_from_file else None, generator, args.out, device=device,
        tile=args.tile, halo=args.halo, skip_air=args.skip_air,
    )
    print(args.out)
    return 0


def _run_segment(args: argparse.Namespace) -> int:
    from .sr_segment import SegmentOptions, segment_samples  # pylint: disable=import-outside-toplevel

    if not 0.0 < args.min_fraction <= 1.0:
        raise SystemExit("--min-fraction must be in (0, 1]")
    options = SegmentOptions(
        model=args.model, out_root=args.out, work_dir=args.work_dir, device=args.device,
        tile=args.tile, halo=args.halo, skip_air=args.skip_air, crop=args.crop,
        crop_margin=args.crop_margin, min_fraction=args.min_fraction,
        output_format=args.output_format, save_fractions=args.save_fractions,
        keep_upscaled=args.keep_upscaled, overwrite=args.overwrite,
        voxel_size=args.voxel_size, apply_dicom_rescale=args.apply_dicom_rescale,
        incisor_margin_shrink_mm=args.incisor_margin_shrink, workers=args.workers,
        strict=args.strict, cache_dir=args.cache_dir, train=_train_config(args),
        resume_training=args.resume_training,
    )
    return segment_samples(args.input, options, report_path=args.validation_report)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose and args.quiet:
        raise SystemExit("--verbose and --quiet are mutually exclusive")

    log_dir = None
    if args.log:
        if args.command == "upscale":
            log_dir = os.path.dirname(os.path.abspath(args.out))
        elif args.command in ("train", "segment"):
            log_dir = os.path.abspath(args.out)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, verbose=args.verbose, quiet=args.quiet, color=not args.no_color)

    if args.command == "check":
        from .check import run_check  # pylint: disable=import-outside-toplevel

        return run_check(args.device, args.work_dir)
    if args.command == "train":
        return _run_train(args)
    if args.command == "upscale":
        return _run_upscale(args)
    return _run_segment(args)
