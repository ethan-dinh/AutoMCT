"""
Plot the intensity/density histogram of a converted microCT volume.

Usage:
    python plot_density_histogram.py <nrrd_file> [options]

Reads the NRRD memory-mapped and histograms a strided subsample, so a
multi-gigabyte volume costs a few hundred MB of RAM rather than the full
array. The subsample is uniform across the volume, so the shape of the
distribution is preserved; only the counts are scaled down.

What the peaks usually are, on these scans:

    - A single very tall spike, often a third of all voxels, is the air band
      after --noise-floor flattened it to one constant. It is not tissue and
      not an artifact; it is that stage working. Its location is the fill
      value, and <stem>_meta.json records the fraction cleared under
      "_noise_floor".
    - A broad low peak above that is soft tissue, pulp and mounting medium.
    - The high shoulder is mineralised tissue: dentine, then enamel, which on
      a rodent incisor is the densest real material present.
    - Anything pinned to the int16 rails (-32768 / 32767) is neither: those are
      reconstruction artifacts (rotation-axis ringing and streaks), and the ISQ
      reader replaces them on read unless --no-trim-saturated was passed.

With --mu-scaling (or a value found in <stem>_meta.json) the x axis is
converted from stored counts to linear attenuation in 1/cm, which is what makes
the values comparable between scans of differing gain.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

INT16 = np.iinfo(np.int16)


def _load_subsample(path: Path, max_dim: int) -> tuple[np.ndarray, dict]:
    """Memory-map *path* and return a strided subsample plus its NRRD header."""
    import nrrd

    header = nrrd.read_header(str(path))
    shape = tuple(int(n) for n in header["sizes"])

    # nrrd.read() would materialise the whole volume. For an uncompressed raw
    # NRRD the data block is a flat array at a fixed offset, so it can be
    # mapped instead; compressed files have to go through the reader.
    encoding = header.get("encoding", "raw")
    if encoding == "raw" and header.get("data file") is None:
        with open(path, "rb") as fh:
            blob = fh.read()
        offset = blob.index(b"\n\n") + 2
        dtype = np.dtype(_nrrd_dtype(header)).newbyteorder(
            "<" if "little" in header.get("endian", "little") else ">"
        )
        volume = np.memmap(
            path, dtype=dtype, mode="r", offset=offset, shape=shape[::-1]
        )
    else:
        volume = nrrd.read(str(path))[0]

    steps = tuple(slice(None, None, max(1, n // max_dim)) for n in volume.shape)
    return np.asarray(volume[steps]), header


def _nrrd_dtype(header: dict) -> str:
    """Map an NRRD 'type' field to a numpy dtype string."""
    mapping = {
        "signed char": "i1", "int8": "i1", "int8_t": "i1",
        "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
        "short": "i2", "short int": "i2", "signed short": "i2", "int16": "i2",
        "ushort": "u2", "unsigned short": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        "float": "f4", "double": "f8",
    }
    key = str(header["type"]).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported NRRD type: {header['type']!r}")
    return mapping[key]


def _read_meta(nrrd_path: Path) -> dict:
    """Load <stem>_meta.json beside *nrrd_path*, if the converter wrote one."""
    for candidate in (
        nrrd_path.with_name(nrrd_path.stem + "_meta.json"),
        nrrd_path.with_name(nrrd_path.stem.replace("_prefiltered", "") + "_meta.json"),
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def find_pad_value(sample: np.ndarray, *, min_fraction: float = 0.05) -> int | None:
    """Return the rotation-pad fill value in *sample*, or None if there is none.

    ``--reorient`` resamples onto a new axis-aligned grid, so the rotated
    specimen sits in it as a tilted box and the corners fall outside the
    original volume. ``affine_transform`` fills those with a single constant --
    the background level, 2050 counts on the scan this was written for -- and on
    a strongly tilted scan that pad can be a third of all voxels.

    It is not a measurement, and it distorts every global statistic: on
    f0004019 the pad *is* the volume median, so the median of the whole scan is
    an artifact of rotation geometry rather than of tissue.

    A pad is identified by being one exact value that is both common and
    concentrated at the volume's faces while absent from its centre. Genuine air
    fails the second test: it is noisy, so it spreads over neighbouring values
    and appears throughout the background rather than only at the edges. On a
    commercial scan checked for comparison the most common value covered 14.8%
    of voxels but sat at only 0.12-0.17 at the faces, so it was correctly left
    alone.
    """
    values, counts = np.unique(sample, return_counts=True)
    top = int(np.argmax(counts))
    fraction = counts[top] / sample.size
    if fraction < min_fraction:
        return None

    candidate = values[top]
    mask = sample == candidate

    edge = float(
        np.mean([
            mask[:10].mean(), mask[-10:].mean(),
            mask[:, :10].mean(), mask[:, -10:].mean(),
            mask[:, :, :10].mean(), mask[:, :, -10:].mean(),
        ])
    )
    mid = tuple(n // 2 for n in sample.shape)
    core = mask[
        mid[0] - 10:mid[0] + 10, mid[1] - 10:mid[1] + 10, mid[2] - 10:mid[2] + 10
    ].mean()

    # Padding fills the border almost completely and never reaches the middle.
    if edge > 0.5 and core < 0.01:
        return int(candidate)
    return None


def plot_histogram(
    nrrd_path: Path,
    out_path: Path | None = None,
    *,
    bins: int = 512,
    max_dim: int = 400,
    mu_scaling: float | None = None,
    clip_percentile: float | None = 99.9,
    log_scale: bool = True,
    drop_pad: bool = True,
) -> Path:
    """Histogram the volume at *nrrd_path* and write a PNG."""
    sample, header = _load_subsample(nrrd_path, max_dim)
    meta = _read_meta(nrrd_path)

    if mu_scaling is None:
        mu_scaling = meta.get("mu_scaling") or None

    pad_value = find_pad_value(sample) if drop_pad else None
    pad_fraction = 0.0
    if pad_value is not None:
        keep = sample != pad_value
        pad_fraction = 1.0 - float(keep.mean())
        log.info(
            "Excluding %.2f%% of voxels at the rotation-pad value %d — it is "
            "fill from --reorient, not measured signal.",
            100 * pad_fraction, pad_value,
        )
        sample = sample[keep]

    values = sample.astype(np.float32).ravel()

    rails = int(((values == INT16.min) | (values == INT16.max)).sum())
    if rails:
        log.warning(
            "%d sampled voxel(s) sit on an int16 rail — these are reconstruction "
            "artifacts, not density.", rails,
        )

    unit = "stored counts"
    if mu_scaling:
        values = values / float(mu_scaling)
        unit = "linear attenuation (1/cm)"

    # The mineralised tail is long and sparse; without trimming it the plot is
    # mostly empty space. Percentile limits keep the informative range on screen.
    lo = float(values.min())
    hi = float(np.percentile(values, clip_percentile)) if clip_percentile else float(values.max())
    if hi <= lo:
        hi = float(values.max())

    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(centres, counts, step="mid", alpha=0.35, color="#4878CF")
    ax.step(centres, counts, where="mid", color="#2B4C7E", linewidth=1.2)

    # The noise-floor stage collapses the whole air band onto one value, which
    # shows up as a spike an order of magnitude above everything else. Label it
    # so it is not mistaken for a tissue population.
    floor = meta.get("_noise_floor") or {}
    if floor.get("fill") is not None:
        fill = float(floor["fill"]) / float(mu_scaling) if mu_scaling else float(floor["fill"])
        if lo <= fill <= hi:
            ax.axvline(fill, color="#C44E52", linestyle="--", linewidth=1.0)
            ax.annotate(
                f"noise floor\n({floor.get('fraction_cleared', 0):.1%} of voxels)",
                xy=(fill, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points",
                color="#C44E52", fontsize=8, va="top",
            )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("voxel count (log)")
    else:
        ax.set_ylabel("voxel count")

    ax.set_xlabel(unit)
    ax.set_title(f"{nrrd_path.stem} — density distribution")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.margins(x=0.01)

    pad_note = (
        f"rotation pad excluded: {pad_fraction:.1%} at {pad_value}\n"
        if pad_value is not None else ""
    )
    stats = (
        f"sampled {values.size:,} of {np.prod([int(n) for n in header['sizes']]):,} voxels\n"
        f"{pad_note}"
        f"median {np.median(values):.4g} · p95 {np.percentile(values, 95):.4g} · "
        f"p99.9 {np.percentile(values, 99.9):.4g}\n"
        f"full range [{values.min():.4g}, {values.max():.4g}]"
    )
    ax.text(
        0.99, 0.97, stats, transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCC", alpha=0.85),
    )

    out_path = out_path or nrrd_path.with_name(nrrd_path.stem + "_histogram.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot the density histogram of a converted microCT volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("nrrd_file", type=Path)
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    p.add_argument("--bins", type=int, default=512)
    p.add_argument("--max-dim", type=int, default=400,
                   help="Max voxels per axis in the strided subsample")
    p.add_argument("--mu-scaling", type=float, default=None,
                   help="Divide counts by this to plot 1/cm (default: from _meta.json)")
    p.add_argument("--clip-percentile", type=float, default=99.9,
                   help="Upper x limit as a percentile; 0 to show the full range")
    p.add_argument("--linear", action="store_true", help="Linear y axis instead of log")
    p.add_argument("--keep-pad", action="store_false", dest="drop_pad",
                   help="Keep the constant fill --reorient pads the rotated grid "
                        "with. It is not measured signal and can be a third of "
                        "all voxels, so it is excluded by default")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.nrrd_file.is_file():
        raise SystemExit(f"Not a file: {args.nrrd_file}")

    plot_histogram(
        args.nrrd_file,
        args.out,
        bins=args.bins,
        max_dim=args.max_dim,
        mu_scaling=args.mu_scaling,
        clip_percentile=args.clip_percentile or None,
        log_scale=not args.linear,
        drop_pad=args.drop_pad,
    )


if __name__ == "__main__":
    main()
