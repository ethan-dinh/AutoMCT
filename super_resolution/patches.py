"""
Training data cut from the scans themselves.

There is no ground truth at twice the scanner's resolution, so the model is
trained one scale down: a patch of the real scan is the high-resolution target,
and the same patch block-averaged by the scale factor is the input. Having
learnt that mapping, the generator is applied to the real scan to go one scale
*up*. This is self-supervised ("internal") super-resolution (Shocher et al.
2018, ZSSR; Zhao et al. 2021, SMORE for MRI). It rests on the assumption that
the tissue looks alike across the two scales -- sound for bone and enamel
boundaries, and the reason nothing here is ever measured on the upscaled
volume.

Block-averaging also averages away noise, which would leave the training
inputs cleaner than the real scan the model is later run on. Each input
therefore gets fresh noise at the scan's own measured level (see
:func:`estimate_noise_sigma`), randomised per patch so the model does not rely
on one exact level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .volume_io import IntensityScale, Scan, tissue_threshold

logger = logging.getLogger(__name__)

Shape3 = tuple[int, int, int]


@dataclass
class TrainingVolume:
    """One scan, prepared for patch sampling."""

    name: str
    data: np.ndarray
    intensity: IntensityScale
    noise_sigma: float
    spacing: tuple[float, float, float]
    spacing_known: bool
    train_coords: np.ndarray
    val_coords: np.ndarray
    holdout: tuple[int, int] | None


def estimate_noise_sigma(
    data: np.ndarray,
    intensity: IntensityScale,
    coords: np.ndarray,
    rng: np.random.Generator,
    *,
    blocks: int = 24,
    size: int = 24,
) -> float:
    """
    Noise standard deviation, in normalised units, measured inside tissue.

    The residual of each voxel from its 3x3x3 mean is dominated by noise
    wherever the tissue is locally smooth, and its median absolute deviation
    ignores the minority of voxels on edges. For white noise that residual has
    variance sigma^2 * 26/27, which is corrected for. Blocks are taken inside
    tissue because air in many reconstructions is clipped flat and would
    report no noise at all. Texture pushes the estimate up slightly; the
    per-patch randomisation during training absorbs that.
    """
    if len(coords) == 0:
        return 0.0
    half = size // 2
    residuals = []
    for index in rng.choice(len(coords), size=min(blocks, len(coords)), replace=False):
        centre = coords[index]
        lo = np.clip(centre - half, 0, np.maximum(np.asarray(data.shape) - size, 0))
        block = intensity.to_unit(data[tuple(slice(int(a), int(a) + size) for a in lo)])
        if min(block.shape) < 3:
            continue
        residual = block - ndimage.uniform_filter(block, size=3, mode="reflect")
        residuals.append(residual[1:-1, 1:-1, 1:-1].ravel())
    if not residuals:
        return 0.0
    values = np.concatenate(residuals)
    mad = np.median(np.abs(values - np.median(values)))
    return float(1.4826 * mad / np.sqrt(26.0 / 27.0))


def prepare_volume(
    scan: Scan,
    hr_patch: Shape3,
    rng: np.random.Generator,
    *,
    holdout_fraction: float = 0.1,
    max_coords: int = 2_000_000,
) -> TrainingVolume:
    """
    Normalisation, noise level, tissue locations and a held-out slab for one scan.

    The held-out slab is a band of axial slices through the middle of the scan
    (where the tissue is) that training never samples, so validation measures
    how the model does on anatomy it has not seen. A scan too short to spare
    one gets none, and its validation patches come from the training region.
    """
    data = scan.data
    if any(n < p for n, p in zip(data.shape, hr_patch)):
        raise ValueError(
            f"{scan.name}: volume {data.shape} is smaller than the training patch "
            f"{hr_patch}; lower --lr-patch."
        )
    intensity = IntensityScale.fit(data)
    threshold = tissue_threshold(data)

    stride = max(1, int(np.ceil((data.size / 8_000_000) ** (1 / 3))))
    coarse = data[::stride, ::stride, ::stride] > threshold
    coords = (np.argwhere(coarse) * stride).astype(np.int32)
    if len(coords) == 0:
        logger.warning("%s: no voxels above the tissue threshold; sampling uniformly", scan.name)
    if len(coords) > max_coords:
        coords = coords[rng.choice(len(coords), size=max_coords, replace=False)]

    depth = data.shape[0]
    patch_z = hr_patch[0]
    holdout = None
    if depth >= 3 * patch_z:
        thickness = max(int(round(depth * holdout_fraction)), patch_z)
        start = (depth - thickness) // 2
        holdout = (start, start + thickness)
    else:
        logger.warning(
            "%s: only %d slices, too few to hold out a validation slab; "
            "validation will reuse the training region", scan.name, depth,
        )

    if holdout is not None and len(coords):
        in_slab = (coords[:, 0] >= holdout[0]) & (coords[:, 0] < holdout[1])
        train_coords, val_coords = coords[~in_slab], coords[in_slab]
    else:
        train_coords, val_coords = coords, coords

    noise_sigma = estimate_noise_sigma(data, intensity, coords, rng)
    logger.info(
        "%s: intensity %.4g..%.4g -> 0..1, tissue threshold %.4g, noise sigma %.4f "
        "(normalised), %d tissue sample points, held-out slices %s",
        scan.name, intensity.low, intensity.high, threshold, noise_sigma,
        len(coords), holdout,
    )
    return TrainingVolume(
        name=scan.name, data=data, intensity=intensity, noise_sigma=noise_sigma,
        spacing=scan.geometry.spacing, spacing_known=scan.spacing_from_file,
        train_coords=train_coords, val_coords=val_coords, holdout=holdout,
    )


def _is_isotropic(spacing: tuple[float, float, float], tolerance: float = 0.01) -> bool:
    return max(spacing) <= min(spacing) * (1 + tolerance)


class PatchSampler:
    """
    Draws high-resolution training patches from one or more scans.

    Most patches are centred on tissue (``foreground_fraction``); micro-CT
    scans are mostly air, and a uniformly sampled batch would spend its
    capacity learning to upscale nothing. Training patches never overlap a
    scan's held-out slab.
    """

    def __init__(
        self,
        volumes: list[TrainingVolume],
        hr_patch: Shape3,
        rng: np.random.Generator,
        *,
        foreground_fraction: float = 0.9,
        augment: bool = True,
    ):
        if not volumes:
            raise ValueError("PatchSampler needs at least one volume")
        self.volumes = volumes
        self.hr_patch = tuple(int(p) for p in hr_patch)
        self.rng = rng
        self.foreground_fraction = foreground_fraction
        self.augment = augment
        # Larger scans contribute proportionally more patches.
        sizes = np.array([v.data.size for v in volumes], dtype=float)
        self.weights = sizes / sizes.sum()

    # -- corner selection ---------------------------------------------------

    def _clamp(self, corner: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.clip(corner, 0, np.asarray(shape) - np.asarray(self.hr_patch))

    def _overlaps_holdout(self, volume: TrainingVolume, z: int) -> bool:
        if volume.holdout is None:
            return False
        return z < volume.holdout[1] and z + self.hr_patch[0] > volume.holdout[0]

    def _uniform_train_corner(self, volume: TrainingVolume) -> np.ndarray:
        shape = volume.data.shape
        corner = np.array([self.rng.integers(0, n - p + 1) for n, p in zip(shape, self.hr_patch)])
        if volume.holdout is not None:
            # Choose z from the slices above or below the slab, weighted by room.
            below = max(0, volume.holdout[0] - self.hr_patch[0] + 1)
            above = max(0, shape[0] - self.hr_patch[0] + 1 - volume.holdout[1])
            if below + above > 0:
                pick = int(self.rng.integers(0, below + above))
                corner[0] = pick if pick < below else volume.holdout[1] + (pick - below)
        return corner

    def _train_corner(self, volume: TrainingVolume) -> np.ndarray:
        shape = volume.data.shape
        if len(volume.train_coords) and self.rng.random() < self.foreground_fraction:
            for _ in range(10):
                point = volume.train_coords[self.rng.integers(len(volume.train_coords))]
                offset = np.array([self.rng.integers(0, p) for p in self.hr_patch])
                corner = self._clamp(point - offset, shape)
                if not self._overlaps_holdout(volume, int(corner[0])):
                    return corner
        return self._uniform_train_corner(volume)

    def _val_corner(self, volume: TrainingVolume) -> np.ndarray:
        shape = volume.data.shape
        coords = volume.val_coords
        if len(coords):
            point = coords[self.rng.integers(len(coords))]
            offset = np.array([self.rng.integers(0, p) for p in self.hr_patch])
            corner = self._clamp(point - offset, shape)
        else:
            corner = np.array([self.rng.integers(0, n - p + 1) for n, p in zip(shape, self.hr_patch)])
        if volume.holdout is not None:
            corner[0] = np.clip(corner[0], volume.holdout[0], volume.holdout[1] - self.hr_patch[0])
        return corner

    # -- extraction ---------------------------------------------------------

    def _extract(self, volume: TrainingVolume, corner: np.ndarray) -> np.ndarray:
        region = tuple(slice(int(c), int(c) + p) for c, p in zip(corner, self.hr_patch))
        return volume.intensity.to_unit(volume.data[region])

    def _augment(self, patch: np.ndarray, volume: TrainingVolume) -> np.ndarray:
        for axis in range(3):
            if self.rng.random() < 0.5:
                patch = np.flip(patch, axis=axis)
        # Axes can only be exchanged when they mean the same physical thing:
        # the same voxel size, and a cubic patch so the shape is unchanged.
        if _is_isotropic(volume.spacing) and len(set(self.hr_patch)) == 1:
            patch = np.transpose(patch, self.rng.permutation(3))
        return np.ascontiguousarray(patch)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """A training batch: (patches (B, 1, Z, Y, X) float32, noise sigma per patch (B,))."""
        patches, sigmas = [], []
        for index in self.rng.choice(len(self.volumes), size=batch_size, p=self.weights):
            volume = self.volumes[int(index)]
            patch = self._extract(volume, self._train_corner(volume))
            if self.augment:
                patch = self._augment(patch, volume)
            patches.append(patch)
            sigmas.append(volume.noise_sigma)
        return np.stack(patches)[:, None], np.asarray(sigmas, dtype=np.float32)

    def validation_set(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """A fixed set of held-out patches (no augmentation), spread across the scans."""
        patches, sigmas = [], []
        for i in range(count):
            volume = self.volumes[i % len(self.volumes)]
            patches.append(self._extract(volume, self._val_corner(volume)))
            sigmas.append(volume.noise_sigma)
        return np.stack(patches)[:, None], np.asarray(sigmas, dtype=np.float32)
