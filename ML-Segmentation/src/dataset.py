"""
Dataset and transforms for mandible segmentation training.

Loads reoriented_volume.tif as the input image and combines the three binary
masks (incisor, bone, molar) into a single multi-class label volume:
    0 = background
    1 = incisor
    2 = bone
    3 = molar

Heavy augmentation is applied during training to compensate for the small
dataset size (9 samples).
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
import torch
from monai.data import Dataset
from monai.transforms import (
    ClassesToIndicesd,
    Compose,
    CropForegroundd,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
    ToTensord,
    Transform,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "background", 1: "incisor", 2: "bone", 3: "molar"}

# Max class-index entries saved per class. 50 K is ample for 24 balanced
# crops per volume; keeping it small avoids GB-scale index files.
_MAX_IDX_PER_CLASS = 50_000


def discover_samples(data_dir: str | Path) -> list[dict]:
    """
    Scan *data_dir* for sample folders and return a list of dicts with file paths.

    Expected layout per sample::

        <sample>/reoriented_volume.tif
        <sample>/masks/incisor_mask.tif
        <sample>/masks/bone_mask.tif
        <sample>/masks/molar_mask.tif
    """
    data_dir = Path(data_dir)
    samples = []
    for sample_dir in sorted(data_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        vol = sample_dir / "reoriented_volume.tif"
        masks_dir = sample_dir / "masks"
        incisor = masks_dir / "incisor_mask.tif"
        bone    = masks_dir / "bone_mask.tif"
        molar   = masks_dir / "molar_mask.tif"
        if vol.exists() and incisor.exists() and bone.exists() and molar.exists():
            samples.append({
                "name":    sample_dir.name,
                "image":   str(vol),
                "incisor": str(incisor),
                "bone":    str(bone),
                "molar":   str(molar),
            })
        else:
            logger.warning("Skipping %s — missing files", sample_dir.name)
    logger.info("Discovered %d samples in %s", len(samples), data_dir)
    return samples


class LoadAndCombineMasksd(Transform):
    """
    Loads image + three binary masks, combines masks into a single uint8 label
    volume, and normalises the image to float32 [0, 1].

    Precedence when masks overlap: molar > bone > incisor.
    Inherits from monai.transforms.Transform so PersistentDataset treats it as
    a deterministic (cacheable) transform.
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        name = d.get("name", "?")

        t0 = time.perf_counter()
        img = tifffile.imread(d["image"]).astype(np.float32)
        logger.info("  [%-20s]  imread image   %.1fs  shape=%s  %.0f MB",
                    name, time.perf_counter() - t0, img.shape, img.nbytes / 1e6)

        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - p1) / max(float(p99 - p1), 1e-6)
        d["image"] = img[np.newaxis]  # (1, D, H, W)

        tm = time.perf_counter()
        # bool intermediates (1 byte/voxel) vs int64 (8 bytes/voxel)
        incisor = tifffile.imread(d["incisor"]) > 0
        bone    = tifffile.imread(d["bone"])    > 0
        molar   = tifffile.imread(d["molar"])   > 0
        logger.info("  [%-20s]  imread masks   %.1fs", name, time.perf_counter() - tm)

        label = np.zeros(incisor.shape, dtype=np.uint8)
        label[incisor] = 1
        label[bone]    = 2
        label[molar]   = 3
        d["label"] = label[np.newaxis]  # (1, D, H, W)

        for k in ("incisor", "bone", "molar"):
            d.pop(k, None)

        logger.info("  [%-20s]  load+combine   %.1fs total", name, time.perf_counter() - t0)
        return d


# ── Deterministic transforms (slow — load + pad + crop) ───────────────────────

def _det_transform(patch_size: tuple[int, int, int]) -> Compose:
    # ClassesToIndicesd is intentionally absent: storing all voxel indices
    # inflates cache files to gigabytes. They are small when saved separately
    # with max_samples_per_class (see warm_disk_cache).
    return Compose([
        LoadAndCombineMasksd(),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=8),
    ])


# ── Random transforms (fast — applied per-batch from cached arrays) ───────────

def _rand_train_transform(
    patch_size: tuple[int, int, int],
    samples_per_volume: int,
    pos_ratio: float,
    class_ratios: list[float] | None = None,
) -> Compose:
    """Augmentation pipeline applied on top of the disk-cached arrays.

    class_ratios: sampling weight per class [bg, incisor, bone, molar].
    Defaults to [0, 2, 1, 4] — aggressively oversamples molars to counter
    the severe class imbalance (molars have far fewer voxels than bone).
    label_cls_indices is provided by _NpyCacheDataset so ClassesToIndicesd
    is not needed here.
    """
    if class_ratios is None:
        class_ratios = [1, 2, 1, 5]  # bg kept; molar upweighted vs bone
    return Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            ratios=class_ratios,
            num_classes=4,
            num_samples=samples_per_volume,
            indices_key="label_cls_indices",
            allow_smaller=True,
        ),
        # Flips: O(1) array-view, free
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # 90° rotations: stride reindex only, no interpolation
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 2)),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
        # Continuous rotation + scale: covers inter-animal size and pose variation.
        # 90° flips above give discrete symmetries; this fills in the continuous gaps.
        # mode="nearest" for label avoids fractional class values.
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),  # ±15°
            scale_range=(0.15, 0.15, 0.15),                      # ±15%
            padding_mode="zeros",
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.3,
            sigma_range=(5, 8),
            magnitude_range=(20, 60),
            padding_mode="zeros",
        ),
        # Coarse dropout: zero random 3-D cuboids so the model can't memorise
        # local texture patches — the main overfitting mode with 9 samples.
        RandCoarseDropoutd(
            keys=["image"],
            holes=4,
            spatial_size=(16, 16, 16),
            fill_value=0.0,
            prob=0.3,
        ),
        # Intensity augmentation — probabilities raised to 0.5 for small dataset
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
            prob=0.3,
        ),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.75, 1.5), prob=0.5),
        ToTensord(keys=["image", "label"]),
    ])


# ── Npy disk cache ─────────────────────────────────────────────────────────────

def _sample_hash(sample: dict, patch_size: tuple) -> str:
    """Stable 16-char hex hash based on image path + patch size + preprocess version."""
    key = sample["image"] + str(patch_size) + "v2"  # v2: percentile normalization
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class _AugDataset(torch.utils.data.Dataset):
    """
    Pre-loads all training samples fully into CPU RAM at construction time.

    np.load() (no mmap) reads each .npy file once upfront, so every
    __getitem__ call is pure CPU compute with zero disk I/O. This avoids
    the thousands of random page faults that mmap causes on Windows when a
    DataLoader worker reads a non-contiguous 96³ crop from a 1-3 GB file.

    Peak RAM = sum of image + label arrays for all samples in this fold.
    Memory is released when train_fold() deletes the dataset at fold end.
    """

    def __init__(
        self,
        samples: list[dict],
        patch_size: tuple[int, int, int],
        samples_per_volume: int,
        pos_ratio: float,
        cache_dir: Path,
        class_ratios: list[float] | None = None,
    ) -> None:
        cache_dir = Path(cache_dir)
        t0        = time.perf_counter()
        total_mb  = 0.0
        self._data: list[dict] = []

        for s in samples:
            h   = _sample_hash(s, patch_size)
            ts  = time.perf_counter()
            img = np.load(str(cache_dir / f"{h}_img.npy"))   # full load — no mmap
            lbl = np.load(str(cache_dir / f"{h}_lbl.npy"))
            idx = np.load(str(cache_dir / f"{h}_idx.npz"), allow_pickle=True)
            cls_indices = [idx[f"c{k}"] for k in range(4)]
            mb = (img.nbytes + lbl.nbytes) / 1e6
            total_mb += mb
            logger.info("  Loaded %-20s  shape=%s  %.0f MB  %.1fs",
                        s["name"], img.shape, mb, time.perf_counter() - ts)
            self._data.append({
                "image":             img,
                "label":             lbl,
                "label_cls_indices": cls_indices,
                "name":              s["name"],
            })

        logger.info("All %d samples in RAM  total=%.0f MB  %.1fs",
                    len(self._data), total_mb, time.perf_counter() - t0)
        self._rand_tx = _rand_train_transform(patch_size, samples_per_volume, pos_ratio, class_ratios)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        t0     = time.perf_counter()
        result = self._rand_tx(self._data[index])
        logger.debug("  _AugDataset[%d] augment=%.3fs", index, time.perf_counter() - t0)
        return result


def warm_disk_cache(
    samples: list[dict],
    patch_size: tuple[int, int, int],
    cache_dir: Path,
    num_workers: int = 4,
    on_done=None,
) -> None:
    """
    Pre-populate the npy cache for all samples in parallel.

    Saves three files per sample:
      {hash}_img.npy  — float32 image array (memmap-friendly)
      {hash}_lbl.npy  — uint8  label array  (memmap-friendly)
      {hash}_idx.npz  — subsampled class indices (~1 MB, normal load)

    Already-cached entries are skipped on re-runs.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    det_tx = _det_transform(patch_size)
    cls_tx = ClassesToIndicesd(
        keys="label", num_classes=4, indices_postfix="_cls_indices",
        max_samples_per_class=_MAX_IDX_PER_CLASS,
    )

    def _process(s: dict) -> None:
        h        = _sample_hash(s, patch_size)
        img_path = cache_dir / f"{h}_img.npy"
        lbl_path = cache_dir / f"{h}_lbl.npy"
        idx_path = cache_dir / f"{h}_idx.npz"

        if img_path.exists() and lbl_path.exists() and idx_path.exists():
            logger.info("  [%-20s]  already cached  img=%.0f MB",
                        s["name"], img_path.stat().st_size / 1e6)
            return

        logger.info("  [%-20s]  starting ...", s["name"])
        t0 = time.perf_counter()

        d  = det_tx(s)
        t1 = time.perf_counter()
        logger.info("  [%-20s]  det_transform   %.1fs  cropped=%s",
                    s["name"], t1 - t0, d["image"].shape)

        np.save(img_path, d["image"])
        np.save(lbl_path, d["label"])
        t2 = time.perf_counter()
        logger.info("  [%-20s]  saved npy       %.1fs  img=%.0f MB  lbl=%.0f MB",
                    s["name"], t2 - t1,
                    img_path.stat().st_size / 1e6,
                    lbl_path.stat().st_size / 1e6)

        d2 = cls_tx(d)
        cls_indices = d2["label_cls_indices"]
        np.savez(idx_path, **{f"c{k}": cls_indices[k] for k in range(len(cls_indices))})
        logger.info("  [%-20s]  done            %.1fs total  idx=%.1f MB",
                    s["name"], time.perf_counter() - t0,
                    idx_path.stat().st_size / 1e6)

    def _warm(s: dict) -> None:
        _process(s)
        if on_done is not None:
            on_done(s)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = [pool.submit(_warm, s) for s in samples]
        for fut in as_completed(futs):
            fut.result()  # re-raise any exceptions immediately


# ── Public API ────────────────────────────────────────────────────────────────

def get_train_transforms(
    patch_size: tuple[int, int, int],
    samples_per_volume: int,
    pos_ratio: float,
) -> Compose:
    """Full pipeline (det + rand) for use without disk caching."""
    det  = _det_transform(patch_size)
    rand = _rand_train_transform(patch_size, samples_per_volume, pos_ratio)
    return Compose(det.transforms + rand.transforms)


def get_val_transforms(patch_size: tuple[int, int, int]) -> Compose:
    return Compose([
        LoadAndCombineMasksd(),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])


def build_datasets(
    train_samples: list[dict],
    val_samples: list[dict],
    patch_size: tuple[int, int, int],
    samples_per_volume: int,
    pos_ratio: float,
    cache_dir: Path | str,
    class_ratios: list[float] | None = None,
) -> tuple[torch.utils.data.Dataset, Dataset]:
    """
    Build training and validation datasets backed by the npy disk cache.

    Deterministic transforms run once at warm_disk_cache time and are stored
    as .npy files. Each __getitem__ mmaps the files (instant) and applies
    random augmentation on the small 96³ crop region, keeping peak RAM
    proportional to batch size rather than dataset size.
    """
    train_ds = _AugDataset(
        train_samples, patch_size, samples_per_volume, pos_ratio,
        Path(cache_dir), class_ratios,
    )
    val_ds = Dataset(data=[], transform=None)  # val runs via sliding_window_inference
    return train_ds, val_ds
