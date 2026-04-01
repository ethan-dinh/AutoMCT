# Mandible Segmentation Pipeline

Segments three structures from a mouse mandible MicroCT BMP stack:
- **Incisor** — the continuously growing incisor tooth
- **Bone** — the mandibular bone body
- **Molar** — the molar tooth/teeth

---

## Usage

```bash
python main.py --input_path /path/to/samples [--log] [--visualize]
```

| Flag | Short | Description |
|---|---|---|
| `--input_path` | `-i` | Root directory containing one sub-folder per sample |
| `--log` | `-l` | Enable coloured console logging |
| `--visualize` | `-v` | Open napari 3D viewer after each segmentation |

**Expected input layout:**
```
/path/to/samples/
├── Sample_01/
│   ├── slice_0001.bmp
│   ├── slice_0002.bmp
│   └── ...
├── Sample_02/
│   └── ...
└── Compressed/          ← skipped automatically
```

Files matching `*spr.bmp` (scout/projection views) are excluded automatically.

**Output** is written to `./segmentation_results/<sample_name>/`:
```
segmentation_results/
└── Sample_01/
    ├── incisor_volume.tif
    ├── bone_volume.tif
    └── molar_volume.tif
```

The script prompts interactively for each sample. If a sample has already been segmented, it offers the option to visualize existing results and/or re-segment.

---

## Pipeline Overview

```
BMP stack
    │
    ▼
[1] Load volume                  data_io/loaders.py
    │  load_bmp_stack()
    │  Reads all *.bmp files (sorted by name), excluding *spr.bmp.
    │  Returns a (Z, Y, X) uint8 array.
    │
    ▼
[2] Normalize intensities        preprocessing/filters.py
    │  normalize_volume()
    │  Min-max normalization → float64 in [0, 1].
    │
    ▼
[3] Estimate bone intensity      preprocessing/filters.py
    │  find_min_intensity_of_bone()
    │  Applies Gaussian blur (σ=10) to the 30 middle slices,
    │  runs Otsu segmentation on each, and returns the minimum
    │  mean intensity across the largest region in each slice.
    │  This gives a sample-adaptive bone intensity floor.
    │
    ▼
[4] Background removal           pipeline.py
    │  Voxels below 25 % of the bone floor are zeroed out.
    │  Produces preprocessed_volume.
    │
    ▼
[5] Reorientation                preprocessing/reorientation.py
    │  reorient_mandible()
    │  Standardizes the anterior–posterior axis so that downstream
    │  segmentation can assume a fixed scan direction.
    │
    │  a) Binary mask via Otsu on non-zero voxels.
    │  b) Tight 3D bounding box logged for context.
    │  c) SVD on mask voxel coordinates → first principal axis
    │     (axis of maximum spatial variance = long / AP axis).
    │  d) Axis alignment:
    │       • If the principal axis deviates ≤ 15° from a canonical
    │         axis, apply a lossless axis permutation (no interpolation).
    │       • Otherwise apply a scipy affine rotation (linear interpolation).
    │  e) Incisor tip detection: thresholds at the 97th percentile of
    │     bone-masked voxels, labels connected components, and selects
    │     the most intense compact cluster (≤ 2 % of mask size) as the
    │     enamel tip candidate.
    │  f) If the tip centroid is above the axis-0 midpoint, the volume
    │     is flipped along axis 0 so the tip ends up at low Z.
    │
    │  The same transform is applied to the companion raw BMP volume
    │  so spatial correspondence is preserved for final intensity masking.
    │
    ▼
[6] Incisor segmentation         segmentation/incisor.py
    │  segment_incisor()
    │  Iterates slices from posterior → anterior:
    │    • Per-slice contrast stretch (percentile clip, p=10–98).
    │    • Otsu threshold + label connected components.
    │    • First valid slice: pick region with highest mean intensity
    │      (must exceed 50 % of bone floor).
    │    • Subsequent slices: pick region whose centroid is closest
    │      to the previous slice's centroid.
    │    • If a region doubles in area (likely fusion with adjacent
    │      structure), erode iteratively (up to 5×) until the region
    │      splits, then re-select by centroid proximity.
    │  Returns a boolean 3D mask.
    │
    ▼
[7] Incisor removal              pipeline.py
    │  Incisor mask is dilated by radius=2 voxels and zeroed out
    │  from the volume. The remaining volume is re-thresholded at
    │  80 % of the bone floor (higher bar for molar enamel, which
    │  is fully mineralised).
    │
    ▼
[8] Bone + molar segmentation    segmentation/molar_bone.py
    │  segment_molar_bone()
    │  • Gaussian blur (σ=1) + Otsu threshold slice-by-slice.
    │  • Morphological closing (ball r=1) to fill small gaps.
    │  • 3D connected-component labeling.
    │  • Largest component → bone mask.
    │  • All remaining components → candidate molar material;
    │    discard any component smaller than 50 % of the largest
    │    molar component.
    │  Returns (bone_mask, molar_mask).
    │
    ▼
[9] Save & visualize             data_io/loaders.py
       save_ct_volume_as_tiff()   visualization/viewers.py
       Writes compressed BigTIFF files (zlib).
       Optional napari viewer shows all three structures overlaid
       on the original grayscale volume.
```

---

## Module Reference

```
segment_mandible/
├── main.py                  CLI entry point; interactive sample selection
├── pipeline.py              Orchestrates steps 1–7; returns segmented volumes
│
├── data_io/
│   └── loaders.py           load_bmp_stack, load_tiff, save_ct_volume_as_tiff
│
├── preprocessing/
│   ├── filters.py           normalize_volume, gaussian_filter_volume,
│   │                        percentile_clip, rescale_to_unit,
│   │                        cut_bridges_slice, find_min_intensity_of_bone
│   └── reorientation.py     reorient_mandible
│
├── segmentation/
│   ├── utils.py             segment_slice/volume, label_slice/3d_volume,
│   │                        convert_to_binary, get_largest_region,
│   │                        get_region_intensity, erode/dilate_mask,
│   │                        morphological_closing
│   ├── incisor.py           segment_incisor
│   └── molar_bone.py        segment_molar_bone
│
├── visualization/
│   └── viewers.py           create_3d_visualization (napari)
│
└── tests/
    └── test_segmentation.py
```

---

## Adding New Structures

1. Add a new segmentation module under `segmentation/`, e.g. `segmentation/enamel.py`.
2. Export the function from `segmentation/__init__.py`.
3. Call it from `pipeline.py` after the incisor removal step (step 6), following the same mask → `np.where` → save pattern.

## Adding New I/O Formats

Add a loader/saver to `data_io/loaders.py` (or a new file in `data_io/`) and export it from `data_io/__init__.py`.
