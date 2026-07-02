# Mandible Segmentation Pipeline

Segments three structures from mouse mandible microCT scans:
- **Incisor** — the continuously growing incisor tooth
- **Bone** — the mandibular bone body
- **Molar** — the molar teeth (optionally split into individual instances)

---

## Usage

```bash
python main.py --input_path /path/to/samples [--log] [--visualize] [--debug] [--out /path/to/output]
```

| Flag | Short | Description |
|---|---|---|
| `--input_path` | `-i` | Path to a `.nrrd`/`.tif` file, or a root directory containing one sub-folder per sample |
| `--log` | `-l` | Enable coloured console logging |
| `--visualize` | `-v` | Open napari 3D viewer after each segmentation |
| `--debug` | `-d` | Open napari viewer after every major pipeline step for inspection |
| `--out` | | Output root directory (default: `./segmentation_results`) |

### Supported input formats

The pipeline accepts several input layouts:

| Input | How it's handled |
|---|---|
| Directory of `.bmp` slices | Loaded as a (Z, Y, X) stack via `load_bmp_stack()`. Files matching `*spr.bmp` are excluded. |
| `.nrrd` file | Loaded and transposed from NRRD (X, Y, Z) to pipeline (Z, Y, X) convention. |
| `.tif` / `.tiff` file | Loaded directly as a 3D volume. |
| Directory containing any of the above | Auto-detected in priority order: `.nrrd` > `.tif`/`.tiff` > `.bmp` stack. |

**Expected directory layout (BMP mode):**
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

**Output** is written to `<out_root>/<sample_name>/`:
```
segmentation_results/
└── Sample_01/
    ├── reoriented_volume.tif
    ├── volumes/
    │   ├── incisor_volume.tif
    │   ├── bone_volume.tif
    │   └── molar_volume.tif
    └── masks/
        ├── incisor_mask.tif
        ├── bone_mask.tif
        └── molar_mask.tif
```

All volumes are saved as zlib-compressed BigTIFF. Masks are uint8 (0/1).

### Interactive test TUI

An interactive TUI for per-sample inspection lives in `tests/`:

```bash
python tests/run.py                                                    # native file/folder picker
python tests/test_pipeline.py --data /path/to/file.nrrd [--out /path/to/output] [--debug]
python tests/test_pipeline.py --data /path/to/samples   [--out /path/to/output] [--debug]
```

`run.py` opens a small dialog asking whether you want to pick a **single volume file** (`.nrrd`, `.tif`, `.tiff`) or a **folder of samples**, then launches the TUI. Passing a single file directly to `test_pipeline.py` is also supported.

After each segmentation, a napari viewer opens and a menu lets you **Accept** (save), **Re-segment**, **Skip**, or **Quit**. Requires the optional `questionary` package for a nicer menu (falls back to numbered prompts).

---

## Pipeline Overview

```
Input volume (BMP stack / NRRD / TIFF)
    │
    ▼
[1] Load volume                       data_io/loaders.py
    │  load_bmp_stack / load_nrrd / load_tiff
    │  Auto-detects format. Returns a (Z, Y, X) array.
    │
    ▼
[2] Normalize intensities             preprocessing/filters.py
    │  normalize_volume()
    │  Min-max normalization → float64 in [0, 1].
    │
    ▼
[3] Denoise                           preprocessing/filters.py
    │  non_local_means_filter()
    │    Per-slice 2D NLM with auto-estimated noise level.
    │    Preserves edges while suppressing stochastic noise.
    │    Parallelized across CPU cores via ThreadPoolExecutor.
    │  tv_denoise_volume()
    │    Per-slice Total Variation denoising (Chambolle).
    │    Enforces piecewise-constant regions for cleaner
    │    histogram peaks downstream.
    │
    ▼
[4] Threshold estimation              preprocessing/filters.py
    │  find_min_intensity_of_bone()
    │    Gaussian blur (σ=10) on 30 middle slices, Otsu per slice,
    │    returns the minimum mean intensity of the largest region.
    │  find_threshold(strategy="knee")
    │    Background/bone separation via elbow detection on the
    │    histogram's descending slope.
    │  find_threshold(strategy="valley")
    │    Conservative threshold at the valley between background
    │    and bone histogram peaks (used for incisor isolation).
    │
    ▼
[5] Background removal                pipeline.py
    │  Voxels below the knee threshold are zeroed out.
    │  Small disconnected islands (< 3500 voxels) are removed.
    │
    ▼
[6] Reorientation                     preprocessing/reorientation.py
    │  reorient_mandible()
    │  Standardizes the anterior–posterior axis:
    │    a) Binary mask via Otsu on non-zero voxels.
    │    b) SVD on mask coordinates → principal axis.
    │    c) If ≤ 15° from a canonical axis: lossless axis permutation.
    │       Otherwise: scipy affine rotation (linear interpolation).
    │    d) Incisor tip detection (97th percentile, compact cluster).
    │    e) Flip along axis 0 if tip is above midpoint.
    │  Companion raw volume is co-transformed.
    │
    ▼
[7] CLAHE contrast enhancement        preprocessing/filters.py
    │  clahe_volume()
    │  Adaptive histogram equalization (clip_limit=0.01) applied
    │  slice-by-slice to the preprocessed volume.
    │  Amplifies local intensity differences between the dense
    │  incisor tip and surrounding bone before Otsu thresholding.
    │  Parallelized across CPU cores via ThreadPoolExecutor.
    │  Only used as input to incisor segmentation — the raw
    │  preprocessed volume is preserved for all other steps.
    │
    ▼
[8] Conservative incisor isolation    pipeline.py
    │  Re-threshold the CLAHE volume at the valley threshold (more aggressive).
    │  Segment incisor in this conservative volume.
    │  Remove the incisor, dilate + clean the remaining bone mask
    │  (slice-by-slice island removal, min 2500 voxels).
    │  Subtract bone mask from the CLAHE volume to isolate
    │  the incisor region free of surrounding bone.
    │
    ▼
[9] Incisor segmentation              segmentation/incisor.py
    │  segment_incisor()
    │  Iterates slices posterior → anterior:
    │    • Per-slice contrast stretch (percentile clip, p=10–98).
    │    • Otsu threshold + connected-component labeling.
    │    • First slice: pick region with highest mean intensity
    │      (must exceed 50% of bone floor).
    │    • Subsequent slices: pick by centroid proximity.
    │    • Area-doubling guard: iterative erosion (up to 5×) to
    │      split fused structures, then re-select by centroid.
    │  Returns a boolean 3D mask.
    │
    ▼
[10] Incisor removal                  pipeline.py
    │  Dilate incisor mask (radius=2) and zero out from volume.
    │
    ▼
[11] Bone + molar segmentation        segmentation/molar_bone.py
    │  segment_molar_bone()
    │  Primary — enamel-anchored region growing:
    │    • After incisor removal, enamel (brightest tissue) only
    │      exists in molars.
    │    • Adaptively locate the enamel threshold: find the knee on
    │      the descending slope of the dominant (bone) histogram
    │      peak, then refit on the tail beyond it to locate the
    │      enamel peak itself.
    │    • Label enamel clusters; keep top N (≤ max_molars) by mean
    │      intensity as one seed per tooth.
    │    • Each seed is dilated (radius=2) then grown via binary
    │      propagation into the full foreground, with thin bone
    │      bridges cut beforehand so seeds can't leak between teeth.
    │    • Foreground not reached by any seed → bone.
    │  Fallback (if no enamel seeds found) — erosion-based:
    │    • Iteratively erode foreground until ≥ 2 components appear.
    │    • Highest max-intensity component → molar; rest → bone.
    │  Molar mask is dilated (radius=3) to capture porous structure.
    │
    ▼
[12] Save & visualize                 data_io/loaders.py, visualization/viewers.py
       save_ct_volume_as_tiff()       Compressed BigTIFF (zlib).
       save_mask_as_tiff()            uint8 binary masks.
       napari viewer (optional)       Overlays all structures on grayscale volume.
```

---

## Module Reference

```
segment_mandible/
├── main.py                  CLI entry point; batch processing with interactive prompts
├── pipeline.py              Orchestrates steps 1–10; returns segmented volumes
│
├── data_io/
│   ├── __init__.py
│   └── loaders.py           load_bmp_stack, load_nrrd, load_tiff,
│                             save_ct_volume_as_tiff, save_mask_as_tiff, save_as_nrrd
│
├── preprocessing/
│   ├── __init__.py
│   ├── filters.py           normalize_volume, gaussian_filter_volume,
│   │                        non_local_means_filter, tv_denoise_volume,
│   │                        clahe_volume, rescale_to_unit, cut_bridges_slice,
│   │                        find_min_intensity_of_bone, find_threshold
│   └── reorientation.py     reorient_mandible
│
├── segmentation/
│   ├── __init__.py
│   ├── utils.py             segment_slice/volume, label_slice/3d_volume,
│   │                        convert_to_binary, get_largest_region,
│   │                        get_region_intensity, erode/dilate_mask,
│   │                        morphological_closing, remove_small_islands
│   ├── incisor.py           segment_incisor
│   ├── molar_bone.py        segment_molar_bone
│   └── postprocessing.py    postprocess_incisor, split_molars
│                             (available helpers, not yet wired into pipeline.py)
│
├── visualization/
│   ├── __init__.py
│   └── viewers.py           create_3d_visualization (napari)
│
└── tests/
    ├── __init__.py
    ├── run.py               Zero-config wrapper for test_pipeline.py
    └── test_pipeline.py     Interactive TUI for per-sample segmentation & inspection
```

---

## Adding New Structures

1. Add a new segmentation module under `segmentation/`, e.g. `segmentation/enamel.py`.
2. Export the function from `segmentation/__init__.py`.
3. Call it from `pipeline.py` after the incisor removal step, following the same mask → `np.where` → save pattern.

## Adding New I/O Formats

Add a loader/saver to `data_io/loaders.py` (or a new file in `data_io/`) and export it from `data_io/__init__.py`.
