# Mandible Segmentation Pipeline

Segments three structures from mouse mandible microCT scans:
- **Incisor** — the continuously growing incisor tooth
- **Bone** — the mandibular bone body
- **Molar** — the molar teeth (optionally split into individual instances)

---

## Usage

```bash
python main.py --input_path /path/to/samples [--log] [--visualize] [--debug] \
               [--out /path/to/output] [--format tif|nrrd] [--voxel-size 0.02] \
               [--incisor-margin-shrink 0.05]
```

| Flag | Short | Description |
|---|---|---|
| `--input_path` | `-i` | Path to a `.nrrd`/`.tif` file, a directory of `.bmp` or DICOM slices for a single sample, or a root directory containing one sub-folder per sample |
| `--log` | `-l` | Also write a timestamped run log into the output directory. The console is always logged; the file always records `DEBUG` detail regardless of the console level |
| `--verbose` | | Log at `DEBUG`: per-slice tracking decisions and every individual check, including those that passed |
| `--quiet` | `-q` | Log at `WARNING`: only problems and failed checks |
| `--no-color` | | Disable coloured console output (also implied when stderr is not a TTY or `NO_COLOR` is set) |
| `--strict` | | Abort a sample as soon as a stage's validation checks fail, instead of writing an output known to be wrong |
| `--validation-report` | | Path for the JSON check report (default: `<out>/validation_report.json`) |
| `--visualize` | `-v` | Open napari 3D viewer after each segmentation |
| `--debug` | `-d` | Open napari viewer after every major pipeline step for inspection |
| `--out` | | Output root directory (default: `./segmentation_results`) |
| `--format` | `-f` | Output container: `tif` (default) or `nrrd`. Both bake in the voxel size |
| `--voxel-size` | | Override the voxel size in mm — one value for isotropic (`0.02`) or three as `dz,dy,dx` (`0.03,0.01,0.01`) |
| `--incisor-margin-shrink` | | Trim this much off the incisor surface in mm (default `0`, no shrink). Incisor only |
| `--apply-dicom-rescale` | | DICOM inputs only: apply the header's RescaleSlope/Intercept so saved voxel values are in physical units (default: raw stored values) |
| `--workers` | | Threads for the margin-shrink distance transform (default: one per core) |

### Supported input formats

The pipeline accepts several input layouts:

| Input | How it's handled |
|---|---|
| Directory of `.bmp` slices | Loaded as a (Z, Y, X) stack via `load_bmp_stack()`. Files matching `*spr.bmp` are excluded. |
| Directory of DICOM slices | Assembled into a (Z, Y, X) volume via `load_dicom_series()`, using the series' own headers for slice order, voxel values and spacing (see below). |
| `.nrrd` file | Loaded and transposed from NRRD (X, Y, Z) to pipeline (Z, Y, X) convention. |
| `.tif` / `.tiff` file | Loaded directly as a 3D volume. |
| Directory containing any of the above | Auto-detected in priority order: `.nrrd` > `.tif`/`.tiff` > DICOM > `.bmp` stack. |

DICOM is detected by the `DICM` magic number rather than by file extension, so
a series whose slices have no extension — the common scanner export — is picked
up as-is. DICOM is tried before `.bmp` because a folder holding both is an
export whose BMPs are preview renderings, while the DICOMs carry the header
geometry and the full stored bit depth.

#### What is read from the DICOM headers

A DICOM series spreads its geometry across one header per slice, so three
things are recovered from the headers rather than assumed:

- **Voxel values** — kept as stored. `RescaleSlope` / `RescaleIntercept` are
  **not** applied by default, matching `tools/convert_DICOM_NRRD.py`, so
  window/level stays free to adjust downstream. This does not affect
  segmentation: the volume is min-max normalized before any threshold is taken,
  so an affine rescale cannot move a boundary — it only changes the numbers
  written into the saved volumes. Pass `--apply-dicom-rescale` (or
  `apply_rescale=True` to `load_dicom_series()`) to convert to physical units;
  slope/intercept are then read per slice, so a series with a varying intercept
  still lands on one scale. The dtype follows the data: an unrescaled series
  keeps its stored width so peak memory is not doubled, an integer-valued
  rescale stays integer, and a fractional one becomes float32.
- **Slice order** — slices are sorted by physical position (`ImagePositionPatient`
  projected onto the slice normal, which is correct for oblique series too),
  falling back to `InstanceNumber`. Filenames are never trusted for ordering.
- **Voxel spacing** — in-plane from `PixelSpacing`; the Z step is measured as the
  median gap between consecutive slice positions rather than read from
  `SliceThickness`, which describes the reconstructed slab and disagrees with
  the true spacing for overlapping or gapped reconstructions. A non-uniform
  step logs a warning. `SpacingBetweenSlices` then `SliceThickness` are the
  fallbacks when positions are missing.

If a directory holds more than one series, the largest is loaded and the rest
are skipped with a warning — mixing them would interleave unrelated slices into
one volume.

**Single slice-stack sample** — point `--input_path` directly at a folder of
slices, either BMP or DICOM:
```
/path/to/Sample_01/          /path/to/Sample_02/
├── slice_0001.bmp           ├── IM_0001
├── slice_0002.bmp           ├── IM_0002
└── ...                      └── ...
```

**Batch of samples** — point `--input_path` at the root; each sub-folder is treated
as one slice-stack sample (BMP or DICOM), or contains its own `.nrrd`/`.tif`:
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

**Output** is written to `<out_root>/<sample_name>/`, with the extension set by
`--format` (`.tif` shown here, `.nrrd` otherwise):
```
segmentation_results/
└── Sample_01/
    ├── reoriented_volume.tif
    ├── voxel_size.json
    ├── volumes/
    │   ├── incisor_volume.tif
    │   ├── bone_volume.tif
    │   └── molar_volume.tif
    └── masks/
        ├── incisor_mask.tif
        ├── bone_mask.tif
        └── molar_mask.tif
```

TIFF volumes are zlib-compressed BigTIFF; NRRD volumes are gzip-compressed.
Masks are uint8 (0/1) in either format.

### Voxel size and physical units

Both output formats record the voxel size, so a result opens at true physical
scale instead of as a unitless voxel grid:

- **NRRD** stores a full `space directions` matrix (LPS, mm), matching what
  `tools/convert_ISQ_NRRD.py` and `tools/convert_DICOM_NRRD.py` write. A
  segmentation therefore overlays the scan it came from in 3D Slicer / ITK-SNAP.
- **TIFF** has no 3-D geometry field, so the ImageJ convention is used:
  `XResolution`/`YResolution` for the in-plane size plus a `spacing=` entry in
  the ImageDescription for the Z step. Fiji reads this back as the voxel depth.
- `voxel_size.json` records the same numbers in plain text for analysis scripts
  that measure in physical units without reopening a multi-GB volume.

The spacing is read from the input's header — the `space directions` matrix for
NRRD, the resolution tags for TIFF, and `PixelSpacing` plus the inter-slice
displacement for DICOM. **Reorientation permutes the
array axes**, so the per-axis spacing is permuted with it — an anisotropic scan
keeps the correct physical size on each output axis. Inputs that carry no
spacing (BMP stacks, TIFFs without resolution tags) fall back to 1 mm isotropic
and log a warning; pass `--voxel-size` to supply the real value. An override is
stated in the *input* frame and is mapped through the same reorientation.

### Incisor margin shrink

`--incisor-margin-shrink` erodes the finished incisor mask by a physical
distance, e.g. `--incisor-margin-shrink 0.05` for 50 µm.

The incisor/bone boundary is not a clean step: voxels straddling it are a
partial-volume blend of both tissues. For measurements that read intensities
inside the mask — mineral density, enamel thickness — that rim biases the
result toward bone. Shrinking pulls the mask back to voxels that are
unambiguously incisor.

Points worth knowing:

- **Incisor only.** Bone and molar masks are never shrunk.
- **The margin is physical, not a voxel count.** It is divided by the voxel
  size per axis, so the same value trims the same real distance on scans of
  different resolution, and an anisotropic scan is trimmed by an ellipsoid
  rather than a ball. Where no voxel size is available, the value is read as a
  voxel count instead.
- **Only the output shrinks.** Bone/molar segmentation still subtracts the
  *untrimmed* incisor, so the shaved rim is discarded rather than relabelled as
  bone — no structure claims those voxels.
- **It degrades safely.** A margin finer than one voxel leaves the mask
  unshrunk and logs a warning. A margin large enough to erase the mask raises
  an error naming the specimen's deepest point, so an unusable value fails
  immediately instead of after a long run.
- **It is parallel.** The shrink is a Euclidean distance transform (O(volume)
  regardless of margin width), split across overlapping Z slabs — each padded
  by the margin so results are bit-identical to the serial path. Roughly 6x on
  12 cores; `--workers 1` forces serial.

Because it removes real tissue from the mask, keep the value at or below the
partial-volume width (roughly one to two voxels) and hold it fixed across every
sample in a comparison — the shrink changes volume and surface measurements.
At an 8 µm voxel that is roughly `0.008`–`0.016`; values like `0.5` are a
half-millimetre off every surface and will erase a mouse incisor entirely.

### Automated validation

Every run checks its own output. The pipeline has no ground truth to score
against, so each check is a *plausibility* test: it encodes something known
about a mouse hemimandible and flags an output that violates it. The aim is to
catch what is obvious to a human glancing at a napari window — the mask
swallowed the whole jaw, the incisor broke into pieces, the seed latched onto
a molar — without anyone having to open that window for every sample.

Checks run automatically; there is no flag to enable them. Results go to the
log, to a per-run table, and to `validation_report.json`.

**Why the incisor needs this.** The segmentation is a chain of greedy per-slice
decisions: pick the brightest region on the seed slice, then follow the nearest
centroid forward. Every link can fail quietly. The seed can land on a molar
cusp, the centroid chase can hop onto bone where the two touch, and the final
region grow can leak through a partial-volume bridge and return the entire
hemimandible as "incisor". All three produce a mask that looks like *a*
structure — the failure is only visible if you know the shape an incisor
should have.

| Stage | Checks |
|---|---|
| **Loaded volume** | 3D shape; no axis too thin to be a complete load; no NaN/Inf; non-zero dynamic range; not a single constant value |
| **Background removal** | Foreground fraction within a plausible band (threshold neither erased the specimen nor kept the background); specimen dominates the surviving components |
| **Reorientation** | Axis 0 really is the longest; cross-section is X ≥ Y; a tip-driven transform was actually recorded |
| **Incisor isolation** | Bone removal retained a sensible share of the CLAHE volume — not everything (no isolation happened), not nothing (the incisor went with the bone) |
| **Incisor seed** | A seed was found at all; it is small enough to be a tooth cross-section rather than a slab of bone; it sits near the posterior end |
| **Incisor mask** | Share of foreground; single connected component; slice coverage along the jaw; no interior gaps; elongation; slice-to-slice area stability; centroid trajectory smoothness; brighter than surrounding bone; physical volume in mm³ |
| **Margin shrink** | Enough of the mask survived; the shrink did not pinch the tooth apart |

Each result is `PASS`, `WARN` or `FAIL`. A **warning** means unusual, worth an
eyeball. A **failure** means the output is very likely wrong and should not be
used as-is. The `--strict` flag turns a failure into an abort for that sample;
without it the sample still finishes, which is what you want while judging
whether a threshold is calibrated right for your data. `main.py` exits `1` if
any sample failed, so a shell-driven batch can notice without parsing the log.

At the end of a run, a table names the samples needing attention:

```
Segmentation validation
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Sample    ┃ Status ┃ Checks ┃ Problems                         ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Sample_01 │ OK     │     22 │ —                                │
│ Sample_02 │ FAIL   │     21 │ Incisor mask/share of foreground │
└───────────┴────────┴────────┴──────────────────────────────────┘
```

with each problem then spelled out in full underneath:

```
Sample_02 — [Incisor mask] share of foreground: incisor is 74.1% of the
segmented foreground (881,204 of 1,189,533 voxels) — the mask has grown into
the surrounding bone; it is not an incisor alone
```

**Tuning.** The thresholds are deliberately loose — set to catch gross failure,
not to enforce a tight morphometric prior. Every one is a keyword argument on
the `check_*` function, so a study with unusual anatomy (a young mouse, a
knockout with a stunted incisor) can widen them at the call site in
`pipeline.py` rather than editing `validation/checks.py`. If a check fires
consistently on data you have inspected and trust, that is the threshold
needing adjustment, not the data.

### Logging

Console output goes through `rich` when available — level-coloured, aligned,
with wrapped messages — and falls back to ANSI-coloured plain text otherwise.
Log records are routed through `tqdm.write`, so progress bars are cleared and
redrawn instead of being smeared across the scrollback. Colour is dropped
automatically when stderr is not a TTY or `NO_COLOR` is set.

With `--log`, the file handler always records at `DEBUG` regardless of the
console level: when a run turns out to have gone wrong, the detail is already
on disk and the run need not be repeated with `--verbose` to find out why.


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
Input volume (BMP stack / DICOM series / NRRD / TIFF)
    │
    ▼
[1] Load volume                       data_io/loaders.py, data_io/dicom.py
    │  load_bmp_stack / load_dicom_series / load_nrrd / load_tiff
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
    │  Input: the whole background-removed foreground minus the
    │  dilated incisor (not the incisor-descent volume, which is cut
    │  far above bone and dentin intensity).
    │  Primary — enamel-disconnection threshold search:
    │    • After incisor removal, enamel (brightest tissue) only
    │      exists in molars.
    │    • Enamel threshold = valley below the enamel peak of the
    │      LOG foreground histogram. Enamel is ~3% of voxels, too
    │      small to register as a peak on a linear histogram.
    │    • Bone anchor = foreground outside the enamel bounding box
    │      grown by 4% of the long axis. A fixed anchor makes
    │      "enamel attached to bone" monotone in the threshold.
    │    • Coarse sweep up from the foreground floor (10% steps)
    │      until enamel detaches from the anchor, then bisect the
    │      bracket (0.5% tolerance). ~12 labellings in total.
    │    • Molars = enamel-bearing components at that threshold,
    │      regrown by a 2-voxel rim (marker watershed) to recover
    │      the surface the threshold shaved off.
    │    • Rest of the foreground → bone.
    │  Fallback (no enamel, or it never detaches) — erosion-based:
    │    • Iteratively erode foreground until ≥ 2 components appear.
    │    • Highest max-intensity component → molar; rest → bone.
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
├── logging_setup.py         setup_logging — rich console + DEBUG file log,
│                            tqdm-safe, re-entrant
│
├── validation/
│   ├── __init__.py
│   ├── checks.py            check_loaded_volume, check_preprocessed_volume,
│   │                        check_reorientation, check_isolation_volume,
│   │                        check_incisor_seed, check_incisor_mask,
│   │                        check_shrink; CheckReport / CheckResult / Severity
│   └── summary.py           ValidationSummary, SampleValidation —
│                            end-of-run table and validation_report.json
│
├── data_io/
│   ├── __init__.py
│   ├── dicom.py             load_dicom_series, is_dicom_dir, find_dicom_files,
│   │                        spacing_from_dicom_headers
│   ├── loaders.py           load_bmp_stack, load_nrrd, load_tiff,
│   │                        save_volume, save_mask, save_as_nrrd,
│   │                        save_ct_volume_as_tiff, save_mask_as_tiff
│   └── spacing.py           spacing_from_nrrd_header, spacing_from_tiff,
│                            permute_spacing_for_record
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
    ├── test_pipeline.py     Interactive TUI for per-sample segmentation & inspection
    └── test_validation.py   Checks the checks: synthetic good/bad masks with a
                             known verdict (`python tests/test_validation.py`)
```

---

## Adding New Structures

1. Add a new segmentation module under `segmentation/`, e.g. `segmentation/enamel.py`.
2. Export the function from `segmentation/__init__.py`.
3. Call it from `pipeline.py` after the incisor removal step, following the same mask → `np.where` → save pattern.

## Adding New I/O Formats

Add a loader/saver to `data_io/loaders.py` (or a new file in `data_io/`) and export it from `data_io/__init__.py`.
