# Longitudinal Incisor Analysis

Two stages, run in order, that together turn a segmented mouse incisor into a **mineralization
trajectory** — density as a function of position along the tooth:

| Stage | Script | Takes | Produces |
|---|---|---|---|
| **1. Reslice** | [cmpr.py](cmpr.py) | Segmented incisor volume | `<sample>_resliced_volume.nrrd` — z straightened along the tooth |
| **2. Quantify** | [enamel_density.py](enamel_density.py) | That reslice | Per-slice enamel density, ratio, and status |
| **3. Locate** *(optional)* | [detect_landmarks.py](detect_landmarks.py), [map_landmarks.py](map_landmarks.py) | The segmentation + stage 1's transform | Anatomical landmarks (M1/M2/M3, eruption) found, or picked, and placed on the trajectory |

The mouse incisor grows continuously, so its length *is* a developmental time axis: tissue at the
cervical loop was deposited most recently, tissue at the incisal tip is fully mature. Plotting
density against position therefore traces the entire mineralization trajectory in a single
specimen — from unmineralized secretory matrix, through maturation, to mature enamel.

That only works if position along the array is position along the *tooth*. The incisor is curved,
so a raw z-slice cuts it obliquely and mixes developmental stages within one plane. Stage 1 removes
the curve; stage 2 assumes it is gone. The two are documented separately below.

```bash
# Stage 1 -- straighten
python cmpr.py -i ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_incisor_volume.nrrd" \
    -o ".../CMPR" --log

# Stage 2 -- quantify
python enamel_density.py -i ".../CMPR/Odam-CRE-A_resliced_volume.nrrd" \
    -o ".../RESULTS" --log

# Stage 3 -- locate anatomy on the curve (optional): found from the segmentation...
python detect_landmarks.py -t ".../Segments/Odam-CRE-A/Odam-CRE-A_cmpr_transform.npz" \
    --incisor-volume ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_incisor_volume.nrrd" \
    --bone-volume ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_bone_volume.nrrd" \
    --molar-volume ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_molar_volume.nrrd" \
    --marks-out ".../RESULTS/Odam-CRE-A_marks.csv" -o ".../RESULTS/Odam-CRE-A_landmarks.csv"

# ...or picked by hand in napari
python map_landmarks.py -t ".../Segments/Odam-CRE-A/Odam-CRE-A_cmpr_transform.npz" \
    --profile ".../RESULTS/enamel_density_profile.csv" \
    -o ".../RESULTS/Odam-CRE-A_landmarks.csv"
```

---

# Stage 1 — Curved Multi-Planar Reconstruction (`cmpr.py`)

`cmpr.py` resamples the incisor along its own curve, so that every output slice is a true
perpendicular cross-section and the output z-axis is arc length down the tooth.

It is **intensity-preserving**: the resampled voxel values are the scanner's own, never rounded,
rescaled, or coerced. That is what stage 2 requires, since it fits Gaussians to slice histograms and
reports peak centres to sub-unit precision.

This is [cmpr.py](cmpr.py), in this folder — the one stage 2 reads. Other copies of this geometry
exist elsewhere in the repo for segmentation work; they are not interchangeable with this one, and
running one of those instead silently degrades the intensities the density fit depends on.

## What "intensity-preserving" means here

Three properties, all about not corrupting the numbers the density fit reads:

1. **Interpolation happens in float, and stays float.** `map_coordinates` returns the *input's*
   dtype, so resampling an int16 volume rounds every interpolated voxel back to a whole number.
   Measured on an int16 phantom with dentin-like statistics (μ = 143, σ = 12): the int16 path yields
   **71 distinct intensities where the float path yields 199 171**, a mean absolute error of 0.25
   intensity units. That is a quantization grid ~2 % of the ~11-unit dentin width stage 2 resolves
   peaks against — imposed on every voxel.
2. **No dtype coercion on save.** Nothing casts the array on the way out — a float32 reslice is
   written as float32. (The segmentation-facing variants floor anything that is not int16/float32
   to int16, which silently truncates a float volume.)
3. **The volume is never masked, thresholded, or rescaled.** A binary mask is derived from the
   volume to drive the *geometry*, but intensities that reach the output are the scanner's own — so
   stage 2's saturation logic, which keys off the volume maximum, stays meaningful.

Interpolation is **trilinear** (`INTERPOLATION_ORDER = 1`), the highest order that cannot overshoot.
Cubic rings at the enamel/air edge, inventing intensities above the scan's ceiling and below its
floor — exactly the out-of-range values stage 2's bounded fit is built to distrust.

## Usage — reslicing

```bash
# A single volume
python cmpr.py -i ".../Segments/Odam-CRE-A/volumes/Odam-CRE-A_incisor_volume.nrrd" \
    -o results/Odam-CRE-A --log

# Every sample under a directory of per-sample folders
python cmpr.py -i ".../Segments" -o results --log
```

`--input` accepts an incisor volume, a sample folder, or a directory of sample folders; in the
latter cases each sample's volume is located by trying `volumes/*_incisor_volume.nrrd`, then the
`.tif` equivalent, then the same names unnested.

### Key flags — reslicing

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | Incisor volume, a sample folder, or a directory of sample folders |
| `--output` | `-o` | `./results` | Directory to write results to |
| `--format` | `-f` | `nrrd` | Container for the reslice (`nrrd` or `tif`) |
| `--slice_size` | | auto | Slice width/height in px. Unset = auto-sized to contain the cross-section |
| `--spacing` | | `1.0` | In-plane sampling step, in voxels per pixel |
| `--margin` | | `1.2` | Safety factor on the auto-sized slice window |
| `--per-sample-dir` | | off | Write each sample into its own `<output>/<sample>/` |
| `--visualize` | `-v` | off | Interactive 3D visualization of centerline and slice planes |
| `--log` | `-l` | off | Coloured console + file logging |

### Output — reslice

```
<output>/
├── <sample>_resliced_volume.nrrd    z straightened along the tooth, float32
└── <sample>_cmpr_transform.npz      the geometry used, for mapping back (stage 3)
```

The `.npz` sidecar records the centerline, the frames, the crop origin, and the
voxel spacing — everything needed to map a position on the reslice back to a position in the
original scan. Stage 2 ignores it; [stage 3](#stage-3--mapping-anatomy-onto-the-curve-map_landmarkspy)
is what reads it.

NRRD is the default because it carries voxel spacing as an explicit `space directions` matrix
rather than TIFF's resolution tags, so the reslice opens at true physical size in Slicer/ITK-SNAP,
and stage 2 reads it through the same `load_nrrd` it uses for the molar cross-check.

The filename carries the sample name deliberately: **stage 2 derives its sample label from the
input basename**, so a bare `resliced_volume.nrrd` makes every specimen's results read as
"resliced_volume".

## How the reslice works, step by step

```
<sample>_incisor_volume.nrrd
    │
[1] Mask + crop ──────────────── binary_mask = volume > 0, closed, holes filled
[2] Skeletonize ──────────────── skeletonize()
[3] Graph diameter path ──────── two-pass Dijkstra -> longest path = centerline
[4] Smooth (B-spline) ────────── smooth_centerline()
[5] Extend to mask extent ────── extend_centerline_to_mask_extent()
[6] Rotation-minimizing frames   compute_local_frames()
[7] Auto-size the window ─────── compute_required_half_extent()
[8] Resample one plane / point   resample_slice(), trilinear, in float
    │
    ▼
<sample>_resliced_volume.nrrd
```

**[1] Mask and crop.** The mask is `volume > 0`, morphologically closed with a radius-2 ball and
hole-filled slice by slice, then both volume and mask are cropped to its bounding box. The volume
is cast to float32 exactly once here — everything downstream samples that array, and
`map_coordinates` preserves its dtype.

**[2–3] Centerline.** The mask is skeletonized, the skeleton voxels become a weighted graph, and the
centerline is the **graph diameter** — found by two-pass Dijkstra (farthest node `u` from an
arbitrary source, then farthest node `v` from `u`, then the path between them). Taking the longest
path discards skeleton side-branches, which are spurs off the main axis rather than tooth.

**[4] Smoothing.** A B-spline (`splprep`/`splev`, smoothing factor `25 × n_points`) removes the
voxel-staircase jitter of the raw skeleton path. Jitter matters here because frames are built from
the *tangent*: a staircased centerline yields a tangent that wobbles slice to slice, and every
wobble tilts a cross-section.

**[5] Extension.** The skeleton stops short of the tooth's two ends, so both ends are extrapolated
outward along their local tangent, in 0.5-voxel steps, until they exit the mask. Without this the
reslice omits the tip and the cervical end — the two regions the trajectory most depends on.

**[6] Frames.** Each slice plane is spanned by two vectors perpendicular to the tangent, propagated
along the curve as a **Rotation Minimizing Frame** (double-reflection method, Wang et al. 2008).
Each frame is derived from the previous one with the minimal twist needed to track the tangent,
rather than independently from a fixed "up" vector. A fixed reference vector flips discontinuously
where the tangent passes near it, and independent per-point frames jitter; either would rotate the
cross-section from slice to slice, which stage 2's angular coherence check (step 7) would then read
as a moving enamel cap.

**[7] Window sizing.** Unless `--slice_size` is given, the window is sized from the data: the
largest distance from the centerline to any mask surface voxel, doubled, times `--margin`. An
undersized window clips the cross-section and truncates its histogram — silently removing the
brightest enamel voxels, which is precisely the population stage 2 is looking for.

**[8] Resampling.** One plane per centerline point, sampled on a `slice_size × slice_size` grid at
`--spacing` voxels per pixel, trilinear, with `FILL_VALUE = 0.0` outside the volume (stage 2 treats
0 as background). The centerline is arc-length parameterized and z advances one centerline point per
slice, so **output z ≈ 1 voxel per slice** — which is what lets stage 2 report position as a simple
percentage along the array.

## What stage 2 inherits from this

Stage 1's quality sets stage 2's ceiling, in specific ways:

- **An oblique reslice** — a bad centerline, or frames that twist — produces elongated
  cross-sections, which inflate the rim fraction and push slices into `unresolvable`.
- **A rotating frame** moves the enamel cap's angular position from slice to slice, which the
  spatial-coherence test can read as artifact rather than tissue.
- **A clipped window** truncates the bright tail, removing the enamel population entirely.
- **Any rescaling or dtype coercion** breaks the saturation check and the sub-unit peak centres.

If stage 2's results look wrong in a way the per-slice statuses do not explain, inspect the reslice
before touching the fit. Open it in Slicer and scroll z: every slice should be a plausible tooth
cross-section, roughly centred, and roughly constant in orientation.

---

# Stage 2 — Enamel density quantification (`enamel_density.py`)

`enamel_density.py` measures **enamel density as a function of position along the incisor**, from
the reslice stage 1 produced.

## The problem this solves

[`segment_enamel_WIP/enamel_seg.py`](../segment_enamel_WIP/enamel_seg.py) tries to *delineate* enamel voxel-by-voxel. Its own README records
where that breaks: enamel is brighter than dentin near the tip and dimmer near the base, so a
per-slice intensity label silently flips partway down z, and precision decays 0.77 → 0.27 with no
warning.

This module never segments anything. It asks a purely statistical question of each slice's
intensity histogram —

> *is there a mineral population distinguishable from dentin, and where is it centred?*

— and, critically, **is allowed to answer "no"**. The position-dependent contrast collapse becomes
an explicit, reported outcome instead of a silent failure.

It also **resolves enamel that is less dense than dentin**, not only enamel that is denser. That
requirement drives most of the design below, because the obvious methods are blind to it — and it
is what makes the secretory stage measurable at all.

---

## Assumptions

These are the premises the measurement rests on. If one is false for your specimen, the
corresponding result is the one to distrust.

1. **The CMPR z-axis follows the tooth.** Each slice is a true cross-section, not an oblique cut.
   This is [stage 1](#stage-1--curved-multi-planar-reconstruction-cmprpy)'s job; a bad reslice
   produces elongated cross-sections and inflated rim fractions here. See
   [What stage 2 inherits](#what-stage-2-inherits-from-this).
2. **Dentin is present in every measurable slice, at a slowly varying density.** This is what makes
   dentin usable as the internal reference. Measured drift on the reference specimen is ~26
   intensity units over ~770 slices — well under one unit per slice.
3. **Enamel, where it exists, forms a labial cap** — one connected region covering one angular
   sector, not a ring around the whole perimeter. This is the assumption that separates real tissue
   from partial-volume artifact (step 7).
4. **Each tissue's intensities are approximately Gaussian about a central value.** Real histograms
   are skewed, so this is an approximation; step 5 explains where that matters and how it is handled.
5. **Intensity is monotonically related to mineral density.** Higher intensity = more mineral. This
   permits ranking and ratios, **not** absolute mineral units — see [Units](#units-and-what-they-mean).
6. **The scan is not clipped.** Verified for the reference specimen (see
   [Saturation](#7-saturation-is-a-property-of-the-data)); the pipeline re-checks per slice.
7. **One specimen, one scanner session.** The dentin anchor and the null are both derived from the
   specimen itself, so nothing is comparable across sessions in raw intensity. The **ratio** is.

---

## Usage — quantification

```bash
python enamel_density.py \
  --input  "/Volumes/Z Drive/Incisor Enamel Analysis/CMPR/Odam-CRE-A_resliced_volume.nrrd" \
  --molar  "/Volumes/Z Drive/Incisor Enamel Analysis/SEG DATA/Odam-CRE-A/volumes/Odam-CRE-A_molar_volume.nrrd" \
  --output "/Volumes/Z Drive/Incisor Enamel Analysis/RESULTS" --log
```

Run the validation suite (synthetic tests always run; real-data tests skip if the volume is not
mounted):

```bash
python test_enamel_density.py [path/to/resliced_volume.nrrd]
```

### Key flags — quantification

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | *(required)* | CMPR resliced incisor volume (`.tif` or `.nrrd`) |
| `--molar` | `-m` | none | Molar volume: supplies the flat dentin baseline (step 6) and an independent cross-check |
| `--output` | `-o` | `<input dir>/enamel_density` | Output directory |
| `--baseline` | | `molar` | Dentin reference: `molar` (flat, needs `--molar`) or `trace` (per-slice core) |
| `--max-baseline-delta` | | `1.5` | Largest tolerated molar-vs-incisor-anchor disagreement, in dentin σ |
| `--orientation` | | `auto` | `auto`, `tip-low-z`, or `tip-high-z` |
| `--stride` | | `1` | Analyze every Nth slice (use `10` for a fast preview) |
| `--erosion` | | `2` | Rim erosion iterations — **do not set to 0**, see step 1 |
| `--max-rim-fraction` | | `0.50` | Rim fraction above which a slice is `unresolvable` |
| `--gate-scale` | | `1.15` | Safety factor on the bootstrapped significance threshold |
| `--log` / `--debug` | `-l` / `-d` | off | Coloured console + file logging / diagnostics |

Full list: `python enamel_density.py --help`.

### Outputs

```
enamel_density/
├── enamel_density_profile.csv    per-slice results, ordered cervical loop -> tip
├── enamel_density_profile.png    enamel density (raw scanner intensity) vs position
├── enamel_density_ratio.png      enamel / dentin ratio vs position
├── enamel_density_run.json       full provenance: anchor, null table, constants
└── run_<timestamp>.log
```

---

## The per-slice outcomes

| Status | Meaning | Plotted as |
|---|---|---|
| `resolved_hi` | A distinct population **denser** than dentin — mineralized enamel | Filled blue markers, connected |
| `resolved_lo` | A distinct population **less dense** than dentin — secretory matrix | Filled red squares, connected |
| `coincident` | The fit is sound but nothing separates — **enamel is at the same density as dentin** | Open orange markers at the dentin level, plus a shaded band |
| `saturated` | Voxels are piled against the scan's intensity ceiling — the value is a **lower bound** | Purple carets, plus a shaded band |
| `insufficient` | Too few voxels to decide | Grey shading |
| `unresolvable` | Cross-section is mostly partial-volume rim; there is no interior to measure. Also a slice just tipward of the enamel onset where the matrix was too thin to measure and the fit fell back to the dentin (see below) | Grey shading |
| `pre_onset` | Cervical to the enamel onset, the profile's density trough: no enamel, so density and ratio are 0 | Grey shading |

**Where enamel starts.** Enamel begins as its least dense form, secretory matrix, and only gets
denser toward the tip, so the profile's density trough is the onset (`trim_pre_onset()`) and
everything cervical to it is `pre_onset`. Just tipward of the onset the matrix is a thin, patchy
layer. Where its fragments fail the solidity test, the fit rejects them as rim and falls back to the
dentin, reporting `coincident` at the dentin's level. Left in, those readings open the profile high;
it then drops to the matrix before the true rise. Until the matrix has risen halfway from its onset
density to the dentin's (`OPENING_RISE`, a median over the next 21 resolved slices), no enamel can
be that dense, so `withdraw_opening_fallbacks()` marks those slices `unresolvable` and keeps their
dentin fit. Tipward of that, enamel does approach dentin, and `coincident` stands. On the P22 TGM2
cohort this withdrew 0–123 slices per specimen, all reading 0.77–0.83 against matrix at 0.47–0.57.

`coincident` is a *measurement*, not a failure: it is the honest report when enamel and dentin have
converged in density. `unresolvable` is likewise a refusal to invent a number where the data cannot
support one.

### Cross-specimen agreement

Run over three WT scans (`Odam-CRE-A/B/C`), the recovered mineralization trajectories agree, which is
the real test — the three teeth differ in cropping, in cervical mineralization, and in whether the
scan is clipped, so nothing here is guaranteed by construction. Mean enamel/dentin ratio over
resolved slices:

| z % | 15–20 | 25–30 | 35–40 | 45–50 | 55–60 | 65–70 | 75–80 | 85–90 | 95–100 |
|---|---|---|---|---|---|---|---|---|---|
| **A** | 0.60 | 0.88 | 1.45 | 1.59 | 1.63 | 1.77 | 1.78 | 1.69 | 1.40 |
| **B** | 0.60 | 0.76 | 1.29 | 1.52 | 1.60 | 1.65 | 1.65 | 1.60 | 1.36 |
| **C** | 0.62 | 0.95 | 1.38 | 1.59 | 1.68 | 1.84 | 1.80 | 1.65 | 1.36 |

All three trace the same curve: matrix at ratio ≈ 0.6, crossing 1.0 near z ≈ 30 %, plateauing at
1.6–1.8. B's `coincident` band at z = 30–35 % is exactly where **its** ratio crosses 1.0 — that is
the status doing its job, not a residual failure.

---

## How the measurement works, step by step

```
resliced_volume.nrrd  (CMPR output, z straightened along the tooth)
    │
    ├─ [0] Specimen calibration, once per volume ── build_anchor(), calibrate_null(),
    │                                               detect_orientation()
    ▼
   per slice:
    │
[1] Rim erosion ─────────────────────── slice_values()
[2] Validity gate ───────────────────── slice_validity()
[3] Pool neighbours until enough voxels  slice_values(window)
[4] Fit one Gaussian, then two ───────── fit_one() / fit_two()
[5] Is the second one real? ──────────── decompose() / gate_for()
[6] Which one is dentin? ─────────────── decompose()
[7] Does the low-density side look like tissue?  spatial_coherence()
[8] Runaway-fit rescue, saturation check  refit_narrow()
    │
    ▼
CSV + PNG, 0% = cervical loop → 100% = tip
```

### [0] Specimen calibration (once)

Three things are established before any slice is measured.

**The dentin anchor.** Voxels are sampled from the middle 30 % of the tooth (`ANCHOR_BAND`), where
the cross-section is widest and the dentin peak is cleanest, and a single Gaussian is fitted. On the
reference specimen this gives **dentin = 129.7, width σ = 11.3**. Every later slice is judged
relative to this.

**The fit bounds.** Both fitted centres are confined to `[anchor − 6σ, bright tail]` — on the
reference, `[62, 248]`. Without bounds the second component slides onto a histogram tail and returns
nonsense (measured: centres of 281.8 and 17.9 on a volume whose maximum is 248).

**The null.** See step 5 — this is what "real" is measured against.

Orientation is also detected here, never assumed. Nothing in the CMPR output fixes which array end
is which, and **the reference specimen has the tip at low z, the reverse of the plotting
convention.** Detection uses the fraction of dense voxels at each end (measured **0.308 at low z
against 0.000 at high z** — unambiguous) plus modal intensity. Taper does *not* discriminate,
because both ends taper; it is logged but not voted on.

### [1] Rim erosion

Two iterations of a 3×3 erosion strip the outer shell of each cross-section before histogramming.

This is mandatory. At a tooth edge, a voxel straddles tissue and air, so it records a blend of the
two — the **partial-volume effect**. Those blended voxels are dimmer than any real tissue. Without
erosion the modal intensity of a tapered slice collapses onto them: measured modes of **81 / 55 / 51**
at three cervical positions, against a true dentin peak above 120. Every real voxel then reads as
"excess" above dentin, fabricating up to **40 % phantom enamel** exactly where enamel is least
mineralized and the answer matters most.

Two iterations restore those modes to **115 / 93 / 67**. Three empties the smallest cross-sections.
`--erosion 0` is a diagnostic setting, not a supported one; the test suite asserts the artifact
reappears without it.

### [2] Validity gate

A cross-section that is *mostly* rim has no interior left to form a second population, so no honest
measurement is possible. Two physical checks:

- **rim fraction** — what share of the slice sits within 2 voxels of the surface — must be ≤ 0.50
- **interior depth** — how far the deepest point is from the surface — must be ≥ 8 voxels

Failing either gives `unresolvable`. This retires the extreme ends on physical grounds rather than
reporting numbers from an all-rim cross-section.

> **Note on the 0.50 threshold.** This was 0.25, set from the mineralized range alone, and it cut
> straight through the secretory zone — where the tooth is genuinely thinner but the tissue is still
> perfectly resolvable. The secretory cap survives to rim = 0.47 with a single coherent component.
> Since the *shape* check in step 7 is what actually separates tissue from artifact, the rim gate
> only needs to decide whether a slice can be measured at all.

### [3] Pool neighbouring slices until the histogram is usable

A single thin slice may hold too few voxels for a stable fit. The window grows symmetrically in z
(up to ±12 slices) until it holds **6000** voxels. This is safe because tissue properties vary
slowly along z (assumption 2). The window used is recorded per slice as `z_window`, so a wide window
is visible as lost z-resolution rather than hidden.

Below 250 voxels even after pooling, the slice is `insufficient`.

### [4] Fit one Gaussian, then two

Build a smoothed intensity histogram, then fit it twice: once with a single bell curve (the
"dentin only" explanation) and once with two (the "dentin plus something else" explanation).

The two-component fit is seeded from **both sides** of the dentin peak and at two separations, and
the best-fitting result wins. That symmetry is essential: nothing in the search prefers the dense
side, which is what allows enamel *less* dense than dentin to be found at all.

### [5] Is the second component real?

This is the core statistical decision, and the one that was most recently corrected.

Adding a second bell curve **always** fits better — more knobs always fit better. The question is
whether it fits *enough* better to be worth believing. Measure the improvement as the drop in χ²
(chi-square), a standard measure of total mismatch between model and data:

```
D  =  χ²(one curve)  −  χ²(two curves)
```

Large D means the second component explains something real. But how large is large? That depends on
the slice, so the threshold is **bootstrapped from the specimen's own dentin**: draw voxels from a
known dentin-only pool, run the identical two-curve fit, and record how much improvement pure noise
produces. The 99th percentile of that becomes the gate. A slice must beat its own null.

The null is a 2-D table over **sample size** and **dentin peak width**, because both matter — a
broad peak is intrinsically harder for a second component to improve upon — and each slice is judged
against its own local width, interpolated from the table.

> **Why D is a raw difference and not a ratio.** The textbook form divides by the reduced χ² of the
> two-component fit — an F-like normalization. That is only valid when the model can actually reach
> the data, and a sum of two Gaussians cannot: real tissue histograms carry skew and shoulders no
> Gaussian mixture removes (assumption 4). So the denominator ends up measuring *model inadequacy*,
> not noise — and the slices with the strongest enamel are the least Gaussian, so they got the
> largest denominators.
>
> The effect was to invert the comparison. Measured on the reference specimen:
>
> | | raw improvement | χ²/dof | normalized D |
> |---|---|---|---|
> | dentin-only null | 199 | 0.21 | **931** |
> | real slice, mid-tooth | 16,117 | 28.4 | **568** |
>
> A slice with **80× more evidence** for two peaks scored *lower* than pure noise. That single
> division flattened 898 of 916 slices onto the dentin line, discarding fitted enamel of 218 / 210 /
> 203 across the mid-tooth. Undivided, both sides are counted in the same units and separate
> cleanly: null p99 = 263 against real slices of 750–16,500.

Two further sanity requirements: the components must be separated by at least 0.75 dentin widths,
and the smaller must hold at least 2 % of the mass. Failing any of these gives `coincident`.

### [6] Which component is dentin?

Not by position — that would presuppose the answer. Not by mass either: at the mature tip enamel is
the *larger* population, and taking the heavier component as dentin inverted the pair there,
reporting `dentin = 230` with "enamel" at 136.

Dentin is identified by **continuity**: it is the one tissue present along the entire tooth at a
slowly varying density (assumption 2), so the component closer to the local dentin reference is
dentin, and the other is enamel. **Which side enamel falls on remains a pure output of the fit.**

> **The reference must be local, not global.** Comparing against one specimen-wide dentin mode
> assumes dentin holds a single density along the whole tooth. It does not, and the violation is
> specimen-specific: measured drift of the dentin core away from the global anchor is mild on
> Odam-CRE-A but reaches −20 intensity units on −B and −C, which are less mineralized cervically.
>
> Because this is a nearest-neighbour test with no margin, once drift passes roughly half the
> component separation the two roles simply swap. At Odam-CRE-C `z=880` the fit returns 116.4
> (82 % of the mass, one connected region covering 74 % of the cross-section — the tooth body) and
> 142.1 (a thin bright band). Against a global anchor of 129.5 the distances are 13.1 and 12.6, so
> **the tooth body loses by 0.5 intensity units**, is relabelled "enamel", and is then discarded by
> the low-side rim check as a ring. That single inversion produced **38 coincident slices on each of
> B and C against 4 on A** — the entire reason those two scans looked broken.

**The dentin trace.** `trace_dentin()` measures dentin per slice instead of assuming it, by reading
the **deep core** of each cross-section: every foreground voxel ≥ 6 voxels from the tissue surface.
Enamel is an outer cap over a dentin core (assumption 3), so the deepest tissue is dentin wherever
dentin exists at all. Median intensity by depth shell rises monotonically inward and flattens past
depth 6 on all three scans:

| | `[2-4)` | `[4-6)` | `[6-8)` | `[8-12)` |
|---|---|---|---|---|
| A, z ≈ 50 % | 132.3 | 131.4 | 130.5 | 131.0 |
| B, z ≈ 15 % | 71.7 | 97.2 | 111.4 | 117.8 |
| C, z ≈ 15 % | 80.8 | 98.9 | 114.4 | 113.7 |

The plateau is dentin; the rise below it is rim and, cervically, unmineralized matrix. Note B and C
at z ≈ 15 %: the outer shells read 72–99 while the core holds 111–118. A histogram-peak rule has to
*guess* which of those populations is dentin and gets it wrong specimen by specimen; the depth rule
simply reads the core. The resulting trace matches the measured core to **median 0.0 and worst-case
≈ 2 intensity units** on all three scans.

Two guards keep the trace honest. Slices whose core has shrunk below 30 voxels are skipped and
interpolated across. Slices whose core runs more than **3 σ above the global anchor** are dropped as
*enamel* core, not dentin — at the incisal tip the dentin horn tapers out and the deepest remaining
tissue is enamel, which would otherwise re-create the same inversion at the other end of the tooth
(measured "dentin = 217, enamel = 157" at z = 95 % on B before the ceiling was added). Real dentin
holds below +2.5 σ and the jump to enamel clears +3.7 σ in one step, so the two never overlap.

The traced value is written to the `dentin_reference` column so the reference is auditable per slice.

**The molar baseline (default).** The 3 σ ceiling above is a *step* detector, and it only catches a
step. On the P82 controls the tip contamination arrives as a gradual ramp that creeps in underneath
it: the trace tops out at 150.2 against a ceiling of 158.2 (+2.3 σ), and only 11 of 177 sampled
points are rejected. Measured on P82-Odam-CTR-1, tip end:

| z % | 80 | 84 | 88 | 90 | 92 | 94 | 96 |
|---|---|---|---|---|---|---|---|
| `dentin_reference` | 128.7 | 136.6 | 142.3 | 149.0 | 150.2 | 150.2 | 150.2 |
| `dentin_sigma` | 13.5 | 14.0 | 14.3 | 15.9 | 17.7 | 22.4 | 19.3 |
| `enamel_fraction` | 0.24 | 0.31 | 0.37 | 0.44 | 0.57 | 0.73 | 0.94 |
| `separation_sigma` | 6.84 | 5.96 | 5.52 | 4.82 | 4.31 | 2.60 | 2.71 |

The reference climbs 22 units while its own width inflates 65 % and the two components close from
6.8 σ to 2.6 σ apart. That is one population being read as two — not dentin mineralizing — and it is
the same enamel-core inversion the ceiling exists to prevent, arriving too slowly to trip it.

The deeper problem is that the trace is **circular at the tip**: it derives the dentin reference from
the very voxels it is trying to classify, at exactly the place where dentin has run out.
`molar_baseline()` breaks the circularity by taking dentin from the **molar** — fully mature,
geometrically thick, and not part of the structure whose composition is being measured — and holding
it flat along z, which is what assumption 2 says dentin should be anyway.

The molar value is validated against the mid-shaft incisor anchor before use. Both are reads of the
same tissue on the same scan, so they must agree; a disagreement beyond **1.5 σ** means the two
regions are not on a common intensity scale (differing beam hardening, or a mis-segmented molar) and
the baseline is refused, falling back to the trace — wrong only at the tip beats wrong everywhere.
Measured agreement is 0.30–0.42 σ on all three P82 controls:

| scan | molar dentin | incisor anchor | Δ |
|---|---|---|---|
| P82-Odam-CTR-1 | 119.9 | 124.5 | +4.6 (0.41 σ) |
| P82-Odam-CTR-2 | 119.6 | 122.9 | +3.3 (0.30 σ) |
| P82-Odam-CTR-3 | 118.2 | 121.7 | +3.5 (0.33 σ) |

Select with `--baseline molar` (the default, requires `--molar`) or `--baseline trace` for the old
behaviour; the choice and its diagnostics are recorded under `dentin_baseline` in the run JSON.

**The reference is also the ratio's denominator.** `enamel_ratio` is enamel over the dentin
*reference*, not over the slice's own fitted dentin component. This matters most exactly where the
fit is weakest. As the dentin horn tapers out near the tip the fitted dentin component has less and
less dentin to work with and drifts upward into the enamel peak, while enamel itself barely moves
(P82-Odam-CTR-1):

| z % | 88.6 | 91.5 | 94.3 |
|---|---|---|---|
| fitted `dentin_mode` | 137.7 | 142.3 | 149.3 |
| `enamel_density` | 217.1 | 218.5 | 228.5 |
| ratio ÷ fitted peak | 1.577 | 1.535 | 1.530 |
| ratio ÷ reference | **1.811** | **1.823** | **1.906** |

Dividing by the drifting peak makes the ratio *fall* toward the tip purely because the denominator is
climbing — reading as a less mineralized tip when the tip is the most mineralized part of the tooth.
Dividing by the fixed reference makes the denominator mean the same thing on every slice, which is
what makes ratios comparable along z and between specimens.

This also keeps the degenerate tip slices usable rather than discarding them. Beyond z ≈ 94 % the
decomposition can no longer cleanly separate two populations (separation 2.6 σ, `enamel_fraction`
0.94) — but the enamel numerator is still a real read of real tissue, and a fixed denominator does
not require the fit to have found a dentin population to divide by.

### [7] Does a low-density component look like real tissue?

Partial-volume blur pulls surface voxels *down*, never up. So a component **less dense** than dentin
is the one that can be counterfeited by rim, and it alone gets a shape check. (Applied symmetrically,
these checks rejected genuine mineralized enamel.)

The discriminator is **thickness**, measured as *solidity*: the fraction of the component surviving
two 3×3 erosions. Partial-volume rim is not a layer at all — it is the one-to-two-voxel blur across
the tissue boundary, so it **cannot have an interior**. Two erosions strip 2 voxels from every edge,
annihilating rim and leaving the core of anything genuinely thicker:

| | solidity |
|---|---|
| true cervical rim (z ≈ 0–10 %) | **0.00 – 0.01** |
| taper, mixed (z ≈ 10–15 %) | 0.01 – 0.07 |
| secretory matrix (z ≈ 15–20 %) | **0.07 – 0.30** |
| secretory matrix (z ≈ 20–30 %) | **0.10 – 0.39** |

A low-density candidate passes if solidity ≥ 0.10. Shape — angular extent ≥ 0.45 and at most 12
connected regions — is checked **only for components too thin to settle on thickness alone**, since
a thin component that also wraps the perimeter is rim on every available measure.

> **Angular extent alone is not sufficient, and using it alone discarded the secretory stage of two
> of three specimens.** Real enamel is a labial cap on the reference scan, but B and C lay matrix
> down around *most of the perimeter* at z = 10–30 %: measured angular concentration **0.02–0.15** on
> slices where the low component is 27–78 % of the cross-section and is plainly a thick layer, not a
> fringe. The old ≥ 0.45 rule rejected all of it. Solidity rescues these (0.20–0.38) while still
> refusing true rim (0.00), because it asks about the component's own geometry rather than where it
> sits in the tooth — so it stays valid wherever the secretory zone falls in a given specimen.

> **Depth is not the discriminator either, and using it discarded the entire secretory stage.** An
> earlier rule required a low-density component to sit ≥ 5 voxels deep. But enamel matrix is laid
> down as a thin *outer layer over dentin* — it is legitimately shallow, measured at 3.0–4.1 voxels
> through the secretory zone, against true rim at 3.6. The two overlap, so depth cannot separate
> them. Thickness can.

### [8] Runaway-fit rescue, and the saturation check

**Runaway fits.** On thin distal slices the second Gaussian can widen until it absorbs the enamel
peak's skewed upper shoulder, which walks its centre off the peak and onto the fit bound. Signature:
a fitted width 6× the dentin width, and a centre above every voxel in the slice — measured as
**247.7 reported on slices whose brightest voxel is 235–239**, while the real enamel mode sat plainly
at ~229.

Such a fit is **retried, not discarded**: `refit_narrow` constrains the width and seeds from the
empirical dense mode, then a polish pass releases the width once the centre is anchored. (Leaving the
width clamped biases centres ~10 units low.) If the constrained refit still cannot find a locatable
peak, the slice is `unresolvable`.

**Saturation** is tested **against the data, not the fit**: it requires more than 2 % of a slice's
voxels to actually sit at the intensity ceiling. See below for why this distinction matters.

It is also tested **before the significance and shape gates**, because a clipped histogram cannot
support those gates in the first place. Clipping collapses the bright population into a single spike,
the two-component fit puts both components on it (separation 0.00), and D goes sharply *negative* —
measured **−6037 and −7956** on Odam-CRE-B's two tip-most slices, where 26 % of voxels sit at 254.0.
Gating first reported those as `coincident`, i.e. "enamel is the same density as dentin", which is
not what a censored measurement means. Odam-CRE-A is unclipped (its maximum is 247.7 with a single
voxel there) and reaches this branch nowhere, so the ordering only affects genuinely clipped scans.

---

## Why it is built this way

Every decision above replaced something simpler that was built first and **failed a measurement**.
The rejected alternatives, briefly:

### One-sided template subtraction

Anchoring a dentin fit to its left flank and looking for excess on the right gives excellent
above-dentin sensitivity. It also **structurally cannot** detect enamel less dense than dentin: such
a component hides under the very flank used to anchor the fit. Incompatible with the requirement.

### A mirrored-molar dentin template

Building the dentin template by mirroring the molar's uncontaminated left flank leaves a **+4.2 %
systematic residual** on a dentin-only null — six times a plain Gaussian's +0.67 % — which alone
produced a **66 % false-positive rate**.

### Relative χ² improvement as the test statistic

A second Gaussian improves χ² by **~40 % even on pure dentin**, because it absorbs
smoothing/discretization mismatch rather than detecting a component. The null threshold landed at
0.81 and **nothing real ever cleared it** — detection was 0 % everywhere. (This is a different
failure from the normalized-D problem in step 5, and both point the same way: the statistic has to
be an absolute improvement, judged against a self-calibrated null.)

### A synthetic-Gaussian null

A null built from an idealized Gaussian put the threshold far too high and clipped genuine mid-tooth
enamel, because real dentin carries more histogram shape than a Gaussian. The null is bootstrapped
from the specimen's own dentin voxels, stretched about their mode to emulate broader peaks while
preserving that real shape.

### <a name="7-saturation-is-a-property-of-the-data"></a>Saturation as a property of the fit

The earlier rule flagged `saturated` when the *fitted centre* landed near the intensity ceiling and
the component was wide. That censored **61 perfectly good slices**, because a runaway fit lands near
the ceiling whether or not the data does.

Checking the pre-extraction volume settled it — **this scan is not clipped anywhere**:

| Check | Result |
|---|---|
| Voxels at the maximum (248) | **14** of 5,468,939 — 0.0003 % |
| Histogram approaching the ceiling | Smooth decline: 235 → 13887, 241 → 4388, 245 → 479, 248 → **14** |
| Enamel peak | **223**, upper half-max closes at **237** — 11 units of headroom |
| Container | int16 (ceiling 32767); 8-bit source maxes at 255 — data tops out at 248 |

A genuinely clipped scan shows a **spike** of thousands of voxels piled at the ceiling. This one
shows a clean falling tail. Saturation is therefore now declared only when the voxels themselves
pile up (> 2 % within 3 units of the maximum), and the tip plateau is confirmed as real biology
rather than a measurement hitting a wall.

---

## Measured performance

Synthetic data with a known answer, from `test_enamel_density.py`:

| Check | Result |
|---|---|
| Detection at 3.0 σ separation, enamel above / below dentin | **100 % / 100 %** |
| Detection at 2.0 σ separation, enamel above / below dentin | **100 % / 100 %** |
| False positives on pure dentin (n = 500 … 20 000) | **0.0 – 1.7 %** |
| Recovered centre, +2.5 σ | true 150 → **149.5** |
| Recovered centre, −2.5 σ | true 90 → **89.7** |

Recovery accuracy matters more than raw detection rate here, because the recovered centre *is* the
reported density.

**Independent cross-check.** The molar has a known, unambiguous answer and is measured with the same
machinery: **dentin 123.5, enamel 188.4**, separation 5.73 σ — a ratio of **1.53**, consistent with
the incisor's mature-tip regime.

Reference specimen `Odam-CRE-A`, all 1328 slices:

| Outcome | Count | Span | Density | Ratio |
|---|---|---|---|---|
| `resolved_hi` (mineralized enamel) | **938** | 27 – 98 % | 137 – 247 | 1.09 – 1.87 |
| `resolved_lo` (secretory matrix) | **149** | 14 – 28 % | 71 – 121 | 0.58 – 0.89 |
| `coincident` | 36 | 25 – 29 % | — | 1.00 |
| `unresolvable` | 205 | the two extreme ends | — | — |
| `saturated` | **0** | — | — | — |

The full validation suite is **31/31 passing**.

---

## Reading the profile

The curve traces amelogenesis end to end:

1. **Secretory zone (~14–28 %, red).** Enamel matrix is being laid down but barely mineralized, so it
   is *less dense than dentin* — ratio 0.58 rising to 0.89. This is real tissue and a real
   measurement, not a failure.
2. **The crossover (~25–29 %, orange).** Mineralizing enamel passes *through* dentin density. Here
   the two genuinely coincide and `coincident` is the correct answer.
3. **Maturation (~29–70 %, blue).** Density climbs steeply as mineral replaces matrix.
4. **Mature plateau (~70–98 %).** Ratio levels off around 1.8, easing slightly toward the tip.

The Spearman ρ reported in the log is computed over the **mineralized slices only**. Pooling both
regimes would measure the gap between two different tissue states rather than the maturation trend.

---

## <a name="units-and-what-they-mean"></a>Interpreting the output

- **Units are scanner intensity, not mg HA/cm³.** Converting needs a hydroxyapatite phantom scanned
  alongside the specimen; this scan has none. The **ratio** plot is self-calibrating (dentin is the
  shared reference tissue) and is the better figure for comparing specimens or sessions.
- **Secretory readings measure mineral, not matrix.** A density of 74 means sparsely mineralized
  tissue. X-ray attenuation does not see protein, so this is not a measurement of matrix content.
- **Check `z_window` in the CSV.** A wide pooling window means that point traded z-resolution for
  histogram signal.
- **Check `solidity` on any `resolved_lo` row.** This is the primary evidence that the low-density
  component is real tissue: a genuine matrix layer scores ≥ 0.10 (typically 0.20–0.38), while
  partial-volume rim scores 0.00–0.01. `angular_concentration` is secondary — a low value alone does
  **not** mean rim, since B and C lay matrix around most of the perimeter at concentrations of
  0.02–0.15.
- **Check `dentin_reference` against `dentin_mode`.** These should track each other within a few
  intensity units. A large gap means the fitted dentin has wandered off the traced core, and the
  role assignment on that row deserves a look.
- **A negative Spearman ρ is a red flag** for an inverted orientation, and is warned about in the log.

---

## Low-contrast and hypomineralized specimens

**This is the most important caveat in the document.** Some specimens have enamel only slightly
denser than dentin, or nearly identical to it along the whole tooth. The algorithm degrades
gracefully there, but **the quantity you should report changes**, and using the obvious one will
manufacture a false result.

### Where detection stops

Measured on synthetic data with a known answer, realistic non-Gaussian dentin, 20 % enamel fraction,
13 000 voxels per slice:

| True enamel/dentin ratio | Detected | Recovered centre | Error |
|---|---|---|---|
| 1.35 | 100 % | accurate | −0.3 |
| 1.26 | 100 % | accurate | −0.4 |
| 1.17 | 100 % | accurate | −1.4 |
| 1.13 | 100 % | accurate | +1.9 |
| 1.09 | 93 % | biased high | **+5.5** |
| 1.07 | 67 % | biased high | **+12.5** |
| 1.04 | 13 % | biased high | **+13.2** |
| ≤ 1.02 | 0 % | — | — |

Down to **ratio ≈ 1.13 the measurement is trustworthy**: essentially every slice resolves and the
recovered density is accurate to ~2 intensity units. That covers a substantially hypomineralized
tooth.

### The trap below ratio ≈ 1.10

Once detection becomes partial, **only the slices that happen to fluctuate favourably clear the
gate** — and those are exactly the slices whose fitted enamel centre sits too high. The reported
mean stops tracking the truth and floors out:

| True ratio | % resolved | **Mean *reported* ratio** |
|---|---|---|
| 1.17 | 100 % | 1.17 |
| 1.13 | 100 % | 1.14 |
| 1.09 | 85 % | **1.15** |
| 1.07 | 62 % | **1.15** |
| 1.04 | 15 % | **1.14** |

A severely affected tooth and a mildly affected one both report ≈ 1.15. **Averaging the
`resolved` slices of a low-contrast specimen will understate the phenotype**, and comparing that
average against a control will understate the genotype effect — the direction that loses real
findings.

### What to report instead

The signal that survives is the **fraction of the tooth that can be resolved at all**, which
degrades smoothly and monotonically:

| True ratio | 1.35 | 1.26 | 1.17 | 1.13 | 1.09 | 1.07 | 1.04 | 1.02 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| **% coincident** | 0 % | 0 % | 0 % | 0 % | 15 % | 38 % | 85 % | 100 % | 100 % |

So for any specimen with a substantial `coincident` band:

1. **Report `% coincident` (or `% resolved`) as the primary outcome**, not mean enamel density. It is
   a valid ordinal severity measure across the entire range, including where density estimation has
   failed completely.
2. **Report mean density only over the span where the same specimen resolves ~100 %**, and state
   that span.
3. **Compare genotypes over a common z-range**, since resolvability varies along the tooth.
4. A tooth reported **100 % `coincident` is a real, strong finding** — enamel indistinguishable from
   dentin — not a failed run. That is the outcome the `coincident` status exists to express.

### The guards that tell you which case you are in

- **The molar cross-check (`--molar`) is not optional for this work.** The molar is fully mature in
  every animal, so it must always separate strongly. Measured across all six specimens on this
  drive, it does — ratio **1.48 – 1.57**, separation **4.7 – 6.0 σ**:

  | | A | B | C | Hifla-A | Hifla-B | Hifla-C |
  |---|---|---|---|---|---|---|
  | molar ratio | 1.53 | 1.56 | 1.56 | 1.57 | 1.48 | 1.53 |
  | separation (σ) | 5.73 | 5.57 | 5.06 | 6.01 | 4.72 | 5.40 |

  This confirms scanner response is comparable across the set, so an incisor difference is biology
  rather than acquisition. **A weak molar means distrust the scan, not the genotype.**
- **The dentin anchor survives low contrast.** Tested with enamel at 20 % and 45 % of the
  cross-section across separations 0–6 σ, the fitted anchor stays within ±0.6 of the truth in every
  case but one: at **45 % enamel and 1.5 σ separation** the mass rule in `build_anchor` picked the
  wrong component and the anchor shifted **+14.5**. That combination — enamel nearly half the
  cross-section *and* barely separated — should be checked against the logged anchor value before
  the run is trusted.

---

## Known limitations

- **No absolute calibration.** Without a phantom, only relative comparisons are meaningful.
- **Four slices near 85–90 % still deviate > 8 intensity units from their neighbours.** The
  constrained refit in step 8 is an iterative search and does not converge identically on every thin
  distal slice. Spot-check that region if it bears on your conclusions.
- **The extreme ends remain `unresolvable`** (below ~14 % and above ~98 %). Those cross-sections are
  genuinely too thin to support a distinct interior population. Loosening `--max-rim-fraction`
  further will produce numbers there, but they should be treated as indicative only.
- **The dentin anchor comes from the specimen's own mid-z band.** A specimen whose mid-tooth dentin
  is itself abnormal would shift the reference; the molar cross-check is the guard against this.
- **Detection power falls off** below ~1.0 σ separation (enamel/dentin ratio ≈ 1.09) and for enamel
  fractions under ~5 % of the cross-section. Slices near that limit report `coincident`, which is a
  statement about *separability*, not proof that the two tissues are identical. See
  [Low-contrast and hypomineralized specimens](#low-contrast-and-hypomineralized-specimens) for what
  to report when a specimen sits near or below that limit — the mean of the resolved slices is the
  wrong statistic there.
- **Single reference specimen.** Every threshold quoted above was measured on `Odam-CRE-A`. The
  self-calibrating ones (anchor, null, fit bounds) adapt per specimen; the shape thresholds
  (`MIN_LOW_SIDE_CONCENTRATION`, `MAX_ENAMEL_SIGMA_RATIO`) are fixed and should be re-checked against
  a second specimen before being trusted broadly.

---

# Stage 3 — Mapping anatomy onto the curve (`map_landmarks.py`)

Stages 1–2 produce density against *position*, but that position is an index into a resliced
volume — it carries no anatomy. This stage answers the question that makes the profile
interpretable: **what is actually at 34 % along the tooth?** Pick landmarks on the original scan,
get back where each one falls on the trajectory.

This stage is optional. Nothing in stages 1–2 depends on it.

## Finding the landmarks automatically (`detect_landmarks.py`)

The landmarks can be found from the segmentation instead of picked, a whole cohort in one run
(MicroCT Studio's Landmarks step runs this per specimen). A mark set by hand always wins: pass the
marks so far with `--marks`, and only the landmarks nobody marked are filled in. `--map-only` maps
the marks as they are, after adjusting some by hand.

- **Eruption** — where the incisor's enamel comes out of the bone. The tooth is followed along the
  centerline in this stage's transform, and in each cross-section 120 rays radiate outward from the
  tooth; those that start on **enamel** — where the tooth's outer 0.024 mm reaches the incisor's
  enamel threshold, the valley under the enamel peak of its own histogram — are followed for bone.
  Inside the jaw every one of them meets it (the bone lies a median 0.12 mm from the tooth). Eruption
  is the first cross-section, walking tipward from the last one where bone covered all of the
  enamel, where **no ray from the enamel touches bone**, nor does for the next 0.09 mm, so a gap in
  a thin plate does not free it early. The 3° at each edge of the enamel band are left out: the
  band thins to nothing there, and the bone those rays clip held KO_F_1's eruption to z 1667.
  On P22_TGM2_KO_F_1 it is z 1615, six slices from the plane marked by eye (1621). On the 12 P22
  TGM2 specimens with a usable scan it falls **424 ± 23 slices past the front of M1's crown**. On
  the one flagged for motion the bone never covers the incisor's enamel, and no eruption is placed.
- **M1–M3** — each crown along the jaw, by the molar step's own split
  (`quantify_molar/molar_enamel.py`'s `molar_crowns`), less the slices where the crown in front still
  shows. Placed only when the split finds all three teeth. Identical to the molar step's placement on
  the 11 specimens it had placed.

The bone has to be bone. Masks written before the segmentation learnt to leave out the soft tissue
round the teeth hold a sleeve of it round the incisor, which touches the enamel long after the
bone has stopped. The detector cleans the bone as the segmentation now does
(`segmentation.molar_bone.clean_bone_mask`); a mask already clean passes through unchanged.
`--report` writes what was found as JSON: each landmark detected (a hand-marked one too, for going
back to), the bone cleanup, the enamel threshold, and for every cross-section how many rays start on
enamel and what share of them touch bone.

## The mapping already exists

The reslice is not an independent volume. Slice `i` was sampled from the plane through
`centerline[i]`, spanned by that point's frame vectors:

```
original_zyx = bbox_min + centerline[i] + u·n[i] + v·b[i]
```

Every term is computed during stage 1. `cmpr.py` writes them to
`<sample>_cmpr_transform.npz` so they survive the run — without that sidecar the geometry is
discarded when the process exits, and the reslice has no way back to the anatomy it came from.

**`bbox_min` is the term most easily lost.** The centerline is computed on the *cropped* volume, so
a transform reconstructed later from the reslice alone is offset by the crop origin — a fixed
translation that shifts every landmark, consistently enough to look plausible and be wrong.

## Usage

```bash
python map_landmarks.py \
  --transform ".../Segments/Odam-CRE-A/Odam-CRE-A_cmpr_transform.npz" \
  --profile   ".../RESULTS/enamel_density_profile.csv" \
  --output    ".../RESULTS/Odam-CRE-A_landmarks.csv"
```

napari opens on the **reoriented volume** — the whole hemimandible — with the molar and bone
segmentations overlaid and the centerline drawn in cyan. Scroll to a slice and press the landmark's
key:

| Key | Marks |
|---|---|
| `Q` | eruption |
| `G` / `H` | where M1 **starts** / **ends** |
| `J` / `K` | where M2 starts / ends |
| `N` / `W` | where M3 starts / ends |
| `U` | undo last mark |
| `Shift-U` | toggle 2-D / 3-D |

**Every key marks exactly one z, and pressing it again just moves that mark.** There is no press
count to keep track of and no hidden "waiting for the other end" state — if you are unsure whether a
boundary registered, look at the plane, and press the key again to correct it.

A molar marked at only one end is still written out, as a single position with a warning, rather
than discarded.

### Why these keys

They look arbitrary because they are constrained. **napari binds most single letters and the digits
`1`-`4` to layer-mode switches at the viewer level** — `1` is `activate_image_pan_zoom_mode`, `2` is
`activate_image_transform_mode`, and `E`, `A`, `D` are mode switches too. napari's binding wins while
the image layer is selected, which it must be for the scroll wheel to move through z.

A collision is **silent**: the key switches the image layer's mode instead of marking a landmark, and
nothing raises. The only symptom is a key that appears not to work.

In a default install only `G H J K N Q U W` are free, so the keys above are drawn from that set and
grouped by hand position rather than chosen as mnemonics. `assert_keys_available()` re-checks them
against napari's own shortcut table at startup and warns on any conflict, so this cannot recur
silently if napari's defaults change.

Each mark draws itself as a plane across the specimen, coloured by landmark (eruption yellow, M1
green, M2 cyan, M3 magenta) and labelled with which boundary it is. The planes render in **both**
slice view and 3-D, so `Shift-U` shows every mark at once against the whole tooth with the centerline for
reference, and you can toggle straight back to keep picking. `--start-3d` opens in 3-D.

You never draw anything by hand: the planes are generated from your keypresses, and the layer is
held read-only so a stray click cannot add one that no mark corresponds to. Close the window to
write the CSV.

## What you pick on

The volume stage 1 reslices is the *incisor* volume, and it is masked to the incisor alone — every
other structure is zeroed out. **The molars and the alveolar bone are not in it**, so they cannot be
picked there.

Picking therefore happens on `<sample>_reoriented_volume.nrrd` instead. It is the same array shape
and the same coordinate frame as the incisor volume — the segmentation pipeline reorients first and
then masks, never cropping — so a z marked on the reoriented volume is directly comparable to the
centerline, with no transformation between them.

It is found automatically next to the transform, in this order:

```
<sample>/<sample>_reoriented_volume.nrrd       <- preferred: all tissue present
<sample>/volumes/<sample>_incisor_volume.nrrd  <- fallback, warns: incisor only
```

The molar and bone segmentations (`volumes/<sample>_molar_volume`, `..._bone_volume`) are overlaid
when present, so a molar's start and end can be read off the molar segmentation rather than judged
from greyscale. Each is shape-checked against the display volume and skipped on a mismatch — an
overlay that does not line up looks authoritative while pointing at the wrong voxels.

### Key flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--transform` | `-t` | *(required)* | `<sample>_cmpr_transform.npz` from stage 1 |
| `--volume` | `-v` | discovered | Volume to pick on — defaults to the sample's reoriented volume |
| `--sample-root` | | inferred | Sample's segmentation folder, if not discoverable from `--transform` |
| `--output` | `-o` | `<transform dir>/<sample>_landmarks.csv` | Output CSV |
| `--points` | | none | Re-map a saved landmark CSV instead of opening the viewer |
| `--molars` | | none | The molar step's `<sample>_molar_enamel.json`: M1-M3 from its tooth split's crowns, for any molar not picked (see the script's docstring) |
| `--profile` | | none | `enamel_density_profile.csv`, to read the orientation from |
| `--orientation` | | from profile | Override: `tip-low-z` or `tip-high-z` |
| `--no-overlays` | | off | Do not overlay the molar and bone segmentations |
| `--start-3d` | | off | Open in 3-D view |
| `--downsample` | | `1` | Subsample the *display* volume (marks are rescaled back) |

## Landmarks are z, not points

A landmark here is a **z-plane**, or a pair of them: M1 occupies a range of z, eruption happens at
one z. Picking is therefore done with the slice slider and a keypress rather than by clicking a 3-D
position — the y and x of a click would be discarded anyway, and asking for them suggests they
carry meaning.

Two kinds of mark:

- **event** — one z. `z_percent_start` is its position; the end and length columns are blank.
- **span** — a start and an end z. Reported as a range on the trajectory, with a physical length.

**Volume z is not arc length along the tooth.** The incisor curves, so equal steps in z are unequal
steps along the centerline — on a test curve, two 50-voxel z steps came out as 0.5173 mm and
0.5118 mm of tooth. Converting between them is the whole point of the mapping.

`z_residual_vox` reports how far the matched centerline point sits from the plane you marked. It is
normally sub-voxel. It grows only where the centerline runs steeply through z — near a sharply
curving tip — and there a z-plane is a poor description of position along the tooth regardless.
A large residual means the mark is outside the centerline's z range entirely.

## Spans are ordered by trajectory, not by z

A span's start and end are reported in **trajectory order**, which is not always the order you
marked them. On a `tip-low-z` specimen, increasing volume z means *decreasing* `z_percent`, so the
mark made at lower z is the span's *end*. The tool reorders so `z_percent_start < z_percent_end`
always holds; without that, every molar's extent silently reverses on half the specimens.

### Output

| Column | Meaning |
|---|---|
| `landmark` | The name you typed or cycled to |
| `kind` | `span` or `event` |
| `z_percent_start` | **Position on the trajectory** — matches the density profile's x-axis |
| `z_percent_end` | End of a span; blank for an event |
| `z_percent_mid` | Midpoint — the single number to annotate a plot with |
| `extent_percent` | How much of the trajectory a span covers |
| `z_start` / `z_end` | The raw volume z you marked |
| `slice_start` / `slice_end` | Corresponding CMPR slice indices |
| `arc_length_*_mm`, `length_mm` | Physical distance along the tooth |
| `z_residual_vox` | Distance from the marked plane to the matched centerline point |
| `source` | `auto` for a landmark `detect_landmarks.py` found, `manual` for a pick (`teeth`: a molar the molar step placed, before detection did) |

Rows are sorted by `z_percent_start`, so the CSV reads in the same order as the profile. Re-mapping
reads back `z_start`/`z_end` — the raw picks — so a saved CSV recomputes against a new reslice
rather than carrying stale percentages.


## Orientation

`z_percent` depends on which end of the array is the tip, and that is **detected per specimen**, not
fixed — the reference specimen has its tip at low z, the reverse of the plotting convention. Get it
backwards and every landmark mirrors to the wrong end of the tooth, with a result that still looks
perfectly reasonable.

So the orientation is read from the density profile's own CSV (`--profile`), recovered from whether
its `z` and `z_percent` columns run together or opposed. Pass `--orientation` to override. With
neither, the tool assumes `tip-high-z` and logs a warning saying so.

## Verification

- **Projection.** On a straight synthetic centerline, `project_z` returns the exact slice with zero
  residual. On a curved one, arc length exceeds the z extent (301.9 vs 240.0 voxels) and equal z
  steps give unequal arc steps — confirming the curvature is actually accounted for rather than
  assumed away. A z beyond the centerline's range reports a large residual instead of clamping
  silently.
- **Span ordering.** The `tip-low-z` case is tested explicitly: the same marks produce a span that
  is the mirror of the `tip-high-z` result and still satisfies `start < end`. Marks given
  end-before-start normalize too.
- **Picker.** Every keybinding is driven against a stub viewer — each boundary key, re-pressing one
  to move its mark, a half-marked molar surviving as a point, undo across interleaved keys, and the
  2-D/3-D toggle. The plane layer is checked to appear and disappear with the marks, to carry one
  label per boundary, and to stay read-only across redraws — assigning `.data` silently resets
  napari's `editable` flag, so that is asserted rather than assumed. A 2× display downsample
  rescales marks back to full-resolution z.
- **Keybinding collisions.** All nine keys are checked against napari's live shortcut table, and the
  guard itself is tested against two known-bad cases: `1` (which napari binds to
  `activate_image_pan_zoom_mode`) and `!`/`@` (which napari normalises to the *same* empty key, so
  only one would bind). Normalisation is delegated to napari's own parser rather than reimplemented,
  because a string compare misses both.
- **Round-trip.** A written CSV re-maps to identical percentages, spans and events alike.
- **Real layers, not a permissive stub.** The picker's tests construct genuine napari layer objects
  behind a stubbed viewer, so layer-construction errors surface in the test rather than on a real
  specimen. Run with `python test_map_landmarks.py`.
- **The transform sidecar** is checked end-to-end through the real pipeline on a synthetic curved
  tooth: **100 % of centerline points land inside the tooth in original-volume coordinates**, the
  assertion that fails if the crop offset is dropped.

## Limitations

- **A z-plane is not a cross-section.** Marks are planes perpendicular to the volume's z axis, while
  the CMPR slices are perpendicular to the *centerline*. Where the tooth runs obliquely the two
  differ, and `z_residual_vox` is the visible symptom. For most of the incisor — which lies along z
  in the reoriented frame — the difference is sub-voxel.
- **Picking precision is yours.** The tool reports the z you marked exactly; that says nothing about
  how well you identified the anatomy. For a molar's start and end, the meaningful uncertainty is
  where you judge the tooth to begin, which the mapping cannot improve on.
- **`--downsample` coarsens picking.** Marks are rescaled back to full resolution, but a 2×
  downsample means each mark is at best 2 voxels of precision.
- **Molar spans are z-extents, not incisor tissue.** A molar's span says which part of the incisor's
  trajectory sits alongside it. It remains a spatial correspondence, not a claim that the molar and
  that stretch of incisor share a developmental stage.
- **One specimen per transform.** The tool refuses a volume whose shape does not match the
  transform's recorded original shape, catching the common mistake of pairing a transform with the
  wrong scan.
