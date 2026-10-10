# Super-resolution segmentation

Segments a micro-CT scan on a finer grid than it was scanned at, then puts the
result back on the original scan.

```
scan.nrrd ──► 3-D generative model ──► upscaled volume (2x per axis, temporary)
                                                │
                                                ▼
                                 segment_mandible (unchanged)
                                                │
               masks on the fine grid ◄─────────┘
                        │  majority vote per original voxel
                        ▼
      masks on the ORIGINAL grid, applied to the ORIGINAL intensities
```

The upscaled volume is only an input to segmentation. Every intensity written
to the outputs comes from the original scan; the upscaled file is deleted when
the run finishes.

---

## Contents

1. [How it works](#how-it-works)
2. [Read this before trusting the results](#read-this-before-trusting-the-results)
3. [Setup on a PC](#setup-on-a-pc) — Docker (recommended) or conda
4. [Usage](#usage)
5. [Outputs](#outputs)
6. [Troubleshooting](#troubleshooting)
7. [Tests and code map](#tests-and-code-map)

---

## How it works

### The model

A volumetric **ESRGAN** (Wang et al. 2018): a generator built from
residual-in-residual dense blocks, trained first for fidelity (L1 plus an L1 on
spatial gradients) and then with a small adversarial term from a 3-D PatchGAN
critic, using the relativistic-average GAN loss. Everything is 3-D, so the
added detail is consistent across slices rather than invented slice by slice.

Two changes from the original make it safe to segment with:

- **It can only add sub-voxel detail.** The last layer projects the output
  onto the volumes whose `s×s×s` block averages equal the input *exactly*. A CT
  voxel is, to first order, the mean attenuation over its cell (the
  partial-volume model). So this says the fine voxels inside one scanned voxel
  must average to what the scanner measured. The network can redistribute
  intensity within a voxel, which is where a boundary between tissues sits, but
  it cannot add or remove density at the scanned resolution.
- **It starts as interpolation.** The network predicts detail on top of a
  trilinear upsample. An undertrained model degrades to interpolation, not to
  noise.

### Training without high-resolution data

There is no ground truth at twice the scanner's resolution, so the model is
trained one scale down. A patch of the real scan is the target. The same patch,
block-averaged by the scale factor, is the input, with noise added back to the
scan's own measured level. The model learns that step and is then applied one
scale *up*. This is self-supervised ("internal") super-resolution, as in ZSSR
(Shocher et al. 2018) and SMORE (Zhao et al. 2021). A model can be trained on
the scan being segmented, or once on several representative scans and reused.

A slab of slices through the middle of each training scan is held out. The
training log reports the model against plain trilinear interpolation there, in
dB of PSNR. A model that does not beat trilinear is flagged.

### Back to the original grid

`segment_mandible` returns its masks cropped and reoriented. Every step of that
reorientation is lossless and recorded, so it is undone exactly. For every
original voxel the code then counts how many of its `s³` fine voxels each
structure covers:

- **Tissue or not:** a voxel is tissue if at least half of it is (`--min-fraction`).
- **Which structure:** the structure covering most of the voxel. Ties go to
  molar, then bone, then incisor.

Tissue is decided on the *total* first so that a voxel split between incisor
and bone is not dropped. Voting each structure separately would punch holes
along every interface. The same reorientation is then applied to the original
scan, so the masks line up with it exactly.

---

## Read this before trusting the results

- **Memory is the limit.** At `--scale 2` the pipeline segments 8× the voxels.
  `segment_mandible` peaks at roughly 17–18 bytes per voxel of what it is
  given: its non-local-means stage holds about four float32 copies at once.
  For this lab's scans:

  | Scan | Voxels | Upscaled (2×) | Peak RAM to segment | Temp disk |
  |---|---|---|---|---|
  | Smallest BMP stack (P27RWT) | 0.37 G | 3.0 G | ~50 GB | 3 GB |
  | Head-3 `R-Mandible.nrrd` (uint16) | 0.64 G | 5.2 G | ~93 GB | 10 GB |
  | Largest BMP stack (Odam-CRE-Hifla-C) | 0.91 G | 7.3 G | ~123 GB | 7 GB |

  Every run logs this estimate against the machine's RAM before it starts, and
  warns when it will not fit. `--crop` upscales only the specimen's bounding box
  and can cut these numbers substantially.

- **segment_mandible was tuned at the native resolution.** Several of its
  parameters are counted in voxels, not millimetres. Examples: the 3,500-voxel
  minimum island size, the 2,500-pixel per-slice minimum, dilation radii of 2
  voxels, and the 5×5 NLM patches. At 2× each of these covers ⅛ of the volume,
  ¼ of the area, or ½ of the distance it did before. Differences from a native
  run therefore come from this as well as from the extra detail. **Before
  adopting this for a study, segment a few scans both ways and compare the
  masks.**

- **The model is trained one scale down and applied one scale up.** That
  relies on bone, enamel and dentin looking alike at the two scales. The
  held-out score measures the step it was trained on, not the step it is used
  for; nothing can measure that without a genuinely higher-resolution scan of
  the same specimen.

- **The adversarial phase sharpens, and can invent texture.** Its weight is
  small and the consistency projection limits it, but `model_l1.pt` (the
  fidelity-only model saved before that phase) is kept for comparison, and
  `--gan-iters 0` skips the phase entirely.

- **Give every scan its true voxel size.** When `segment_mandible` knows the
  spacing, its incisor checks measure jumps in millimetres, so they mean the
  same thing at any resolution. Without a spacing they count voxels, which at
  2× are half the size they were tuned on. A failed check swaps in the
  pipeline's fallback incisor mask, so this changes results, not just the
  report. BMP stacks carry no spacing, so pass `--voxel-size` (e.g. `0.008` for
  the 8 µm scans). A spacing is never invented: a scan without one is upscaled
  without one. Also check NRRD headers, since a converted file can declare
  1 mm voxels.

- **One model per voxel size.** A scan whose voxel size differs from the
  training scans' by more than 10% produces a warning. Train on scans like the
  ones you segment.

- **Runtimes are not known yet.** None of this has been run on a real scan or
  a GPU. The training log prints iterations per second; judge the schedule from
  that on your machine.

---

## Setup on a PC

Both routes end with `check`, which reports the GPU, memory and disk, confirms
`segment_mandible` imports, and runs a tiny forward and backward pass on the
device. Run it before anything long.

### Option A — Docker (recommended)

The image bundles Python, a CUDA build of PyTorch, and every dependency of
`segment_mandible` and this package. Nothing needs installing on the PC except
Docker and the GPU driver.

**1. Prerequisites**

| PC | Install |
|---|---|
| **Windows 10/11** | An up-to-date NVIDIA driver, then [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) with the WSL 2 backend (its default). GPU access through WSL 2 needs nothing further. |
| **Linux** | An NVIDIA driver, [Docker Engine](https://docs.docker.com/engine/install/), and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). |

Check the GPU is visible to Docker (this pulls a small CUDA image once):

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

**2. Choosing the base image**

PyTorch builds target specific GPU generations, so pick the base image for
your card. `nvidia-smi` shows the card and driver version.

| GPU | Driver | `BASE_IMAGE` |
|---|---|---|
| RTX 20xx / 30xx / 40xx / 50xx, A-series, H100 **(default)** | ≥ 570 for RTX 50xx; any recent driver otherwise | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` |
| GTX 10xx (Pascal) or older | ≥ 525 | `pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime` |
| Turing or newer, and you want the newest PyTorch | ≥ 580 | `pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime` |

A wrong choice shows up in `check` as "no kernel image is available for
execution on the device" (card too new or too old for the build) or "driver
version is insufficient" (update the driver or choose an older CUDA).

**3. Configure folders.** From the repository root:

```bash
cp super_resolution/.env.example super_resolution/.env
```

Edit `super_resolution/.env` to point at the folders on the PC:

- `DATA_DIR` holds the scans. It is mounted read-only at `/data`.
- `WORK_DIR` holds the temporary upscaled volume, at `/work`. It must be a
  fast local disk with free space of 8× the largest scan.
- **Do not point `WORK_DIR` at a Box or OneDrive folder**: it would try to
  sync a 10 GB temporary file on every run.

**4. Build and check**

```bash
docker compose -f super_resolution/compose.yaml build
docker compose -f super_resolution/compose.yaml run --rm sr check
```

The build sends only `segment_mandible/` and `super_resolution/` to Docker
(see `Dockerfile.dockerignore`), and it fails early if anything cannot import.
It needs the BuildKit builder, which is the default in any recent Docker. The
legacy builder would upload the whole repository, `data/` included.

**5. Run.** Paths are the container's (`/data`, `/results`, `/models`):

```bash
# Train once on a few representative 8 µm scans (BMP stacks need --voxel-size)
docker compose -f super_resolution/compose.yaml run --rm sr \
    train -i /data/Odam-CRE-A /data/WTP25-1_8um_2K_reoriented --voxel-size 0.008 -o /models/cohort_x2

# Segment every sample in /data with it
docker compose -f super_resolution/compose.yaml run --rm sr \
    segment -i /data -m /models/cohort_x2/model.pt -o /results --voxel-size 0.008 --log
```

`--voxel-size` overrides the spacing of every input in the run, so run scans
of different resolutions separately. Leave it off for NRRD or DICOM files
whose headers are right.

Without compose, the equivalent is:

```bash
docker build -f super_resolution/Dockerfile -t microct-sr .
docker run --rm --gpus all -v D:/scans:/data:ro -v D:/sr_results:/results \
    -v D:/sr_models:/models -v E:/scratch:/work microct-sr check
```

On a CPU-only machine, remove the `deploy:` block from `compose.yaml`, or drop
`--gpus all`. That works, but training and upscaling are very slow.

On a Linux host, files written by the container belong to root. Add
`--user "$(id -u):$(id -g)"` to `docker run` to keep your own ownership.

### Option B — the lab's conda environment

`env/environment.yml` already contains everything this package needs.

```bash
conda env create -f env/environment.yml      # once
conda activate microct-analysis
# Windows/Linux + NVIDIA: replace the CPU PyTorch with a CUDA build,
# exactly as env/README.md ("GPU Support") describes, e.g.
pip install "torch>=2.0" --index-url https://download.pytorch.org/whl/cu128
python -m super_resolution check
```

Run every command from the **repository root**. The package finds
`segment_mandible/` next to itself, and its commands are `python -m
super_resolution ...`.

---

## Usage

```
python -m super_resolution check    [--device auto|cuda|cuda:1|mps|cpu]
python -m super_resolution train    -i SCAN [SCAN ...] -o MODEL_DIR [options]
python -m super_resolution upscale  -i SCAN -m MODEL_DIR/model.pt -o OUT.nrrd
python -m super_resolution segment  -i SCAN_OR_FOLDER [-m MODEL_DIR/model.pt] [-o RESULTS]
```

A scan is anything `segment_mandible` reads: a `.nrrd` or `.tif`, a folder
holding a DICOM series or BMP stack, or, for `segment`, a folder of such
samples. `--help` on each subcommand lists every option.

### `train`

Learns from one or more scans. Several scans make one shared model, and all
of them are held in RAM while it trains.

| Option | Default | |
|---|---|---|
| `--scale` | `2` | Factor per axis, or `z,y,x` (e.g. `1,2,2` for in-plane only). Voxel count grows by the product. |
| `--pretrain-iters` / `--gan-iters` | 5000 / 5000 | Fidelity phase, then adversarial phase (`0` skips it). |
| `--lr-patch`, `--batch-size` | 32, 4 | Training patch edge at scan resolution; lower both if the GPU runs out of memory. |
| `--features`, `--growth`, `--blocks` | 32, 16, 4 | Network size (~2.2 M parameters by default). |
| `--adv-weight`, `--grad-weight` | 1e-3, 0.5 | Loss weights relative to L1. |
| `--resume` | | Continue from `MODEL_DIR/last_state.pt` after an interruption. |

The command writes these files to `MODEL_DIR`:

- `model.pt`: the final model.
- `model_l1.pt`: the model at the end of the fidelity phase.
- `history.json`: held-out PSNR against trilinear at every validation.
- `last_state.pt`: resumable state.

### `segment`

The whole pipeline for one sample or a folder of them. Samples that already
have results are skipped unless `--overwrite` is given. Without `-m`, it
first trains a model on each sample using the `train` options, saves it in
`<results>/<sample>/sr_model/`, and reuses it on later runs.

| Option | Default | |
|---|---|---|
| `--min-fraction` | `0.5` | Share of an original voxel that must be tissue on the fine grid. |
| `--crop` / `--crop-margin` | off / 24 | Upscale only the specimen's bounding box (plus margin). Saves RAM, disk and time, but changes the intensity histogram `segment_mandible` sets its thresholds from. |
| `--tile` / `--halo` | 64 / 10 | Inference tile size and context, in scan voxels; lower `--tile` on a small GPU. |
| `--no-skip-air` | | Also run the network on tiles that are air throughout (by default they are copied, which is exact and faster). |
| `--work-dir` | `<results>/_sr_work` | Where the temporary upscaled volume goes (`/work` in Docker). |
| `--format` | `tif` | `tif` or `nrrd`, as in `segment_mandible`. |
| `--save-fractions` | | Also save each structure's fine-voxel counts per original voxel. |
| `--keep-upscaled` | | Keep the upscaled volume to inspect it. |
| `--incisor-margin-shrink`, `--strict`, `--cache-dir`, `--workers`, `--voxel-size`, `--apply-dicom-rescale` | | Passed to `segment_mandible` with the same meaning. |

### `upscale`

Writes the upscaled volume only, as an uncompressed NRRD that overlays the
original in 3D Slicer. Use it to look at what the model does before trusting it
with a cohort.

---

## Outputs

`segment` writes the same layout as `segment_mandible`, using its own writers,
plus a few additions:

```
<results>/
├── validation_report.json            segment_mandible's plausibility checks
└── <sample>/
    ├── <sample>_reoriented_volume.tif    original scan, reoriented like segment_mandible's output
    ├── <sample>_voxel_size.json
    ├── masks/<sample>_{incisor,bone,molar}_mask.tif      0/1, original grid
    ├── volumes/<sample>_{incisor,bone,molar}_volume.tif  original intensities inside each mask
    ├── <sample>_labels_input_frame.nrrd  one label map (1 incisor, 2 bone, 3 molar) in the scan's
    │                                     own orientation and geometry: overlays the raw scan in Slicer
    ├── <sample>_sr_summary.json          model, scale, crop, reorientation, per-structure counts, timings
    ├── sr_fractions/…_fine_voxels.tif    (--save-fractions) fine voxels of each original voxel per structure
    └── sr_model/                         (only when trained by `segment`) model.pt, history.json, …
```

`<sample>_sr_summary.json` gives each structure's volume two ways:

- **`volume_mm3`**: whole voxels of the binary mask.
- **`subvoxel_volume_mm3`**: from the fine-grid segmentation, so partial
  voxels count partially. This is the finer estimate.

Both are `null` when the scan has no voxel size. `voxels` and
`subvoxel_voxels` give the same two measures as voxel counts.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check` fails with "no kernel image is available" | The PyTorch build does not support your GPU: pick the base image from the table in [Choosing the base image](#option-a--docker-recommended) and rebuild. |
| "CUDA driver version is insufficient" | Update the NVIDIA driver, or use a base image with an older CUDA. |
| `CUDA out of memory` while upscaling | `--tile 48` (or 32). While training: `--batch-size 2`, `--lr-patch 24`. |
| Run killed or the PC swaps during segmentation | Too little RAM for the upscaled volume (the log estimates it up front): add `--crop`, or use a machine with more memory. |
| "needs N GB but … has M GB free" | `--work-dir` (compose: `WORK_DIR`) on a disk with room for 8× the scan. |
| "The model does not beat trilinear interpolation" | Train longer, or on more or more representative scans. Do not segment with that model. |
| Voxel-size warning on `segment` | The model was trained on scans with a different voxel size; train on matching scans. BMP stacks need `--voxel-size`. |
| Docker build is slow and sends gigabytes | The legacy builder is ignoring `Dockerfile.dockerignore`. Use `docker buildx build` or enable BuildKit. |

---

## Tests and code map

```bash
python -m pytest super_resolution/tests                    # conda environment
docker compose -f super_resolution/compose.yaml run --rm --entrypoint python sr -m pytest super_resolution/tests
```

The tests use tiny networks and synthetic phantoms on the CPU and take seconds.
They check the following:

- Block-averaging the output reproduces the input.
- Tiled inference matches a single pass.
- The upscaled NRRD reads back through `segment_mandible`'s loader in the right
  place.
- Masks survive the real `reorient_mandible` round trip exactly.
- Training resumes where it stopped.
- Every subcommand runs end to end. For this check `segment_mandible` is
  replaced by a stand-in that uses its real loader and reorientation.

| Module | Role |
|---|---|
| `model.py` | Generator, critic, losses, consistency projection, checkpoints, device choice |
| `config.py` | `TrainConfig` (no torch import, so the CLI and `check` load without it) |
| `patches.py` | Training patches from the scans: tissue-biased sampling, held-out slab, noise estimate |
| `train.py` | Fidelity then adversarial training, EMA weights, validation against trilinear, resume |
| `upscale.py` | Tiled inference into a memory-mapped NRRD; range clipping; air tiles |
| `mask_mapping.py` | Undo/replay the reorientation record; fine-voxel counts; the vote |
| `sr_segment.py` | The end-to-end `segment` command |
| `volume_io.py` | Loading via `segment_mandible`, NRRD geometry, specimen crop, memory/disk queries |
| `classical.py` | Puts `segment_mandible` on the path; imports its pipeline without napari |
| `check.py`, `cli.py` | The `check` command; argument parsing |
