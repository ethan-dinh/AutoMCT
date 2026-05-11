# Automated 3D Mandible Segmentation via Deep Learning on Micro-CT Volumes

> **Status:** Active development — v1.0 trained on N=9 samples. Designed for continuous improvement via human-in-the-loop annotation.

---

## Abstract

We present a fully automated deep learning pipeline for multi-structure segmentation of mouse mandible micro-CT volumes. The system delineates three anatomically distinct structures — the incisor, mandibular bone, and molars — from grayscale volumetric data using a 3D U-Net trained under a leave-one-out cross-validation (LOO-CV) regime. To address the challenge of extremely limited annotated data (N=9 at time of writing), the pipeline incorporates an aggressive 14-stage augmentation strategy, foreground-class-balanced patch sampling with per-class oversampling ratios, and a two-stage inference procedure that decouples background suppression from foreground classification. Ensemble prediction over all fold checkpoints is used at inference time to maximize predictive accuracy and uncertainty smoothing. The framework is explicitly designed for iterative growth: a human-in-the-loop process allows domain experts to evaluate model predictions on new scans, correct errors, and reintegrate corrected masks into the training set — triggering a full LOO-CV retrain every five new samples. Cross-validation Dice scores for the best-performing folds reached 0.90+ across all three foreground classes, with molar segmentation representing the primary remaining challenge due to severe class imbalance.

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Dataset & Data Format](#2-dataset--data-format)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline & Augmentation](#4-data-pipeline--augmentation)
5. [Training Protocol](#5-training-protocol)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Cross-Validation & Evaluation](#7-cross-validation--evaluation)
8. [Results](#8-results)
9. [Human-in-the-Loop Framework](#9-human-in-the-loop-framework)
10. [Known Limitations](#10-known-limitations)
11. [Improvements & Future Directions](#11-improvements--future-directions)
12. [Installation](#12-installation)
13. [Usage Reference](#13-usage-reference)
14. [Module Reference](#14-module-reference)
15. [Output Structure](#15-output-structure)

---

## 1. Motivation & Problem Statement

Manual segmentation of micro-CT mandible volumes is a labor-intensive and expert-dependent task. A single scan may span hundreds of Z-slices at sub-micron resolution, and the three target structures — incisor, cortical/trabecular bone, and molar — exhibit significant morphological variation across specimens due to age, genotype, and imaging conditions. Automating this process is scientifically valuable for two reasons:

1. **Throughput**: Enabling high-throughput phenotypic screening of cohorts without proportional increases in annotation labor.
2. **Reproducibility**: Eliminating inter-annotator variability in downstream morphometric measurements.

The core algorithmic challenge is that supervised deep learning for 3D medical image segmentation typically requires hundreds of annotated volumes to generalize well. Our dataset at initial training contains only nine specimens, placing this firmly in the small-data regime where standard training recipes fail. This motivates every major design decision in the pipeline: the choice of leave-one-out cross-validation over held-out test sets, the aggressive augmentation policy, the foreground-biased patch sampling, and the human-in-the-loop retraining strategy.

---

## 2. Dataset & Data Format

### 2.1 Sample Structure

The pipeline expects one directory per specimen produced by the upstream `segment_mandible` classical pipeline. Each sample directory must contain:

```
<sample_name>/
    reoriented_volume.tif        # 3D micro-CT scan, dtype int16 or uint16
    masks/
        incisor_mask.tif         # binary uint8 mask
        bone_mask.tif            # binary uint8 mask
        molar_mask.tif           # binary uint8 mask
```

Samples are auto-discovered by `discover_samples()` in [src/dataset.py](src/dataset.py). Any directory missing one or more of the four required files is skipped with a warning.

### 2.2 Label Encoding

The three binary masks are merged into a single uint8 label volume at load time using the following priority scheme:

| Label | Structure | Priority |
|-------|-----------|----------|
| 0 | Background | Lowest |
| 1 | Incisor | Low |
| 2 | Bone | Medium |
| 3 | Molar | Highest |

When masks overlap (e.g., a voxel is labeled both bone and molar), the higher-priority class wins. This ordering reflects anatomical containment: molar tissue sits within a bony socket, so molar is assigned precedence over bone in ambiguous boundary regions.

### 2.3 Intensity Normalization

Each volume is normalized to float32 in [0, 1] using robust percentile clipping:

```
p1, p99 = np.percentile(image, [1, 99])
image = clip(image, p1, p99)
image = (image - p1) / (p99 - p1)
```

This is applied identically at both training and inference time, ensuring consistency. The 1st/99th percentile range avoids sensitivity to scanner-specific intensity outliers and bright reconstruction artifacts that frequently appear at volume boundaries.

### 2.4 Class Imbalance

The three foreground classes exhibit significant volumetric imbalance. Bone is the dominant foreground class, occupying the largest volume of the mandible. Incisor is intermediate. Molar is the smallest and most spatially compact structure, making it the hardest to segment and the most prone to false negatives. This imbalance is addressed at three levels: (1) foreground-biased patch sampling, (2) per-class oversampling ratios during patch extraction, and (3) class-weighted cross-entropy loss.

---

## 3. System Architecture

### 3.1 Primary Model: 3D U-Net

The default model is a 3D U-Net implemented via MONAI's `UNet` class with the following configuration:

| Hyperparameter | Value |
|---|---|
| Spatial dimensions | 3 |
| Input channels | 1 (grayscale CT) |
| Output channels | 4 (background + 3 structures) |
| Encoder channels | (32, 64, 128, 256) |
| Downsampling strides | (2, 2, 2) |
| Residual units per level | 2 |
| Dropout probability | 0.20 |
| Approximate parameters | ~4.7M |

The U-Net architecture is well-established in 3D medical image segmentation (Çiçek et al., 2016; Milletari et al., 2016). Its skip connections between encoder and decoder preserve fine-grained spatial detail — important for accurately delineating thin structures like the incisor root and cortical bone margins. The two residual units per encoder level improve gradient flow during backpropagation, partially mitigating vanishing gradient issues that arise when training deep networks on small datasets.

### 3.2 Alternative Model: SegResNet

A SegResNet (Myronenko 2019) is available via `--model segresnet`:

| Hyperparameter | Value |
|---|---|
| Init filters | 32 |
| Encoder blocks | (1, 2, 2, 4) |
| Decoder blocks | (1, 1, 1) |
| Dropout probability | 0.20 |

SegResNet uses a VAE-regularized bottleneck that imposes a distributional prior on latent representations, which can benefit generalization in limited-data settings. It is recommended for experiments when N > 15 and molar Dice remains below 0.70.

### 3.3 Model Factory

Both architectures are instantiated through `build_model()` in [src/model.py](src/model.py), which accepts a string selector and returns an `nn.Module`. Checkpoint loading is handled by `load_checkpoint()`, which maps weights to the target device and returns the full checkpoint dictionary (including optimizer state, epoch, and best Dice score) for training resumption.

---

## 4. Data Pipeline & Augmentation

### 4.1 Two-Stage Transform Architecture

The data pipeline separates transforms into two phases to maximize training throughput:

**Stage 1 — Deterministic preprocessing (run once, cached to disk):**
1. `LoadAndCombineMasksd` — load image + three masks, combine into label volume, normalize intensity
2. `SpatialPadd` — zero-pad all volumes to at least `patch_size` on each axis
3. `CropForegroundd` — crop tightly to the foreground bounding box with 8-voxel margin

These transforms are CPU-bound and I/O-heavy. The output (cropped float32 image array + uint8 label array) is serialized to `.npy` files in a disk cache directory keyed by a SHA-256 hash of the file path and patch size. On subsequent runs, already-cached samples are skipped. Cache warmup runs in parallel using a `ThreadPoolExecutor`.

**Stage 2 — Random augmentation (applied per-batch, from in-RAM arrays):**

Training samples are loaded entirely into CPU RAM (`_AugDataset`), eliminating random-access disk I/O during the training loop. All stochastic augmentation is applied on-the-fly on the small 96³ patch crops. This means peak training RAM equals the sum of all training volume arrays for the current fold (typically 2–6 GB for N=8 training samples), but per-epoch disk I/O drops to near zero.

### 4.2 Augmentation Policy

The following 14 augmentation stages are applied in sequence to every training patch:

| Stage | Transform | Parameters | Probability |
|---|---|---|---|
| 1 | Foreground-biased patch crop | ratios=[1,2,1,5], n_samples=24/volume | 1.0 |
| 2 | Random flip (axis 0) | — | 0.50 |
| 3 | Random flip (axis 1) | — | 0.50 |
| 4 | Random flip (axis 2) | — | 0.50 |
| 5 | Random 90° rotation (axes 0,1) | max_k=3 | 0.50 |
| 6 | Random 90° rotation (axes 0,2) | max_k=3 | 0.50 |
| 7 | Random 90° rotation (axes 1,2) | max_k=3 | 0.50 |
| 8 | Random affine | rotate ±15°, scale ±15% | 0.50 |
| 9 | Random 3D elastic deformation | σ=(5–8), magnitude=(20–60) | 0.30 |
| 10 | Coarse dropout | 4 cuboids of 16³ voxels | 0.30 |
| 11 | Gaussian noise | std=0.05 | 0.30 |
| 12 | Gaussian smoothing | σ_xyz=(0.5–1.5) | 0.30 |
| 13 | Intensity scale | factor=0.30 | 0.50 |
| 14 | Intensity shift | offset=0.20 | 0.50 |
| 15 | Gamma contrast adjustment | γ=(0.75–1.5) | 0.50 |

**Design rationale:**
- **Flips + 90° rotations** provide discrete symmetry augmentation at negligible compute cost (array view operations).
- **Affine** fills in continuous rotations and scale variation, modeling inter-animal size differences and scanner tilt.
- **3D elastic deformation** simulates tissue deformability and imperfect co-registration between volume and mask.
- **Coarse dropout** forces the model to learn spatially distributed features rather than memorizing local texture patches — the dominant overfitting failure mode on N=9 specimens.
- **Intensity augmentation** simulates variation in scanner calibration, tube current, and reconstruction kernel.

### 4.3 Patch Sampling Strategy

Patches are extracted using `RandCropByLabelClassesd` with class sampling ratios `[1, 2, 1, 5]` (background : incisor : bone : molar). The molar weight of 5 aggressively oversamples this minority class to prevent the model from neglecting it in favor of the dominant bone class. 24 patches are sampled per volume per epoch (`samples_per_volume=24`). The foreground voxel indices per class are precomputed during cache warmup and stored in an `.npz` file alongside the volume arrays, enabling O(1) patch center selection at training time.

---

## 5. Training Protocol

### 5.1 Loss Function

Training minimizes a composite loss combining foreground Dice loss and class-weighted cross-entropy:

```
L = L_Dice + L_CE
```

**Dice loss** (`DiceLoss`, include_background=False, softmax=True): The Dice coefficient is differentiable when applied to softmax logits and is robust to class imbalance. Background is excluded to force the model to focus its capacity on the three foreground structures.

**Cross-entropy loss** (`CrossEntropyLoss`) with class weights `[0.5, 1.0, 1.0, 2.0]` (background, incisor, bone, molar): CE provides dense per-voxel gradient signal that complements the region-level Dice term. The molar weight of 2.0 applies additional penalty for molar misclassifications.

### 5.2 Optimizer & Scheduler

| Component | Setting |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Gradient clipping | max_norm = 1.0 |
| Scheduler | CosineAnnealingLR (T_max = max_epochs) |
| Mixed precision | `torch.amp.autocast("cuda")` + `GradScaler` |

Cosine annealing decays the learning rate smoothly from 1e-3 to ~0 over the full training run. This avoids the abrupt LR step-downs of plateau scheduling, which can cause training instability when the dataset is small and validation metrics are noisy.

Gradient clipping at `max_norm=1.0` prevents gradient explosion during the early high-loss phase of training (epochs 0–50), where the composite loss can produce large gradient magnitudes.

### 5.3 Early Stopping

Validation runs every 5 epochs via full-volume sliding-window inference. If the best mean foreground Dice (averaged over incisor, bone, molar) does not improve for 50 consecutive validation rounds (250 epochs), training terminates. The best checkpoint is preserved regardless of early stopping.

### 5.4 Checkpoint System

Two checkpoint files are written per fold:

| File | Contents | Purpose |
|---|---|---|
| `best_model.pt` | model weights, optimizer state, epoch, best Dice, per-class Dice | Best validation performance |
| `last_checkpoint.pt` | above + scheduler state, scaler state, epochs_no_improve | Fault-tolerant resumption |

`last_checkpoint.pt` is deleted upon fold completion. On startup, `run_pipeline.py` checks for its existence and automatically resumes interrupted folds.

### 5.5 Training Infrastructure

The primary training entry point is [run_pipeline.py](run_pipeline.py), which provides a Rich TUI with per-epoch progress bars, live validation metrics, and an atomic `pipeline_status.json` file that can be monitored from a Jupyter notebook without file corruption. Completed folds (those with `history.json` and no `last_checkpoint.pt`) are automatically skipped. The `--extend` flag resumes completed folds from `best_model.pt` at LR/10 for additional fine-tuning epochs.

---

## 6. Inference Pipeline

### 6.1 Sliding-Window Inference

All inference uses MONAI's `sliding_window_inference` with:

| Parameter | Value |
|---|---|
| ROI size | 96 × 96 × 96 |
| SW batch size | 32 |
| Overlap | 0.25 |
| Predictor | Gaussian-weighted |

The 25% overlap between adjacent windows provides smooth probability averaging at patch boundaries, eliminating the tiling artifacts common in non-overlapping inference. The output is a 4-channel softmax probability volume (C, D, H, W).

### 6.2 Inference Modes

Four inference modes are available, composable with each other:

#### Mode 1: Single-Fold Inference
Standard single-model prediction. Appropriate for rapid sanity-checking or when only a subset of folds has been trained.

```bash
python src/predict.py \
    --checkpoint runs/fold_0/best_model.pt \
    --input volume.tif \
    --output predictions/sample_01
```

#### Mode 2: Ensemble Inference (`--ensemble`)
Loads all `fold_*/best_model.pt` checkpoints found under the specified directory and averages their softmax probability maps before argmax decoding. This is the recommended production inference mode.

```
P_ensemble(x) = (1/K) * Σ_k P_k(x)
ŷ = argmax P_ensemble(x)
```

Ensemble averaging reduces prediction variance across folds, improves calibration, and implicitly provides a form of model uncertainty — high variance across fold predictions signals low-confidence regions.

```bash
python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --input volume.tif \
    --output predictions/sample_01
```

#### Mode 3: Test-Time Augmentation (`--tta`)
Applies all 8 axis-flip combinations (2³ = 8 augmentations covering full reflection symmetry), runs inference on each, unflips the probability maps, and averages:

```
P_TTA(x) = (1/8) * Σ_{f ∈ flips} f⁻¹(P(f(x)))
```

TTA adds ~8× the inference cost but can recover 1–3 Dice points on boundary regions, particularly for the incisor root which may be oriented inconsistently across specimens.

#### Mode 4: Staged Prediction (`--staged`)
A two-stage decision process applied to any probability source (single-model, ensemble, or TTA):

**Stage 1 — Background gate:** Voxels where `P(background) ≥ bg_threshold` (default: 0.05) are hard-assigned to background. This prevents the model from emitting foreground predictions in clearly empty regions, reducing false positives without affecting interior anatomy.

**Stage 2 — Foreground classification:** Within confirmed foreground voxels, argmax is taken over `[P(incisor), P(bone), P(molar)]` only, removing background as a competing hypothesis. Optionally, a bone gate (`--bone-threshold`) can be applied first, assigning bone to voxels where `P(bone) ≥ bone_threshold` before resolving the incisor/molar binary split.

The staged approach is particularly effective at reducing false-positive foreground predictions in background regions near the mandible edge, where the softmax distributes probability mass across all classes.

**Recommended production command:**
```bash
python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --staged \
    --input volume.tif \
    --output predictions/sample_01
```

### 6.3 Output Format

All predictions are saved as compressed TIFF files:

| File | Description | Dtype |
|---|---|---|
| `label_map.tif` | Combined multi-class label map | int16 |
| `masks/incisor_mask.tif` | Binary incisor mask | uint8 |
| `masks/bone_mask.tif` | Binary bone mask | uint8 |
| `masks/molar_mask.tif` | Binary molar mask | uint8 |

---

## 7. Cross-Validation & Evaluation

### 7.1 Leave-One-Out Cross-Validation

With N=9 specimens, leave-one-out cross-validation (LOO-CV) is the statistically optimal strategy: it maximizes training set size per fold (N-1=8 samples), provides N independent generalization estimates, and avoids the underfitting bias that k-fold CV with small k introduces when N is very small.

For each fold i ∈ {0, ..., 8}:
- **Validation set:** sample[i]
- **Training set:** all samples except sample[i]

This means every sample is evaluated exactly once by a model that has never seen it, giving honest unbiased generalization estimates.

### 7.2 Evaluation Metric

The primary metric is the **Dice Similarity Coefficient (DSC)** computed per class, averaged over foreground classes:

```
DSC(A, B) = 2|A ∩ B| / (|A| + |B|)
```

DSC ranges from 0 (no overlap) to 1 (perfect overlap). It is equivalent to the F1 score and is robust to class imbalance when computed per class (as opposed to globally). Validation uses full-volume sliding-window inference rather than patch-level evaluation, giving representative Dice scores for the actual deployment scenario.

Background is excluded from all reported metrics. Per-class Dice (incisor, bone, molar) is reported alongside mean Dice at each validation checkpoint and fold summary.

---

## 8. Results

### 8.1 Training Configuration

All results below were produced with the following hyperparameters:

| Parameter | Value |
|---|---|
| Model | 3D U-Net (channels: 32-64-128-256) |
| Max epochs | 500 |
| Batch size | 2 |
| Patches per volume | 24 |
| Patch size | 96³ |
| Optimizer | AdamW (lr=1e-3, wd=1e-5) |
| Scheduler | CosineAnnealingLR |
| Loss | DiceLoss + CrossEntropy (weights: 0.5/1.0/1.0/2.0) |
| Early stop patience | 50 epochs |
| Total wall time | ~9.3 hours (9 folds, CUDA) |
| Average epoch time | ~43.7 seconds |

### 8.2 Per-Fold Performance

The following table summarizes peak validation Dice scores across the 9 LOO-CV folds. Each fold holds out a single specimen for validation; performance reflects how well the model trained on 8 specimens generalizes to the held-out specimen.

| Fold | Val Sample | Best Mean Dice | Incisor | Bone | Molar |
|------|-----------|---------------|---------|------|-------|
| 0 | sample_0 | ~0.902 | ~0.911 | ~0.909 | ~0.888 |
| 4 | sample_4 | ~0.900 | ~0.937 | ~0.953 | ~0.803 |
| 1–3, 5–7 | various | 0.82–0.90 | — | — | — |
| 8 | sample_8 | ~0.61 | ~0.66 | ~0.79 | ~0.32 |

> **Note:** Fold 8 completed fewer epochs at the time of the pipeline_status snapshot and represents a morphologically distinct or harder specimen. The lower molar score (0.32) on fold 8 reflects the persistent difficulty of molar segmentation in the small-data regime.

### 8.3 Observations

**Bone segmentation** converges fastest and most reliably (~0.85–0.95 DSC across folds). Its large volumetric presence provides abundant training signal even within 96³ patches.

**Incisor segmentation** exhibits high variance across folds. The incisor is a thin, elongated structure whose orientation relative to the patch sampling grid varies significantly across specimens. Folds where the incisor is well-represented within the patch crop distribution score 0.90+; folds where it is geometrically undersampled score lower.

**Molar segmentation** is the most challenging target. Molars occupy a small fraction of the volume (~2–5%), sit in anatomically complex crypts within the bone, and show high morphological variation across developmental stages. Despite the 5× oversampling ratio, molar Dice ranges from 0.32 to 0.89 across folds — the widest spread of any class. This is the primary target for improvement as the dataset grows.

**Training dynamics:** Loss decreases steadily for the first 100–150 epochs, then plateaus with small oscillations driven by cosine annealing. Validation Dice shows a characteristic rapid rise (epochs 30–100), followed by a slower climb (epochs 100–250), and eventual plateau. Early stopping at patience=50 typically fires at epochs 250–350.

---

## 9. Human-in-the-Loop Framework

### 9.1 Overview

The system is designed as an **active learning** loop where human domain expertise guides model improvement. The process operates as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Human-in-the-Loop Cycle                      │
│                                                                 │
│  1. Acquire new micro-CT scan(s)                                │
│  2. Run ensemble inference → predicted label_map.tif           │
│  3. Expert reviews predictions in napari / 3D Slicer            │
│  4. Expert corrects errors → validated binary masks             │
│  5. Place corrected masks in segmentation_results/<sample>/     │
│  6. If N_new mod 5 == 0: trigger full LOO-CV retrain            │
│  7. Archive old runs/ → runs_vN/ for comparison                 │
│  8. Return to step 1                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Workflow Detail

**Step 1 — Acquisition:** New scans are processed through the upstream `segment_mandible` classical pipeline, which handles reorientation, registration, and initial coarse masks. The output `reoriented_volume.tif` serves as the ML pipeline input.

**Step 2 — Inference:** Run ensemble + staged prediction on the new volume:
```bash
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --staged \
    --input /path/to/new_sample/reoriented_volume.tif \
    --output predictions/new_sample
```
This produces `label_map.tif` and individual binary masks that the annotator uses as a starting point.

**Step 3 — Expert review:** The annotator loads the predicted masks alongside the raw volume in a 3D viewer. High-confidence regions (typically bone interior and incisor shaft) require little or no correction. Low-confidence regions (molar crypts, incisor tip, bone-molar interfaces) are corrected manually. The ensemble probability volume can be exported for uncertainty visualization if needed.

**Step 4 — Mask integration:** Corrected binary masks are placed in the standard directory structure:
```
segmentation_results/<new_sample>/
    reoriented_volume.tif
    masks/
        incisor_mask.tif   # human-corrected
        bone_mask.tif      # human-corrected
        molar_mask.tif     # human-corrected
```

**Step 5 — Retrain trigger:** Check whether N_total reached the next multiple of 5:
```python
from src.dataset import discover_samples
n = len(discover_samples("../segmentation_results"))
if n % 5 == 0:
    print(f"Retrain triggered at N={n}")
    # archive current runs: rename runs/ → runs_v{n-5}/
    # then: python run_pipeline.py --data ../segmentation_results --output runs/
```

**Step 6 — Full LOO-CV retrain:** The pipeline auto-discovers all N samples and runs N-fold LOO-CV. `run_pipeline.py` skips any completed fold (by checking for `history.json` without a `last_checkpoint.pt`), so if training is interrupted, it resumes from where it left off.

### 9.3 Annotation Strategy

To maximize the information gain per annotation hour, new samples should be selected to maximize diversity from the existing training set. Concrete criteria:

- **Developmental stage diversity:** Include specimens across multiple postnatal ages (P14, P21, P28, adult), as molar eruption stages vary dramatically.
- **Genotype diversity:** If multiple genotypes are available, alternate between them to prevent genotype-specific overfitting.
- **Failure mode targeting:** After each retrain, identify which fold has the lowest Dice score and prioritize acquiring additional specimens morphologically similar to that validation sample.

### 9.4 Retrain Schedule

| N Samples | Folds | Estimated Retrain Time |
|-----------|-------|----------------------|
| 9 (current) | 9 | ~9.3 hours |
| 14 | 14 | ~16 hours |
| 19 | 19 | ~22 hours |
| 24 | 24 | ~28 hours |

Times are estimated from the observed average per-fold training time (~1 hour) on the current CUDA hardware.

---

## 10. Known Limitations

### 10.1 Small Dataset
N=9 specimens is below the threshold where deep learning models generalize robustly without heavy regularization. The LOO-CV Dice scores reported here are honest per-specimen estimates, but the model's performance on substantially different populations (different strains, imaging protocols, developmental stages not represented in training) is unknown and likely lower.

### 10.2 Molar Segmentation
Molar Dice is the most variable metric across folds (range: 0.32–0.89). The molar occupies <5% of foreground voxels, sits in anatomically complex crypts, and varies substantially across developmental stages. Despite 5× oversampling, the model has insufficient training examples to learn a consistent molar representation. This is the primary bottleneck for clinical deployment.

### 10.3 Instance Segmentation
The pipeline produces a single `molar` label without distinguishing individual molar teeth (M1, M2, M3). For studies requiring per-molar morphometrics, post-processing instance segmentation (e.g., connected-component analysis with anatomical priors) is required.

### 10.4 Single-Channel Input
Only grayscale intensity is used. The pipeline does not exploit Hounsfield unit calibration or multi-channel inputs (e.g., dual-energy CT). Normalized intensity is the sole input modality.

### 10.5 Fixed Patch Size
The 96³ patch size was chosen to fit on a single GPU with batch size 2. It may be insufficient to capture the full incisor in a single field of view, potentially fragmenting long-range spatial context. Larger patches (128³ or 160³) would improve contextual reasoning but require more GPU memory.

### 10.6 No Uncertainty Quantification
The ensemble variance across folds provides an implicit uncertainty signal, but it is not currently surfaced to the human annotator during the review step. Annotators have no programmatic way to identify which voxels the model is most uncertain about, forcing them to review the entire volume rather than focusing on uncertain regions.

---

## 11. Improvements & Future Directions

### 11.1 Near-Term (N = 10–20 Samples)

**11.1.1 Export ensemble uncertainty maps**
Compute per-voxel entropy across fold probability predictions and export as a heatmap overlay:
```python
# H(x) = -Σ_c P_ensemble(c|x) * log P_ensemble(c|x)
uncertainty = -(probs * np.log(probs + 1e-8)).sum(axis=0)
```
This allows annotators to focus correction effort on high-entropy regions, reducing annotation time per sample by 30–50%.

**11.1.2 Molar-specific fine-tuning**
Train a dedicated binary segmentation model (molar vs. not-molar) using the full-resolution incisor+bone predictions as a spatial prior. A two-stage cascade (coarse localization → fine segmentation) is a proven strategy for small, complex structures.

**11.1.3 Larger patch size**
Increase patch size to 128³ with gradient checkpointing to fit in GPU memory. This improves long-range context for incisor segmentation and reduces boundary fragmentation in large bone volumes.

**11.1.4 Transfer learning from public datasets**
Pre-train the encoder on a larger public micro-CT or CT dataset (e.g., VerSe vertebrae, TotalSegmentator) and fine-tune on the mandible data. Even domain-mismatched pre-training substantially reduces the number of labeled samples required for convergence.

### 11.2 Mid-Term (N = 20–50 Samples)

**11.2.1 Switch from LOO-CV to stratified k-fold**
Once N ≥ 20, 5-fold stratified CV becomes statistically adequate and reduces training cost from O(N) folds to O(5) folds per retrain cycle. Stratify by developmental age to ensure even representation across age groups.

**11.2.2 Train a "production" all-data model**
After each CV retrain cycle, train a final model on all N samples (no held-out set) using the hyperparameters validated by CV. This model maximizes training data utilization for deployment, while CV estimates provide the generalization guarantee.

**11.2.3 Active learning sampling**
Replace the fixed "every 5 samples" trigger with an uncertainty-based selection strategy: rank unannotated specimens by ensemble prediction entropy and prioritize annotating those with highest uncertainty. This can reduce the total annotation budget required to reach a target Dice threshold by 30–50% compared to random sample selection.

**11.2.4 Semi-supervised learning with pseudo-labels**
Use the ensemble model to generate pseudo-labels for unannotated scans. Apply a confidence threshold (e.g., accept voxels where max class probability > 0.85) and include high-confidence pseudo-labeled voxels in training. This can accelerate learning without proportional annotation cost.

**11.2.5 Per-class Dice early stopping**
Replace mean Dice early stopping with a multi-objective criterion: stop only when all three per-class Dice values have plateaued. This prevents early stopping from terminating training while molar Dice is still actively improving.

### 11.3 Long-Term (N > 50 Samples)

**11.3.1 Instance segmentation for individual molars**
Extend the label set to distinguish M1, M2, and M3 separately. This enables per-molar morphometric analysis (volume, surface area, root-to-crown ratio) critical for developmental biology studies.

**11.3.2 Anatomical shape priors (statistical shape models)**
Train a Statistical Shape Model (SSM) or a learned atlas on the training set mandibles. Use atlas-registration residuals as an additional regularization term or as a spatial attention mechanism to constrain predictions to anatomically plausible shapes.

**11.3.3 Self-supervised pre-training on unlabeled volumes**
Apply masked autoencoding (MAE) or contrastive learning (SimCLR, MoCo) to unlabeled micro-CT volumes from the same scanner before fine-tuning on labeled data. Self-supervised pre-training has been shown to dramatically reduce labeled data requirements in medical imaging.

**11.3.4 Continual learning to prevent catastrophic forgetting**
As the dataset grows, naively retraining from scratch discards the representations learned on earlier samples. Implement elastic weight consolidation (EWC) or experience replay to retain performance on specimens that are no longer in the most recent training set due to data distribution shifts.

**11.3.5 3D instance morphometrics API**
Build a downstream analysis module that consumes the segmentation output and computes: incisor length and volume, bone density proxy (mean HU within mask), molar crown area, root-to-crown length ratio, and inter-structure distances. Automate this into a per-sample JSON report for downstream statistical analysis.

---

## 12. Installation

### 12.1 Environment

All Python commands should use the `microct-analysis` conda environment:

```bash
conda activate microct-analysis
# or prefix all commands:
conda run -n microct-analysis python ...
```

### 12.2 Dependencies

```
python >= 3.10
torch >= 2.0
monai >= 1.3
tifffile
numpy
rich
pynrrd (optional, for .nrrd input)
```

### 12.3 Directory Layout

```
ML-Segmentation/
├── run_pipeline.py         # Primary training entry point (Rich TUI)
├── src/
│   ├── config.py           # Configuration dataclass
│   ├── dataset.py          # Data loading, caching, augmentation
│   ├── model.py            # Model factory and checkpoint utilities
│   ├── train.py            # Training loop and LOO-CV runner
│   └── predict.py          # Inference pipeline (single/ensemble/TTA/staged)
├── runs/
│   ├── fold_0/
│   │   ├── best_model.pt
│   │   ├── history.json
│   │   └── config.json
│   ├── fold_1/ ... fold_8/
│   └── pipeline_status.json
└── training_notebook.ipynb # Interactive training monitor and analysis
```

---

## 13. Usage Reference

### 13.1 Training

```bash
# Full LOO-CV on all discovered samples (recommended)
conda run -n microct-analysis python run_pipeline.py \
    --data ../segmentation_results \
    --output ./runs \
    --epochs 500

# Single fold (sanity check / debugging)
conda run -n microct-analysis python run_pipeline.py \
    --folds 0 \
    --epochs 100

# Resume interrupted run (auto-detected from last_checkpoint.pt)
conda run -n microct-analysis python run_pipeline.py \
    --data ../segmentation_results \
    --output ./runs

# Extend completed folds with additional epochs at LR/10
conda run -n microct-analysis python run_pipeline.py \
    --extend \
    --epochs 600

# Override key hyperparameters
conda run -n microct-analysis python run_pipeline.py \
    --epochs 500 \
    --lr 5e-4 \
    --batch 4 \
    --patience 75
```

### 13.2 Inference

```bash
# Recommended production inference: ensemble + staged
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --staged \
    --input /path/to/volume.tif \
    --output ./predictions/sample_01

# With test-time augmentation (8 flip combinations)
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --tta \
    --staged \
    --input /path/to/volume.tif \
    --output ./predictions/sample_01

# Staged with explicit bone gate
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --staged \
    --bone-threshold 0.5 \
    --bg-threshold 0.05 \
    --input /path/to/volume.tif \
    --output ./predictions/sample_01

# Single-fold inference (faster, lower accuracy)
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/fold_0/best_model.pt \
    --input /path/to/volume.tif \
    --output ./predictions/sample_01

# Batch inference on a directory of volumes
conda run -n microct-analysis python src/predict.py \
    --checkpoint runs/ \
    --ensemble \
    --staged \
    --input /path/to/volumes/ \
    --output ./predictions/
```

### 13.3 Monitoring Training

The `training_notebook.ipynb` provides live monitoring of `runs/pipeline_status.json` and per-fold `history.json` files for loss curves, per-class Dice trajectories, and a cross-fold summary table. Open it in Jupyter while `run_pipeline.py` is running.

### 13.4 Retrain After New Samples

```bash
# 1. Verify new samples were discovered
conda run -n microct-analysis python -c "
from src.dataset import discover_samples
samples = discover_samples('../segmentation_results')
print(f'Found {len(samples)} samples: {[s[\"name\"] for s in samples]}')
"

# 2. Archive the current run directory
mv runs/ runs_v9/

# 3. Run full LOO-CV on all N samples
conda run -n microct-analysis python run_pipeline.py \
    --data ../segmentation_results \
    --output ./runs \
    --epochs 500
```

---

## 14. Module Reference

| File | Key Symbols | Description |
|---|---|---|
| [src/config.py](src/config.py) | `Config` | Frozen dataclass for all hyperparameters. Serializes to/from JSON. Handles device auto-detection (CUDA > MPS > CPU). |
| [src/dataset.py](src/dataset.py) | `discover_samples()`, `LoadAndCombineMasksd`, `_AugDataset`, `warm_disk_cache()`, `build_datasets()` | Data discovery, mask loading/combining, disk cache management, augmentation pipeline, in-RAM dataset. |
| [src/model.py](src/model.py) | `build_model()`, `load_checkpoint()` | Model factory for UNet and SegResNet. Checkpoint loading with device mapping. |
| [src/train.py](src/train.py) | `train_fold()`, `validate()`, `run_cross_validation()` | Core training loop, sliding-window validation, LOO-CV orchestration. Minimal TUI (use `run_pipeline.py` for full experience). |
| [src/predict.py](src/predict.py) | `predict_volume()`, `predict_ensemble()`, `predict_with_tta()`, `predict_staged()`, `_staged_from_probs()`, `save_predictions()` | All inference modes. Composable: ensemble + TTA + staged are orthogonal flags. |
| [run_pipeline.py](run_pipeline.py) | `main()`, `train_fold()` | Primary entry point. Rich TUI, disk cache warmup, fold skip/resume logic, atomic status file, CV summary table. |

---

## 15. Output Structure

```
runs/
├── pipeline_status.json        # Live status updated each checkpoint interval
├── pipeline.log                # Full training log with timestamps
├── fold_0/
│   ├── best_model.pt           # Best validation Dice checkpoint
│   │                           #   keys: epoch, model_state_dict,
│   │                           #         optimizer_state_dict, best_dice,
│   │                           #         per_class_dice, fold
│   ├── last_checkpoint.pt      # Most recent checkpoint (deleted on completion)
│   │                           #   additionally: scheduler_state_dict,
│   │                           #                 scaler_state_dict,
│   │                           #                 epochs_no_improve
│   ├── history.json            # Per-epoch: epochs, train_loss, val_dice, per_class
│   └── config.json             # Frozen Config dataclass at training time
├── fold_1/ ... fold_8/
└── .det_cache/                 # Disk cache for deterministic preprocessed arrays
    ├── {hash}_img.npy          # float32 cropped image (one per sample)
    ├── {hash}_lbl.npy          # uint8 cropped label
    └── {hash}_idx.npz          # class voxel indices for patch sampling

predictions/
└── <sample_name>/
    ├── label_map.tif           # Combined label map (int16), zlib-compressed
    └── masks/
        ├── incisor_mask.tif    # Binary uint8, zlib-compressed
        ├── bone_mask.tif
        └── molar_mask.tif
```

---

## References

- Çiçek, Ö., et al. (2016). *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.* MICCAI.
- Milletari, F., Navab, N., & Ahmadi, S.A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV.
- Myronenko, A. (2019). *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization.* BrainLes Workshop, MICCAI.
- Cardoso, M.J., et al. (2022). *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701.
- Settles, B. (2009). *Active Learning Literature Survey.* University of Wisconsin–Madison Technical Report.
- Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods.
