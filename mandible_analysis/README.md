# **Project Plan: Deep Learning Segmentation of Micro-CT Data Without Labeled Ground Truth**

## **Objective**

Develop a deep learning pipeline to segment micro-CT data in the absence of labeled ground truth. The goal is to generate high-quality segmentations using pseudo-labeling, self-supervised pretraining, and minimal manual annotation.

---

## **Phase 1: Data Preparation and Pseudo-Labeling**

### Tasks:

1. **Load and preprocess 3D micro-CT `.bmp` volumes** using Python.
2. **Apply classical segmentation algorithms** (e.g., Otsu thresholding, region growing, Frangi filters) to generate pseudo-labels.
3. **Visually verify** pseudo-labels for structural consistency (e.g., bone, cartilage).

### Example:

```python
from skimage.filters import threshold_otsu
thresh = threshold_otsu(slice_2d)
pseudo_labels = slice_2d > thresh
```

### Deliverables:

* Initial pseudo-labeled dataset
* Visual overlays of raw vs. labeled slices

---

## **Phase 2: Train Initial Supervised Model on Pseudo-Labels**

### Tasks:

1. Train a 2D U-Net (e.g., using `segmentation_models_pytorch`) on pseudo-labeled slices.
2. Apply model predictions across the volume.
3. Iterate:

   * Identify label failures
   * Refine pseudo-labels or correct a subset manually
   * Retrain model with improved data

### Tools:

* `PyTorch`, `segmentation_models_pytorch`
* `skimage`, `opencv`, `napari`

---

## **Phase 3: Self-Supervised Pretraining (Optional but Recommended)**

### Tasks:

1. Pretrain an encoder using SSL methods on the entire unlabeled micro-CT volume.
2. Techniques may include:

   * SimCLR / BYOL (contrastive)
   * MAE / Swin-MAE (masked modeling)
   * Rotation prediction or jigsaw tasks
3. Fine-tune the encoder in the U-Net model with existing pseudo-labels.

### Libraries:

* `MONAI`, `solo-learn`, `torchssl`, `lightly`

---

## **Phase 4: Transfer Learning and Model Generalization**

### Tasks:

1. Evaluate pretrained models (e.g., `nnU-Net`, MONAI zoo) on your dataset.
2. Freeze encoder layers, fine-tune decoder with existing annotations.
3. Compare performance vs. custom-trained models.

---

## **Phase 5: Active Learning (If Manual Annotation is Feasible)**

### Tasks:

1. Annotate a minimal number of slices (10–20) using `napari`, `ITK-SNAP`, or `Label Studio`.
2. Train a model (e.g., DeepLabV3 or U-Net) on these labels.
3. Use uncertainty sampling to suggest new slices to annotate.

---

## **Phase 6: Weakly Supervised Alternatives (If Annotations Are Sparse)**

### Tasks:

1. Generate coarse labels via:

   * Bounding boxes
   * Points
   * Scribbles
2. Train a weakly supervised model using methods such as:

   * MIL, CAM-based learning
   * `PointRend`
   * GrabCut + U-Net hybrids
3. Refine labels via post-processing (e.g., CRFs or morphology)

---

## **Technology Stack**

| Component           | Libraries/Tools                                              |
| ------------------- | ------------------------------------------------------------ |
| **Data Loading**    | `PIL`, `tifffile`, `numpy`                                   |
| **Visualization**   | `napari`, `matplotlib`                                       |
| **Preprocessing**   | `scikit-image`, `opencv`                                     |
| **Segmentation**    | `PyTorch`, `segmentation_models_pytorch`, `MONAI`, `nnU-Net` |
| **SSL/Transfer**    | `solo-learn`, `lightly`, `MONAI`, `torchssl`                 |
| **Pseudo-labeling** | `threshold_otsu`, `frangi`, `watershed`                      |
| **Annotation GUI**  | `napari`, `Label Studio`, `ITK-SNAP`                         |

---

## **Immediate Next Steps**

1. Apply classical threshold-based segmentation on a subset of slices.
2. Generate pseudo-labels and train a baseline U-Net.
3. Evaluate results visually and define criteria for label refinement or model iteration.

---

Let me know if you'd like this exported as a PDF or Markdown project spec, or if you'd like a starter training script using U-Net and pseudo-labels.
