# MONAI-for-Brain-Tumor-Segmentation
just using MONAI and build 3D U-Net for brain tumor segmentation that view through what will happen
# Brain Tumor Segmentation with MONAI (3D U-Net)

This project explores how different data augmentation strategies affect the segmentation performance of a 3D U-Net model trained on the **BraTS dataset**, using the **MONAI** medical imaging framework.

---

## Project Overview

This experiment was designed to answer a simple question:

> 💭 *"How do different data augmentations (like flip, noise, or affine) influence model stability and Dice performance in 3D medical image segmentation?"*

The goal was not to achieve state-of-the-art accuracy, but to **build a fully reproducible baseline**, understand **training behavior under different augmentations**, and gain insight into how **data diversity** impacts model generalization.

---

## Core Ideas

| Component | Description |
|------------|-------------|
| **Task** | Binary brain tumor segmentation (tumor vs. background) |
| **Dataset** | BraTS 2021 (`.nii.gz` volumes, 4 modalities per case) |
| **Framework** | [MONAI](https://monai.io/) + PyTorch |
| **Model** | 3D U-Net with encoder–decoder skip connections |
| **Loss Function** | `DiceCELoss` (Dice + CrossEntropy) |
| **Evaluation Metric** | Dice coefficient (0–1, higher is better) |
| **Hardware** | CPU / GPU (tested on local environment, MONAI CPU mode supported) |

---

##  Implementation Workflow

### 1. **Dataset Preparation**
- Download the BraTS dataset (`imagesTr` and `labelsTr` folders).
- Convert labels to binary masks (`tumor > 0 → 1`).
- Verify shape alignment between image and label volumes.

### 2. **Baseline Training (No Augmentation)**
- Input normalization with `NormalizeIntensityd`.
- Patch-based training using `RandCropByPosNegLabeld`:
  - Patch size: `96 × 96 × 96`
  - Sampling ratio: `pos=3`, `neg=1`
- Model: 3D U-Net with 4 down-sampling levels.
- Loss: Dice + CrossEntropy (`DiceCELoss`).
- Optimizer: Adam (`lr=3e-4` with `ReduceLROnPlateau` scheduler).

### 3. **Experiment: Adding Augmentation**
Compared variants:
- `Baseline` – no augmentation  
- `+Flip` – random axis flipping  
- `+Flip+Noise` – add Gaussian noise  
- `+Affine+Elastic` – geometric warping  
Each variant was trained with the same seed, hyperparameters, and number of epochs for fair comparison.

### 4. **Evaluation**
- Used `sliding_window_inference` for full-volume Dice evaluation.  
- Dice metric computed with `DiceMetric(include_background=False)`.

---

## Results Summary

| Experiment | Description | Observed Dice Trend |
|-------------|--------------|----------------------|
| Baseline (no aug) | Stable convergence, moderate overfitting | ~0.45 |
| +Flip / +Noise | Slight regularization, small Dice variation | 0.4–0.46 |
| +Affine / +Elastic | Higher randomness, unstable if dataset small | 0.3–0.4 |

 **Takeaway:** Augmentation helps regularization **only when enough cases and foreground patches exist**.  
With limited data, heavy transforms may cause underfitting or instability.

---

##  Key Learnings

1. **Patch sampling matters more than augmentation** when training on small datasets.  
2. **Too strong augmentation (e.g., affine+elastic)** may distort anatomical structures and harm Dice.  
3. **Balanced data sampling** (foreground vs. background) has a direct impact on model stability.  
4. **DiceCELoss** provides smoother convergence than standalone Dice or BCE.

---

##  Future Improvements

| Goal | Next Step |
|------|------------|
| **1. Data scale-up** | Train on full BraTS dataset (20–50 cases) to evaluate generalization. |
| **2. Augmentation tuning** | Systematically vary augmentation probability (`prob=0.2–0.7`) and intensity. |
| **3. Multi-class segmentation** | Extend from binary (tumor vs. background) to sub-regions (edema, enhancing core, necrotic). |
| **4. Visualization** | Add MONAI’s `NiftiSaver` or `matplotlib` slices to visualize predictions. |
| **5. Automation** | Wrap training & evaluation into a single reproducible script with config files. |

---

##  Environment

- Python 3.10  
- MONAI 1.x  
- PyTorch ≥ 2.0  
- NumPy, Matplotlib, Nibabel  

```bash
conda create -n monai_cpu python=3.10
conda activate monai_cpu
pip install monai torch torchvision nibabel matplotlib
