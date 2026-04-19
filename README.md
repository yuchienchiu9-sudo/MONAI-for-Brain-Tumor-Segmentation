# Brain Tumor Segmentation with MONAI (3D U-Net)

A 3D brain tumor segmentation project built with **MONAI** and **PyTorch**, 
trained on the **BraTS dataset**. This project explores how different data 
augmentation strategies affect segmentation stability and Dice performance, 
and documents the debugging process that improved Val Dice from unstable 
results to a stable **0.91+**.

---

## Project Background

This project was originally developed as a course mini-project, 
then revisited and significantly improved after graduation. 
The initial version suffered from training instability and 
fluctuating Dice scores. After systematic debugging, 
the root causes were identified and resolved.

---

## Bugs Found & Fixed

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Dice score highly unstable | Validation used only 1 case (BRATS_001) | Changed to full val set (97 cases) |
| Poor normalization | `ScaleIntensity` applied globally across 4 MRI modalities | Replaced with `NormalizeIntensityd(channel_wise=True)` |
| Gradient explosion | No gradient clipping in some versions | Added `clip_grad_norm_(model.parameters(), 1.0)` |
| Slow / no convergence | Learning rate inconsistent across versions | Unified to `lr=2e-4` with `ReduceLROnPlateau` |

---

## Experiment Design

Three augmentation strategies were compared under identical conditions:
- Same random seed (`SEED=42`)
- Same train/val split (387 train / 97 val)
- Same model architecture and hyperparameters
- Same number of epochs (120)

| Version | Augmentation | Best Val Dice | Best Epoch |
|---------|-------------|--------------|------------|
| Baseline | None | 0.9141 | 105 |
| Flip Only | RandFlip (prob=0.5) | TBD | TBD |
| Affine + Flip | RandAffine + RandFlip | 0.9135 | 85 |

### Key Findings
- Augmentation significantly **speeds up early convergence**
  (Epoch 5: Affine+Flip 0.79 vs Baseline 0.64)
- With sufficient training data (387 cases), 
  final Dice scores converge to similar levels (~0.91)
- Augmentation benefit is expected to be more pronounced 
  with smaller datasets

---

## Model Architecture

```python
UNet(
    spatial_dims=3,
    in_channels=4,       # BraTS: T1, T1ce, T2, FLAIR
    out_channels=1,      # Binary: tumor vs background
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
# Parameters: ~4M
```

---

## Training Details

| Config | Value |
|--------|-------|
| Dataset | BraTS (484 cases total) |
| Train / Val Split | 387 / 97 |
| Patch Size | 96 × 96 × 96 |
| Batch Size | 1 |
| Loss Function | DiceCELoss (Dice + CrossEntropy) |
| Optimizer | Adam (lr=2e-4, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (patience=6, factor=0.5) |
| Inference | Sliding window (ROI=128³, overlap=0.5) |
| Hardware | RTX 4060 8GB |

---

## Results

### Affine + Flip
<img width="448" height="513" alt="{D991F9EB-2D63-404F-AA32-AF18C4E76C83}" src="https://github.com/user-attachments/assets/8c237a0e-d2f3-41ee-9654-95825a2ae2fe" />
<img width="927" height="374" alt="{28B1D5AF-B348-4019-B0B7-817070864B45}" src="https://github.com/user-attachments/assets/8d979e4b-7a9c-4a96-ae55-0a525fdce6f7" />



### Baseline
<img width="471" height="367" alt="{24A6485C-356C-4A5A-8698-713EA8963D04}" src="https://github.com/user-attachments/assets/ea9c674c-0189-46a8-b07f-1dcfe2987fef" />

<img width="1017" height="411" alt="{EBF5D2EF-81A2-4DE2-9543-1BDFE0DD8C86}" src="https://github.com/user-attachments/assets/795babfb-95e7-4d1a-b26b-6c15c8a3fff3" />


---

## Dataset

BraTS dataset from Medical Segmentation Decathlon:
- Link: https://decathlon-10.grand-challenge.org/
- Direct: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

---

## Environment

```bash
conda create -n monai_gpu python=3.10
conda activate monai_gpu
pip install monai torch torchvision nibabel matplotlib
```

- Python 3.10
- MONAI 1.x
- PyTorch ≥ 2.0
- CUDA supported (tested on RTX 4060)

---

## Future Work

- Complete Flip Only experiment and finalize 3-way comparison
- Extend to multi-class segmentation (edema, enhancing tumor, necrotic core)
- Try SwinUNETR (Transformer-based architecture)
- Add prediction visualization with matplotlib slices
