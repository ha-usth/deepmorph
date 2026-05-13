# Insect Wing Landmark Detection: MAE + Few-shot Pipeline

A 3-stage pipeline for landmark detection on insect wing images,
combining Self-Supervised Learning (MAE) and Few-shot Fine-tuning.

## Overview

```
[Stage 1] MAE Pretraining             [Stage 2] Few-shot Fine-tuning        [Stage 3] Predict
  ~20K UNLABELED images    →              5 LABELED images       →            New images
  (Droso-big, mosquito, ...)               (Bactro/Fly/...)                       ↓
        ↓                                        ↓                           .txt files
  Encoder weights                        Fine-tuned model                 (iMorph format)
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements: GPU with ≥8GB VRAM (≥16GB recommended for ViT-Base).

## Data Preparation

### Stage 1 (pretraining): unlabeled images
Folders only need to contain image files (.jpg, .png, .bmp, etc.):
```
data_pretrain/
├── droso_big/
│   ├── 0001.bmp
│   └── ...
└── mosquito_nolte/
    ├── img001.jpg
    └── ...
```

Recommended data sources:
- **Droso-big**: https://gigadb.org/dataset/100706
- **Mosquito repository (Nolte 2025)**: https://www.nature.com/articles/s41597-025-05043-3
- Other datasets from your original paper

### Stage 2 (fine-tuning): labeled images
iMorph format:
```
Bactro/
├── 001.bmp
├── 001.txt    # one line per landmark: "x y"
├── 002.bmp
├── 002.txt
└── ...
```

## Running the Pipeline

### Stage 1: MAE Pretraining

Edit `CONFIG` in `stage1_pretrain_mae.py`:
- `data_dirs`: list of directories containing unlabeled images
- `epochs`: 200 (minimum); 400 if time allows
- `batch_size`: depends on GPU (64 for 16GB VRAM)

```bash
python stage1_pretrain_mae.py
```

Output: `checkpoints/mae_pretrain_final.pth` (~330MB for ViT-Base)

**Estimated time**: 1–3 days on a single RTX 3090 GPU, depending on image count.

### Stage 2: Few-shot Fine-tuning

Edit `CONFIG` in `stage2_finetune_landmark.py`:
- `mae_checkpoint`: path to the file from Stage 1
- `target_data_dir`: target dataset (e.g. Bactro)
- `n_shots`: 1, 3, 5, 10, or 15 (run multiple times with different values!)
- `num_landmarks`: number of landmarks per dataset (Droso-small: 15, Bactro: 12, Fly/Diacha: 10)

```bash
python stage2_finetune_landmark.py
```

Output: `checkpoints/finetune_best_n{N}.pth` + MRE report.

**Time**: ~10–30 minutes per configuration.

### Stage 3: Prediction

Edit `CONFIG` in `stage3_predict.py`:
- `finetune_checkpoint`: model from Stage 2
- `input_dir`: directory of images to predict

```bash
python stage3_predict.py
```

Output: `predictions/` directory containing .txt files in iMorph format.
Can be opened in iMorph GUI for manual correction.

## Experimental Design for Paper

### Main Comparison Table (proposed)

| Method | Pretrain data | n=1 | n=3 | n=5 | n=10 | n=15 |
|---|---|---|---|---|---|---|
| iMorph (original) | - | x.x | x.x | x.x | x.x | x.x |
| ViT + ImageNet | ImageNet | x.x | x.x | x.x | x.x | x.x |
| **MAE-Wing (ours)** | **Wing images** | **x.x** | **x.x** | **x.x** | **x.x** | **x.x** |

### Evaluation Datasets
Run Stage 2+3 for 5 datasets from the original paper:
- Droso-small (15 landmarks)
- Droso-big (15 landmarks)
- Fly (10 landmarks)
- Bactro (12 landmarks)
- Diacha (10 landmarks)

### Recommended Ablation Studies
1. **Mask ratio**: 50%, 60%, 75%, 85%
2. **Pretrain data size**: 1K, 5K, 10K, 20K images
3. **Encoder freeze vs unfreeze**
4. **Model size**: ViT-Small vs ViT-Base
5. **Heatmap sigma**: 1, 2, 3, 4

## Important Notes

### On the MRE metric
MRE is measured at `image_size=224`. To **compare with the original paper**, rescale:
```python
MRE_original_resolution = MRE_at_224 * (original_resolution / 224)
```
Example for Droso-small (1400×900): scale ≈ 1400/224 ≈ 6.25.

### On data leakage
Be careful: if pretraining uses the same dataset as fine-tuning (e.g. pretrain on Droso-big then fine-tune on Droso-small), this must be clearly explained in the paper. Recommended split:
- Pretrain on: Droso-big + mosquito + others
- Fine-tune on: Droso-small / Fly / Bactro / Diacha (not included in pretrain)

### On reproducibility
- Set seed (already in code)
- Save config into checkpoint (already done)
- Run each setting 3–5 times with different seeds, report mean ± std

## File Structure

```
insect_wing_mae/
├── README.md                     # This file
├── requirements.txt
├── utils.py                      # Heatmap, MRE, helpers
├── stage1_pretrain_mae.py        # Stage 1 script
├── stage2_finetune_landmark.py   # Stage 2 script
├── stage3_predict.py             # Stage 3 script
├── models/
│   ├── mae.py                    # MAE architecture
│   └── landmark_head.py          # Landmark head + full model
└── datasets/
    ├── unlabeled_dataset.py      # For pretraining (unlabeled)
    └── landmark_dataset.py       # For fine-tuning (labeled)
```

## References

- He et al. "Masked Autoencoders Are Scalable Vision Learners" CVPR 2022
- Geldenhuys et al. "Deep learning approaches to landmark detection in tsetse wing images" PLOS Comp Bio 2023
- Nolte et al. "ITHILDIN: Automated landmark and semilandmark annotation for wing geometric morphometrics in Diptera" bioRxiv 2026
- Nolte et al. "Comprehensive Mosquito Wing Image Repository" Sci Data 2025
- Nguyen et al. "A lightweight keypoint matching framework..." Ecological Informatics 2022
