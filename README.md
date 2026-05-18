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
Folders only need to contain image files (.jpg, .png, .bmp, etc.). The annotation files (.txt) will be ignored during pretraining.
```
train_pool/
├── droso_big/
│   ├── 0001.bmp
│   └── ...
└── mosquito_nolte/
    ├── img001.jpg
    └── ...
```

Recommended data sources:
- **iMorph datasets: droso-small, droso-big, ...**: https://github.com/morphometrics/iMorph
- **Droso-big**: https://gigadb.org/dataset/100706
- **Mosquito repository (Nolte 2025)**: https://www.nature.com/articles/s41597-025-05043-3
- Other datasets from your original paper

### Stage 2 (fine-tuning): labeled images
iMorph format:
```
droso_big/
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
