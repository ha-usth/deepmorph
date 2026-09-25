# Insect Wing Landmark Detection: MAE + Few-shot Pipeline

A 3-stage pipeline for landmark detection on insect wing images,
combining Self-Supervised Learning (MAE) and Few-shot Fine-tuning.

## Overview

```
[Stage 1] MAE Pretraining             [Stage 2] Few-shot Fine-tuning        [Stage 3] Predict
      UNLABELED images    →               LABELED images       →            New images
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

**Estimated time**: 1h-48hs on a single RTX 3090 GPU, depending on image count.

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

## Reproducing the Paper Experiments

The three `stageN_*.py` scripts above run one configuration at a time and are
useful for a single dataset. The two `e*` scripts below are what the paper's
tables were produced with: they sweep protocols, datasets and random seeds,
cache every completed run, and write the result tables directly.

### `e1_pretrain_mae.py` — Stage 1 under three pretraining protocols

The encoder is the only thing that changes between the three protocols; Stages
2 and 3 are identical throughout. The protocols differ solely in what the
encoder saw before few-shot fine-tuning:

| Protocol | Stage 1 | Pretraining pool |
|---|---|---|
| `PC` | none — the public MAE-ImageNet checkpoint is used as released | – |
| `PA` | continued MAE pretraining on the whole unlabeled biological pool | all datasets |
| `PB` | same, but the target dataset is excluded (leave-one-dataset-out) | all but the target |

`PB` is the protocol that answers the practical question: it measures how well
the encoder transfers to a dataset whose images it has never seen, which is the
situation for a newly collected or rare species.

```bash
python e1_pretrain_mae.py --protocol PC                                   # no training, exports ImageNet
python e1_pretrain_mae.py --protocol PA                                   # ~15 min
python e1_pretrain_mae.py --protocol PB --exclude ./train_pool/sea_bass   # one run per target
```

`PA` and `PC` pools do not depend on the target, so one checkpoint each serves
every dataset. `PB` does, so its checkpoints carry the held-out dataset name:

```
checkpoints/mae_PC_s0_imagenet_only.pth
checkpoints/mae_PA_s0_best.pth
checkpoints/mae_PB_sea_bass_s0_best.pth
```

Useful options: `--total-steps` (default 2000 optimizer steps, not epochs —
one epoch over a few-hundred-image pool is only about six steps), `--keep`
(which artefacts to leave on disk; the default `best log` writes 2 files
instead of 6), `--seed`, and `--target` when several directories are excluded
at once. Each run also writes a `*_log.json` with the configuration, the pool
composition and the validation-loss curve.

### `e2_finetune_eval.py` — Stage 2 + Stage 3, swept and scored

One invocation fine-tunes on N annotated images, predicts on the held-out test
set, and scores the predictions, repeating over protocols, shot counts and
seeds:

```bash
python e2_finetune_eval.py                                    # defaults from CONFIG
python e2_finetune_eval.py --dataset droso_small sea_bass --repeats 5
python e2_finetune_eval.py --protocols PA PC --head regression
```

Datasets are declared in `CONFIG['datasets']`. Each entry carries its own
`num_landmarks`, and may override any global setting for itself — `n_shots`
above all, since the labelled pools range from 10 images to 150 and one budget
does not fit all. Precedence is command-line flag > dataset entry > global
default.

Because Stage 3 is deterministic, repeating it over one fine-tuned model would
report a standard deviation of exactly zero. The variance worth reporting comes
from which N images are drawn and from the fine-tuning itself, so repeat *i*
runs Stage 2 with seed *i* and then Stage 3 on its result.

Two metrics are reported, both computed at the original image resolution from
the `.txt` files Stage 3 actually wrote:

- **MRE** — mean Euclidean error in pixels, comparable with published results
  on the same dataset.
- **NME** — the same errors divided by each image's own diagonal, in percent.
  Because the normaliser is per image, NME is comparable *between* datasets;
  MRE in pixels is not, the resolutions here ranging from 632×480 to 1935×2400.

Results go to `results/e2_<dataset>/`: `summary.csv` (mean ± std per protocol,
the table format used in the paper), `runs.csv` (one row per run) and
`results.json`. Completed runs are cached and skipped on the next invocation,
so an interrupted sweep resumes where it stopped; `--force` recomputes only the
runs matching the current selection and leaves the rest untouched.

## Repository Layout

```
deepmorph/
├── README.md                          # This file
├── requirements.txt
├── utils.py                           # Heatmap generation, MRE, sub-pixel decoding
│
├── e1_pretrain_mae.py                 # Stage 1 under protocols PA / PB / PC
├── e2_finetune_eval.py                # Stage 2 + 3 swept over datasets and seeds
├── bench_efficiency.py                # Parameters, FLOPs, latency, memory
├── migrate_protocol_names.py          # One-off protocol relabelling helper
│
├── stage1_pretrain_mae.py             # Single-configuration Stage 1
├── stage2_finetune_landmark.py        # Single-configuration Stage 2 (heatmap head)
├── stage2_finetune_landmark_reg.py    # Stage 2 ablation: coordinate regression head
├── stage2_finetune_landmark_dino.py   # Stage 2 ablation: DINOv2 encoder
├── stage3_predict.py                  # Single-configuration Stage 3
├── stage3_dino_predict.py             # Stage 3 for the DINOv2 variant
│
├── models/
│   ├── mae.py                         # MAE architecture
│   ├── imagenet_mae_loader.py         # Downloads and remaps the public checkpoint
│   ├── landmark_head.py               # Heatmap head + full model
│   ├── regression_head.py             # Coordinate-regression head (ablation)
│   └── dinov2_backbone.py             # DINOv2 encoder behind the MAE interface
└── datasets/
    ├── unlabeled_dataset.py           # Stage 1 (unlabeled)
    └── landmark_dataset.py            # Stage 2 (labeled)
```

## References

- He et al. "Masked Autoencoders Are Scalable Vision Learners" CVPR 2022
- Geldenhuys et al. "Deep learning approaches to landmark detection in tsetse wing images" PLOS Comp Bio 2023
- Nolte et al. "ITHILDIN: Automated landmark and semilandmark annotation for wing geometric morphometrics in Diptera" bioRxiv 2026
- Nolte et al. "Comprehensive Mosquito Wing Image Repository" Sci Data 2025
- Nguyen et al. "A lightweight keypoint matching framework..." Ecological Informatics 2022
