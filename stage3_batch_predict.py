#!/usr/bin/env python3
"""
batch_predict.py - Run stage3_predict_arg.py for multiple n_shots checkpoints.

Expects checkpoints named:  finetune_<dataset>_best_n<n_shots>_size<image_size>.pth
Saves predictions to:       ./predictions/<dataset>_n<n_shots>/

Usage:
    python batch_predict.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
# n_shots_list     = [5] 
# dataset          = 'cepha'       # must match the folder name used in stage2
# num_landmarks    = 19
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [3, 5, 7, 9, 10, 15, 20] #Bactro
# dataset          = 'bactro'       # must match the folder name used in stage2
#num_landmarks    = 12
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [3, 5, 7, 10, 15, 20, 30] #
# dataset          = 'diacha'       # must match the folder name used in stage2
# num_landmarks    = 10
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [5, 7, 10, 15, 20, 30, 150, 300, 500] #Tsetse
# dataset          = 'tsetse'       # must match the folder name used in stage2
# num_landmarks    = 11
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [3, 5, 7, 9, 10, 15, 20] #
# dataset          = 'diacha'       # must match the folder name used in stage2
# num_landmarks    = 10
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [3, 5, 7, 10, 15, 20] 
# dataset          = 'sea_bass'       # must match the folder name used in stage2
# num_landmarks    = 11
# image_size       = 512
# heatmap_size     = 128

# n_shots_list     = [3,5, 7, 10, 15, 20, 25]
# dataset          = 'droso-281'       # must match the folder name used in stage2
# num_landmarks    = 12
# image_size       = 512
# heatmap_size     = 128



# n_shots_list     = [3, 4, 5, 6, 8, 10] 
# dataset          = 'fly'       # must match the folder name used in stage2
# num_landmarks    = 10
# image_size       = 512
# heatmap_size     = 128

n_shots_list     = [3,5, 7, 10, 15, 20, 25]
dataset          = 'droso_big'       # must match the folder name used in stage2
num_landmarks    = 15
image_size       = 512
heatmap_size     = 128

input_dir        = f'./data_predict/{dataset}'
checkpoints_dir  = './checkpoints'
output_base_dir  = './predictions'

model_size       = 'base'
embed_dim        = 768
# ──────────────────────────────────────────────────────────────────────────────


def parse_mre(output: str):
    """Extract overall MRE from stage3 stdout. Returns float or None."""
    for line in output.splitlines():
        if line.strip().startswith('Overall :'):
            try:
                return float(line.split(':')[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    return None


IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']


def count_images(directory):
    d = Path(directory)
    return sum(len(list(d.glob(ext))) for ext in IMG_EXTS)


def run_predict(n_shots):
    ckpt_name = f"finetune_{dataset}_best_n{n_shots}_size{image_size}.pth"
    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
    output_dir = os.path.join(output_base_dir, f"{dataset}_n{n_shots}")

    if not os.path.exists(ckpt_path):
        print(f"[SKIP] Checkpoint not found: {ckpt_path}")
        return None, None

    cmd = [
        sys.executable, 'stage3_predict_arg.py',
        '--checkpoint',    ckpt_path,
        '--input_dir',     input_dir,
        '--output_dir',    output_dir,
        '--num_landmarks', str(num_landmarks),
        '--image_size',    str(image_size),
        '--heatmap_size',  str(heatmap_size),
        '--model_size',    model_size,
        '--embed_dim',     str(embed_dim),
        '--max_test', str(0)
    ]

    print(f"\n{'='*60}")
    print(f"n_shots={n_shots}  |  checkpoint: {ckpt_name}")
    print(f"output  : {output_dir}")
    print(f"{'='*60}")

    n_images = count_images(input_dir)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')

    if result.returncode != 0:
        print(f"[ERROR] n_shots={n_shots} failed (return code {result.returncode})")
        return None, None

    mre = parse_mre(result.stdout)
    avg_ms = (elapsed / n_images * 1000) if n_images > 0 else 0.0
    print(f"[DONE]  n_shots={n_shots}  "
          + (f"MRE={mre:.2f} px  " if mre is not None else "MRE=n/a  ")
          + f"avg {avg_ms:.1f} ms/img  ({n_images} images)")
    return mre, avg_ms


if __name__ == '__main__':
    results = {}   # n_shots -> (mre, avg_ms)

    for n_shots in n_shots_list:
        mre, avg_ms = run_predict(n_shots)
        if mre is not None or avg_ms is not None:
            results[n_shots] = (mre, avg_ms)

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY  —  dataset: {dataset}  |  image_size: {image_size}")
    print(f"{'='*60}")
    print(f"{'n_shots':>10}  {'MRE (px)':>12}  {'ms/img':>10}")
    print(f"{'-'*36}")
    for n_shots in n_shots_list:
        if n_shots in results:
            mre, avg_ms = results[n_shots]
            mre_str = f"{mre:.2f}" if mre is not None else "n/a"
            ms_str  = f"{avg_ms:.1f}" if avg_ms is not None else "n/a"
            print(f"{n_shots:>10}  {mre_str:>12}  {ms_str:>10}")
        else:
            print(f"{n_shots:>10}  {'skipped/error':>12}  {'':>10}")
    print(f"{'='*60}")
