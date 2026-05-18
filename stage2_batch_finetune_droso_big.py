#!/usr/bin/env python3
"""
batch_finetune.py - Run stage2_finetune_arg.py multiple times with different n_shots values

Usage:
    python batch_finetune.py
"""

import subprocess
import sys
import os

# List of n_shots values to try
n_shots_list = [3,5, 7, 10, 15, 20, 25]

# Common arguments (adjust as needed)
mae_checkpoint = './checkpoints/mae_pretrain_continue_final_all.pth'
data_dir = './data_finetune/droso_big'
num_landmarks = 15
finetune_image_size = 512
heatmap_size = 128
save_dir = './checkpoints'

batch_size = 2
epochs = 300
lr_head = 5e-4
lr_encoder = 5e-6
weight_decay = 0.05
freeze_epochs = 30
eval_every = 10
use_tta = True
seed = 42

def run_finetune(n_shots):
    cmd = [
        sys.executable, 'stage2_finetune_arg.py',
        '--mae_checkpoint', mae_checkpoint,
        '--data_dir', data_dir,
        '--num_landmarks', str(num_landmarks),
        '--n_shots', str(n_shots),
        '--finetune_image_size', str(finetune_image_size),
        '--heatmap_size', str(heatmap_size),
        '--save_dir', save_dir,
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr_head', str(lr_head),
        '--lr_encoder', str(lr_encoder),
        '--weight_decay', str(weight_decay),
        '--freeze_epochs', str(freeze_epochs),
        '--eval_every', str(eval_every),
        '--seed', str(seed)
    ]
    
    if use_tta:
        cmd.append('--use_tta')
    
    print(f"Running fine-tuning with n_shots={n_shots}")
    print("Command:", ' '.join(cmd))
    
    # Run the command
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode != 0:
        print(f"Error running with n_shots={n_shots}, return code: {result.returncode}")
    else:
        print(f"Completed fine-tuning with n_shots={n_shots}")

if __name__ == '__main__':
    for n_shots in n_shots_list:
        run_finetune(n_shots)
        print("-" * 50)