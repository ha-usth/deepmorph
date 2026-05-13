"""
stage1_pretrain_mae.py - STAGE 1: MAE Pretraining

Supports two modes:
    1. 'from_scratch': Train MAE from scratch on insect wing images
    2. 'continue_from_imagenet': Start from MAE-ImageNet checkpoint (Facebook AI),
       then continue pretraining on insect wing images (RECOMMENDED)
       - Leverages ~1.4M already-learned ImageNet images
       - Converges much faster (50-100 epochs instead of 200-400)
       - Usually gives better results when domain dataset < 100K images

Output: Encoder weights saved to checkpoint, used in Stage 2.

Usage:
    python stage1_pretrain_mae.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.imagenet_mae_loader import load_imagenet_mae_weights
from datasets.unlabeled_dataset import UnlabeledWingDataset
from utils import set_seed


# ============== CONFIG ==============
CONFIG = {
    # =========================================================================
    # PRETRAINING MODE (choose one)
    # =========================================================================
    'pretrain_mode': 'continue_from_imagenet',  # or 'from_scratch'

    # Only used when mode = 'continue_from_imagenet'
    'imagenet_cache_dir': None,  # None = use ~/.cache/mae_imagenet

    # =========================================================================
    # DATA
    # =========================================================================
    'data_dirs': [
        # './data_pretrain/bactro',
        # './data_pretrain/diacha',
        # './data_pretrain/droso_big',
        # './data_pretrain/droso-281',
        # './data_pretrain/droso_small',
        # './data_pretrain/fly',
        # './data_pretrain/hindwing',
        './data_pretrain/sea_bass',
        # './data_pretrain/tsetse',
        # './data_pretrain/cepha',
        # './data_pretrain/cepha400',

    ],
    'image_size': 224,           # must be 224 if continue_from_imagenet
                                  # (or the model will interpolate pos_embed)

    # =========================================================================
    # TRAINING
    # =========================================================================
    'batch_size': 32,
    'num_workers': 4,
    'lr': 1.5e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'mask_ratio': 0.75,
    'model_size': 'base',         # must be 'base'/'large'/'huge' if continuing
                                   # ('small' only works with from_scratch)

    # Epoch count differs between modes:
    'epochs_from_scratch': 200,
    'epochs_continue': 50,        # fewer epochs needed since already pretrained

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================
    'save_dir': './checkpoints',
    'save_every': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def adjust_learning_rate(optimizer, epoch, config, total_epochs):
    """Cosine LR schedule with warmup."""
    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        progress = (epoch - config['warmup_epochs']) / (total_epochs - config['warmup_epochs'])
        lr = config['lr'] * 0.5 * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, imgs in enumerate(pbar):
        imgs = imgs.to(device)
        loss, _, _ = model(imgs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def build_model(config):
    """
    Build MAE model from config.

    If mode = 'continue_from_imagenet': load weights from Facebook checkpoint.
    If mode = 'from_scratch': use random init.
    """
    # Validate config
    if config['pretrain_mode'] == 'continue_from_imagenet':
        if config['model_size'] not in ['base', 'large', 'huge']:
            raise ValueError(
                f"continue_from_imagenet only supports model_size='base'/'large'/'huge', "
                f"got '{config['model_size']}'. "
                f"To use 'small', switch to 'from_scratch' mode."
            )

    # Initialize architecture
    if config['model_size'] == 'base':
        model = mae_vit_base(
            img_size=config['image_size'],
            mask_ratio=config['mask_ratio']
        )
    elif config['model_size'] == 'small':
        model = mae_vit_small(
            img_size=config['image_size'],
            mask_ratio=config['mask_ratio']
        )
    else:
        raise NotImplementedError(
            f"model_size='{config['model_size']}' not implemented in models/mae.py. "
            f"Only 'base' and 'small' are available. "
            f"To use ImageNet large/huge, add mae_vit_large/huge."
        )

    # Load ImageNet weights if needed
    if config['pretrain_mode'] == 'continue_from_imagenet':
        print("\n=== Loading MAE-ImageNet weights ===")
        load_imagenet_mae_weights(
            model,
            model_size=config['model_size'],
            cache_dir=config['imagenet_cache_dir'],
            strict=False,
            verbose=True,
        )
    elif config['pretrain_mode'] == 'from_scratch':
        print("\n=== Training from scratch (random init) ===")
    else:
        raise ValueError(
            f"pretrain_mode must be 'from_scratch' or 'continue_from_imagenet', "
            f"got '{config['pretrain_mode']}'"
        )

    return model


def main():
    set_seed(42)
    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)

    # Choose epoch count based on mode
    if config['pretrain_mode'] == 'continue_from_imagenet':
        total_epochs = config['epochs_continue']
    else:
        total_epochs = config['epochs_from_scratch']

    print(f"=== Configuration ===")
    print(f"  Mode: {config['pretrain_mode']}")
    print(f"  Model: ViT-{config['model_size'].capitalize()}")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Image size: {config['image_size']}")
    print(f"  Batch size: {config['batch_size']}")

    # 1. Load dataset
    print(f"\n=== Step 1: Load dataset ===")
    dataset = UnlabeledWingDataset(
        root_dirs=config['data_dirs'],
        image_size=config['image_size']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    # 2. Build model
    print(f"\n=== Step 2: Initialize model ===")
    model = build_model(config)
    model = model.to(config['device'])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    # 3. Optimizer
    effective_lr = config['lr'] * config['batch_size'] / 256
    optimizer = AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95),
    )

    # 4. Training loop
    print(f"\n=== Step 3: Start pretraining ===")
    mode_tag = 'continue' if config['pretrain_mode'] == 'continue_from_imagenet' else 'scratch'

    for epoch in range(total_epochs):
        lr = adjust_learning_rate(optimizer, epoch, config, total_epochs)
        avg_loss = train_one_epoch(model, dataloader, optimizer,
                                    config['device'], epoch)

        print(f"Epoch {epoch}: loss={avg_loss:.4f}, lr={lr:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % config['save_every'] == 0 or epoch == total_epochs - 1:
            ckpt_path = os.path.join(
                config['save_dir'],
                f'mae_pretrain_{mode_tag}_epoch{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(
        config['save_dir'],
        f'mae_pretrain_{mode_tag}_final.pth'
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, final_path)
    print(f"\n✓ Pretraining complete ({mode_tag}). "
          f"Encoder weights at: {final_path}")


if __name__ == '__main__':
    main()
