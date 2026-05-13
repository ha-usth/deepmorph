"""
stage2_finetune_landmark.py - Stage 2: Few-shot Fine-tuning
                              (Supports HIGH-RESOLUTION FINETUNING)

Supports a finetune image_size different from the pretrain image_size.

Example:
    Stage 1 pretrain at 224 (fast)
    Stage 2 finetune at 512 (much lower MRE)

    -> When loading the encoder from stage 1, pos_embed is automatically
       interpolated from 14x14 -> 32x32 patches.

Goal: Fine-tune the MAE encoder + landmark head with only a few labeled
      images (3-15).

Usage:
    python stage2_finetune_landmark.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.landmark_head import WingLandmarkModel
from models.imagenet_mae_loader import _interpolate_pos_embed
from datasets.landmark_dataset import LandmarkDataset, select_few_shot_subset
from utils import set_seed, heatmaps_to_coords, compute_MRE


# ============== RECOMMENDED CONFIG FOR 32GB GPU ==============
CONFIG = {
    # Path to MAE checkpoint from Stage 1
    'mae_checkpoint': './checkpoints/mae_pretrain_continue_final_all.pth',

    # ==== IMAGE SIZE: pretrain 224, finetune 512 ====
    'pretrain_image_size': 224,    # image_size used in stage 1
    'finetune_image_size': 512,    # image_size for stage 2 (higher = lower MRE)

    # Heatmap size = finetune_image_size / 4
    'heatmap_size': 128,            # 384 / 4 = 96
    'sigma': 3,                  # scale with heatmap_size (sigma 2.0 for 96, 3.0 for 128)

    # Target dataset
    'target_data_dir': './data_pretrain/sea_bass',

    # Few-shot setup
    'n_shots': 100,              # number of labeled images for finetuning (3-15 recommended)
    'num_landmarks': 11,

    # Model config
    'patch_size': 16,
    'embed_dim': 768,              # ViT-Base
    'model_size': 'base',

    # Training
    'batch_size': 2,               # 8 fits 512x512 on 32GB VRAM
    'gradient_accumulation_steps': 4,  # effective batch = 8 * 4 = 32
    'lr_head': 5e-4,
    'lr_encoder': 5e-6,            # lower LR since encoder is already well-pretrained
    'weight_decay': 0.05,
    'epochs': 300,                 # more epochs needed for high-res training
    'freeze_encoder_epochs': 30,   # freeze for the first 30 epochs

    # Eval
    'eval_every': 10,

    'save_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,

    # Test-time augmentation (free 1-2 pixel improvement)
    'use_tta': True,
}


def load_pretrained_encoder_with_resize(checkpoint_path, target_image_size,
                                         model_size='base', verbose=True):
    """
    Load MAE encoder from stage 1, interpolating pos_embed if target_image_size
    differs from the pretrain image_size.

    Args:
        checkpoint_path: .pth file from stage 1
        target_image_size: desired image_size for finetuning (e.g. 512)
        model_size: 'base' or 'small'
        verbose: print details
    Returns:
        MAE model with loaded weights and resized pos_embed
    """
    if verbose:
        print(f"\n=== Loading encoder from stage 1 ===")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Target image_size: {target_image_size}")

    # Build model at target_image_size (NOT the pretrain size)
    if model_size == 'base':
        model = mae_vit_base(img_size=target_image_size)
    else:
        model = mae_vit_small(img_size=target_image_size)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    # Infer pretrain size from config if available
    if 'config' in ckpt and 'image_size' in ckpt['config']:
        pretrain_size = ckpt['config']['image_size']
    else:
        # Infer from pos_embed shape
        old_pos_embed = state_dict['pos_embed']  # (1, num_patches+1, embed_dim)
        old_num_patches = old_pos_embed.shape[1] - 1
        old_grid = int(math.sqrt(old_num_patches))
        pretrain_size = old_grid * 16  # patch_size = 16

    if verbose:
        print(f"  Pretrain image_size: {pretrain_size}")

    # Interpolate pos_embed if sizes differ
    if pretrain_size != target_image_size:
        if verbose:
            print(f"  Interpolating pos_embed: {pretrain_size} -> {target_image_size}")

        target_pos_embed_shape = model.pos_embed.shape
        state_dict['pos_embed'] = _interpolate_pos_embed(
            state_dict['pos_embed'], target_pos_embed_shape
        )

        if 'decoder_pos_embed' in state_dict:
            target_dec_pos_shape = model.decoder_pos_embed.shape
            state_dict['decoder_pos_embed'] = _interpolate_pos_embed(
                state_dict['decoder_pos_embed'], target_dec_pos_shape
            )

    # Load weights
    msg = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"  Loaded encoder. Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")

    return model


def heatmap_loss(pred_heatmaps, gt_heatmaps):
    """MSE loss between predicted and ground-truth heatmaps."""
    return F.mse_loss(pred_heatmaps, gt_heatmaps)


def evaluate(model, dataloader, device, image_size, heatmap_size, use_tta=False):
    """
    Compute MRE on the test set.

    Args:
        use_tta: if True, apply test-time augmentation (horizontal flip)
                 for ~1-2 pixel free improvement.
    """
    model.eval()
    all_pred_coords = []
    all_gt_coords = []

    scale = image_size / heatmap_size

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            gt_coords = batch['coords']        # (B, K, 2) at image_size scale
            orig_sizes = batch['orig_size']    # (B, 2) = [orig_w, orig_h]

            heatmaps = model(imgs)

            # TTA: horizontal flip
            if use_tta:
                imgs_flipped = torch.flip(imgs, dims=[-1])
                heatmaps_flipped = model(imgs_flipped)
                heatmaps_flipped = torch.flip(heatmaps_flipped, dims=[-1])
                heatmaps = (heatmaps + heatmaps_flipped) / 2

            pred_coords = heatmaps_to_coords(heatmaps)
            pred_coords = pred_coords * scale  # → image_size scale

            # Rescale to original image resolution  (B, 1, 2) broadcast over K
            scale_to_orig = (orig_sizes / image_size).unsqueeze(1).to(pred_coords.device)  # (B, 1, 2)
            pred_coords = pred_coords * scale_to_orig
            gt_coords   = gt_coords.to(pred_coords.device) * scale_to_orig

            all_pred_coords.append(pred_coords.cpu())
            all_gt_coords.append(gt_coords.cpu())

    if not all_pred_coords:
        return None, float('inf')

    all_pred = torch.cat(all_pred_coords, dim=0)
    all_gt = torch.cat(all_gt_coords, dim=0)

    MRE_per_lm, MRE_overall = compute_MRE(all_pred, all_gt)
    return MRE_per_lm, MRE_overall


def train_one_epoch(model, dataloader, optimizer, device, accum_steps=1):
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        imgs = batch['image'].to(device)
        gt_heatmaps = batch['heatmaps'].to(device)

        pred_heatmaps = model(imgs)
        loss = heatmap_loss(pred_heatmaps, gt_heatmaps) / accum_steps
        loss.backward()

        # Update every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(dataloader)


def main():
    config = CONFIG
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    print(f"=== Configuration ===")
    print(f"  Pretrain image_size: {config['pretrain_image_size']}")
    print(f"  Finetune image_size: {config['finetune_image_size']}")
    print(f"  Heatmap size: {config['heatmap_size']}")
    print(f"  Batch size: {config['batch_size']} "
          f"(effective: {config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"  N-shot: {config['n_shots']}")
    print(f"  Use TTA: {config['use_tta']}")

    # ===== 1. Few-shot train/test split =====
    train_list, test_list = select_few_shot_subset(
        config['target_data_dir'],
        n_shots=config['n_shots'],
        seed=config['seed']
    )
    print(f"\nTrain: {len(train_list)} images | Test: {len(test_list)} images")

    # ===== 2. Datasets at finetune resolution =====
    train_dataset = LandmarkDataset(
        data_dir=config['target_data_dir'],
        image_list=train_list,
        num_landmarks=config['num_landmarks'],
        image_size=config['finetune_image_size'],
        heatmap_size=config['heatmap_size'],
        sigma=config['sigma'],
        augment=True,
    )
    test_dataset = LandmarkDataset(
        data_dir=config['target_data_dir'],
        image_list=test_list,
        num_landmarks=config['num_landmarks'],
        image_size=config['finetune_image_size'],
        heatmap_size=config['heatmap_size'],
        sigma=config['sigma'],
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)

    # ===== 3. Load encoder with pos_embed interpolation =====
    mae = load_pretrained_encoder_with_resize(
        config['mae_checkpoint'],
        target_image_size=config['finetune_image_size'],
        model_size=config['model_size'],
        verbose=True,
    )

    # ===== 4. Build full model (encoder + landmark head) =====
    model = WingLandmarkModel(
        mae_encoder=mae,
        num_landmarks=config['num_landmarks'],
        embed_dim=config['embed_dim'],
        patch_size=config['patch_size'],
        img_size=config['finetune_image_size'],
        heatmap_size=config['heatmap_size'],
        freeze_encoder=True,
    )
    model = model.to(config['device'])

    if torch.cuda.is_available():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\nModel: {n_params:.1f}M params")
        torch.cuda.reset_peak_memory_stats()

    # ===== 5. Optimizer (head only first) =====
    optimizer = AdamW(model.head.parameters(),
                      lr=config['lr_head'],
                      weight_decay=config['weight_decay'])

    # ===== 6. Training loop =====
    best_MRE = float('inf')

    for epoch in range(config['epochs']):
        # Unfreeze encoder after N epochs
        if epoch == config['freeze_encoder_epochs']:
            print("\n=== Unfreezing encoder ===")
            for p in model.encoder.parameters():
                p.requires_grad = True

            optimizer = AdamW([
                {'params': model.encoder.parameters(), 'lr': config['lr_encoder']},
                {'params': model.head.parameters(), 'lr': config['lr_head']},
            ], weight_decay=config['weight_decay'])

        train_loss = train_one_epoch(
            model, train_loader, optimizer, config['device'],
            accum_steps=config['gradient_accumulation_steps']
        )

        # Periodic eval
        if (epoch + 1) % config['eval_every'] == 0:
            MRE_per_lm, MRE_overall = evaluate(
                model, test_loader, config['device'],
                config['finetune_image_size'], config['heatmap_size'],
                use_tta=config['use_tta'],
            )

            if MRE_per_lm is None:
                print(f"Epoch {epoch}: loss={train_loss:.4f} (test set is empty)")
            else:
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, "
                          f"MRE={MRE_overall:.2f}, peak_mem={peak_mem:.1f}GB")
                else:
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, MRE={MRE_overall:.2f}")

                if MRE_overall < best_MRE:
                    best_MRE = MRE_overall
                    ckpt_path = os.path.join(
                        config['save_dir'],
                        f"finetune_best_n{config['n_shots']}_size{config['finetune_image_size']}.pth"
                    )
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'MRE': MRE_overall,
                        'MRE_per_lm': MRE_per_lm,
                        'config': config,
                    }, ckpt_path)
                    print(f"  Saved best: MRE={MRE_overall:.2f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")

    # Final report
    print(f"\n=== FINAL RESULTS ===")
    print(f"  Finetune image_size: {config['finetune_image_size']}")
    print(f"  N-shot: {config['n_shots']}")
    print(f"  Best MRE: {best_MRE:.2f} pixels (at original image resolution)")


if __name__ == '__main__':
    main()
