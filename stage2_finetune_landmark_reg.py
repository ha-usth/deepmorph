"""
stage2_finetune_landmark_reg.py - Stage 2 (Ablation): Direct Regression

Replaces the heatmap head with a regression head:
    MAE encoder features -> MLP -> (K, 2) coordinates directly

Compare with stage2_finetune_landmark.py (heatmap) for ablation study.

Usage:
    python stage2_finetune_landmark_reg.py
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
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.imagenet_mae_loader import _interpolate_pos_embed
from models.regression_head import RegressionLandmarkModel
from datasets.landmark_dataset import LandmarkDataset, select_few_shot_subset
from utils import set_seed, compute_MRE


# ============================================================
CONFIG = {
    'mae_checkpoint': './checkpoints/mae_pretrain_continue_final_all.pth',

    'pretrain_image_size': 224,
    'finetune_image_size': 512,

    'target_data_dir': './data_finetune/sea_bass',

    'n_shots': 100,
    'num_landmarks': 11,

    'patch_size': 16,
    'embed_dim': 768,
    'model_size': 'base',

    'batch_size': 2,
    'gradient_accumulation_steps': 4,
    'lr_head': 5e-4,
    'lr_encoder': 5e-6,
    'weight_decay': 0.05,
    'epochs': 300,
    'freeze_encoder_epochs': 30,

    'eval_every': 10,

    'save_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}


# ============================================================
# Loss
# ============================================================

def regression_loss(pred_coords_norm, gt_coords, image_size):
    """
    Smooth L1 loss on coordinates normalized to [0, 1].

    Args:
        pred_coords_norm : (B, K, 2) model output, sigmoid already applied
        gt_coords        : (B, K, 2) GT coordinates at image_size pixels
        image_size       : scalar used to normalize GT
    """
    gt_norm = gt_coords / image_size          # (B, K, 2) in [0, 1]
    return F.smooth_l1_loss(pred_coords_norm, gt_norm)


# ============================================================
# Helpers
# ============================================================

def load_pretrained_encoder_with_resize(checkpoint_path, target_image_size,
                                        model_size='base', verbose=True):
    if verbose:
        print(f"\n=== Loading encoder from stage 1 ===")
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"  Target size: {target_image_size}")

    if model_size == 'base':
        model = mae_vit_base(img_size=target_image_size)
    else:
        model = mae_vit_small(img_size=target_image_size)

    ckpt       = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    if 'config' in ckpt and 'image_size' in ckpt['config']:
        pretrain_size = ckpt['config']['image_size']
    else:
        old_num_patches = state_dict['pos_embed'].shape[1] - 1
        pretrain_size   = int(math.sqrt(old_num_patches)) * 16

    if verbose:
        print(f"  Pretrain size: {pretrain_size}")

    if pretrain_size != target_image_size:
        if verbose:
            print(f"  Interpolating pos_embed: {pretrain_size} -> {target_image_size}")
        state_dict['pos_embed'] = _interpolate_pos_embed(
            state_dict['pos_embed'], model.pos_embed.shape)
        if 'decoder_pos_embed' in state_dict:
            state_dict['decoder_pos_embed'] = _interpolate_pos_embed(
                state_dict['decoder_pos_embed'], model.decoder_pos_embed.shape)

    msg = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"  Loaded. Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")
    return model


def evaluate(model, dataloader, device, image_size):
    """MRE at image_size scale (for fair comparison with heatmap variant)."""
    model.eval()
    all_pred, all_gt = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs      = batch['image'].to(device)
            gt_coords = batch['coords']              # (B, K, 2) at image_size

            pred_norm = model(imgs)                  # (B, K, 2) in [0, 1]
            pred_px   = pred_norm.cpu() * image_size # back to pixel scale

            all_pred.append(pred_px)
            all_gt.append(gt_coords)

    if not all_pred:
        return None, float('inf')

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)
    return compute_MRE(all_pred, all_gt)


def train_one_epoch(model, dataloader, optimizer, device, image_size, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        imgs      = batch['image'].to(device)
        gt_coords = batch['coords'].to(device)  # (B, K, 2) at image_size

        pred_norm = model(imgs)
        loss = regression_loss(pred_norm, gt_coords, image_size) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(dataloader)


# ============================================================
# Main
# ============================================================

def main():
    config = CONFIG
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    print("=== Ablation: Direct Regression (no heatmap) ===")
    print(f"  Finetune image_size : {config['finetune_image_size']}")
    print(f"  N-shot              : {config['n_shots']}")
    print(f"  Num landmarks       : {config['num_landmarks']}")

    # 1. Data split
    train_list, test_list = select_few_shot_subset(
        config['target_data_dir'],
        n_shots=config['n_shots'],
        seed=config['seed'],
    )
    print(f"\nTrain: {len(train_list)} images | Test: {len(test_list)} images")

    # 2. Datasets
    # heatmap_size / sigma are not used for regression, but LandmarkDataset
    # still needs valid values to produce batch['coords'] at the correct scale.
    train_dataset = LandmarkDataset(
        data_dir=config['target_data_dir'],
        image_list=train_list,
        num_landmarks=config['num_landmarks'],
        image_size=config['finetune_image_size'],
        heatmap_size=config['finetune_image_size'] // 4,
        sigma=2.0,
        augment=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=0)

    test_loader = None
    if test_list:
        test_dataset = LandmarkDataset(
            data_dir=config['target_data_dir'],
            image_list=test_list,
            num_landmarks=config['num_landmarks'],
            image_size=config['finetune_image_size'],
            heatmap_size=config['finetune_image_size'] // 4,
            sigma=2.0,
            augment=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=0)

    # 3. Model
    mae = load_pretrained_encoder_with_resize(
        config['mae_checkpoint'],
        target_image_size=config['finetune_image_size'],
        model_size=config['model_size'],
    )
    model = RegressionLandmarkModel(
        mae_encoder=mae,
        num_landmarks=config['num_landmarks'],
        embed_dim=config['embed_dim'],
        freeze_encoder=True,
    ).to(config['device'])

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel: {n_params:.1f}M params total")
    n_head = sum(p.numel() for p in model.head.parameters()) / 1e6
    print(f"       {n_head:.2f}M params in regression head")

    # 4. Optimizer
    optimizer = AdamW(model.head.parameters(),
                      lr=config['lr_head'],
                      weight_decay=config['weight_decay'])

    # 5. Training loop
    best_MRE = float('inf')

    for epoch in range(config['epochs']):
        if epoch == config['freeze_encoder_epochs']:
            print("\n=== Unfreezing encoder ===")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = AdamW([
                {'params': model.encoder.parameters(), 'lr': config['lr_encoder']},
                {'params': model.head.parameters(),    'lr': config['lr_head']},
            ], weight_decay=config['weight_decay'])

        train_loss = train_one_epoch(
            model, train_loader, optimizer, config['device'],
            image_size=config['finetune_image_size'],
            accum_steps=config['gradient_accumulation_steps'],
        )

        if (epoch + 1) % config['eval_every'] == 0:
            if test_loader is None:
                print(f"Epoch {epoch}: loss={train_loss:.4f} (no test set)")
                continue

            MRE_per_lm, MRE_overall = evaluate(
                model, test_loader, config['device'],
                image_size=config['finetune_image_size'],
            )

            if MRE_per_lm is None:
                print(f"Epoch {epoch}: loss={train_loss:.4f} (test set is empty)")
            else:
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, "
                          f"MRE={MRE_overall:.2f}, peak_mem={peak_mem:.1f}GB")
                else:
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, "
                          f"MRE={MRE_overall:.2f}")

                if MRE_overall < best_MRE:
                    best_MRE = MRE_overall
                    ckpt_path = os.path.join(
                        config['save_dir'],
                        f"finetune_reg_best_n{config['n_shots']}"
                        f"_size{config['finetune_image_size']}.pth"
                    )
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'MRE': MRE_overall,
                        'MRE_per_lm': MRE_per_lm,
                        'config': config,
                        'head_type': 'regression',
                    }, ckpt_path)
                    print(f"  Saved best: MRE={MRE_overall:.2f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")

    # 6. Final report
    print(f"\n=== FINAL RESULTS (Regression) ===")
    print(f"  N-shot         : {config['n_shots']}")
    print(f"  Image size     : {config['finetune_image_size']}")
    print(f"  Best MRE      : {best_MRE:.2f} px  (at {config['finetune_image_size']}px scale)")


if __name__ == '__main__':
    main()
