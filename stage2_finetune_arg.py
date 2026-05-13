"""
stage2_finetune_arg.py - Stage 2: Few-shot Fine-tuning with CLI arguments

Usage:
    python stage2_finetune_arg.py \
        --mae_checkpoint ./checkpoints/mae_pretrain_continue_final_global.pth \
        --data_dir       ./data_finetune/droso_small \
        --num_landmarks  15 \
        --n_shots        5
    python stage2_finetune_arg.py --mae_checkpoint ./checkpoints/mae_pretrain_continue_final_cepha.pth --data_dir .\data_finetune\cepha400 --num_landmarks 19 --n_shots 30 
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.landmark_head import WingLandmarkModel
from models.imagenet_mae_loader import _interpolate_pos_embed
from datasets.landmark_dataset import LandmarkDataset, select_few_shot_subset
from utils import set_seed, heatmaps_to_coords, compute_MRE


def parse_args():
    parser = argparse.ArgumentParser(description='DeepMorph Stage 2: Few-shot Fine-tuning')

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument('--mae_checkpoint', type=str, default='./checkpoints/mae_pretrain_continue_final_all.pth',
                        help='Path to MAE checkpoint from Stage 1 (.pth)')
    parser.add_argument('--data_dir', type=str, default='./data_finetune/tsetse',
                        help='Directory containing labeled images + .txt landmark files')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save best checkpoint (default: ./checkpoints)')

    # ── Dataset ────────────────────────────────────────────────────────────────
    parser.add_argument('--num_landmarks', type=int, default=11,
                        help='Number of landmark classes in the target dataset (default: 11)')
    parser.add_argument('--n_shots', type=int, default=100,
                        help='Number of labeled images for fine-tuning (default: 100)')

    # ── Image / heatmap sizes ──────────────────────────────────────────────────
    parser.add_argument('--finetune_image_size', type=int, default=512,
                        help='Input image size for fine-tuning (default: 512)')
    parser.add_argument('--heatmap_size', type=int, default=128,
                        help='Output heatmap size (default: 128, = image_size/4)')
    parser.add_argument('--sigma', type=float, default=3,
                        help='Gaussian heatmap sigma in heatmap pixels (default: 3)')

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'small'],
                        help='MAE model size (default: base)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='ViT patch size (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='ViT embedding dimension (default: 768 for base)')

    # ── Training ───────────────────────────────────────────────────────────────
    parser.add_argument('--epochs', type=int, default=300,
                        help='Total training epochs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps (default: 4, effective batch=32)')
    parser.add_argument('--lr_head', type=float, default=5e-4,
                        help='Learning rate for the landmark head (default: 5e-4)')
    parser.add_argument('--lr_encoder', type=float, default=5e-6,
                        help='Learning rate for the encoder after unfreezing (default: 5e-6)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='AdamW weight decay (default: 0.05)')
    parser.add_argument('--freeze_epochs', type=int, default=30,
                        help='Epochs to keep encoder frozen before end-to-end training (default: 30)')

    # ── Eval / misc ────────────────────────────────────────────────────────────
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate on test set every N epochs (default: 10)')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use test-time augmentation (horizontal flip, default: on)')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false',
                        help='Disable test-time augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: auto)')

    return parser.parse_args()


def load_pretrained_encoder_with_resize(checkpoint_path, target_image_size,
                                         model_size='base', verbose=True):
    if verbose:
        print(f"\n=== Loading encoder from Stage 1 ===")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Target image_size: {target_image_size}")

    if model_size == 'base':
        model = mae_vit_base(img_size=target_image_size)
    else:
        model = mae_vit_small(img_size=target_image_size)

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    if 'config' in ckpt and 'image_size' in ckpt['config']:
        pretrain_size = ckpt['config']['image_size']
    else:
        old_pos_embed = state_dict['pos_embed']
        old_num_patches = old_pos_embed.shape[1] - 1
        old_grid = int(math.sqrt(old_num_patches))
        pretrain_size = old_grid * 16

    if verbose:
        print(f"  Pretrain image_size: {pretrain_size}")

    if pretrain_size != target_image_size:
        if verbose:
            print(f"  Interpolating pos_embed: {pretrain_size} -> {target_image_size}")
        state_dict['pos_embed'] = _interpolate_pos_embed(
            state_dict['pos_embed'], model.pos_embed.shape
        )
        if 'decoder_pos_embed' in state_dict:
            state_dict['decoder_pos_embed'] = _interpolate_pos_embed(
                state_dict['decoder_pos_embed'], model.decoder_pos_embed.shape
            )

    msg = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"  Loaded. Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")
    return model


def heatmap_loss(pred_heatmaps, gt_heatmaps):
    return F.mse_loss(pred_heatmaps, gt_heatmaps)


def evaluate(model, dataloader, device, image_size, heatmap_size, use_tta=False):
    model.eval()
    all_pred_coords = []
    all_gt_coords = []
    scale = image_size / heatmap_size

    with torch.no_grad():
        for batch in dataloader:
            imgs      = batch['image'].to(device)
            gt_coords = batch['coords']
            orig_sizes = batch['orig_size']

            heatmaps = model(imgs)

            if use_tta:
                imgs_flipped    = torch.flip(imgs, dims=[-1])
                heatmaps_flipped = model(imgs_flipped)
                heatmaps_flipped = torch.flip(heatmaps_flipped, dims=[-1])
                heatmaps = (heatmaps + heatmaps_flipped) / 2

            pred_coords = heatmaps_to_coords(heatmaps) * scale

            scale_to_orig = (orig_sizes / image_size).unsqueeze(1).to(pred_coords.device)
            pred_coords = pred_coords * scale_to_orig
            gt_coords   = gt_coords.to(pred_coords.device) * scale_to_orig

            all_pred_coords.append(pred_coords.cpu())
            all_gt_coords.append(gt_coords.cpu())

    if not all_pred_coords:
        return None, float('inf')

    MRE_per_lm, MRE_overall = compute_MRE(
        torch.cat(all_pred_coords, dim=0),
        torch.cat(all_gt_coords, dim=0),
    )
    return MRE_per_lm, MRE_overall


def train_one_epoch(model, dataloader, optimizer, device, accum_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        imgs        = batch['image'].to(device)
        gt_heatmaps = batch['heatmaps'].to(device)

        loss = heatmap_loss(model(imgs), gt_heatmaps) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

    return total_loss / len(dataloader)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"=== Configuration ===")
    print(f"  MAE checkpoint   : {args.mae_checkpoint}")
    print(f"  Data dir         : {args.data_dir}")
    print(f"  Finetune size    : {args.finetune_image_size}")
    print(f"  Heatmap size     : {args.heatmap_size}")
    print(f"  N-shot           : {args.n_shots}")
    print(f"  Num landmarks    : {args.num_landmarks}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Batch size       : {args.batch_size} "
          f"(effective: {args.batch_size * args.grad_accum})")
    print(f"  Use TTA          : {args.use_tta}")

    # ===== 1. Train / test split =====
    train_list, test_list = select_few_shot_subset(
        args.data_dir, n_shots=args.n_shots, seed=args.seed
    )
    print(f"\nTrain: {len(train_list)} images | Test: {len(test_list)} images")

    # ===== 2. Datasets =====
    train_dataset = LandmarkDataset(
        data_dir=args.data_dir,
        image_list=train_list,
        num_landmarks=args.num_landmarks,
        image_size=args.finetune_image_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        augment=True,
    )
    test_dataset = LandmarkDataset(
        data_dir=args.data_dir,
        image_list=test_list,
        num_landmarks=args.num_landmarks,
        image_size=args.finetune_image_size,
        heatmap_size=args.heatmap_size,
        sigma=args.sigma,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ===== 3. Load encoder =====
    mae = load_pretrained_encoder_with_resize(
        args.mae_checkpoint,
        target_image_size=args.finetune_image_size,
        model_size=args.model_size,
    )

    # ===== 4. Build model =====
    model = WingLandmarkModel(
        mae_encoder=mae,
        num_landmarks=args.num_landmarks,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        img_size=args.finetune_image_size,
        heatmap_size=args.heatmap_size,
        freeze_encoder=True,
    ).to(args.device)

    if torch.cuda.is_available():
        print(f"\nModel: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
        torch.cuda.reset_peak_memory_stats()

    # ===== 5. Optimizer (head only first) =====
    optimizer = AdamW(model.head.parameters(),
                      lr=args.lr_head, weight_decay=args.weight_decay)

    # ===== 6. Training loop =====
    best_MRE  = float('inf')
    folder_name = Path(args.data_dir).name
    ckpt_name = f"finetune_{folder_name}_best_n{args.n_shots}_size{args.finetune_image_size}.pth"

    for epoch in range(args.epochs):

        if epoch == args.freeze_epochs:
            print("\n=== Unfreezing encoder ===")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = AdamW([
                {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
                {'params': model.head.parameters(),    'lr': args.lr_head},
            ], weight_decay=args.weight_decay)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.device, accum_steps=args.grad_accum
        )

        if (epoch + 1) % args.eval_every == 0:
            MRE_per_lm, MRE_overall = evaluate(
                model, test_loader, args.device,
                args.finetune_image_size, args.heatmap_size,
                use_tta=args.use_tta,
            )

            if MRE_per_lm is None:
                print(f"Epoch {epoch}: loss={train_loss:.4f} (test set empty)")
            else:
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, "
                          f"MRE={MRE_overall:.2f}, peak_mem={peak_mem:.1f}GB")
                else:
                    print(f"Epoch {epoch}: loss={train_loss:.4f}, MRE={MRE_overall:.2f}")

                if MRE_overall < best_MRE:
                    best_MRE = MRE_overall
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'MRE':        MRE_overall,
                        'MRE_per_lm': MRE_per_lm,
                        'config':     vars(args),
                    }, os.path.join(args.save_dir, ckpt_name))
                    print(f"  Saved best: MRE={MRE_overall:.2f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")

    # When the test set is empty no checkpoint was saved during training.
    # Save the final model so inference is still possible.
    if best_MRE == float('inf'):
        ckpt_path = os.path.join(args.save_dir, ckpt_name)
        torch.save({
            'model_state_dict': model.state_dict(),
            'MRE':        None,
            'MRE_per_lm': None,
            'config':     vars(args),
        }, ckpt_path)
        print(f"\nTest set was empty — saved final model to {ckpt_path}")

    print(f"\n=== FINAL RESULTS ===")
    print(f"  N-shot   : {args.n_shots}")
    if best_MRE == float('inf'):
        print(f"  Best MRE : n/a (no test set)")
    else:
        print(f"  Best MRE : {best_MRE:.2f} px (original resolution)")


if __name__ == '__main__':
    main()
