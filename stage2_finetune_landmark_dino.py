"""
stage2_finetune_landmark_dino.py - Stage 2 with a DINOv2 encoder.

Ablation counterpart of stage2_finetune_landmark.py: the MAE encoder is
replaced by a DINOv2 ViT, everything downstream is unchanged. The training
loop, the heatmap loss and the evaluation are imported from the MAE script
rather than copied, so the two branches cannot drift apart and any difference
in the reported error is attributable to the representation alone.

DINOv2 uses 14x14 patches, so the default input is 448 = 14 x 32. That gives
the same 32x32 token grid and the same 128x128 heatmap as the MAE setup at
512 px, leaving the input side as the only geometric difference.

Usage:
    python stage2_finetune_landmark_dino.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dinov2_backbone import DINOv2Backbone, heatmap_size_for
from models.landmark_head import WingLandmarkModel
from datasets.landmark_dataset import LandmarkDataset, select_few_shot_subset
from utils import set_seed
# Reused verbatim so the DINOv2 and MAE branches share one optimisation path.
from stage2_finetune_landmark import train_one_epoch, evaluate


CONFIG = {
    'dino_variant': 'vitb14',      # vits14 / vitb14 / vitl14
    'finetune_image_size': 448,    # must be a multiple of 14; 448 = 14 x 32
    'heatmap_size': 128,           # (448 // 14) * 4
    'sigma': 3,

    'target_data_dir': './train_pool/droso_small',
    'n_shots': 15,
    'num_landmarks': 15,

    'patch_size': 14,
    'embed_dim': 768,              # 384 for vits14, 1024 for vitl14

    'batch_size': 2,
    'gradient_accumulation_steps': 4,
    'lr_head': 5e-4,
    'lr_encoder': 5e-6,
    'weight_decay': 0.05,
    'epochs': 300,
    'freeze_encoder_epochs': 30,
    'eval_every': 10,
    'use_tta': True,

    'save_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}


def main():
    config = CONFIG
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    size = config['finetune_image_size']
    expected = heatmap_size_for(size)
    if config['heatmap_size'] != expected:
        print(f"[warn] heatmap_size {config['heatmap_size']} does not match the "
              f"head's output for a {size}px input; using {expected}")
        config['heatmap_size'] = expected

    print("=== Stage 2 (DINOv2 ablation) ===")
    print(f"  Backbone   : DINOv2 {config['dino_variant']}")
    print(f"  Image size : {size}  ({size // 14}x{size // 14} tokens)")
    print(f"  Heatmap    : {config['heatmap_size']}")
    print(f"  N-shot     : {config['n_shots']}")

    train_list, test_list = select_few_shot_subset(
        config['target_data_dir'], n_shots=config['n_shots'],
        seed=config['seed'])
    print(f"\nTrain: {len(train_list)} images | Val: {len(test_list)} images")

    def make(image_list, augment):
        return LandmarkDataset(
            data_dir=config['target_data_dir'], image_list=image_list,
            num_landmarks=config['num_landmarks'], image_size=size,
            heatmap_size=config['heatmap_size'], sigma=config['sigma'],
            augment=augment)

    train_loader = DataLoader(make(train_list, True),
                              batch_size=config['batch_size'], shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(make(test_list, False),
                             batch_size=config['batch_size'], shuffle=False,
                             num_workers=0)

    backbone = DINOv2Backbone(config['dino_variant'], img_size=size)
    model = WingLandmarkModel(
        mae_encoder=backbone, num_landmarks=config['num_landmarks'],
        embed_dim=config['embed_dim'], patch_size=config['patch_size'],
        img_size=size, heatmap_size=config['heatmap_size'],
        freeze_encoder=True).to(config['device'])

    optimizer = AdamW(model.head.parameters(), lr=config['lr_head'],
                      weight_decay=config['weight_decay'])

    best_MRE = float('inf')
    for epoch in range(config['epochs']):
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
            accum_steps=config['gradient_accumulation_steps'])

        if (epoch + 1) % config['eval_every'] == 0:
            MRE_per_lm, MRE_overall = evaluate(
                model, test_loader, config['device'], size,
                config['heatmap_size'], use_tta=config['use_tta'])
            if MRE_per_lm is None:
                print(f"Epoch {epoch}: loss={train_loss:.4f} (val set empty)")
                continue
            print(f"Epoch {epoch}: loss={train_loss:.4f}, MRE={MRE_overall:.2f}")
            if MRE_overall < best_MRE:
                best_MRE = MRE_overall
                path = os.path.join(
                    config['save_dir'],
                    f"finetune_dino_best_n{config['n_shots']}_size{size}.pth")
                torch.save({'model_state_dict': model.state_dict(),
                            'MRE': MRE_overall, 'MRE_per_lm': MRE_per_lm,
                            'config': config, 'head_type': 'heatmap',
                            'backbone': f"dinov2_{config['dino_variant']}"},
                           path)
                print(f"  Saved best: MRE={MRE_overall:.2f}")
        else:
            print(f"Epoch {epoch}: loss={train_loss:.4f}")

    print(f"\n=== FINAL (DINOv2 {config['dino_variant']}) ===")
    print(f"  Best MRE: {best_MRE:.2f} px (at original image resolution)")


if __name__ == '__main__':
    main()
