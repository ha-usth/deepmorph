"""
stage3_predict_arg.py - Stage 3: Prediction (Inference) with CLI arguments

Usage:
    python stage3_predict_arg.py --checkpoint ./checkpoints/finetune_best.pth \
                                  --input_dir ./data/tsetse \
                                  --output_dir ./predictions/tsetse \
                                  --num_landmarks 11
    e.g.:    python stage3_predict_arg.py --checkpoint .\checkpoints\finetune_cepha400_best_n30_size512.pth --input_dir .\data_pretrain\cepha400 --output_dir .\prediction\cepha400 --num_landmarks 19
    #python stage3_predict_arg.py --checkpoint .\checkpoints\finetune_fly_best_n5_size512.pth --input_dir .\data_predict\fly --output_dir .\predictions\fly --num_landmarks 10
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.landmark_head import WingLandmarkModel
from models.regression_head import RegressionLandmarkModel
from utils import heatmaps_to_coords, compute_MRE


def parse_args():
    parser = argparse.ArgumentParser(description='DeepMorph Stage 3: Landmark Prediction')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images to predict')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predicted .txt landmark files')

    parser.add_argument('--num_landmarks', type=int, default=15,
                        help='Number of landmark classes (default: 15)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size used during fine-tuning (default: 512)')
    parser.add_argument('--heatmap_size', type=int, default=128,
                        help='Heatmap output size (default: 128)')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='ViT patch size (default: 16)')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='ViT embedding dimension (default: 768)')
    parser.add_argument('--model_size', type=str, default='base', choices=['base', 'small'],
                        help='MAE model size (default: base)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on (default: auto)')
    parser.add_argument('--max_test', type=int, default=20,
                        help='Maximum number of test images to evaluate (default: 20, 0 = all)')

    return parser.parse_args()


def load_finetuned_model(ckpt_path, config):
    """Load model from Stage 2. Auto-detects heatmap vs regression."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    head_type = ckpt.get('head_type', 'heatmap')
    config['head_type'] = head_type

    if config['model_size'] == 'base':
        mae = mae_vit_base(img_size=config['image_size'])
    else:
        mae = mae_vit_small(img_size=config['image_size'])

    if head_type == 'regression':
        model = RegressionLandmarkModel(
            mae_encoder=mae,
            num_landmarks=config['num_landmarks'],
            embed_dim=config['embed_dim'],
            freeze_encoder=False,
        )
        print(f"Loaded REGRESSION model from {ckpt_path}")
    else:
        model = WingLandmarkModel(
            mae_encoder=mae,
            num_landmarks=config['num_landmarks'],
            embed_dim=config['embed_dim'],
            patch_size=config['patch_size'],
            img_size=config['image_size'],
            heatmap_size=config['heatmap_size'],
            freeze_encoder=False,
        )
        print(f"Loaded HEATMAP model from {ckpt_path}")

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if ckpt.get('MRE') is not None:
        print(f"  (Test MRE at save time: {ckpt['MRE']:.2f})")
    return model


def predict_one_image(model, img_path, config, device):
    """
    Predict landmarks for one image.

    Returns:
        landmarks: np.array (K, 2) - coordinates at ORIGINAL image resolution
    """
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    img_resized = img.resize(
        (config['image_size'], config['image_size']), Image.BILINEAR
    )

    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    if config.get('head_type', 'heatmap') == 'regression':
        coords_image = output[0].cpu().numpy() * config['image_size']
    else:
        coords_heatmap = heatmaps_to_coords(output)[0].cpu().numpy()
        coords_image = coords_heatmap * (config['image_size'] / config['heatmap_size'])

    coords_orig = coords_image.copy()
    coords_orig[:, 0] *= orig_w / config['image_size']
    coords_orig[:, 1] *= orig_h / config['image_size']

    return coords_orig


def load_gt_landmarks(txt_path, num_landmarks):
    """Read ground truth from iMorph .txt file. Returns None if file does not exist."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return None
    pts = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    if len(pts) < num_landmarks:
        return None
    return np.array(pts[:num_landmarks], dtype=np.float32)


def save_landmarks_imorph_format(landmarks, output_path):
    """Save landmarks in iMorph format (one 'x y' per line)."""
    with open(output_path, 'w') as f:
        for x, y in landmarks:
            f.write(f"{x:.2f} {y:.2f}\n")


def main():
    args = parse_args()

    config = {
        'finetune_checkpoint': args.checkpoint,
        'input_dir':           args.input_dir,
        'output_dir':          args.output_dir,
        'num_landmarks':       args.num_landmarks,
        'image_size':          args.image_size,
        'heatmap_size':        args.heatmap_size,
        'patch_size':          args.patch_size,
        'embed_dim':           args.embed_dim,
        'model_size':          args.model_size,
        'device':              args.device,
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    model = load_finetuned_model(config['finetune_checkpoint'], config)
    model = model.to(config['device'])

    input_dir = Path(config['input_dir'])
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_paths.extend(list(input_dir.glob(ext)))
    image_paths.sort()

    total = len(image_paths)
    if args.max_test and len(image_paths) > args.max_test:
        image_paths = image_paths[:args.max_test]
    print(f"Found {total} images — evaluating {len(image_paths)}.")

    all_pred = []
    all_gt   = []

    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            landmarks = predict_one_image(model, img_path, config, config['device'])

            output_path = Path(config['output_dir']) / (img_path.stem + '.txt')
            save_landmarks_imorph_format(landmarks, output_path)

            gt = load_gt_landmarks(
                input_dir / img_path.with_suffix('.txt').name,
                config['num_landmarks']
            )
            if gt is not None:
                all_pred.append(landmarks)
                all_gt.append(gt)
        except Exception as e:
            print(f"Error on image {img_path.name}: {e}")

    print(f"\nDone. Results saved to: {config['output_dir']}")

    # ===== MRE =====
    if all_pred:
        pred_t = torch.from_numpy(np.stack(all_pred))  # (N, K, 2)
        gt_t   = torch.from_numpy(np.stack(all_gt))    # (N, K, 2)
        MRE_per_lm, MRE_overall = compute_MRE(pred_t, gt_t)

        print(f"\n=== MRE (original resolution, px) — N={len(all_pred)} images ===")
        print(f"Overall : {MRE_overall.item():.2f} px")
        for i, v in enumerate(MRE_per_lm):
            print(f"  LM {i+1:02d}: {v.item():.2f} px")
    else:
        print("\n(No GT .txt files found in input_dir — skipping MRE.)")


if __name__ == '__main__':
    main()
