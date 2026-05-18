"""
stage3_dino_predict.py - Stage 3: Prediction (Inference) for DINOv2-based models

Goal: Use a model trained by stage2_dino_finetune_landmark.py to predict
      landmarks on new images.
Output format: .txt files matching iMorph format (one 'x y' per line).

Usage:
    python stage3_dino_predict.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.landmark_head import WingLandmarkModel
from utils import heatmaps_to_coords, compute_MRE


# ============== CONFIGURATION ==============
CONFIG = {
    # Checkpoint produced by stage2_dino_finetune_landmark.py
    'finetune_checkpoint': './checkpoints/finetune_sea_bass_best_n25_size512.pth',

    'input_dir':  './data_predict/sea_bass',   # images to predict
    'output_dir': './predictions/drososea_bass',    # where .txt files are saved

    # Must match the training config used in stage2_dino_finetune_landmark.py
    'image_size':    518,
    'heatmap_size':  148,   # = (518 // 14) * 4 = 37 * 4
    'patch_size':    14,    # DINOv2 always uses 14×14 patches
    'embed_dim':     768,   # ViT-B (384 for ViT-S, 1024 for ViT-L)
    'num_landmarks': 11,

    # DINOv2 model variant — must match what was used during training
    'dino_model':    'dinov2_vitb14',

    # Optional torch.hub cache dir (None → default ~/.cache/torch/hub)
    'hub_cache_dir': None,

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ---------------------------------------------------------------------------
# DINOv2 encoder wrapper  (identical to the one in stage2_dino_finetune_landmark.py)
# ---------------------------------------------------------------------------

class DINOv2Encoder(torch.nn.Module):
    """
    Thin wrapper around a DINOv2 ViT that exposes the same
    ``forward_features(imgs)`` interface as the MAE encoder used in
    WingLandmarkModel.

    Returns (B, num_patches + 1, embed_dim) with the CLS token at index 0.
    """

    def __init__(self, model_name: str = 'dinov2_vitb14', hub_cache_dir=None):
        super().__init__()

        if hub_cache_dir is not None:
            torch.hub.set_dir(hub_cache_dir)

        print(f"\n=== Loading DINOv2 backbone: {model_name} ===")
        self.dino = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True,
            verbose=False,
        )
        print(f"  Loaded {model_name} from torch.hub")

        self.embed_dim  = self.dino.embed_dim
        self.patch_size = self.dino.patch_size

    def forward_features(self, imgs):
        out   = self.dino.forward_features(imgs)
        cls   = out['x_norm_clstoken'].unsqueeze(1)   # (B, 1, D)
        patch = out['x_norm_patchtokens']              # (B, N, D)
        return torch.cat([cls, patch], dim=1)          # (B, N+1, D)

    def forward(self, imgs):
        return self.forward_features(imgs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_finetuned_model(ckpt_path, config):
    """
    Rebuild the DINOv2 + landmark-head architecture and load the saved weights.

    The checkpoint's 'config' dict is used to auto-fill model hyper-parameters
    when they are present, falling back to the local CONFIG values otherwise.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Prefer values that were baked into the checkpoint at training time
    saved_cfg = ckpt.get('config', {})
    image_size    = saved_cfg.get('finetune_image_size', config['image_size'])
    heatmap_size  = saved_cfg.get('heatmap_size',        config['heatmap_size'])
    patch_size    = saved_cfg.get('patch_size',          config['patch_size'])
    embed_dim     = saved_cfg.get('embed_dim',           config['embed_dim'])
    num_landmarks = saved_cfg.get('num_landmarks',       config['num_landmarks'])
    dino_model    = saved_cfg.get('dino_model',          config['dino_model'])

    print(f"\n=== Loading checkpoint ===")
    print(f"  Path          : {ckpt_path}")
    print(f"  DINOv2 model  : {dino_model}")
    print(f"  Image size    : {image_size}")
    print(f"  Heatmap size  : {heatmap_size}")
    print(f"  Patch size    : {patch_size}")
    print(f"  Embed dim     : {embed_dim}")
    print(f"  Num landmarks : {num_landmarks}")

    # Build the encoder (weights loaded from hub, then overridden by checkpoint)
    dino_encoder = DINOv2Encoder(
        model_name=dino_model,
        hub_cache_dir=config['hub_cache_dir'],
    )

    # Build the full model
    model = WingLandmarkModel(
        mae_encoder=dino_encoder,
        num_landmarks=num_landmarks,
        embed_dim=embed_dim,
        patch_size=patch_size,
        img_size=image_size,
        heatmap_size=heatmap_size,
        freeze_encoder=False,
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    if ckpt.get('MRE') is not None:
        print(f"  Test MRE at save time: {ckpt['MRE']:.2f} px")

    # Propagate resolved values back so predict_one_image can use them
    config['image_size']    = image_size
    config['heatmap_size']  = heatmap_size
    config['num_landmarks'] = num_landmarks

    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict_one_image(model, img_path, config, device):
    """
    Predict landmarks for one image.

    Returns:
        landmarks: np.ndarray  (K, 2) — coordinates at ORIGINAL image resolution
    """
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    img_resized = img.resize(
        (config['image_size'], config['image_size']), Image.BILINEAR
    )

    img_arr    = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps = model(img_tensor)   # (1, K, heatmap_size, heatmap_size)

    coords_heatmap = heatmaps_to_coords(heatmaps)[0].cpu().numpy()
    coords_image   = coords_heatmap * (config['image_size'] / config['heatmap_size'])

    coords_orig = coords_image.copy()
    coords_orig[:, 0] *= orig_w / config['image_size']
    coords_orig[:, 1] *= orig_h / config['image_size']

    return coords_orig


def load_gt_landmarks(txt_path, num_landmarks):
    """Read ground truth from iMorph .txt file. Returns None if file missing."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = CONFIG
    os.makedirs(config['output_dir'], exist_ok=True)

    model = load_finetuned_model(config['finetune_checkpoint'], config)
    model = model.to(config['device'])

    input_dir   = Path(config['input_dir'])
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_paths.extend(list(input_dir.glob(ext)))
    image_paths.sort()

    print(f"\nFound {len(image_paths)} images in {input_dir}")

    all_pred = []
    all_gt   = []

    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            landmarks = predict_one_image(model, img_path, config, config['device'])

            output_path = Path(config['output_dir']) / (img_path.stem + '.txt')
            save_landmarks_imorph_format(landmarks, output_path)

            gt = load_gt_landmarks(
                input_dir / img_path.with_suffix('.txt').name,
                config['num_landmarks'],
            )
            if gt is not None:
                all_pred.append(landmarks)
                all_gt.append(gt)

        except Exception as e:
            print(f"Error on {img_path.name}: {e}")

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
