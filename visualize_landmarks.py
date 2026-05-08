"""
visualize_landmarks.py - Visualize predicted vs ground truth landmarks.

Predicted landmarks : "+" marker (red)
Ground truth        : "o" marker (green)

Usage:
    python visualize_landmarks.py                  # process all images
    python visualize_landmarks.py --show           # display each image interactively
    python visualize_landmarks.py --img 001.bmp    # single image
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless default; --show switches to TkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ============== CONFIGURATION ==============
CONFIG = {
    'image_dir': './data_predict/droso-new',   # original images + GT .txt files
    'pred_dir':  './predictions/droso-new',    # predicted .txt files
    'vis_dir':   './visualizations/droso-new', # output PNG directory
    'num_landmarks': 12,
    'marker_size': 10,   # marker size in points
    'linewidth':   1.5,
}


# ======================================

def load_landmarks(txt_path):
    """Read iMorph .txt file (one 'x y' per line). Returns None if file does not exist."""
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return None
    pts = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=np.float32) if pts else None


def draw_cross(ax, x, y, size, color, lw):
    """Draw a + marker at (x, y)."""
    ax.plot([x - size, x + size], [y, y], color=color, lw=lw)
    ax.plot([x, x], [y - size, y + size], color=color, lw=lw)


def visualize(img_path, pred_path, gt_path, vis_path, cfg, show=False):
    img = Image.open(img_path).convert('RGB')
    pred = load_landmarks(pred_path)
    gt   = load_landmarks(gt_path)

    if pred is None and gt is None:
        print(f"[SKIP] No data for {img_path.name}")
        return

    ms  = cfg['marker_size']
    lw  = cfg['linewidth']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(img_path.name, fontsize=10)
    ax.axis('off')

    # Ground truth — green circles
    if gt is not None:
        for i, (x, y) in enumerate(gt):
            ax.plot(x, y, 'o', markersize=ms, markerfacecolor='none',
                    markeredgecolor='#00e676', markeredgewidth=lw)
            ax.text(x + ms * 0.6, y - ms * 0.6, str(i + 1),
                    color='#00e676', fontsize=6, va='bottom')

    # Predicted — red crosses
    if pred is not None:
        cross_size = ms * 0.5  # half-arm length in data coords (pixels)
        # Convert ms from points to data pixels (approximate)
        dpi = fig.dpi
        fig_w_px, fig_h_px = fig.get_size_inches() * dpi
        img_w, img_h = img.size
        pts_per_px_x = fig_w_px / img_w * (72 / dpi)
        cross_data = ms / (pts_per_px_x if pts_per_px_x > 0 else 1)

        for i, (x, y) in enumerate(pred):
            draw_cross(ax, x, y, cross_data, color='#ff1744', lw=lw)
            ax.text(x + cross_data * 0.8, y + cross_data * 0.8, str(i + 1),
                    color='#ff1744', fontsize=6, va='top')

    # Legend
    legend_handles = []
    if gt is not None:
        legend_handles.append(
            mpatches.Patch(color='#00e676', label='Ground truth (o)'))
    if pred is not None:
        legend_handles.append(
            mpatches.Patch(color='#ff1744', label='Predicted (+)'))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=8, framealpha=0.7)

    fig.tight_layout(pad=0)

    if show:
        plt.show()

    vis_path = Path(vis_path)
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',  default=CONFIG['image_dir'])
    parser.add_argument('--pred_dir',   default=CONFIG['pred_dir'])
    parser.add_argument('--vis_dir',    default=CONFIG['vis_dir'])
    parser.add_argument('--img',        default=None,
                        help='Process a single image by filename (e.g. 001.bmp)')
    parser.add_argument('--show',       action='store_true',
                        help='Display each image interactively (requires display)')
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg')

    cfg = CONFIG.copy()
    image_dir = Path(args.image_dir)
    pred_dir  = Path(args.pred_dir)
    vis_dir   = Path(args.vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if args.img:
        image_paths = [image_dir / args.img]
    else:
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
            image_paths.extend(sorted(image_dir.glob(ext)))

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Visualizing {len(image_paths)} images -> {vis_dir}")
    for img_path in image_paths:
        pred_path = pred_dir  / img_path.with_suffix('.txt').name
        gt_path   = image_dir / img_path.with_suffix('.txt').name
        vis_path  = vis_dir   / (img_path.stem + '_vis.png')

        visualize(img_path, pred_path, gt_path, vis_path, cfg, show=args.show)
        print(f"  {img_path.name}")

    print(f"\nDone. Visualizations saved to: {vis_dir}")


if __name__ == '__main__':
    main()
