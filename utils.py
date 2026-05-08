"""
utils.py - Shared utility functions for the full pipeline
"""
import torch
import numpy as np


def generate_heatmap(coord, image_size, sigma=4.0):
    """
    Generate a Gaussian heatmap for one landmark.

    Args:
        coord: (x, y) landmark coordinate (already scaled to heatmap size)
        image_size: (H, W) heatmap dimensions
        sigma: Gaussian width
    Returns:
        heatmap: tensor (H, W)
    """
    H, W = image_size
    x, y = coord

    # Build coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )

    # Compute Gaussian
    heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap

# def heatmaps_to_coords(heatmaps, sigma=2.5):
#     """
#     Convert heatmaps to sub-pixel coordinates via weighted average of pixels
#     within radius=sigma around the peak activation. Avoids the quantization error of argmax and the pull toward background
#     activations that a global weighted average suffers from.
#     Args:
#         heatmaps: (B, K, H, W)
#         sigma: radius in heatmap pixels around the peak to include
#     Returns:
#         coords: (B, K, 2) float (x, y) per landmark
#     """
#     B, K, H, W = heatmaps.shape

#     # Peak position via argmax
#     flat    = heatmaps.reshape(B, K, -1)
#     max_idx = flat.argmax(dim=-1)                   # (B, K)
#     peak_y  = (max_idx // W).float()                # (B, K)
#     peak_x  = (max_idx %  W).float()                # (B, K)

#     # Coordinate grid
#     yy, xx = torch.meshgrid(
#         torch.arange(H, dtype=torch.float32, device=heatmaps.device),
#         torch.arange(W, dtype=torch.float32, device=heatmaps.device),
#         indexing='ij',
#     )  # (H, W)

#     # Distance from each landmark's peak: broadcast (B, K, 1, 1) vs (H, W)
#     dy = yy - peak_y.unsqueeze(-1).unsqueeze(-1)    # (B, K, H, W)
#     dx = xx - peak_x.unsqueeze(-1).unsqueeze(-1)    # (B, K, H, W)
#     in_radius = (dx ** 2 + dy ** 2 <= sigma ** 2).float()

#     masked = heatmaps * in_radius
#     total  = masked.sum(dim=[-2, -1]).clamp(min=1e-8)   # (B, K)

#     x_coord = (xx * masked).sum(dim=[-2, -1]) / total   # (B, K)
#     y_coord = (yy * masked).sum(dim=[-2, -1]) / total   # (B, K)

#     return torch.stack([x_coord, y_coord], dim=-1)      # (B, K, 2)
def heatmaps_to_coords(heatmaps):
    """
    Convert heatmaps to coordinates via argmax.

    Args:
        heatmaps: tensor (B, K, H, W) - K is the number of landmark classes
    Returns:
        coords: tensor (B, K, 2) - (x, y) per landmark
    """
    B, K, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.reshape(B, K, -1)
    max_idx = heatmaps_flat.argmax(dim=-1)

    y = (max_idx // W).float()
    x = (max_idx % W).float()

    coords = torch.stack([x, y], dim=-1)  # (B, K, 2)
    return coords


def compute_mrse(pred_coords, gt_coords):
    """
    Compute Mean Radial Square Error — preserves the metric from the original paper.

    Args:
        pred_coords: (N, K, 2) predicted coordinates
        gt_coords: (N, K, 2) ground truth coordinates
    Returns:
        mrse_per_landmark: (K,) MRSE per landmark class
        mrse_overall: scalar overall MRSE
    """
    # Euclidean distance per landmark per image
    distances = torch.sqrt(((pred_coords - gt_coords) ** 2).sum(dim=-1))  # (N, K)

    mrse_per_landmark = distances.mean(dim=0)  # (K,)
    mrse_overall = distances.mean()

    return mrse_per_landmark, mrse_overall


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
