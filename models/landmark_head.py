"""
models/landmark_head.py - Landmark detection head on top of MAE encoder

Design: Heatmap regression — predicts one heatmap per landmark class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LandmarkHead(nn.Module):
    """
    Heatmap regression head.

    Takes features from ViT encoder (B, num_patches, embed_dim)
    -> reshape to 2D feature map
    -> upsample through Conv layers
    -> output K heatmaps (K = number of landmark classes)
    """

    def __init__(self,
                 embed_dim=768,
                 num_landmarks=15,
                 patch_size=16,
                 img_size=224,
                 heatmap_size=56):  # 56 = 224 / 4
        super().__init__()

        self.num_landmarks = num_landmarks
        self.patch_size = patch_size
        self.img_size = img_size
        self.heatmap_size = heatmap_size

        # Number of patches per side
        self.grid_size = img_size // patch_size  # e.g. 14

        # Feature pyramid: upsample from 14x14 -> 56x56
        self.upsample = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Output K heatmaps
        self.heatmap_pred = nn.Conv2d(128, num_landmarks, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: (B, num_patches+1, embed_dim) - ViT encoder output
                      including CLS token
        Returns:
            heatmaps: (B, K, heatmap_size, heatmap_size)
        """
        # Drop CLS token
        x = features[:, 1:, :]  # (B, num_patches, embed_dim)

        # Reshape to 2D feature map
        B = x.shape[0]
        x = rearrange(x, 'b (h w) d -> b d h w', h=self.grid_size, w=self.grid_size)

        # Upsample
        x = self.upsample(x)

        # Predict heatmaps
        heatmaps = self.heatmap_pred(x)  # (B, K, heatmap_size, heatmap_size)

        return heatmaps


class WingLandmarkModel(nn.Module):
    """
    Full model: pretrained MAE encoder + landmark head.
    """

    def __init__(self, mae_encoder, num_landmarks=15,
                 embed_dim=768, patch_size=16, img_size=224,
                 heatmap_size=56, freeze_encoder=False):
        super().__init__()

        self.encoder = mae_encoder
        self.head = LandmarkHead(
            embed_dim=embed_dim,
            num_landmarks=num_landmarks,
            patch_size=patch_size,
            img_size=img_size,
            heatmap_size=heatmap_size,
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("⚠ Encoder is FROZEN (head only)")

    def forward(self, imgs):
        # Extract features — forward_features (no masking, no decoder)
        features = self.encoder.forward_features(imgs)
        # Predict heatmaps
        heatmaps = self.head(features)
        return heatmaps
