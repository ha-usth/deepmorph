"""
models/regression_head.py - Direct coordinate regression head.
"""
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Input  : ViT features (B, num_patches+1, embed_dim)
    Output : normalized coordinates (B, K, 2) in [0, 1]

    Uses CLS token + global-average-pool of patch tokens, concatenated,
    then passed through a 3-layer MLP.
    """

    def __init__(self, embed_dim=768, num_landmarks=15):
        super().__init__()
        self.num_landmarks = num_landmarks
        in_dim = 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_landmarks * 2),
        )

    def forward(self, features):
        cls   = features[:, 0]
        patch = features[:, 1:].mean(1)
        x = torch.cat([cls, patch], dim=-1)
        x = self.mlp(x)
        return torch.sigmoid(x.reshape(x.shape[0], self.num_landmarks, 2))


class RegressionLandmarkModel(nn.Module):
    """MAE encoder + RegressionHead."""

    def __init__(self, mae_encoder, num_landmarks=15,
                 embed_dim=768, freeze_encoder=False):
        super().__init__()
        self.encoder = mae_encoder
        self.head    = RegressionHead(embed_dim=embed_dim,
                                      num_landmarks=num_landmarks)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("⚠ Encoder đã bị FREEZE (chỉ train head)")

    def forward(self, imgs):
        return self.head(self.encoder.forward_features(imgs))
