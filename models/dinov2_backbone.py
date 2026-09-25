"""
models/dinov2_backbone.py - DINOv2 encoder behind the MAE encoder's interface.

WingLandmarkModel only needs an object exposing forward_features(imgs) that
returns (B, 1 + num_patches, embed_dim) with the CLS token at index 0, so the
DINOv2 ablation reuses the heatmap head and the whole Stage 2/3 pipeline
unchanged.

Resolution note. DINOv2 uses 14x14 patches, so the input side must be a
multiple of 14 and 512 is not one. The default here is 448 = 14 x 32, which
reproduces the MAE setup's 32x32 token grid exactly: same 1024 patch tokens,
same 128x128 heatmap after the head's two 2x upsamplings, same head parameter
count, same sigma. The only remaining difference is the input side (448 vs
512), which keeps the ablation about the representation rather than about the
head geometry. Use 518 = 14 x 37 instead if matching DINOv2's native
pretraining resolution matters more; the head then emits 148x148 heatmaps.
"""
import torch
import torch.nn as nn

# name -> (timm model id, torch.hub id, embed dim)
DINOV2_MODELS = {
    'vits14': ('vit_small_patch14_dinov2.lvd142m', 'dinov2_vits14', 384),
    'vitb14': ('vit_base_patch14_dinov2.lvd142m',  'dinov2_vitb14', 768),
    'vitl14': ('vit_large_patch14_dinov2.lvd142m', 'dinov2_vitl14', 1024),
}


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT exposing the MAE encoder's forward_features() contract."""

    def __init__(self, variant='vitb14', img_size=448, verbose=True):
        super().__init__()
        if variant not in DINOV2_MODELS:
            raise ValueError(f"variant must be one of {sorted(DINOV2_MODELS)}, "
                             f"got '{variant}'")
        timm_id, hub_id, embed_dim = DINOV2_MODELS[variant]

        if img_size % 14:
            raise ValueError(
                f"DINOv2 uses 14x14 patches, so img_size must be a multiple of "
                f"14; {img_size} is not. Nearest valid sizes: "
                f"{(img_size // 14) * 14} and {(img_size // 14 + 1) * 14}.")

        import timm
        if verbose:
            print(f"\n=== Loading DINOv2 {variant} @ {img_size}px ===")
        # timm interpolates the pretrained positional embedding to img_size.
        self.dino = timm.create_model(timm_id, pretrained=True, num_classes=0,
                                      img_size=img_size)
        self.embed_dim = embed_dim
        self.patch_size = 14
        self.img_size = img_size
        self.grid = img_size // 14
        if verbose:
            n = sum(p.numel() for p in self.dino.parameters()) / 1e6
            print(f"  {timm_id}: {n:.1f}M params, "
                  f"{self.grid}x{self.grid} = {self.grid ** 2} patch tokens")

    def forward_features(self, imgs):
        """(B, 3, H, W) -> (B, 1 + num_patches, embed_dim), CLS at index 0."""
        return self.dino.forward_features(imgs)

    def forward(self, imgs):
        return self.forward_features(imgs)


def heatmap_size_for(img_size):
    """The head upsamples the token grid 4x, so this is its output side."""
    return (img_size // 14) * 4
