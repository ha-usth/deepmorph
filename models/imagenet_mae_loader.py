"""
models/imagenet_mae_loader.py - Download and load MAE checkpoints pretrained on ImageNet

Source: https://github.com/facebookresearch/mae
Available checkpoints:
    - ViT-Base:  mae_pretrain_vit_base.pth        (~330MB)
    - ViT-Large: mae_pretrain_vit_large.pth       (~1.3GB)
    - ViT-Huge:  mae_pretrain_vit_huge.pth        (~2.5GB)

Usage:
    >>> mae = mae_vit_base(img_size=224)
    >>> load_imagenet_mae_weights(mae, 'base')
"""
import os
import sys
from pathlib import Path
import urllib.request
import torch

# Official URLs from Facebook AI Research
MAE_URLS = {
    'base': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
    'large': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
    'huge': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
}

DEFAULT_CACHE_DIR = Path.home() / '.cache' / 'mae_imagenet'


def download_mae_checkpoint(model_size='base', cache_dir=None):
    """
    Download MAE checkpoint from Facebook AI if not already cached.

    Args:
        model_size: 'base', 'large', or 'huge'
        cache_dir: directory to cache files (default: ~/.cache/mae_imagenet)
    Returns:
        Path to the checkpoint file
    """
    if model_size not in MAE_URLS:
        raise ValueError(
            f"model_size must be one of {list(MAE_URLS.keys())}, "
            f"got '{model_size}'"
        )

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = f'mae_pretrain_vit_{model_size}.pth'
    filepath = cache_dir / filename

    if filepath.exists():
        print(f"✓ Checkpoint already cached: {filepath}")
        return filepath

    url = MAE_URLS[model_size]
    print(f"Downloading MAE-ImageNet checkpoint ({model_size})...")
    print(f"  URL: {url}")
    print(f"  Destination: {filepath}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Progress: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # newline after progress bar
        print(f"✓ Download complete: {filepath}")
    except Exception as e:
        # Remove partial file on error
        if filepath.exists():
            filepath.unlink()
        raise RuntimeError(f"Error downloading checkpoint: {e}")

    return filepath


def load_imagenet_mae_weights(model, model_size='base', cache_dir=None,
                                strict=False, verbose=True):
    """
    Load weights from MAE-ImageNet checkpoint into our model.

    Because the Facebook checkpoint uses slightly different layer names
    than our implementation, this function automatically remaps them.

    Args:
        model: instance of MaskedAutoencoderViT (from models/mae.py)
        model_size: 'base', 'large', or 'huge'
        cache_dir: cache directory
        strict: True = error on key mismatch; False = skip mismatches
        verbose: print loading details
    Returns:
        model (with loaded weights)
    """
    ckpt_path = download_mae_checkpoint(model_size, cache_dir)

    if verbose:
        print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Facebook checkpoint structure: {'model': state_dict, 'epoch': ...}
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Remap layer names from Facebook -> our implementation
    new_state_dict = _remap_keys(state_dict, model)

    # Load
    msg = model.load_state_dict(new_state_dict, strict=strict)

    if verbose:
        n_loaded = len([k for k in new_state_dict if k in model.state_dict()])
        n_total = len(model.state_dict())
        print(f"✓ Loaded {n_loaded}/{n_total} layers from MAE-ImageNet")
        if msg.missing_keys:
            print(f"  Missing keys ({len(msg.missing_keys)}): "
                  f"{msg.missing_keys[:3]}{'...' if len(msg.missing_keys) > 3 else ''}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys ({len(msg.unexpected_keys)}): "
                  f"{msg.unexpected_keys[:3]}{'...' if len(msg.unexpected_keys) > 3 else ''}")

    return model


def _remap_keys(fb_state_dict, our_model):
    """
    Map layer names from Facebook MAE -> our implementation.

    Key differences:
    - Facebook uses 'blocks.X' for Transformer blocks (encoder & decoder)
    - We use 'encoder_blocks.X' and 'decoder_blocks.X'
    - Facebook uses 'norm' and 'decoder_norm'; we use 'encoder_norm', 'decoder_norm'
    - Facebook uses separate Attention/MLP; we use nn.MultiheadAttention
    """
    new_state_dict = {}
    our_keys = set(our_model.state_dict().keys())

    for fb_key, value in fb_state_dict.items():
        # === Mapping rules ===
        new_key = fb_key

        # Encoder blocks: blocks.X.* -> encoder_blocks.X.*
        if fb_key.startswith('blocks.'):
            new_key = 'encoder_' + fb_key

        # Encoder norm: norm.* -> encoder_norm.*
        elif fb_key.startswith('norm.'):
            new_key = 'encoder_norm.' + fb_key[len('norm.'):]

        # Decoder blocks and norm keep their names (decoder_blocks.X.*, decoder_norm.*)
        # CLS token, pos_embed, patch_embed, mask_token also keep their names

        # Handle attention layer remapping
        # Facebook: attn.qkv.weight, attn.qkv.bias, attn.proj.weight, attn.proj.bias
        # We use nn.MultiheadAttention: in_proj_weight, in_proj_bias, out_proj.*
        new_key = _remap_attention_keys(new_key)

        # Handle MLP layer remapping
        # Facebook: mlp.fc1, mlp.fc2
        # We use: mlp.0 (Linear), mlp.2 (second Linear)
        new_key = _remap_mlp_keys(new_key)

        # Only keep keys that exist in our model
        if new_key in our_keys:
            # Check shape matches
            our_shape = our_model.state_dict()[new_key].shape
            if value.shape == our_shape:
                new_state_dict[new_key] = value
            else:
                # Shape mismatch — usually pos_embed when image_size != 224
                if 'pos_embed' in new_key:
                    new_state_dict[new_key] = _interpolate_pos_embed(value, our_shape)
                # Skip other mismatches

    return new_state_dict


def _remap_attention_keys(key):
    """
    Map attention keys:
        ...attn.qkv.weight -> ...attn.in_proj_weight
        ...attn.qkv.bias   -> ...attn.in_proj_bias
        ...attn.proj.weight -> ...attn.out_proj.weight
        ...attn.proj.bias   -> ...attn.out_proj.bias
    """
    if '.attn.qkv.weight' in key:
        return key.replace('.attn.qkv.weight', '.attn.in_proj_weight')
    if '.attn.qkv.bias' in key:
        return key.replace('.attn.qkv.bias', '.attn.in_proj_bias')
    if '.attn.proj.weight' in key:
        return key.replace('.attn.proj.weight', '.attn.out_proj.weight')
    if '.attn.proj.bias' in key:
        return key.replace('.attn.proj.bias', '.attn.out_proj.bias')
    return key


def _remap_mlp_keys(key):
    """
    Map MLP keys:
        ...mlp.fc1.weight -> ...mlp.0.weight
        ...mlp.fc1.bias   -> ...mlp.0.bias
        ...mlp.fc2.weight -> ...mlp.2.weight
        ...mlp.fc2.bias   -> ...mlp.2.bias
    """
    replacements = {
        '.mlp.fc1.weight': '.mlp.0.weight',
        '.mlp.fc1.bias': '.mlp.0.bias',
        '.mlp.fc2.weight': '.mlp.2.weight',
        '.mlp.fc2.bias': '.mlp.2.bias',
    }
    for old, new in replacements.items():
        if old in key:
            return key.replace(old, new)
    return key


def _interpolate_pos_embed(pos_embed_old, target_shape):
    """
    Interpolate positional embedding when image size differs from 224.

    Example: pretrained at 224 (14x14 patches), target 384 (24x24 patches)
    -> interpolate pos_embed from 14x14 -> 24x24.
    """
    import math
    import torch.nn.functional as F

    # pos_embed shape: (1, num_patches + 1, embed_dim)
    # Position 0 is the CLS token

    cls_token = pos_embed_old[:, :1, :]
    pos_tokens = pos_embed_old[:, 1:, :]

    old_num = pos_tokens.shape[1]
    new_num = target_shape[1] - 1  # subtract CLS

    if old_num == new_num:
        return pos_embed_old  # no interpolation needed

    embed_dim = pos_tokens.shape[-1]
    old_size = int(math.sqrt(old_num))
    new_size = int(math.sqrt(new_num))

    # Reshape to 2D grid and interpolate
    pos_tokens = pos_tokens.reshape(1, old_size, old_size, embed_dim)
    pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # (1, D, H, W)
    pos_tokens = F.interpolate(
        pos_tokens, size=(new_size, new_size),
        mode='bicubic', align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embed_dim)

    new_pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
    print(f"  Interpolated pos_embed: {old_size}x{old_size} -> {new_size}x{new_size}")
    return new_pos_embed
