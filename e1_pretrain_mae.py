"""
e1_pretrain_mae.py - EXPERIMENT 1: continued MAE pretraining under 3 protocols

Answers the reviewer question "if a new rare species has only 10 images, would
this approach still hold?" by making the Stage 1 pretraining pool an explicit,
swappable variable:

    PC  imagenet_only   no domain pretraining at all (MAE-ImageNet as-is)
    PA  full pool       all biological data, target dataset INCLUDED
    PB  leave-one-out   all biological data, target dataset EXCLUDED

All three write a checkpoint in the same format, so stage2_finetune_landmark.py
consumes them interchangeably via its 'mae_checkpoint' config key.

Differences from stage1_pretrain_mae.py, all aimed at the few-hundred-image
regime where the goal is to gain domain knowledge WITHOUT forgetting ImageNet:

    - schedule counted in optimizer steps, not epochs (an epoch is ~5 steps here)
    - per-iteration warmup + cosine (stage 1 wastes epoch 0 at lr=0)
    - layer-wise lr decay, so early ImageNet blocks barely move
    - decoder lr x10: the decoder is discarded, let it absorb the domain shift
    - patch_embed frozen: ImageNet low-level filters kept exactly
    - weight decay excluded from norms/biases/tokens
    - held-out unlabeled val split with fixed masking -> principled stopping rule
    - EMA weights, and WiSE checkpoints interpolating back toward ImageNet
    - byte-level leakage guard against the test set

PB depends on which dataset was held out, so its target name goes into the
filename; PC and PA share one pool across every target and keep the short name:

    mae_PC_s0_imagenet_only.pth
    mae_PA_s0_best.pth
    mae_PB_droso_small_s0_best.pth
    mae_PB_sea_bass_s0_best.pth

Usage:
    python e1_pretrain_mae.py                          # uses CONFIG below
    python e1_pretrain_mae.py --protocol PC
    python e1_pretrain_mae.py --protocol PB --exclude ./train_pool/droso_small
    python e1_pretrain_mae.py --protocol PB --exclude ./train_pool/sea_bass
    python e1_pretrain_mae.py --protocol PA --total-steps 4000 --seed 1
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import json
import copy
import time
import random
import hashlib
import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base, mae_vit_small
from models.imagenet_mae_loader import load_imagenet_mae_weights
from utils import set_seed


IMG_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')


# ============================== CONFIG ==============================
CONFIG = {
    # =========================================================================
    # PROTOCOL
    # =========================================================================
    'protocol': 'PA',             # PC/PA/PB — tag only, drives the filename
    'pretrain_mode': 'continue_from_imagenet',
                                  # 'continue_from_imagenet' : PA/PB
                                  # 'imagenet_only'          : PC, exports the
                                  #     ImageNet checkpoint in stage-1 format
                                  #     without training (no GPU time needed)
                                  # 'from_scratch'           : ablation
    'imagenet_cache_dir': None,

    # =========================================================================
    # DATA
    # =========================================================================
    'data_dirs': [
        './train_pool/bactro',
        './train_pool/diacha',
        './train_pool/droso_big',
        './train_pool/droso-281',
        './train_pool/droso_small',
        './train_pool/fly',
        './train_pool/sea_bass',
        './train_pool/tsetse',
        './train_pool/cepha',
    ],

    # Name of the downstream target dataset. It goes into the checkpoint
    # filename, because a PB pool depends on WHICH dataset was held out:
    # without it, PB-for-droso_small and PB-for-sea_bass overwrite each other.
    # None = derive it from exclude_dirs (PC/PA need no tag, their pool is the
    # same whatever the target is).
    'target': None,

    # PB: directories dropped from the pool. Any pool path containing one of
    # these strings is removed, so './train_pool/droso_small' is enough.
    'exclude_dirs': [],

    # Extra directories appended to the pool, e.g. an external unlabeled
    # repository that does not live under train_pool/.
    'include_dirs': [],

    # ---- leakage guard -------------------------------------------------------
    # Pool images byte-identical to a file under these directories are reported
    # (and optionally dropped). train_pool/ and test_set/ currently hold
    # identical copies for bactro, diacha, droso-281, fly and tsetse, so run
    # this once with 'warn' to see the damage before fixing the splits.
    'leakage_guard_dirs': ['./test_set'],
    'on_leak': 'warn',            # 'warn' | 'drop' | 'error'
    'dedup': True,                # drop duplicate images inside the pool itself

    'val_ratio': 0.15,            # held-out unlabeled split -> stopping rule
    'split_seed': 0,              # fixed, independent of the training seed
    'image_size': 224,            # keep 224: matches ImageNet pos_embed exactly

    # =========================================================================
    # AUGMENTATION
    # =========================================================================
    # With a few hundred images this is the main defence against memorisation.
    # RandomResizedCrop is the MAE recipe; keep the aspect range narrow so wing
    # geometry is not distorted beyond what Stage 2's square resize already does.
    'rrc_scale': (0.5, 1.0),
    'rrc_ratio': (0.8, 1.25),
    'color_jitter': 0.2,          # brightness/contrast; matches the stain and
                                  # contrast variability argued in the paper
    'rotation_deg': 10.0,         # 0 to disable

    # =========================================================================
    # BATCHING
    # =========================================================================
    'batch_size': 64,             # physical; 64 fits easily in 32GB at 224/bf16
    'accum_iter': 1,              # effective batch = batch_size * accum_iter
    'num_workers': 4,
    'amp': 'bf16',                # 'bf16' | 'fp16' | 'none'

    # =========================================================================
    # OPTIMISATION
    # =========================================================================
    'blr': 1.5e-4,                # lr = blr * effective_batch / 256
    'layer_decay': 0.75,          # LLRD: encoder block 0 moves slowest
    'decoder_lr_mult': 10.0,      # decoder is thrown away -> let it adapt
    'freeze_patch_embed': True,   # keep ImageNet low-level filters frozen
    'weight_decay': 0.05,         # applied to ndim>=2 weights only
    'betas': (0.9, 0.95),
    'clip_grad': 1.0,
    'mask_ratio': 0.75,           # keep fixed across protocols for comparability
    'model_size': 'base',

    # =========================================================================
    # SCHEDULE (in optimizer steps, not epochs)
    # =========================================================================
    'total_steps': 2000,          # sweep 500/1000/2000/4000 for the paper
    'warmup_steps': 200,          # ~10% of total
    'min_lr': 1e-6,

    # The public mae_pretrain_vit_*.pth holds the ENCODER ONLY (150 tensors, no
    # decoder_*, no mask_token), so the MAE decoder starts random. Its early
    # gradients are meaningless and would corrupt the intact ImageNet encoder.
    # During these first steps the encoder is frozen and only the decoder learns.
    # Set to 0 only if the decoder was itself loaded from a checkpoint.
    'decoder_warmup_steps': 200,

    # =========================================================================
    # AVERAGING / ANCHORING TO IMAGENET
    # =========================================================================
    'ema_decay': 0.999,           # 0 to disable
    'wise_alphas': [0.25, 0.5, 0.75],
                                  # theta = a*theta_domain + (1-a)*theta_imagenet
                                  # a=0 is PC, a=1 is the trained model.
                                  # Sweeping a gives a direct measurement of how
                                  # much ImageNet generality is worth keeping.

    # =========================================================================
    # ARTEFACTS TO KEEP
    # =========================================================================
    # Every .pth is ~450MB and a full run would otherwise leave six of them.
    # Anything not listed here is never written in the first place (except
    # 'best', which the training loop needs to track the best step and which
    # WiSE is built from, so it is removed at the very end instead). Stale
    # files left by an earlier run with the same name are cleaned up too.
    #   'best'   lowest val reconstruction loss -- what Stage 2 should consume
    #   'final'  weights at total_steps; identical to 'best' whenever the val
    #            curve is still improving at the end, which it usually is
    #   'ema'    EMA of the trajectory (skipping it also frees ~450MB of VRAM)
    #   'wise'   the ImageNet-interpolation family, one file per wise_alpha
    #   'log'    JSON log: config, pool report, val curve (a few KB)
    'keep': ['best', 'log'],

    # =========================================================================
    # RUN
    # =========================================================================
    'seed': 0,
    'eval_every': 50,
    'save_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================== DATA ==============================
class WingImageList(Dataset):
    """Unlabeled images from an explicit path list (so splits stay reproducible)."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        for offset in range(len(self.paths)):
            path = self.paths[(idx + offset) % len(self.paths)]
            try:
                return self.transform(Image.open(path).convert('RGB'))
            except Exception as e:
                print(f"Error reading {path}: {e}")
        raise RuntimeError("No readable image in dataset")


def build_transforms(config):
    """Stochastic transform for training, deterministic one for validation."""
    size = config['image_size']
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    train_ops = [
        transforms.RandomResizedCrop(
            size,
            scale=tuple(config['rrc_scale']),
            ratio=tuple(config['rrc_ratio']),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if config['rotation_deg'] > 0:
        train_ops.append(transforms.RandomRotation(
            config['rotation_deg'], interpolation=InterpolationMode.BILINEAR))
    if config['color_jitter'] > 0:
        cj = config['color_jitter']
        train_ops.append(transforms.ColorJitter(brightness=cj, contrast=cj))
    train_ops += [transforms.ToTensor(), norm]

    val_ops = [
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        norm,
    ]
    return transforms.Compose(train_ops), transforms.Compose(val_ops)


def _list_images(dirs):
    out = []
    for root in dirs:
        root = Path(root)
        if not root.exists():
            print(f"  ! missing directory, skipped: {root}")
            continue
        for ext in IMG_EXTS:
            out.extend(root.rglob(ext))
    return out


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def build_pool(config):
    """
    Resolve the pretraining pool for the active protocol.

    Returns (train_paths, val_paths, report) where report is JSON-serialisable
    and belongs in the paper's supplementary material.
    """
    print("=== Step 1: Resolve pretraining pool ===")
    paths = _list_images(list(config['data_dirs']) + list(config['include_dirs']))
    n_raw = len(paths)
    print(f"  Listed {n_raw} images from {len(config['data_dirs'])} directories")

    # ---- protocol exclusion (PB) --------------------------------------------
    excluded = []
    if config['exclude_dirs']:
        keys = [str(Path(d)).replace('\\', '/').lstrip('./')
                for d in config['exclude_dirs']]
        kept = []
        for p in paths:
            s = str(p).replace('\\', '/')
            if any(k in s for k in keys):
                excluded.append(str(p))
            else:
                kept.append(p)
        paths = kept
        print(f"  Excluded {len(excluded)} images "
              f"({', '.join(config['exclude_dirs'])})")

    # ---- hash the pool once, reuse for dedup and for the leakage guard ------
    pool_hash = {}
    if config['dedup'] or config['leakage_guard_dirs']:
        for p in tqdm(paths, desc="  hashing pool", leave=False):
            pool_hash[p] = _md5(p)

    # ---- deduplicate inside the pool ----------------------------------------
    n_dup = 0
    if config['dedup']:
        seen, kept = set(), []
        for p in paths:
            h = pool_hash[p]
            if h in seen:
                n_dup += 1
            else:
                seen.add(h)
                kept.append(p)
        paths = kept
        print(f"  Removed {n_dup} duplicate images inside the pool")

    # ---- leakage guard against the test set ---------------------------------
    n_leak, leaked = 0, []
    if config['leakage_guard_dirs']:
        guard_files = _list_images(config['leakage_guard_dirs'])
        guard_hashes = set()
        for p in tqdm(guard_files, desc="  hashing guard set", leave=False):
            guard_hashes.add(_md5(p))

        kept = []
        for p in paths:
            if pool_hash[p] in guard_hashes:
                n_leak += 1
                leaked.append(str(p))
            else:
                kept.append(p)

        pct = 100.0 * n_leak / max(len(paths), 1)
        if n_leak:
            print(f"  !! LEAKAGE: {n_leak}/{len(paths)} pool images ({pct:.1f}%) "
                  f"are byte-identical to files under "
                  f"{', '.join(config['leakage_guard_dirs'])}")
            by_dir = {}
            for s in leaked:
                key = '/'.join(Path(s).parts[:2])
                by_dir[key] = by_dir.get(key, 0) + 1
            for k, v in sorted(by_dir.items(), key=lambda kv: -kv[1]):
                print(f"       {v:6d}  {k}")

            if config['on_leak'] == 'error':
                raise RuntimeError(
                    f"{n_leak} pool images also appear in the guard set. "
                    f"Fix the split, or set on_leak='warn'/'drop'.")
            elif config['on_leak'] == 'drop':
                paths = kept
                print(f"  Dropped them; pool is now {len(paths)} images")
            else:
                print("  on_leak='warn': images KEPT. Results are transductive "
                      "and must be reported as such.")
        else:
            print(f"  Leakage guard clean ({len(guard_files)} guard files checked)")

    if not paths:
        raise RuntimeError("Pretraining pool is empty after filtering.")

    # ---- train / val split ---------------------------------------------------
    paths = sorted(paths, key=lambda p: str(p))       # deterministic order
    rng = random.Random(config['split_seed'])
    rng.shuffle(paths)
    if config['val_ratio'] > 0 and len(paths) >= 4:
        n_val = max(1, min(int(round(len(paths) * config['val_ratio'])),
                           len(paths) - 1))
    else:
        n_val = 0                                  # pool too small to hold out
    val_paths, train_paths = paths[:n_val], paths[n_val:]

    print(f"  Pool: {len(train_paths)} train / {len(val_paths)} val "
          f"(val_ratio={config['val_ratio']}, split_seed={config['split_seed']})")

    report = {
        'n_listed': n_raw,
        'n_excluded': len(excluded),
        'n_duplicates_removed': n_dup,
        'n_leaked': n_leak,
        'on_leak': config['on_leak'],
        'n_train': len(train_paths),
        'n_val': len(val_paths),
        'leaked_examples': leaked[:20],
    }
    return train_paths, val_paths, report


# ============================== MODEL ==============================
def build_model(config):
    """Build the MAE and load ImageNet weights when the mode asks for them."""
    if config['model_size'] == 'base':
        model = mae_vit_base(img_size=config['image_size'],
                             mask_ratio=config['mask_ratio'])
    elif config['model_size'] == 'small':
        if config['pretrain_mode'] != 'from_scratch':
            raise ValueError("model_size='small' has no ImageNet checkpoint; "
                             "use pretrain_mode='from_scratch'.")
        model = mae_vit_small(img_size=config['image_size'],
                              mask_ratio=config['mask_ratio'])
    else:
        raise NotImplementedError(
            f"model_size='{config['model_size']}' not in models/mae.py "
            f"(only 'base' and 'small' exist).")

    decoder_from_imagenet = False
    if config['pretrain_mode'] in ('continue_from_imagenet', 'imagenet_only'):
        print("\n=== Loading MAE-ImageNet weights ===")
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        load_imagenet_mae_weights(
            model, model_size=config['model_size'],
            cache_dir=config['imagenet_cache_dir'], strict=False, verbose=True)

        # Which halves actually came from the checkpoint?
        after = model.state_dict()
        changed = {k for k, v in after.items()
                   if v.shape == before[k].shape and not torch.equal(v, before[k])}
        enc_keys = [k for k in after if not k.startswith('decoder')
                    and k != 'mask_token']
        dec_keys = [k for k in after if k.startswith('decoder') or k == 'mask_token']
        n_enc = sum(k in changed for k in enc_keys)
        n_dec = sum(k in changed for k in dec_keys)
        decoder_from_imagenet = n_dec > 0
        print(f"  encoder tensors initialised from ImageNet: {n_enc}/{len(enc_keys)}")
        print(f"  decoder tensors initialised from ImageNet: {n_dec}/{len(dec_keys)}")
        if not decoder_from_imagenet:
            print("  NOTE: the public mae_pretrain_vit_*.pth ships the encoder "
                  "only, so the decoder is random.\n"
                  "        Keep decoder_warmup_steps > 0 so its initial "
                  "gradients do not corrupt the encoder.")
    elif config['pretrain_mode'] == 'from_scratch':
        print("\n=== Training from scratch (random init) ===")
    else:
        raise ValueError(f"unknown pretrain_mode '{config['pretrain_mode']}'")

    return model, decoder_from_imagenet


def build_param_groups(model, config):
    """
    AdamW param groups with:
      - layer-wise lr decay over the encoder (block 0 slowest)
      - a single higher lr multiplier for the whole decoder
      - weight decay on ndim>=2 weights only (never on norms, biases, tokens)

    Each group carries 'lr_scale'; set_lr() multiplies the schedule by it.
    """
    depth = len(model.encoder_blocks)
    num_layers = depth + 1
    decay = config['layer_decay']

    def encoder_layer_id(name):
        if name.startswith('patch_embed') or name in ('cls_token', 'pos_embed'):
            return 0
        if name.startswith('encoder_blocks.'):
            return int(name.split('.')[1]) + 1
        return num_layers                      # encoder_norm and anything else

    groups, meta = {}, []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = (p.ndim == 1) or name.endswith('cls_token') or name.endswith('mask_token')
        wd = 0.0 if no_wd else config['weight_decay']

        if name.startswith('decoder') or name == 'mask_token':
            tag, scale = 'decoder', float(config['decoder_lr_mult'])
        else:
            lid = encoder_layer_id(name)
            tag, scale = f'enc{lid:02d}', decay ** (num_layers - lid)

        key = (tag, wd)
        if key not in groups:
            groups[key] = {'params': [], 'weight_decay': wd,
                           'lr_scale': scale, 'name': f'{tag}_wd{wd}'}
        groups[key]['params'].append(p)

    out = [groups[k] for k in sorted(groups.keys())]
    for g in out:
        meta.append((g['name'], len(g['params']), round(g['lr_scale'], 5)))
    print(f"  Param groups ({len(out)}): "
          f"lr_scale {min(m[2] for m in meta):.4f} .. {max(m[2] for m in meta):.1f}")
    return out


def set_lr(optimizer, step, config):
    """Per-iteration warmup + cosine decay, scaled per param group."""
    base_lr = config['_lr']
    warmup, total = config['warmup_steps'], config['total_steps']
    if step < warmup:
        lr = base_lr * (step + 1) / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total - warmup, 1)
        lr = config['min_lr'] + (base_lr - config['min_lr']) * \
            0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_scale', 1.0)
    return lr


# ============================== EMA / WiSE ==============================
class WeightEMA:
    """Exponential moving average over floating-point state-dict entries."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self, model):
        """EMA floats merged back over the live model's state dict."""
        sd = copy.deepcopy(model.state_dict())
        for k, v in self.shadow.items():
            sd[k] = v.to(sd[k].dtype)
        return sd


def cleanup_artefacts(prefix, config):
    """
    Remove the artefacts this run does not keep.

    Also catches files left by an earlier run that used the same prefix, so a
    protocol re-run with fewer wise_alphas does not leave orphans behind. Every
    pattern is anchored on the exact prefix, so a run with a different --tag or
    --seed is never touched.
    """
    keep = set(config['keep'])
    folder, stem = Path(prefix).parent, Path(prefix).name
    doomed = []

    for kind, name in (('best', f"{stem}_best.pth"),
                       ('final', f"{stem}_final.pth"),
                       ('ema', f"{stem}_ema.pth"),
                       ('log', f"{stem}_log.json")):
        if kind not in keep:
            doomed.append(folder / name)

    wanted_wise = ({f"{stem}_wise{a:.2f}.pth" for a in config['wise_alphas']}
                   if 'wise' in keep else set())
    doomed += [p for p in folder.glob(f"{stem}_wise*.pth")
               if p.name not in wanted_wise]

    removed, freed = [], 0
    for p in doomed:
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            removed.append(p.name)

    if removed:
        print(f"  Removed {len(removed)} artefact(s), "
              f"{freed / 1024 ** 3:.2f} GB freed:")
        for name in sorted(removed):
            print(f"    {name}")
    return removed, freed


def wise_interpolate(sd_domain, sd_imagenet, alpha):
    """theta = alpha * domain + (1 - alpha) * imagenet, over float tensors."""
    out = {}
    for k, v in sd_domain.items():
        ref = sd_imagenet.get(k)
        if ref is not None and v.dtype.is_floating_point and ref.shape == v.shape:
            out[k] = (alpha * v.float().cpu() +
                      (1.0 - alpha) * ref.float().cpu()).to(v.dtype)
        else:
            out[k] = v.cpu()
    return out


# ============================== TRAIN / EVAL ==============================
def amp_context(config):
    """Return a factory producing the autocast context for this run."""
    if config['amp'] == 'none' or config['device'] != 'cuda':
        return nullcontext, None
    if config['amp'] == 'bf16':
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("amp='bf16' but this GPU has no bf16 support; "
                               "use 'fp16' or 'none'.")
        return (lambda: torch.autocast('cuda', dtype=torch.bfloat16)), None
    if config['amp'] == 'fp16':
        return (lambda: torch.autocast('cuda', dtype=torch.float16)), \
               torch.amp.GradScaler('cuda')
    raise ValueError(f"amp must be 'bf16'/'fp16'/'none', got '{config['amp']}'")


def _rng_snapshot():
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda_state


def _rng_restore(state):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


@torch.no_grad()
def evaluate(model, loader, device, autocast, seed=1234):
    """
    Validation reconstruction loss with a FIXED masking pattern.

    Without the seed reset the random mask would dominate the epoch-to-epoch
    difference and the curve could not be used as a stopping rule.
    """
    if loader is None or len(loader) == 0:
        return float('nan')

    rng_state = _rng_snapshot()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    total, n = 0.0, 0
    for imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast():
            loss, _, _ = model(imgs)
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    model.train()

    _rng_restore(rng_state)
    return total / max(n, 1)


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path, state_dict, config, extra=None):
    payload = {
        'model_state_dict': state_dict,
        'config': config,          # stage 2 reads config['image_size'] from here
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  saved {path}")


# ============================== MAIN ==============================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', choices=['PC', 'PA', 'PB', 'PS'],
                   help="PC exports ImageNet as-is; PA/PB continue from it; "
                        "PS trains from scratch on biological images only.")
    p.add_argument('--target', default=None,
                   help="downstream dataset name, written into the checkpoint "
                        "filename (default: derived from --exclude)")
    p.add_argument('--exclude', nargs='*', default=None,
                   help="directories to drop from the pool (PB)")
    p.add_argument('--include', nargs='*', default=None,
                   help="extra directories to append to the pool")
    p.add_argument('--total-steps', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--on-leak', choices=['warn', 'drop', 'error'], default=None)
    p.add_argument('--keep', nargs='*',
                   choices=['best', 'final', 'ema', 'wise', 'log'], default=None,
                   help="artefacts to leave on disk (default: best log)")
    p.add_argument('--tag', default=None, help="extra suffix for checkpoint names")
    return p.parse_args()


def apply_args(config, args):
    if args.protocol:
        config['protocol'] = args.protocol
        config['pretrain_mode'] = {
            'PC': 'imagenet_only',
            'PA': 'continue_from_imagenet',
            'PB': 'continue_from_imagenet',
            'PS': 'from_scratch',
        }[args.protocol]
    if args.exclude is not None:
        config['exclude_dirs'] = args.exclude
    if args.include is not None:
        config['include_dirs'] = args.include
    if args.target is not None:
        config['target'] = args.target
    if args.keep is not None:
        config['keep'] = args.keep
    for key, val in (('total_steps', args.total_steps), ('seed', args.seed),
                     ('batch_size', args.batch_size), ('on_leak', args.on_leak)):
        if val is not None:
            config[key] = val
    return config


def main():
    args = parse_args()
    config = apply_args(dict(CONFIG), args)
    set_seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    # A PB checkpoint is only meaningful together with the dataset it held out,
    # so the target name belongs in the filename. PC/PA pools are independent of
    # the target and keep the shorter name.
    target = config.get('target')
    if target is None and config['exclude_dirs']:
        target = Path(str(config['exclude_dirs'][0]).replace('\\', '/')).name
        config['target'] = target

    run = f"mae_{config['protocol']}"
    if target:
        run += f"_{target}"
    run += f"_s{config['seed']}"
    if args.tag:
        run += f"_{args.tag}"
    prefix = os.path.join(config['save_dir'], run)
    device = config['device']

    eff_batch = config['batch_size'] * config['accum_iter']
    config['_lr'] = config['blr'] * eff_batch / 256

    # From-scratch has no ImageNet weights to protect, and three of the defaults
    # actively harm it: a frozen patch_embed would stay at its random
    # initialisation for the whole run, layer-wise decay would let block 0 learn
    # ~40x slower than block 11 for no reason, and the decoder multiplier exists
    # only to keep a pretrained encoder still. Leaving them on would make the
    # ablation look bad for the wrong cause.
    if config['pretrain_mode'] == 'from_scratch':
        forced = {'freeze_patch_embed': False, 'layer_decay': 1.0,
                  'decoder_lr_mult': 1.0}
        changed = {k: v for k, v in forced.items() if config[k] != v}
        config.update(forced)
        if changed:
            print("[from_scratch] nothing pretrained to preserve; forcing "
                  + ", ".join(f"{k}={v}" for k, v in changed.items()) + "\n")

    # Stage 2 needs weights, so never let 'keep' leave only the log behind.
    keep = set(config['keep'])
    if not keep & {'best', 'final', 'ema', 'wise'}:
        keep.add('best')
        config['keep'] = sorted(keep)
        print("[warn] 'keep' listed no weights; 'best' added so Stage 2 has "
              "something to load.")

    print("=== Configuration ===")
    print(f"  Protocol       : {config['protocol']}  ({config['pretrain_mode']})")
    print(f"  Model          : ViT-{config['model_size'].capitalize()} @ "
          f"{config['image_size']}px, mask {config['mask_ratio']}")
    print(f"  Effective batch: {eff_batch} "
          f"({config['batch_size']} x {config['accum_iter']} accum)")
    print(f"  Base lr        : {config['_lr']:.2e}  (blr {config['blr']} * {eff_batch}/256)")
    print(f"  Steps          : {config['total_steps']} "
          f"({config['warmup_steps']} warmup)")
    print(f"  LLRD / dec mult: {config['layer_decay']} / {config['decoder_lr_mult']}")
    print(f"  AMP            : {config['amp']} on {device}")
    print()

    # ---- PC: no training, just re-export ImageNet in stage-1 format ----------
    if config['pretrain_mode'] == 'imagenet_only':
        model, _ = build_model(config)
        path = f"{prefix}_imagenet_only.pth"
        save_checkpoint(path, model.state_dict(), config,
                        extra={'step': 0, 'note': 'no domain pretraining (PC)'})
        print(f"\n[OK] PC baseline ready. Point stage 2 at:\n    {path}")
        return

    # ---- pool ---------------------------------------------------------------
    train_paths, val_paths, pool_report = build_pool(config)
    train_tf, val_tf = build_transforms(config)

    train_loader = DataLoader(
        WingImageList(train_paths, train_tf),
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True, drop_last=False,
        persistent_workers=config['num_workers'] > 0)
    val_loader = DataLoader(
        WingImageList(val_paths, val_tf),
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, drop_last=False,
        persistent_workers=config['num_workers'] > 0) if val_paths else None

    steps_per_epoch = max(len(train_loader) // config['accum_iter'], 1)
    print(f"  {steps_per_epoch} optimizer steps per pass over the pool "
          f"-> {config['total_steps'] / steps_per_epoch:.0f} passes total")

    # ---- model --------------------------------------------------------------
    print("\n=== Step 2: Initialize model ===")
    model, decoder_from_imagenet = build_model(config)
    model = model.to(device)

    # Snapshot the ImageNet init: the WiSE anchor, ~450MB kept on CPU. Only
    # worth holding when WiSE checkpoints are actually going to be written.
    want_wise = ('wise' in keep and config['wise_alphas']
                 and config['pretrain_mode'] == 'continue_from_imagenet')
    sd_imagenet = ({k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}
                   if want_wise else None)

    if config['freeze_patch_embed']:
        for p in model.patch_embed.parameters():
            p.requires_grad = False
        print("  patch_embed frozen (ImageNet low-level filters preserved)")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_all = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_all:.1f}M total, {n_train:.1f}M trainable")

    # ---- optimiser ----------------------------------------------------------
    # Built before any decoder-warmup freezing so the encoder still gets groups.
    param_groups = build_param_groups(model, config)
    optimizer = AdamW(param_groups, lr=config['_lr'], betas=tuple(config['betas']))
    autocast, scaler = amp_context(config)
    # Tracking an EMA costs ~450MB of VRAM and work on every step, so skip it
    # entirely when the EMA checkpoint is not being kept.
    ema = (WeightEMA(model, config['ema_decay'])
           if config['ema_decay'] > 0 and 'ema' in keep else None)

    # ---- decoder-only warmup -------------------------------------------------
    # AdamW skips parameters whose .grad is None, so toggling requires_grad is
    # enough; zero_grad(set_to_none=True) below keeps them at None.
    named = dict(model.named_parameters())
    encoder_trainable = [n for n, p in named.items()
                         if p.requires_grad and
                         not (n.startswith('decoder') or n == 'mask_token')]
    dec_warmup = config['decoder_warmup_steps'] \
        if config['pretrain_mode'] == 'continue_from_imagenet' else 0
    if dec_warmup > 0:
        for n in encoder_trainable:
            named[n].requires_grad = False
        src = "loaded from ImageNet" if decoder_from_imagenet else "RANDOM"
        print(f"  Decoder is {src}; encoder frozen for the first "
              f"{dec_warmup} steps ({len(encoder_trainable)} tensors)")

    # ---- train --------------------------------------------------------------
    print("\n=== Step 3: Pretraining ===")
    history, best_val, best_step = [], float('inf'), -1
    data_iter = infinite(train_loader)
    model.train()
    t0 = time.time()

    pbar = tqdm(range(config['total_steps']), desc="steps")
    for step in pbar:
        if dec_warmup > 0 and step == dec_warmup:
            for n in encoder_trainable:
                named[n].requires_grad = True
            tqdm.write(f"  step {step}: decoder warmup over, encoder unfrozen")

        lr = set_lr(optimizer, step, config)
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        for _ in range(config['accum_iter']):
            imgs = next(data_iter).to(device, non_blocking=True)
            with autocast():
                loss, _, _ = model(imgs)
            loss = loss / config['accum_iter']
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        if config['clip_grad']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if ema is not None:
            ema.update(model)

        pbar.set_postfix({'loss': f'{running:.4f}', 'lr': f'{lr:.2e}'})

        # ---- validation + best checkpoint ----------------------------------
        last = (step == config['total_steps'] - 1)
        if val_loader is not None and ((step + 1) % config['eval_every'] == 0 or last):
            val = evaluate(model, val_loader, device, autocast)
            history.append({'step': step + 1, 'train_loss': running,
                            'val_loss': val, 'lr': lr})
            tqdm.write(f"  step {step + 1:5d}  train {running:.4f}  val {val:.4f}"
                       + ("  <- best" if val < best_val else ""))
            if val < best_val:
                best_val, best_step = val, step + 1
                save_checkpoint(f"{prefix}_best.pth", model.state_dict(), config,
                                extra={'step': best_step, 'val_loss': best_val,
                                       'pool_report': pool_report})

    minutes = (time.time() - t0) / 60
    print(f"\nTraining done in {minutes:.1f} min")

    # ---- final artefacts ----------------------------------------------------
    # Only what 'keep' asks for is written; see cleanup_artefacts() for the rest.
    print("\n=== Step 4: Export checkpoints ===")
    if 'final' in keep:
        save_checkpoint(f"{prefix}_final.pth", model.state_dict(), config,
                        extra={'step': config['total_steps'],
                               'val_loss': (history[-1]['val_loss']
                                            if history else None),
                               'pool_report': pool_report})

    if ema is not None:
        ema_sd = ema.state_dict(model)
        save_checkpoint(f"{prefix}_ema.pth", ema_sd, config,
                        extra={'step': config['total_steps'],
                               'ema_decay': config['ema_decay'],
                               'pool_report': pool_report})

    # WiSE: how much ImageNet do we want to keep? Let Stage 2 answer it.
    if sd_imagenet is not None and config['wise_alphas']:
        # Anchor the interpolation on the best-val weights when we have them.
        base_sd = model.state_dict()
        if best_step > 0:
            base_sd = torch.load(f"{prefix}_best.pth", map_location='cpu',
                                 weights_only=False)['model_state_dict']
        for a in config['wise_alphas']:
            save_checkpoint(f"{prefix}_wise{a:.2f}.pth",
                            wise_interpolate(base_sd, sd_imagenet, a), config,
                            extra={'wise_alpha': a, 'wise_base_step': best_step,
                                   'pool_report': pool_report})

    # ---- log ----------------------------------------------------------------
    log = {'run': run, 'config': {k: v for k, v in config.items()
                                  if not k.startswith('_')},
           'pool_report': pool_report, 'best_val': best_val,
           'best_step': best_step, 'minutes': minutes,
           'decoder_from_imagenet': decoder_from_imagenet,
           'decoder_warmup_steps': dec_warmup,
           'base_lr': config['_lr'], 'history': history}
    with open(f"{prefix}_log.json", 'w') as f:
        json.dump(log, f, indent=2, default=str)

    # ---- tidy up -------------------------------------------------------------
    # Runs last, because WiSE above is built from the 'best' checkpoint even
    # when 'best' itself is not being kept.
    cleanup_artefacts(prefix, config)

    print(f"\n[OK] Protocol {config['protocol']} complete.")
    print(f"  best val loss {best_val:.4f} at step {best_step} "
          f"(of {config['total_steps']})")

    # List what actually survived, so the paths printed can always be used.
    folder, stem = Path(prefix).parent, Path(prefix).name
    kept = sorted(p for p in folder.glob(f"{stem}_*") if p.is_file())
    total = sum(p.stat().st_size for p in kept) / 1024 ** 3
    print(f"\n  Kept {len(kept)} file(s), {total:.2f} GB:")
    for p in kept:
        note = "   <- feed this to stage2_finetune_landmark.py" \
            if p.name.endswith('_best.pth') else ""
        print(f"    {p}{note}")


if __name__ == '__main__':
    main()
