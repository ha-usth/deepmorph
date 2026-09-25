"""
e2_finetune_eval.py - EXPERIMENT 2: Stage 2 + Stage 3 under the Stage 1 protocols

Takes the checkpoints produced by e1_pretrain_mae.py (PC/PA/PB), runs
stage2_finetune_landmark.py then stage3_predict.py on a target dataset, and
reports two metrics as mean +/- std over repeated runs:

    MRE  mean Euclidean error in pixels at the original resolution, the metric
         of the 2022 iMorph paper.

    NME  the same errors normalised by each image's own diagonal, in percent:

             NME = (100 / (N*K)) * sum_i sum_k ||p_hat_ik - p_ik|| / diag_i
             diag_i = sqrt(W_i^2 + H_i^2)

         Because the normaliser is per image, NME is comparable between
         datasets; MRE in pixels is not, the resolutions here running from
         800x331 (sea_bass) to 1935x2400 (cepha).

Data layout, per dataset (droso_small shown):

    ./train_pool/droso_small   38 labelled images
                               -> N are drawn for few-shot training
                               -> the remaining 38-N act as the VALIDATION set
                                  that Stage 2 uses to pick its best checkpoint
    ./test_set/droso_small     99 labelled images, never seen by Stage 1 or 2
                               -> the TEST set Stage 3 is scored on

Several datasets can be evaluated in one invocation. Each gets its own output
directory and its own num_landmarks, declared in CONFIG['datasets'] -- the
value differs per dataset (15 for droso_small, 11 for sea_bass, 19 for cepha)
and a wrong one silently produces nonsense rather than an error.

A dataset entry may also override any other CONFIG key for itself, 'n_shots'
in particular: the labelled pools run from 10 images (fly) to 150 (cepha), so
a single N cannot serve all of them. Precedence is

    command-line flag  >  CONFIG['datasets'][name]  >  CONFIG default

and any N that would leave Stage 2 without a validation set is dropped with a
message instead of failing halfway through the run.

WHY EACH REPEAT RE-RUNS STAGE 2
    Stage 3 is deterministic: same checkpoint, same images, same predictions.
    Running it ten times over one Stage 2 model would give a standard deviation
    of exactly zero. The variance worth reporting comes from which N images are
    drawn and from the fine-tuning itself, so repeat i runs Stage 2 with seed i
    and then Stage 3 on its result. That matches the "mean +/- std over
    independent seeds" protocol in Section 3.3 of the paper.

Usage:
    python e2_finetune_eval.py                       # droso_small, 10 repeats
    python e2_finetune_eval.py --dataset droso_small sea_bass cepha
    python e2_finetune_eval.py --dataset sea_bass --protocols PC PA
    python e2_finetune_eval.py --ckpt ./checkpoints/mae_PA_s0_best.pth --label PA
    python e2_finetune_eval.py --n-shots 1 3 6 11 20 --repeats 3
    python e2_finetune_eval.py --force            # ignore cached results, rerun

Each dataset needs its own PB checkpoint, since PB is defined by holding that
dataset out of the Stage 1 pool:

    python e1_pretrain_mae.py --protocol PB --exclude ./train_pool/sea_bass
    python e1_pretrain_mae.py --protocol PB --exclude ./train_pool/cepha
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import gc
import csv
import json
import math
import time
import shutil
import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_finetune_landmark as s2
import stage2_finetune_landmark_reg as s2reg
import stage3_predict as s3
from utils import compute_MRE

# Prediction heads for the Table 3 ablation. Each entry is the Stage 2 module
# and the filename it writes its best checkpoint to; Stage 3 auto-detects which
# head a checkpoint carries from its stored 'head_type'.
HEADS = {
    'heatmap':    {'module': s2, 'stem': 'finetune_best'},
    'regression': {'module': s2reg, 'stem': 'finetune_reg_best'},
}


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ============================== CONFIG ==============================
CONFIG = {
    # =========================================================================
    # TARGET DATASETS
    # num_landmarks differs per dataset and a wrong value silently produces
    # nonsense, so each entry states its own. Add a row to evaluate a new one.
    #
    # Any other CONFIG key may be repeated inside an entry to override the
    # global value for that dataset only -- 'n_shots' above all, since the
    # labelled pools range from 10 to 150 images and one N does not fit all.
    # An entry that says nothing inherits the global. A command-line flag beats
    # both. The comment on each row is how many labelled images train_dir holds;
    # n_shots must stay below it or Stage 2 is left with no validation set.
    # =========================================================================
    'datasets': {
        'droso_big':   {'train_dir': './train_pool/droso_big',        # 34
                        'test_dir': './test_set/droso_big',
                        'num_landmarks': 15,
                        'n_shots': [15]}, 
        'droso_small': {'train_dir': './train_pool/droso_small',      # 38
                        'test_dir': './test_set/droso_small',
                        'num_landmarks': 15,
                        'n_shots': [15]},  
        'fly':         {'train_dir': './train_pool/fly',              # 10
                        'test_dir': './test_set/fly',
                        'num_landmarks': 10,
                        'n_shots': [8]},     
        'bactro':      {'train_dir': './train_pool/bactro',           # 25
                        'test_dir': './test_set/bactro',
                        'num_landmarks': 12,
                        'n_shots': [10]},    
        'diacha':      {'train_dir': './train_pool/diacha',           # 0 (!)
                        'test_dir': './test_set/diacha',
                        'num_landmarks': 10,
                        'n_shots': [15]},    
        'tsetse':      {'train_dir': './train_pool/tsetse',           # 51
                        'test_dir': './test_set/tsetse',
                        'num_landmarks': 11,
                        'n_shots': [15]},    
        'droso-281':   {'train_dir': './train_pool/droso-281',        # 57
                        'test_dir': './test_set/droso-281',
                        'num_landmarks': 12,
                        'n_shots': [15]},    
        'sea_bass':    {'train_dir': './train_pool/sea_bass',         # 24
                        'test_dir': './test_set/sea_bass',
                        'num_landmarks': 11,
                        'n_shots': [20]},     # 15 would leave only 9 to validate
        'cepha':       {'train_dir': './train_pool/cepha',            # 150
                        'test_dir': './test_set/cepha',
                        'num_landmarks': 19,
                        'n_shots': [20]},     # 15 would leave only 10 to validate
    },
    'run_datasets': [
                    'droso_small',
                     'droso_big',
                     'fly',
                     'bactro',
                     'diacha',
                     'tsetse',
                     'droso-281',
                    'sea_bass',
                    'cepha'
                     ],   # default selection; --dataset overrides

    # =========================================================================
    # STAGE 1 PROTOCOLS -> checkpoint produced by e1_pretrain_mae.py
    # '{dataset}' is substituted per target. PB needs it because its pool is
    # defined by which dataset was held out; PC and PA share one pool.
    # Missing files are skipped with a warning, so you can start with whichever
    # protocols you have already pretrained.
    # =========================================================================
    'protocols': {
        'PC': './checkpoints/mae_PC_s0_imagenet_only.pth',
        'PA': './checkpoints/mae_PA_s0_best.pth',
        'PB': './checkpoints/mae_PB_{dataset}_s0_best.pth',
        'PS': './checkpoints/mae_PS_s0_best.pth',
    },

    # =========================================================================
    # PREDICTION HEAD -- the other axis of the Table 3 ablation
    # 'heatmap' is the framework; 'regression' is the coordinate-regression
    # variant it is compared against. Both can run in one invocation.
    # =========================================================================
    'heads': ['heatmap'],

    # =========================================================================
    # REPEATS
    # =========================================================================
    'n_shots': [15],              # default; a dataset entry may override it
    'repeats': 10,                 # seeds 0..repeats-1
    'seed_base': 0,

    # =========================================================================
    # STAGE 2 HYPERPARAMETERS (mirrors stage2_finetune_landmark.CONFIG)
    # =========================================================================
    'pretrain_image_size': 224,
    'finetune_image_size': 512,
    'heatmap_size': 128,
    'sigma': 3,
    'patch_size': 16,
    'embed_dim': 768,
    'model_size': 'base',
    'batch_size': 2,
    'gradient_accumulation_steps': 4,
    'lr_head': 5e-4,
    'lr_encoder': 5e-6,
    'weight_decay': 0.05,
    'epochs': 300,
    'freeze_encoder_epochs': 30,
    'eval_every': 10,
    'use_tta': True,

    # =========================================================================
    # OUTPUT
    # =========================================================================
    'out_dir': './results/e2_{dataset}',   # '{dataset}' substituted per target
    'keep_checkpoints': False,     # True keeps ~450MB per run
    'quiet_stage2': True,          # Stage 2 chatter goes to a per-run log file
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


# ============================== HELPERS ==============================
def run_key(protocol, n_shots, seed, head='heatmap'):
    # The heatmap key keeps its original shape so results.json files written
    # before the head axis existed still resume.
    tail = '' if head == 'heatmap' else f"|{head}"
    return f"{protocol}|n{n_shots}|s{seed}{tail}"


def run_tag(protocol, n_shots, seed, head='heatmap'):
    """Filename-safe run identifier; also unchanged for the heatmap head."""
    tail = '' if head == 'heatmap' else f"_{head}"
    return f"{protocol}_n{n_shots}_s{seed}{tail}"


def read_landmark_txt(path, num_landmarks):
    """Read an iMorph .txt file. Returns None when unusable."""
    pts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    pts.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    return None
    if len(pts) < num_landmarks:
        return None
    return np.array(pts[:num_landmarks], dtype=np.float32)


def score_predictions(pred_dir, gt_dir, num_landmarks):
    """
    Score the .txt files Stage 3 actually wrote against the ground truth.

    Reading the artefacts back (rather than parsing Stage 3's stdout) checks the
    exact files a user of the released tool would get.

    Two metrics come out of the same distances:

      MRE  mean Euclidean error in pixels at the original resolution. Comparable
           with the 2022 iMorph paper, but not across datasets, whose
           resolutions run from 800x331 (sea_bass) to 1935x2400 (cepha).

      NME  the same errors divided by each image's OWN diagonal
           sqrt(W_i^2 + H_i^2), averaged over landmarks and images, in percent:

               NME = (100 / (N*K)) * sum_i sum_k ||p_hat_ik - p_ik|| / diag_i

           Normalising per image, not by a dataset-wide constant, is what makes
           it comparable across datasets and across mixed-resolution sets.
    """
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    preds, gts, diagonals, missing = [], [], [], []

    for img in sorted(p for p in gt_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS):
        gt_path = img.with_suffix('.txt')
        pred_path = pred_dir / (img.stem + '.txt')
        if not gt_path.exists():
            continue
        if not pred_path.exists():
            missing.append(img.name)
            continue
        gt = read_landmark_txt(gt_path, num_landmarks)
        pr = read_landmark_txt(pred_path, num_landmarks)
        if gt is None or pr is None:
            missing.append(img.name)
            continue
        with Image.open(img) as im:          # header only, pixels never decoded
            w, h = im.size
        gts.append(gt)
        preds.append(pr)
        diagonals.append(math.hypot(w, h))

    if not preds:
        raise RuntimeError(f"No prediction/GT pair could be scored in {pred_dir}")

    pred_t = torch.from_numpy(np.stack(preds))
    gt_t = torch.from_numpy(np.stack(gts))
    per_lm, overall = compute_MRE(pred_t, gt_t)

    dist = torch.sqrt(((pred_t - gt_t) ** 2).sum(dim=-1))          # (N, K) px
    diag = torch.tensor(diagonals, dtype=dist.dtype).unsqueeze(1)  # (N, 1)
    nme = 100.0 * dist / diag                                      # (N, K) %

    return {
        'mre': float(overall),
        'mre_per_lm': [float(v) for v in per_lm],
        'nme': float(nme.mean()),
        'nme_per_lm': [float(v) for v in nme.mean(dim=0)],
        'n_images': len(preds),
        'n_unscored': len(missing),
    }


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def tee_stdout(path, quiet):
    """Send a stage's prints to a log file when quiet; tqdm uses stderr anyway."""
    if not quiet:
        yield
        return
    with open(path, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
        yield


# ============================== STAGE 2 / 3 ==============================
def run_stage2(protocol, mae_ckpt, n_shots, seed, config, paths, head='heatmap'):
    """
    Fine-tune on n_shots images of train_dir and return the best checkpoint.

    Stage 2 always writes a fixed filename, so the file is removed first (a
    stale one would be mistaken for this run's result) and renamed to a unique
    path afterwards.

    The regression module ignores the heatmap-only keys it is handed
    (heatmap_size, sigma, use_tta); setting them is harmless and keeps one code
    path for both heads.
    """
    mod = HEADS[head]['module']
    mod.CONFIG.update({
        'mae_checkpoint': mae_ckpt,
        'target_data_dir': config['train_dir'],
        'n_shots': n_shots,
        'num_landmarks': config['num_landmarks'],
        'seed': seed,
        'save_dir': str(paths['ckpt']),
        'pretrain_image_size': config['pretrain_image_size'],
        'finetune_image_size': config['finetune_image_size'],
        'heatmap_size': config['heatmap_size'],
        'sigma': config['sigma'],
        'patch_size': config['patch_size'],
        'embed_dim': config['embed_dim'],
        'model_size': config['model_size'],
        'batch_size': config['batch_size'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'lr_head': config['lr_head'],
        'lr_encoder': config['lr_encoder'],
        'weight_decay': config['weight_decay'],
        'epochs': config['epochs'],
        'freeze_encoder_epochs': config['freeze_encoder_epochs'],
        'eval_every': config['eval_every'],
        'use_tta': config['use_tta'],
        'device': config['device'],
    })

    tag = run_tag(protocol, n_shots, seed, head)
    default = paths['ckpt'] / (f"{HEADS[head]['stem']}_n{n_shots}"
                               f"_size{config['finetune_image_size']}.pth")
    if default.exists():
        default.unlink()

    log = paths['logs'] / f"stage2_{tag}.log"
    with tee_stdout(log, config['quiet_stage2']):
        mod.main()

    if not default.exists():
        raise RuntimeError(
            f"Stage 2 ({head}) produced no checkpoint for {protocol} "
            f"n={n_shots} seed={seed}. MRE never improved, or the validation "
            f"set was empty. See {log}")

    unique = paths['ckpt'] / f"ft_{config['dataset']}_{tag}.pth"
    shutil.move(str(default), str(unique))

    val_mre = torch.load(unique, map_location='cpu',
                         weights_only=False).get('MRE')
    return unique, (float(val_mre) if val_mre is not None else None)


def run_stage3(ft_ckpt, protocol, n_shots, seed, config, paths, head='heatmap'):
    """
    Predict on the held-out test set, writing iMorph .txt files.

    Stage 3 reads 'head_type' out of the checkpoint itself, so the same call
    serves both heads.
    """
    pred_dir = paths['pred'] / run_tag(protocol, n_shots, seed, head)
    pred_dir.mkdir(parents=True, exist_ok=True)

    s3.CONFIG.update({
        'finetune_checkpoint': str(ft_ckpt),
        'input_dir': config['test_dir'],
        'output_dir': str(pred_dir),
        'image_size': config['finetune_image_size'],
        'heatmap_size': config['heatmap_size'],
        'patch_size': config['patch_size'],
        'embed_dim': config['embed_dim'],
        'num_landmarks': config['num_landmarks'],
        'model_size': config['model_size'],
        'device': config['device'],
    })

    log = paths['logs'] / f"stage3_{run_tag(protocol, n_shots, seed, head)}.log"
    with tee_stdout(log, config['quiet_stage2']):
        s3.main()

    return pred_dir


# ============================== AGGREGATION ==============================
def _stats(values, prefix, num_landmarks, per_lm_values):
    """mean/std/min/max/median for one metric, or Nones when it is absent."""
    if not len(values):
        return {f'{prefix}_{k}': None
                for k in ('mean', 'std', 'sem', 'min', 'max', 'median',
                          'per_lm_mean', 'per_lm_std')}
    v = np.asarray(values, dtype=np.float64)
    lm = np.asarray(per_lm_values, dtype=np.float64)
    # Sample standard deviation (ddof=1) of the per-seed scores, plus the
    # standard error of their mean. A single run has no dispersion to estimate,
    # so both stay None: reporting 0.00 there would read as a perfectly
    # repeatable measurement rather than as one unrepeated run.
    multi = len(v) > 1
    return {
        f'{prefix}_mean': float(v.mean()),
        f'{prefix}_std': float(v.std(ddof=1)) if multi else None,
        f'{prefix}_sem': float(v.std(ddof=1) / math.sqrt(len(v))) if multi else None,
        f'{prefix}_min': float(v.min()),
        f'{prefix}_max': float(v.max()),
        f'{prefix}_median': float(np.median(v)),
        f'{prefix}_per_lm_mean': lm.mean(axis=0).tolist(),
        f'{prefix}_per_lm_std': (lm.std(axis=0, ddof=1).tolist()
                                 if multi else None),
    }


def summarise(rows, num_landmarks):
    """
    mean / std (ddof=1) of MRE and NME per (protocol, n_shots), over seeds.

    NME is tolerated as missing: rows written before the metric existed keep
    their MRE and simply report no NME.
    """
    groups = {}
    for r in rows:
        groups.setdefault((r['protocol'], r.get('head', 'heatmap'),
                           r['n_shots']), []).append(r)

    out = []
    for (protocol, head, n_shots), rs in sorted(
            groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        with_nme = [r for r in rs if r.get('nme') is not None]
        vals = [r['val_mre'] for r in rs if r.get('val_mre') is not None]
        entry = {
            'protocol': protocol,
            'head': head,
            'n_shots': n_shots,
            'runs': len(rs),
            'nme_runs': len(with_nme),
            'val_mre_mean': float(np.mean(vals)) if vals else None,
            'val_mre_scale': rs[0].get('val_mre_scale', 'original'),
            'seeds': sorted(r['seed'] for r in rs),
        }
        entry.update(_stats([r['mre'] for r in rs], 'mre', num_landmarks,
                            [r['mre_per_lm'] for r in rs]))
        entry.update(_stats([r['nme'] for r in with_nme], 'nme', num_landmarks,
                            [r['nme_per_lm'] for r in with_nme]))
        out.append(entry)
    return out


def _cell(s, prefix, digits=2):
    """
    'mean +/- std' for one metric.

    '-' when the metric is absent, and 'mean (1 run)' when there is a single
    seed, since no standard deviation exists to report there.
    """
    mean = s.get(f'{prefix}_mean')
    if mean is None:
        return '-'
    std = s.get(f'{prefix}_std')
    if std is None:
        return f"{mean:.{digits}f} (1 run)"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _per_landmark_table(summary, prefix, title, digits=1):
    rows = [s for s in summary if s.get(f'{prefix}_per_lm_mean')]
    if not rows:
        return
    print(f"\nPer-landmark {title} (mean over runs)")
    k = len(rows[0][f'{prefix}_per_lm_mean'])
    print(f"{'protocol':<14}{'head':<12}{'N':>4}"
          + "".join(f"{i + 1:>7}" for i in range(k)))
    for s in rows:
        cells = "".join(f"{v:>7.{digits}f}" for v in s[f'{prefix}_per_lm_mean'])
        print(f"{s['protocol']:<14}{s.get('head', 'heatmap'):<12}"
              f"{s['n_shots']:>4}{cells}")


def print_summary(summary, config):
    print("\n" + "=" * 86)
    print(f"Results on {config['test_dir']}  (lower is better)")
    print("=" * 86)
    print(f"{'protocol':<14}{'head':<12}{'N':>4}{'runs':>6}"
          f"{'MRE px mean +/- std':>24}{'NME % mean +/- std':>24}{'val':>8}")
    print("-" * 86)
    odd_scale = False
    for s in summary:
        if s['val_mre_mean'] is None:
            val = "-"
        else:
            mark = '' if s.get('val_mre_scale', 'original') == 'original' else '*'
            odd_scale |= bool(mark)
            val = f"{s['val_mre_mean']:.2f}{mark}"
        print(f"{s['protocol']:<14}{s.get('head', 'heatmap'):<12}"
              f"{s['n_shots']:>4}{s['runs']:>6}"
              f"{_cell(s, 'mre'):>24}{_cell(s, 'nme', 3):>24}{val:>8}")
    print("-" * 86)
    if odd_scale:
        print("* this val MRE is at the fine-tuning resolution, not the "
              "original one:\n  the regression Stage 2 does not rescale. Do not "
              "compare it across heads.\n  MRE and NME above are unaffected -- "
              "both are recomputed from the Stage 3 output.")
    print("MRE = pixels at original resolution; comparable with the 2022 paper,")
    print("      but not across datasets of different resolution.")
    print("NME = percent of the image diagonal; comparable across datasets.")
    print("val MRE = Stage 2's own model-selection score on the held-out part "
          "of train_dir")

    # Per-landmark breakdown feeds Section 4.3.3 of the paper.
    _per_landmark_table(summary, 'mre', 'MRE (px)', digits=1)
    _per_landmark_table(summary, 'nme', 'NME (%)', digits=2)


def write_outputs(rows, summary, config, paths):
    with open(paths['out'] / 'results.json', 'w') as f:
        json.dump({'config': config, 'runs': rows, 'summary': summary},
                  f, indent=2, default=str)

    with open(paths['out'] / 'runs.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'protocol', 'head', 'n_shots', 'seed', 'mre',
                    'nme', 'val_mre', 'n_images', 'minutes', 'mae_checkpoint'])
        for r in rows:
            w.writerow([r.get('dataset', ''), r['protocol'],
                        r.get('head', 'heatmap'), r['n_shots'],
                        r['seed'], f"{r['mre']:.4f}",
                        f"{r['nme']:.4f}" if r.get('nme') is not None else '',
                        f"{r['val_mre']:.4f}" if r.get('val_mre') else '',
                        r['n_images'], f"{r['minutes']:.1f}",
                        r['mae_checkpoint']])

    with open(paths['out'] / 'summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['protocol', 'head', 'n_shots', 'runs',
                    'runs',
                    'mre_mean', 'mre_std', 'mre_sem', 'mre_min', 'mre_max',
                    'mre_median',
                    'nme_mean', 'nme_std', 'nme_sem', 'nme_min', 'nme_max',
                    'nme_median',
                    'val_mre_mean'])
        for s in summary:
            def f4(key):
                v = s.get(key)
                return f"{v:.4f}" if v is not None else ''
            w.writerow([s['protocol'], s.get('head', 'heatmap'),
                        s['n_shots'], s['runs'],
                        s['runs'],
                        f4('mre_mean'), f4('mre_std'), f4('mre_sem'),
                        f4('mre_min'), f4('mre_max'), f4('mre_median'),
                        f4('nme_mean'), f4('nme_std'), f4('nme_sem'),
                        f4('nme_min'), f4('nme_max'), f4('nme_median'),
                        f4('val_mre_mean')])

    print(f"\nWritten to {paths['out']}:")
    print(f"  results.json   full records, resumable")
    print(f"  runs.csv       one row per run")
    print(f"  summary.csv    mean +/- std per protocol  <- Table 3 material")


# ============================== MAIN ==============================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', nargs='*', default=None,
                   help="target dataset(s) from CONFIG['datasets'], "
                        "e.g. droso_small sea_bass cepha")
    p.add_argument('--protocols', nargs='*', default=None,
                   help="subset of CONFIG['protocols'] to run, e.g. PC PA")
    p.add_argument('--ckpt', default=None,
                   help="run one arbitrary Stage 1 checkpoint instead")
    p.add_argument('--label', default=None,
                   help="name for --ckpt (default: the file stem)")
    p.add_argument('--head', nargs='*', choices=sorted(HEADS), default=None,
                   help="prediction head(s): heatmap (the framework) and/or "
                        "regression (the Table 3 ablation)")
    p.add_argument('--n-shots', nargs='*', type=int, default=None)
    p.add_argument('--repeats', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--out-dir', default=None,
                   help="output directory template; may contain '{dataset}'")
    p.add_argument('--keep-checkpoints', action='store_true')
    p.add_argument('--verbose', action='store_true',
                   help="let Stage 2/3 print to the console instead of a log")
    p.add_argument('--force', action='store_true',
                   help="recompute the selected runs even if results.json "
                        "already has them; rows outside the selection are kept")
    return p.parse_args()


def apply_args(config, args):
    # Copy each entry: a command-line override edits them, and CONFIG is module
    # state that a second call in the same process would otherwise inherit.
    config['datasets'] = {k: dict(v) for k, v in config['datasets'].items()}

    if args.ckpt:
        label = args.label or Path(args.ckpt).stem
        config['protocols'] = {label: args.ckpt}
    elif args.protocols:
        missing = [p for p in args.protocols if p not in config['protocols']]
        if missing:
            raise SystemExit(f"unknown protocol(s): {missing}. "
                             f"Known: {sorted(config['protocols'])}")
        config['protocols'] = {p: config['protocols'][p] for p in args.protocols}

    if args.dataset:
        missing = [d for d in args.dataset if d not in config['datasets']]
        if missing:
            raise SystemExit(f"unknown dataset(s): {missing}. Known: "
                             f"{sorted(config['datasets'])}. Add an entry to "
                             f"CONFIG['datasets'] with its num_landmarks.")
        config['run_datasets'] = args.dataset

    if args.head:
        config['heads'] = args.head

    # An explicit flag beats both the global default and any per-dataset value,
    # so the per-dataset one is dropped rather than left to win in
    # resolve_dataset().
    for key, val in (('n_shots', args.n_shots), ('repeats', args.repeats),
                     ('epochs', args.epochs)):
        if val:
            config[key] = val
            for spec in config['datasets'].values():
                spec.pop(key, None)
    if args.out_dir:
        config['out_dir'] = args.out_dir
    if args.keep_checkpoints:
        config['keep_checkpoints'] = True
    if args.verbose:
        config['quiet_stage2'] = False
    return config


def resolve_dataset(name, config):
    """Flatten the registry entry for one dataset into a runnable config."""
    dcfg = dict(config)
    dcfg.update(config['datasets'][name])
    dcfg['dataset'] = name
    dcfg['out_dir'] = config['out_dir'].format(dataset=name)
    dcfg['protocols'] = {label: path.format(dataset=name)
                         for label, path in config['protocols'].items()}
    return dcfg


def run_dataset(name, base_config, args):
    """Run every protocol x n_shots x seed for one target dataset."""
    config = resolve_dataset(name, base_config)

    out = Path(config['out_dir'])
    paths = {'out': out, 'ckpt': out / 'ckpt', 'pred': out / 'pred',
             'logs': out / 'logs'}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ---- sanity checks on the data layout -----------------------------------
    train_dir, test_dir = Path(config['train_dir']), Path(config['test_dir'])
    for d in (train_dir, test_dir):
        if not d.exists():
            print(f"[skip] {name}: missing directory {d}")
            return [], []
    n_train = len([p for p in train_dir.iterdir()
                   if p.suffix.lower() in IMG_EXTS
                   and p.with_suffix('.txt').exists()])
    n_test = len([p for p in test_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS
                  and p.with_suffix('.txt').exists()])

    # ---- drop infeasible n_shots --------------------------------------------
    # Stage 2 picks its checkpoint on the images left over after the N draws, so
    # N must stay strictly below the labelled count. Filtering here rather than
    # warning means a bad N costs nothing instead of failing mid-run.
    feasible = [n for n in config['n_shots'] if n < n_train]
    for n in config['n_shots']:
        if n not in feasible:
            print(f"[skip] {name} n_shots={n}: only {n_train} labelled image(s) "
                  f"in {train_dir}, leaving no validation set")
    if not feasible:
        print(f"[skip] {name}: no usable n_shots "
              f"(set one below {n_train} in CONFIG['datasets']['{name}'])")
        return [], []
    config['n_shots'] = feasible

    # ---- resolve protocols ---------------------------------------------------
    available = {}
    for label, ckpt in config['protocols'].items():
        if Path(ckpt).exists():
            available[label] = ckpt
        else:
            print(f"[skip] {label}: checkpoint not found -> {ckpt}")
    if not available:
        print(f"[skip] {name}: no Stage 1 checkpoint found")
        return [], []
    config['protocols'] = available

    seeds = [config['seed_base'] + i for i in range(config['repeats'])]
    total = (len(available) * len(config['heads']) *
             len(config['n_shots']) * len(seeds))

    print(f"\n=== {name} ===")
    print(f"  Landmarks  : {config['num_landmarks']}")
    print(f"  Train pool : {train_dir}  ({n_train} labelled)")
    print(f"  Test set   : {test_dir}  ({n_test} labelled)")
    print(f"  Protocols  : {', '.join(available)}")
    print(f"  Heads      : {', '.join(config['heads'])}")
    print(f"  N-shots    : {config['n_shots']}")
    print(f"  Repeats    : {len(seeds)} (seeds {seeds[0]}..{seeds[-1]})")
    print(f"  Finetune   : {config['finetune_image_size']}px, "
          f"{config['epochs']} epochs, TTA={config['use_tta']}")
    print(f"  Total runs : {total}")
    print()

    # Stage 2 only saves a checkpoint inside its periodic eval, so an
    # eval_every larger than epochs would finish having written nothing.
    if config['eval_every'] > config['epochs']:
        config['eval_every'] = max(1, config['epochs'] // 2)
        print(f"[warn] eval_every > epochs; clamped to {config['eval_every']} "
              f"so Stage 2 still evaluates and saves.")

    # ---- resume --------------------------------------------------------------
    rows, done = [], set()
    results_file = out / 'results.json'
    if results_file.exists():
        try:
            prev = json.load(open(results_file))
            rows = prev.get('runs', [])

            if args.force:
                # Invalidate only the cells this invocation is about to
                # recompute. results.json holds every protocol/head/shot count
                # for the dataset, so dropping all of it would throw away work
                # the command was never asked to touch.
                def targeted(r):
                    return (r['protocol'] in available
                            and r.get('head', 'heatmap') in config['heads']
                            and r['n_shots'] in config['n_shots']
                            and r['seed'] in seeds)

                keep = [r for r in rows if not targeted(r)]
                dropped = len(rows) - len(keep)
                rows = keep
                print(f"--force: discarding {dropped} cached run(s) matching "
                      f"this selection, keeping {len(rows)} other(s)")

            done = {run_key(r['protocol'], r['n_shots'], r['seed'],
                            r.get('head', 'heatmap')) for r in rows}
            if done and not args.force:
                print(f"Resuming: {len(done)} run(s) already in {results_file}")

            # Rows stored before NME existed: rescore them from the prediction
            # .txt files, which are still on disk and cost no GPU time. Without
            # this they would report '-' forever unless rerun with --force.
            stale = [r for r in rows if r.get('nme') is None]
            fixed = 0
            for r in stale:
                pdir = paths['pred'] / run_tag(r['protocol'], r['n_shots'],
                                               r['seed'],
                                               r.get('head', 'heatmap'))
                if not pdir.exists():
                    continue
                try:
                    s = score_predictions(pdir, test_dir,
                                          config['num_landmarks'])
                except Exception:
                    continue
                r['nme'], r['nme_per_lm'] = s['nme'], s['nme_per_lm']
                fixed += 1
            if stale:
                print(f"Backfilled NME for {fixed}/{len(stale)} cached run(s)")
            print()
        except Exception as e:
            print(f"[warn] could not read {results_file}: {e}")

    # ---- run -----------------------------------------------------------------
    idx = 0
    for hd in config['heads']:
        for label, mae_ckpt in available.items():
            for n_shots in config['n_shots']:
                for seed in seeds:
                    idx += 1
                    key = run_key(label, n_shots, seed, hd)
                    line = (f"[{idx}/{total}] {label} {hd} n={n_shots} "
                            f"seed={seed}")
                    if key in done:
                        print(f"{line} -> cached, skipped")
                        continue

                    t0 = time.time()
                    # Full line, not end='': Stage 3's tqdm writes to stderr
                    # and would otherwise overwrite a partial line here.
                    print(line, flush=True)
                    try:
                        ft_ckpt, val_mre = run_stage2(label, mae_ckpt, n_shots,
                                                      seed, config, paths, hd)
                        free_gpu()
                        pred_dir = run_stage3(ft_ckpt, label, n_shots, seed,
                                              config, paths, hd)
                        free_gpu()
                        score = score_predictions(pred_dir, test_dir,
                                                  config['num_landmarks'])
                    except Exception as e:
                        print(f"    FAILED: {type(e).__name__}: {e}")
                        free_gpu()
                        continue

                    minutes = (time.time() - t0) / 60
                    # Both Stage 2 modules now report their model-selection MRE
                    # at the original resolution. Rows written before the
                    # regression script was fixed carry the old scale, so the
                    # field stays rather than being assumed.
                    row = {'dataset': name, 'protocol': label, 'head': hd,
                           'n_shots': n_shots, 'seed': seed,
                           'mae_checkpoint': mae_ckpt,
                           'finetune_checkpoint': str(ft_ckpt),
                           'val_mre': val_mre, 'val_mre_scale': 'original',
                           'minutes': minutes, **score}
                    rows.append(row)
                    done.add(key)

                    extra = (f", {score['n_unscored']} unscored"
                             if score['n_unscored'] else "")
                    print(f"    -> MRE {score['mre']:.2f} px | "
                          f"NME {score['nme']:.3f} % on "
                          f"{score['n_images']} images{extra}  "
                          f"({minutes:.1f} min)")

                    if not config['keep_checkpoints']:
                        Path(ft_ckpt).unlink(missing_ok=True)

                    # Persist after every run so a crash loses no finished work.
                    with open(results_file, 'w') as f:
                        json.dump({'config': config, 'runs': rows,
                                   'summary': summarise(
                                       rows, config['num_landmarks'])},
                                  f, indent=2, default=str)

    if not rows:
        print(f"[skip] {name}: no run completed")
        return [], []

    summary = summarise(rows, config['num_landmarks'])
    for s in summary:
        s['dataset'] = name
    print_summary(summary, config)
    write_outputs(rows, summary, config, paths)
    return rows, summary


def print_combined(all_summary):
    """One table across every dataset, the shape Table 3 needs."""
    print("\n" + "=" * 92)
    print("ALL DATASETS - mean +/- std over seeds")
    print("=" * 92)
    print(f"{'dataset':<14}{'protocol':<14}{'head':<12}{'N':>4}{'runs':>6}"
          f"{'MRE px':>22}{'NME %':>22}")
    print("-" * 92)
    last = None
    for s in sorted(all_summary,
                    key=lambda s: (s['dataset'], s.get('head', 'heatmap'),
                                   s['n_shots'], s['protocol'])):
        label = s['dataset'] if s['dataset'] != last else ''
        last = s['dataset']
        print(f"{label:<14}{s['protocol']:<14}{s.get('head', 'heatmap'):<12}"
              f"{s['n_shots']:>4}{s['runs']:>6}"
              f"{_cell(s, 'mre'):>22}{_cell(s, 'nme', 3):>22}")
    print("-" * 92)
    print("Only NME is comparable between rows of different datasets: MRE is in "
          "pixels,\nand the resolutions differ.")


def main():
    args = parse_args()
    config = apply_args(dict(CONFIG), args)

    names = config['run_datasets']
    unknown = [n for n in names if n not in config['datasets']]
    if unknown:
        raise SystemExit(
            f"run_datasets names {unknown}, which is not in CONFIG['datasets'] "
            f"(active entries: {sorted(config['datasets'])}). Uncomment the "
            f"entry, or set run_datasets / pass --dataset.")

    print("=== Experiment 2: Stage 2 + Stage 3 ===")
    print(f"  Datasets   : {', '.join(names)}")
    print(f"  Protocols  : {', '.join(config['protocols'])}")
    print(f"  N-shots    : {config['n_shots']}  (default)")
    overrides = {n: config['datasets'][n]['n_shots']
                 for n in names if 'n_shots' in config['datasets'][n]}
    if overrides:
        print("               overridden: " +
              ", ".join(f"{k}={v}" for k, v in overrides.items()))
    print(f"  Repeats    : {config['repeats']}")
    print(f"  Finetune   : {config['finetune_image_size']}px, "
          f"{config['epochs']} epochs, TTA={config['use_tta']}")

    all_rows, all_summary = [], []
    for name in names:
        rows, summary = run_dataset(name, config, args)
        all_rows.extend(rows)
        all_summary.extend(summary)

    if not all_summary:
        raise SystemExit("No run completed on any dataset.")

    if len(names) > 1:
        print_combined(all_summary)
        combined = Path(config['out_dir'].format(dataset='all')).parent
        combined.mkdir(parents=True, exist_ok=True)
        with open(combined / 'e2_all_datasets.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'protocol', 'head', 'n_shots', 'runs',
                        'mre_mean', 'mre_std', 'mre_sem', 'mre_min',
                        'mre_max',
                        'nme_mean', 'nme_std', 'nme_sem', 'nme_min',
                        'nme_max',
                        'val_mre_mean'])
            for s in sorted(all_summary, key=lambda s: (s['dataset'],
                                                        s['n_shots'],
                                                        s['protocol'])):
                def f4(key):
                    v = s.get(key)
                    return f"{v:.4f}" if v is not None else ''
                w.writerow([s['dataset'], s['protocol'],
                            s.get('head', 'heatmap'), s['n_shots'],
                            s['runs'],
                            f4('mre_mean'), f4('mre_std'), f4('mre_sem'),
                            f4('mre_min'), f4('mre_max'),
                            f4('nme_mean'), f4('nme_std'), f4('nme_sem'),
                            f4('nme_min'), f4('nme_max'),
                            f4('val_mre_mean')])
        print(f"\nCombined table: {combined / 'e2_all_datasets.csv'}")


if __name__ == '__main__':
    main()
