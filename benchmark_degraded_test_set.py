"""Evaluate one frozen fine-tuned checkpoint on clean and degraded test images.

Run in the same Python environment as Stage 3. Examples (one dataset per run):
    python benchmark_degraded_test_set.py --dataset droso_small --checkpoint PATH_TO_PA_FINETUNED.pth
    python benchmark_degraded_test_set.py --dataset sea_bass --checkpoint PATH_TO_PA_FINETUNED.pth --corruptions gaussian_noise gaussian_blur illumination
    python benchmark_degraded_test_set.py --dataset cepha --checkpoint PATH_TO_PA_FINETUNED.pth --resume
    python benchmark_degraded_test_set.py --plot-only results/degradation_benchmark/droso_small/PA/results.json

Defaults: all eight corruptions, all three levels, all images, PA label.
PA is a user-declared provenance label, NOT inferred from checkpoint weights.
Use a trusted Stage 2 checkpoint (not an MAE pretraining checkpoint).
The same frozen model and paired image set are used for every condition.
Duplicate images are retained. Incomplete annotations are recorded and excluded
consistently from every condition; prediction failures abort instead of silently
changing the evaluated image set. Scores use coordinates rounded to 2 decimals,
matching E2's scoring of Stage 3 prediction files. No training is performed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from make_degraded_test_set import EXTENSIONS, LANDMARK_COUNTS, PRESETS, digest, is_within


LABELS = dict(gaussian_noise='Gaussian\nnoise', gaussian_blur='Gaussian\nblur',
              low_contrast='Low\ncontrast', darken='Darken', brighten='Brighten',
              illumination='Uneven\nillumination', low_resolution='Low\nresolution', jpeg='JPEG')


def write_json(path, value):
    """Atomic replacement so an interrupted write does not destroy progress."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def condition_list(corruptions):
    return [('clean', 'clean', 0)] + [
        (f'{name}/level_{level}', name, level)
        for name in corruptions for level in (1, 2, 3)]


def inventory(root, dataset, conditions, limit=None):
    """Validate pairing, dimensions, GT bytes and fingerprint every input."""
    base = root / dataset
    clean_dir = base / 'clean'
    clean = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not clean:
        raise ValueError(f'No clean images: {clean_dir}')
    names = [p.name for p in clean]
    if len({p.stem.casefold() for p in clean}) != len(clean):
        raise ValueError('Duplicate image stems in clean folder')
    sizes, hashes, gt_bytes, ground_truth, excluded = {}, {}, {}, {}, []
    count = LANDMARK_COUNTS[dataset]
    for path in clean:
        label = path.with_suffix('.txt')
        coords = np.loadtxt(label, ndmin=2, dtype=np.float64)
        if coords.shape[1] != 2 or not np.isfinite(coords).all():
            raise ValueError(f'Invalid annotation: {label}')
        gt_bytes[path.name] = label.read_bytes()
        if coords.shape != (count, 2):
            excluded.append(dict(image=path.name, reason=f'{len(coords)} landmarks; expected {count}'))
        else:
            ground_truth[path.name] = coords
        with Image.open(path) as im:
            sizes[path.name] = list(im.size)
    for condition, _, _ in conditions:
        folder = base / condition
        actual = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if actual != names:
            raise ValueError(f'Image list differs from clean: {folder}')
        entries = []
        for name in names:
            path = folder / name
            label = path.with_suffix('.txt')
            if label.read_bytes() != gt_bytes[name]:
                raise ValueError(f'Ground truth changed: {label}')
            with Image.open(path) as im:
                if list(im.size) != sizes[name]:
                    raise ValueError(f'Image dimensions changed: {path}')
            entries.append([name, digest(path), digest(label)])
        hashes[condition] = entries
    selected = [name for name in names if name in ground_truth][:limit]
    if not selected:
        raise ValueError('No completely annotated images to score')
    return (selected, np.stack([ground_truth[n] for n in selected]),
            np.asarray([sizes[n] for n in selected]), excluded, hashes, len(names))


def load_model(checkpoint, dataset, device, image_size=None):
    """Read training dimensions from the trusted local fine-tuned checkpoint."""
    import torch
    from models.mae import mae_vit_base, mae_vit_small
    from models.landmark_head import WingLandmarkModel
    from models.regression_head import RegressionLandmarkModel

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in ckpt:
        raise ValueError('Expected a Stage 2 fine-tuned checkpoint with model_state_dict')
    training = ckpt.get('config', {})
    saved_size = training.get('finetune_image_size', training.get('image_size'))
    if image_size is not None and saved_size is not None and image_size != saved_size:
        raise ValueError(f'--image-size {image_size} conflicts with checkpoint size {saved_size}')
    size = image_size or saved_size
    if size is None:
        raise ValueError('Checkpoint has no image size; provide --image-size matching training')
    k = LANDMARK_COUNTS[dataset]
    if training.get('num_landmarks', k) != k:
        raise ValueError('Checkpoint landmark count does not match the selected dataset')
    model_size = training.get('model_size', 'base')
    if model_size not in ('base', 'small'):
        raise ValueError(f'Unsupported model size: {model_size}')
    config = dict(image_size=size, heatmap_size=training.get('heatmap_size', size // 4),
                  patch_size=training.get('patch_size', 16),
                  embed_dim=training.get('embed_dim', 768 if model_size == 'base' else 384),
                  num_landmarks=k, model_size=model_size,
                  head_type=ckpt.get('head_type', 'heatmap'))
    if config['head_type'] not in ('heatmap', 'regression'):
        raise ValueError(f"Unsupported head: {config['head_type']}")
    if config['patch_size'] != 16 or size % 16:
        raise ValueError('Stage 3 requires patch size 16 and image size divisible by 16')
    if config['head_type'] == 'heatmap' and config['heatmap_size'] != size // 4:
        raise ValueError('Heatmap size does not match the model output (image_size / 4)')
    mae = (mae_vit_base if model_size == 'base' else mae_vit_small)(img_size=size)
    kwargs = dict(mae_encoder=mae, num_landmarks=k, embed_dim=config['embed_dim'], freeze_encoder=False)
    if config['head_type'] == 'heatmap':
        model = WingLandmarkModel(**kwargs, img_size=size, patch_size=16, heatmap_size=config['heatmap_size'])
    else:
        model = RegressionLandmarkModel(**kwargs)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model.to(device).eval(), config, {
        'mae_checkpoint': str(training.get('mae_checkpoint', 'unknown')),
        'training_directory': str(training.get('train_dir', training.get('data_dir', 'unknown'))),
        'n_shots': training.get('n_shots'), 'seed': training.get('seed'),
    }


def evaluate_condition(model, config, device, folder, names, gt, sizes, archive):
    from stage3_predict import predict_one_image
    from tqdm import tqdm

    predictions = []
    for name in tqdm(names, desc='/'.join(folder.parts[-2:])):
        points = np.asarray(predict_one_image(model, folder / name, config, device), dtype=np.float64)
        if points.shape != gt.shape[1:] or not np.isfinite(points).all():
            raise ValueError(f'Invalid model prediction for {folder / name}')
        # Match the precision of files scored in E2, without rewriting GT files.
        predictions.append(np.array([[float(f'{v:.2f}') for v in xy] for xy in points]))
    predictions = np.stack(predictions)
    errors = np.linalg.norm(predictions - gt, axis=2)
    nme = 100 * errors / np.linalg.norm(sizes, axis=1)[:, None]
    temp = archive.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, image_names=np.asarray(names), predictions=predictions,
                            ground_truth=gt, image_sizes=sizes, errors_px=errors, nme_percent=nme)
    os.replace(temp, archive)
    return dict(n_images=len(names), mre_px=float(errors.mean()), nme_percent=float(nme.mean()),
                mre_per_landmark_px=errors.mean(axis=0).tolist(),
                nme_per_landmark_percent=nme.mean(axis=0).tolist())


def relative_scores(rows):
    clean = rows[0]
    for row in rows:
        row['delta_mre_px'] = row['mre_px'] - clean['mre_px']
        row['relative_mre_increase_percent'] = (
            100 * row['delta_mre_px'] / clean['mre_px'] if clean['mre_px'] else None)
        row['delta_nme_percentage_points'] = row['nme_percent'] - clean['nme_percent']


def plot_results(result_path, metric='mre'):
    """Read saved JSON only; no checkpoint, input images, or torch required."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    result_path = Path(result_path)
    data = json.loads(result_path.read_text(encoding='utf-8'))
    if data['status'] != 'complete':
        raise ValueError('Run is incomplete; finish/resume it before plotting')
    key = 'mre_px' if metric == 'mre' else 'nme_percent'
    names = data['run']['corruptions']
    colors = ['#91b9dd', '#4485ba', '#194f7b']
    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.45 * len(names)), 4.7))
    positions, values, bar_colors = [0], [data['conditions'][0][key]], ['#8a8a8a']
    for group, name in enumerate(names):
        for level in (1, 2, 3):
            row = next(r for r in data['conditions'] if r['corruption'] == name and r['level'] == level)
            positions.append(2 + group * 4 + level - 1)
            values.append(row[key])
            bar_colors.append(colors[level - 1])
    ax.bar(positions, values, color=bar_colors, width=.82)
    ax.axhline(values[0], color='#777777', linestyle='--', linewidth=.8, zorder=0)
    ax.set_xticks([0] + [3 + i * 4 for i in range(len(names))])
    ax.set_xticklabels(['Clean'] + [LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel('MRE (native-image pixels)' if metric == 'mre' else 'NME (% of image diagonal)')
    title = f"{data['run']['dataset']} | {data['run']['protocol']} | {len(data['image_names'])} test images"
    if data['run']['limit'] is not None:
        title += ' | SUBSET RUN'
    ax.set_title(title)
    ax.legend(handles=[Patch(color=colors[i], label=f'Level {i + 1}') for i in range(3)],
              loc='upper left', bbox_to_anchor=(0, 1.0), ncol=3, frameon=False)
    ax.set_ylim(0, max(values) * 1.2 if max(values) else 1)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        target = result_path.parent / f'degradation_{metric}.{suffix}'
        fig.savefig(target, dpi=220, bbox_inches='tight')
        print(f'Saved {target}')
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=sorted(LANDMARK_COUNTS))
    p.add_argument('--checkpoint', type=Path, help='Trusted PA Stage 2 checkpoint for this dataset')
    p.add_argument('--input-root', type=Path, default=Path('results/degraded_test_set'))
    p.add_argument('--output-dir', type=Path, help='Default: results/degradation_benchmark/DATASET/PROTOCOL')
    p.add_argument('--protocol', choices=['PA', 'PB', 'PC'], default='PA', help='Declared provenance label')
    p.add_argument('--corruptions', nargs='+', choices=['all'] + list(PRESETS), default=['all'])
    p.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, etc.')
    p.add_argument('--image-size', type=int, help='Only needed if the checkpoint has no saved image size')
    p.add_argument('--limit', type=int, help='First N eligible images for a smoke test; omit for the paper')
    p.add_argument('--resume', action='store_true', help='Reuse completed conditions with identical inputs/config')
    p.add_argument('--dry-run', action='store_true', help='Validate input pairing without torch or output writes')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--plot-only', type=Path, metavar='RESULTS_JSON', help='Replot saved results without inference')
    p.add_argument('--metric', choices=['mre', 'nme'], default='mre', help='Metric for the bar chart')
    args = p.parse_args()
    if not args.plot_only and (not args.dataset or not args.checkpoint):
        p.error('--dataset and --checkpoint are required unless using --plot-only')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if len(set(args.corruptions)) != len(args.corruptions) or ('all' in args.corruptions and len(args.corruptions) != 1):
        p.error('Use --corruptions all or a list of distinct corruption names')
    return args


def main():
    args = parse_args()
    if args.plot_only:
        plot_results(args.plot_only, args.metric)
        return
    root, checkpoint = args.input_root.resolve(), args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint}')
    output = (args.output_dir or Path('results/degradation_benchmark') / args.dataset / args.protocol).resolve()
    if is_within(output, root) or is_within(root, output) or is_within(checkpoint, output):
        raise ValueError('Output must be separate from input data and checkpoints')
    generator_path = root / 'config.json'
    generator = json.loads(generator_path.read_text(encoding='utf-8')) if generator_path.exists() else None
    if generator is not None and generator.get('status') != 'complete':
        raise ValueError('Degradation generation has not completed. Wait for it before benchmarking.')
    corruptions = list(PRESETS) if args.corruptions == ['all'] else args.corruptions
    conditions = condition_list(corruptions)
    print('Validating paired images and annotations...', flush=True)
    names, gt, sizes, excluded, fingerprints, total = inventory(root, args.dataset, conditions, args.limit)
    print(f'{args.dataset}: {len(names)}/{total} images scored, {len(excluded)} annotation exclusions; '
          f'{len(conditions)} conditions; duplicates retained.')
    for item in excluded:
        print(f"  Excluded in ALL conditions: {item['image']}: {item['reason']}")
    if args.dry_run:
        print('Dry run complete. No inference or output writes; checkpoint contents not loaded.')
        return

    import torch
    import PIL
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    # Fingerprints prevent accidental mixing of different models, data or code on resume.
    run = dict(dataset=args.dataset, protocol=args.protocol, checkpoint=str(checkpoint),
               checkpoint_sha256=digest(checkpoint), input_root=str(root), corruptions=corruptions,
               levels=[1, 2, 3], limit=args.limit, image_size_override=args.image_size, device=device,
               source_sha256={n: digest(Path(__file__).resolve().parent / n) for n in
                              ('benchmark_degraded_test_set.py', 'stage3_predict.py', 'models/mae.py',
                               'models/landmark_head.py', 'models/regression_head.py', 'utils.py')},
               versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                             numpy=np.__version__, pillow=PIL.__version__),
               input_fingerprints=fingerprints,
               generator_config_sha256=digest(generator_path) if generator_path.exists() else None)
    result_path = output / 'results.json'
    if args.resume:
        data = json.loads(result_path.read_text(encoding='utf-8'))
        if data['run'] != run:
            raise ValueError('Resume configuration, code, environment, checkpoint or data changed. Use a new --output-dir.')
        for row in data['conditions']:
            if digest(output / row['archive']) != row['archive_sha256']:
                raise ValueError(f"Saved archive changed: {row['archive']}")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Output is not empty: {output}. Use --resume or a new --output-dir.')
        output.mkdir(parents=True, exist_ok=True)
        data = dict(schema_version=1, status='running', run=run, generator_config=generator,
                    image_names=names, excluded_images=excluded, n_source_images=total,
                    metric_notes='One checkpoint. No between-run SD. Coordinates rounded to 2 decimals as in E2.',
                    conditions=[], started_utc=datetime.now(timezone.utc).isoformat())
        write_json(result_path, data)
    try:
        pending = [c for c in conditions if c[0] not in {r['condition'] for r in data['conditions']}]
        if pending:
            torch.manual_seed(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            model, config, provenance = load_model(checkpoint, args.dataset, device, args.image_size)
            data.update(status='running', inference_config=config, checkpoint_metadata=provenance)
            data.pop('error', None)
            write_json(result_path, data)
            print(f"Model: {config['head_type']}, image size {config['image_size']}, device {device}")
            print(f"Declared protocol: {args.protocol}; saved MAE source: {provenance['mae_checkpoint']}")
            (output / 'arrays').mkdir(exist_ok=True)
            for condition, corruption, level in pending:
                relative_path = f"arrays/{condition.replace('/', '_')}.npz"
                with torch.inference_mode():
                    score = evaluate_condition(model, config, device, root / args.dataset / condition,
                                               names, gt, sizes, output / relative_path)
                row = dict(condition=condition, corruption=corruption, level=level,
                           archive=relative_path, archive_sha256=digest(output / relative_path), **score)
                data['conditions'].append(row)
                relative_scores(data['conditions'])
                write_json(result_path, data)
                print(f"{condition}: MRE={score['mre_px']:.4f} px; NME={score['nme_percent']:.4f}%", flush=True)
        data.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(result_path, data)
    except (Exception, KeyboardInterrupt) as exc:
        data.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        write_json(result_path, data)
        raise
    print(f'Results saved: {result_path}')
    if not args.no_plot:
        plot_results(result_path, args.metric)


if __name__ == '__main__':
    main()
