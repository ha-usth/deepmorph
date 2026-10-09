"""Create paired imaging-degradation benchmarks from held-out landmark datasets.

Examples (run from the repository root):
    python make_degraded_test_set.py --dry-run
    python make_degraded_test_set.py --datasets droso_small sea_bass cepha
    python make_degraded_test_set.py --datasets all --levels 1 2 3
    python make_degraded_test_set.py --datasets droso_small --limit 2 --output-root results/degradation_smoke

Layout:
    OUTPUT/dataset/clean/<stem>.png and <stem>.txt
    OUTPUT/dataset/gaussian_noise/level_1/<stem>.png and <stem>.txt
    OUTPUT/manifest.jsonl, config.json, previews/<dataset>.jpg

Every condition uses the same images and unmodified annotation bytes, including
duplicates. PNG avoids introducing another lossy codec into non-JPEG conditions.
The clean control uses the same PIL RGB conversion as stage3_predict.py. EXIF
orientation is deliberately not applied, matching that loader. No rotation,
translation, crop, or geometric warp is introduced. Low resolution is simulated
by downsampling and restoring the original dimensions. An existing non-empty
output root is refused; source files are never edited.

Evaluate the SAME frozen fine-tuned checkpoints on clean and degraded folders.
Do not retrain, select checkpoints, or choose favorable severity settings using
test errors. Report per-landmark and overall MRE/NME changes relative to this
clean control. These are synthetic imaging stress tests, not a demonstration of
robustness to anatomical/species shift or calibrated real acquisition conditions.

Corruptions are applied at native resolution BEFORE Stage 3's resize to 512.
Resize can attenuate noise/artifacts, especially on larger images. Inspect the
previews and report this order; do not equate a severity level with a measured
physical noise level or identical difficulty across datasets. The three preset
levels are illustrative, fixed increasing perturbation strengths, not expected
monotonic model error. Use --seed with a different output root for another random
realization of gaussian_noise / illumination; deterministic conditions repeat.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
LANDMARK_COUNTS = {'droso_small': 15, 'droso_big': 15, 'fly': 10,
                   'bactro': 12, 'diacha': 10, 'tsetse': 11,
                   'droso-281': 12, 'sea_bass': 11, 'cepha': 19}
PRESETS = {
    'gaussian_noise': [dict(sigma_8bit=s) for s in (5, 15, 30)],
    'gaussian_blur': [dict(sigma_short_side_fraction=s) for s in (.001, .003, .006)],
    'low_contrast': [dict(factor=s) for s in (.75, .50, .25)],
    'darken': [dict(factor=s) for s in (.85, .65, .45)],
    'brighten': [dict(factor=s) for s in (1.15, 1.35, 1.60)],
    'illumination': [dict(amplitude=s) for s in (.15, .30, .50)],
    'low_resolution': [dict(scale=s) for s in (.50, .25, .125)],
    'jpeg': [dict(quality=s, subsampling=2) for s in (75, 40, 15)],
}


def image_seed(seed: int, dataset: str, filename: str, corruption: str) -> int:
    """Stable under iteration order; same noise field across severity levels."""
    key = json.dumps([seed, dataset, filename, corruption], ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big')


def degrade(image: Image.Image, name: str, level: int, seed: int):
    """Return RGB image and exact effective parameters; no input mutation."""
    if image.mode != 'RGB':
        raise ValueError('degrade expects RGB images, matching Stage 3 input conversion')
    if name not in PRESETS or level not in (1, 2, 3):
        raise ValueError(f'Unknown degradation/level: {name}/{level}')
    params = dict(PRESETS[name][level - 1])
    rng = np.random.default_rng(seed)
    if name == 'gaussian_blur':
        params['sigma_native_px'] = min(image.size) * params['sigma_short_side_fraction']
        out = image.filter(ImageFilter.GaussianBlur(params['sigma_native_px']))
    elif name == 'low_contrast':
        out = ImageEnhance.Contrast(image).enhance(params['factor'])
    elif name in ('darken', 'brighten'):
        out = ImageEnhance.Brightness(image).enhance(params['factor'])
    elif name == 'low_resolution':
        small = tuple(max(1, round(s * params['scale'])) for s in image.size)
        params.update(downsampled_size=list(small), down_filter='BOX', up_filter='BICUBIC')
        out = image.resize(small, Image.Resampling.BOX).resize(image.size, Image.Resampling.BICUBIC)
    elif name == 'jpeg':
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=params['quality'], subsampling=params['subsampling'])
        stream.seek(0)
        with Image.open(stream) as decoded:
            out = decoded.convert('RGB').copy()
    else:
        pixels = np.asarray(image, dtype=np.float32)
        if name == 'gaussian_noise':
            pixels = pixels + rng.standard_normal(pixels.shape, dtype=np.float32) * params['sigma_8bit']
            params['noise_model'] = 'independent additive RGB Gaussian, clipped to [0,255]'
        elif name == 'illumination':
            angle = float(rng.uniform(0, 2 * np.pi))
            x = np.linspace(-1, 1, image.width, dtype=np.float32)[None, :]
            y = np.linspace(-1, 1, image.height, dtype=np.float32)[:, None]
            # Smooth directional gain; range bounded by 1 +/- amplitude.
            gradient = (np.cos(angle)*x + np.sin(angle)*y) / (abs(np.cos(angle))+abs(np.sin(angle)))
            pixels = pixels * (1 + params['amplitude'] * gradient[:, :, None])
            params.update(angle_radians=angle, model='multiplicative linear illumination gradient')
        out = Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8))
    assert out.size == image.size and out.mode == 'RGB'
    return out, params


def digest(path: Path) -> str:
    """Compute SHA-256 with bounded memory, including on Python 3.10."""
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def collect(root: Path, datasets: list[str], limit: int | None):
    """Validate before writing; annotation count mismatches are retained/reported."""
    inventory, warnings = {}, []
    for dataset in datasets:
        folder = (root / dataset).resolve()
        if not is_within(folder, root) or not folder.is_dir():
            raise ValueError(f'Invalid dataset directory: {folder}')
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ValueError(f'No images in {folder}')
        # Detect collisions before applying --limit (all outputs use PNG).
        stems = [p.stem.casefold() for p in images]
        if len(stems) != len(set(stems)):
            raise ValueError(f'Duplicate image stems in {folder}; PNG output would collide')
        entries = []
        for img in images[:limit]:
            gt = img.with_suffix('.txt')
            if not gt.is_file():
                raise ValueError(f'Missing annotation: {gt}')
            try:
                coords = np.loadtxt(gt, ndmin=2)
                if coords.size == 0 or coords.shape[1] != 2 or not np.isfinite(coords).all():
                    raise ValueError('Expected finite x y coordinates, one landmark per line')
            except ValueError as exc:
                raise ValueError(f'Invalid annotation {gt}: {exc}') from exc
            expected = LANDMARK_COUNTS.get(dataset)
            if expected is not None and len(coords) != expected:
                warnings.append(f'{dataset}/{gt.name}: {len(coords)} landmarks; expected {expected}. Retained unchanged.')
            # A dry run checks headers only, not a full decode of every image.
            with Image.open(img) as im:
                if getattr(im, 'n_frames', 1) != 1:
                    raise ValueError(f'Multiframe image requires explicit handling: {img}')
                size, mode = im.size, im.mode
            entries.append(dict(image=img, annotation=gt, landmarks=len(coords),
                                expected_landmarks=expected, size=size, source_mode=mode))
        inventory[dataset] = entries
    return inventory, warnings


def save_condition(image, folder, stem, annotation_bytes, output_root):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{stem}.png'
    label = folder / f'{stem}.txt'
    image.save(path, format='PNG')
    label.write_bytes(annotation_bytes)
    return dict(output_image=path.relative_to(output_root).as_posix(),
                output_annotation=label.relative_to(output_root).as_posix(),
                output_image_sha256=digest(path),
                output_annotation_sha256=digest(label))


def preview_tile(image, title):
    tile = Image.new('RGB', (320, 244), 'white')
    small = image.copy()
    small.thumbnail((308, 206))
    tile.paste(small, ((320-small.width)//2, 29+(206-small.height)//2))
    ImageDraw.Draw(tile).text((8, 8), title, fill='black')
    return tile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root', type=Path, default=Path('test_set'))
    parser.add_argument('--output-root', type=Path, default=Path('results/degraded_test_set'))
    parser.add_argument('--datasets', nargs='+', default=['droso_small', 'sea_bass', 'cepha'],
                        help='Dataset folder names, or all')
    parser.add_argument('--degradations', nargs='+', choices=list(PRESETS), default=list(PRESETS))
    parser.add_argument('--levels', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--limit', type=int, help='First N filenames per dataset: smoke tests only, not a full benchmark')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and list counts without writing files')
    args = parser.parse_args(argv)
    src, out = args.input_root.resolve(), args.output_root.resolve()
    if not src.is_dir():
        parser.error(f'Input root does not exist: {src}')
    if is_within(out, src) or is_within(src, out):
        parser.error('Input and output roots must be separate, non-nested directories')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    datasets = (sorted(p.name for p in src.iterdir() if p.is_dir())
                if args.datasets == ['all'] else list(dict.fromkeys(args.datasets)))
    if not datasets or any(Path(d).name != d or d in ('.', '..') for d in datasets):
        parser.error('Use dataset directory names only')
    degradations = list(dict.fromkeys(args.degradations))
    levels = sorted(set(args.levels))
    inventory, warnings = collect(src, datasets, args.limit)
    counts = {d: len(entries) for d, entries in inventory.items()}
    n_conditions = 1 + len(degradations)*len(levels)
    print(json.dumps(dict(images_per_dataset=counts, conditions_per_image=n_conditions,
                         total_output_images=sum(counts.values())*n_conditions,
                         output_root=str(out), annotation_warnings=warnings), indent=2), flush=True)
    if args.dry_run:
        return
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'Output is not empty: {out}. Use a new output root; existing results are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    input_root=str(src), output_root=str(out), datasets=datasets,
                    images_per_dataset=counts, degradations=degradations, levels=levels,
                    presets=PRESETS, seed=args.seed, limit=args.limit,
                    python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__,
                    script_sha256=digest(Path(__file__)), annotation_warnings=warnings,
                    operation_order='decode -> RGB -> native-resolution corruption -> lossless PNG',
                    coordinate_policy='original dimensions; annotations byte-identical; duplicates retained',
                    random_policy='SHA256(global seed, dataset, filename, corruption); shared field across levels',
                    command=sys.argv)
    config_path = out / 'config.json'
    config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    try:
        with (out/'manifest.jsonl').open('w', encoding='utf-8') as manifest:
            for dataset, entries in inventory.items():
                tiles = []
                for index, entry in enumerate(entries):
                    path, gt = entry['image'], entry['annotation']
                    annotation_bytes = gt.read_bytes()
                    with Image.open(path) as loaded:
                        clean = loaded.convert('RGB')
                    common = dict(dataset=dataset, source_image=path.relative_to(src).as_posix(),
                                  source_image_sha256=digest(path),
                                  source_annotation_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                                  width=clean.width, height=clean.height, source_mode=entry['source_mode'],
                                  landmarks=entry['landmarks'], expected_landmarks=entry['expected_landmarks'])
                    jobs = [('clean', 0)] + [(name, level) for name in degradations for level in levels]
                    for name, level in jobs:
                        local_seed = image_seed(args.seed, dataset, path.name, name)
                        image, params = (clean, {}) if name=='clean' else degrade(clean, name, level, local_seed)
                        folder = out/dataset/name
                        if level:
                            folder = folder/f'level_{level}'
                        files = save_condition(image, folder, path.stem, annotation_bytes, out)
                        assert files['output_annotation_sha256']==common['source_annotation_sha256']
                        manifest.write(json.dumps(dict(**common, **files, degradation=name, level=level,
                                                      parameters=params, random_seed=local_seed))+'\n')
                        if index==0:
                            tiles.append(preview_tile(image, f'{name} / L{level}' if level else 'clean'))
                    if (index+1)%10==0 or index+1==len(entries):
                        print(f'{dataset}: {index+1}/{len(entries)} source images complete', flush=True)
                sheet = Image.new('RGB', (320*4, 244*((len(tiles)+3)//4)), '#eeeeee')
                for i, tile in enumerate(tiles):
                    sheet.paste(tile, ((i%4)*320, (i//4)*244))
                (out/'previews').mkdir(exist_ok=True)
                sheet.save(out/'previews'/f'{dataset}.jpg', quality=92)
        metadata.update(status='complete', total_output_images=sum(counts.values())*n_conditions)
    except Exception as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
        config_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
