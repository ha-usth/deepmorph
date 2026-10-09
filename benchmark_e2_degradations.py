"""Evaluate retained E2 checkpoints and aggregate paired degradation metrics.

Example:
    python benchmark_e2_degradations.py --dataset cepha --e2-dir results/e2_robustness_cepha_PA --input-root results/degraded_cepha_e2_verified --output-dir results/degradation_benchmark/cepha/PA_e2_10seeds

Each seed is evaluated using benchmark_degraded_test_set.py. Completed seed
directories are resumed, and a clean-control check against the corresponding
E2 run must pass before aggregation. SD is sample SD across seed-level means.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmark_degraded_test_set import LABELS, write_json
from make_degraded_test_set import digest


def summarize(children, seeds, e2_rows, output):
    order = [r['condition'] for r in children[0]['conditions']]
    names = children[0]['image_names']
    for child in children:
        if child['status'] != 'complete' or child['image_names'] != names:
            raise ValueError('All seeds must be complete and score exactly the same images')
        if [r['condition'] for r in child['conditions']] != order:
            raise ValueError('Condition order differs between seeds')
    fields = ['mre_px', 'nme_percent', 'delta_mre_px', 'delta_nme_percentage_points',
              'relative_mre_increase_percent', 'mre_per_landmark_px', 'nme_per_landmark_percent']
    arrays = {key: np.asarray([[row[key] for row in child['conditions']] for child in children],
                             dtype=float) for key in fields}
    summary = dict(status='complete', dataset=children[0]['run']['dataset'],
                   protocol=children[0]['run']['protocol'], seeds=seeds,
                   n_runs=len(seeds), n_images=len(names), image_names=names,
                   excluded_images=children[0]['excluded_images'],
                   corruptions=children[0]['run']['corruptions'],
                   aggregation='Mean and sample SD (ddof=1) across seed-level metrics; changes paired within seed.',
                   checkpoint_sha256=[c['run']['checkpoint_sha256'] for c in children],
                   n_shots=[r['n_shots'] for r in e2_rows],
                   clean_e2_checks=[dict(seed=s, e2_mre_px=r['mre'],
                                         benchmark_mre_px=c['conditions'][0]['mre_px'])
                                    for s, r, c in zip(seeds, e2_rows, children)],
                   conditions=[], finished_utc=datetime.now(timezone.utc).isoformat())
    for j, source in enumerate(children[0]['conditions']):
        row = {k: source[k] for k in ('condition', 'corruption', 'level')}
        for field, values in arrays.items():
            column = values[:, j]
            # Relative change is undefined if any clean denominator is zero.
            if not np.isfinite(column).all():
                row[field + '_mean'] = row[field + '_sd'] = None
            else:
                row[field + '_mean'] = column.mean(axis=0).tolist()
                row[field + '_sd'] = column.std(axis=0, ddof=1).tolist() if len(seeds) > 1 else None
        summary['conditions'].append(row)
    write_json(output / 'summary.json', summary)
    with (output / 'seed_metrics.npz').open('wb') as stream:
        np.savez_compressed(stream, seeds=np.asarray(seeds), conditions=np.asarray(order), **arrays)
    return summary


def plot(summary, output, metric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    field = 'mre_px' if metric == 'mre' else 'nme_percent'
    colors = ['#91b9dd', '#4485ba', '#194f7b']
    names = summary['corruptions']
    x = [0] + [2 + group * 4 + level for group in range(len(names)) for level in range(3)]
    values = [r[field + '_mean'] for r in summary['conditions']]
    std = [r[field + '_sd'] or 0 for r in summary['conditions']]
    fig, ax = plt.subplots(figsize=(max(8, 2 + 1.45 * len(names)), 4.8))
    ax.bar(x, values, yerr=std, capsize=2.5, color=['#888888'] + colors * len(names),
           width=.82, error_kw={'elinewidth': 1})
    ax.axhline(values[0], color='#777777', linewidth=.8, linestyle='--')
    ax.set_xticks([0] + [3 + i * 4 for i in range(len(names))])
    ax.set_xticklabels(['Clean'] + [LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel('MRE (native-image pixels)' if metric == 'mre' else 'NME (% of image diagonal)')
    ax.set_title(f"{summary['dataset']} | {summary['protocol']} | mean +/- SD across {summary['n_runs']} runs")
    ax.legend(handles=[Patch(color=c, label=f'Level {i + 1}') for i, c in enumerate(colors)],
              ncol=3, loc='upper left', frameon=False)
    ax.set_ylim(0, max(v + s for v, s in zip(values, std)) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'degradation_{metric}_mean_sd.{suffix}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--e2-dir', type=Path, required=True)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--protocol', default='PA', choices=['PA', 'PB', 'PC'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--n-shots', type=int, default=20)
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--wait-for-e2', action='store_true', help='Wait for each E2 run to finish before benchmarking it')
    args = p.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        p.error('Seeds must be distinct')
    source = args.e2_dir / 'results.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / 'batch_status.json'
    status = dict(status='running', dataset=args.dataset, protocol=args.protocol,
                  seeds=args.seeds, completed_seeds=[], e2_results=str(source.resolve()),
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    children, rows = [], []
    try:
        for seed in args.seeds:
            deadline = time.monotonic() + 4 * 3600
            print(f'Checking E2 seed {seed}...', flush=True)
            while True:
                try:
                    e2 = json.loads(source.read_text(encoding='utf-8'))
                except (FileNotFoundError, json.JSONDecodeError):
                    e2 = {'runs': []}
                selected = [r for r in e2['runs'] if r['protocol'] == args.protocol and r['seed'] == seed
                            and r['n_shots'] == args.n_shots and r.get('head', 'heatmap') == 'heatmap'
                            and r.get('dataset', args.dataset) == args.dataset]
                if len(selected) == 1:
                    row = selected[0]
                    rows.append(row)
                    break
                if len(selected) > 1 or not args.wait_for_e2 or time.monotonic() > deadline:
                    raise ValueError(f'Expected one completed E2 run for seed {seed}, got {len(selected)}')
                # The E2 process remains independently visible in its runner log.
                training_log = args.e2_dir / 'runner.log'
                if training_log.exists() and 'Traceback (most recent call last)' in training_log.read_text(encoding='utf-8', errors='replace'):
                    raise RuntimeError('E2 training failed; inspect its runner.log')
                status.update(status='waiting_for_e2', waiting_seed=seed)
                write_json(status_path, status)
                time.sleep(15)
            status.update(status='running', active_seed=seed)
            status.pop('waiting_seed', None)
            write_json(status_path, status)
            folder = args.output_dir / f'seed_{seed:02d}'
            result = folder / 'results.json'
            if not args.summarize_only:
                command = [sys.executable, '-u', 'benchmark_degraded_test_set.py',
                           '--dataset', args.dataset, '--checkpoint', row['finetune_checkpoint'],
                           '--protocol', args.protocol, '--input-root', str(args.input_root),
                           '--output-dir', str(folder), '--no-plot']
                if result.exists():
                    command.append('--resume')
                print(f'Benchmark seed {seed}: {row["finetune_checkpoint"]}', flush=True)
                with (args.output_dir / f'seed_{seed:02d}.log').open('a', encoding='utf-8') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            child = json.loads(result.read_text(encoding='utf-8'))
            if child['status'] != 'complete':
                raise ValueError(f'Seed {seed} is incomplete')
            if child['run']['checkpoint_sha256'] != digest(Path(row['finetune_checkpoint'])):
                raise ValueError(f'Seed {seed}: checkpoint differs from saved benchmark')
            metadata = child['checkpoint_metadata']
            if metadata['seed'] != seed or metadata['n_shots'] != args.n_shots:
                raise ValueError(f'Seed {seed}: checkpoint metadata mismatch')
            if Path(metadata['mae_checkpoint']).resolve() != Path(row['mae_checkpoint']).resolve():
                raise ValueError(f'Seed {seed}: pretraining source mismatch')
            clean = child['conditions'][0]
            if (clean['n_images'] != row['n_images'] or abs(clean['mre_px'] - row['mre']) > 1e-4
                    or abs(clean['nme_percent'] - row['nme']) > 1e-5):
                raise ValueError(f'Seed {seed}: clean score does not match its E2 evaluation')
            children.append(child)
            status['completed_seeds'].append(seed)
            write_json(status_path, status)
            print(f'Seed {seed}: complete; clean MRE matches E2 ({clean["mre_px"]:.6f} px)', flush=True)
        summary = summarize(children, args.seeds, rows, args.output_dir)
        plot(summary, args.output_dir, 'mre')
        plot(summary, args.output_dir, 'nme')
        status.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                      e2_results_sha256=digest(source))
        write_json(status_path, status)
        print(f'Complete: {args.output_dir / "summary.json"}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status='failed', error=str(exc))
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
