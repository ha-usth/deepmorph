"""Render measured MRE for the fully benchmarked seeds in a batch directory.

python plot_degradation_progress.py --results-dir results/degradation_benchmark/droso_small/PA_e2_10seeds
Does not modify benchmark data or shift the clean baseline.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from benchmark_degraded_test_set import LABELS, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir
    batch = json.loads((root / 'batch_status.json').read_text(encoding='utf-8'))
    seeds = batch['completed_seeds']
    if not seeds:
        raise SystemExit('No fully benchmarked seeds yet; run again after a seed completes.')
    children = [json.loads((root / f'seed_{s:02d}' / 'results.json').read_text(encoding='utf-8')) for s in seeds]
    order = [r['condition'] for r in children[0]['conditions']]
    for child in children:
        assert child['status'] == 'complete'
        assert child['image_names'] == children[0]['image_names']
        assert [r['condition'] for r in child['conditions']] == order
    values = np.asarray([[r['mre_px'] for r in c['conditions']] for c in children])
    means = values.mean(axis=0)
    sds = values.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(means)
    names = children[0]['run']['corruptions']
    assert len(order) == 1 + 3 * len(names)
    complete = len(seeds) == len(batch['seeds'])
    out = root / ('chart_final' if complete else f'preview_{len(seeds):02d}_seeds')
    out.mkdir(exist_ok=True)
    summary = dict(dataset=batch['dataset'], protocol=batch['protocol'], seeds=seeds,
                   preliminary=not complete, n_images=len(children[0]['image_names']),
                   snapshot_utc=datetime.now(timezone.utc).isoformat(),
                   metric='Measured MRE in native-image pixels; no baseline adjustment',
                   conditions=[dict(condition=c, mean_px=float(m), sd_px=float(s) if len(seeds)>1 else None)
                               for c, m, s in zip(order, means, sds)])
    write_json(out / 'chart_data.json', summary)
    x = [0] + [2 + 4*g + l for g in range(len(names)) for l in range(3)]
    colors = ['#8eb7d8', '#4384b5', '#184d77']
    fig, ax = plt.subplots(figsize=(14.6, 5.6))
    bars = ax.bar(x, means, yerr=sds if len(seeds)>1 else None, capsize=3,
                  color=['#888888']+colors*len(names), width=.84, error_kw={'elinewidth':1})
    ax.axhline(means[0], color='#777777', linestyle='--', linewidth=1)
    pad = max(means+sds) * .009
    for bar, m, s in zip(bars, means, sds):
        ax.text(bar.get_x()+bar.get_width()/2, m+s+pad, f'{m:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0] + [3+4*g for g in range(len(names))])
    ax.set_xticklabels(['Clean']+[LABELS[n] for n in names], fontsize=10)
    ax.set_ylabel('MRE (native-image pixels)', fontsize=11)
    title = {'droso_small':'Droso-small', 'sea_bass':'Sea-bass', 'cepha':'Cepha'}.get(batch['dataset'],batch['dataset'])
    ax.set_title(title, fontsize=14, pad=14)
    ax.legend(handles=[Patch(color=c,label=f'Level {i+1}') for i,c in enumerate(colors)], ncol=3, frameon=False, loc='upper left')
    ax.set_ylim(0, max(means+sds)*1.22 if max(means+sds)>0 else 1)
    ax.spines[['top','right']].set_visible(False)
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','pdf'):
        path = out / f'{batch["dataset"]}_{batch["protocol"]}_degradation.{ext}'
        fig.savefig(path,dpi=220,bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
