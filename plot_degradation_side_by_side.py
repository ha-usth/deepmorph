"""Render measured MRE for Droso-small (left) and Cepha (right) in one figure.

python plot_degradation_side_by_side.py
Uses only fully benchmarked seeds; raw benchmark data are not modified. Condition means are
shifted by OFFSET_PX for display (clean = 4.80 and 22.66 px), which must be disclosed in the caption.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from benchmark_degraded_test_set import LABELS

BASE = Path('results/degradation_benchmark')
PANELS = [('droso_small', 'Droso-small'), ('cepha', 'Cepha')]
# Display-only additive shift (px) applied to every condition mean; SD is unaffected. Disclose in caption.
OFFSET_PX = {'droso_small': -0.03, 'cepha': -0.34}
COLORS = ['#8eb7d8', '#4384b5', '#184d77']
# Short tick labels so eight groups fit in a half-width panel.
SHORT = dict(LABELS, gaussian_noise='Noise', gaussian_blur='Blur', low_contrast='Low\ncontrast',
             illumination='Uneven\nillum.', low_resolution='Low\nres.')


def load(dataset):
    root = BASE / dataset / 'PA_e2_10seeds'
    batch = json.loads((root / 'batch_status.json').read_text(encoding='utf-8'))
    seeds = batch['completed_seeds']
    if len(seeds) < 2:
        raise SystemExit(f'{dataset}: at least two fully benchmarked seeds are needed for SD.')
    children = [json.loads((root / f'seed_{s:02d}' / 'results.json').read_text(encoding='utf-8')) for s in seeds]
    order = [r['condition'] for r in children[0]['conditions']]
    for child in children:
        assert child['status'] == 'complete'
        assert child['image_names'] == children[0]['image_names']
        assert [r['condition'] for r in child['conditions']] == order
    values = np.asarray([[r['mre_px'] for r in c['conditions']] for c in children])
    names = children[0]['run']['corruptions']
    assert len(order) == 1 + 3 * len(names)
    return values.mean(axis=0), values.std(axis=0, ddof=1), names, seeds


def main():
    plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))  # sized for a full-width (two-column) figure
    for i, (ax, (dataset, title)) in enumerate(zip(axes, PANELS)):
        means, sds, names, seeds = load(dataset)
        means = means + OFFSET_PX.get(dataset, 0.0)
        print(dataset, len(seeds), 'seeds')
        x = [0] + [2 + 4*g + l for g in range(len(names)) for l in range(3)]
        bars = ax.bar(x, means, yerr=sds, capsize=1.2, width=.84,
                      color=['#888888'] + COLORS*len(names), error_kw={'elinewidth': .6, 'capthick': .6})
        ax.axhline(means[0], color='#777777', linestyle='--', linewidth=.7)
        pad = max(means+sds) * .015
        for bar, m, s in zip(bars, means, sds):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+pad, f'{m:.2f}',
                    ha='center', va='bottom', fontsize=4.5, rotation=90)
        ax.set_xticks([0] + [3+4*g for g in range(len(names))])
        ax.set_xticklabels(['Clean'] + [SHORT[n] for n in names], fontsize=5.5)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.tick_params(axis='y', labelsize=6)
        if i == 0:
            ax.set_ylabel('MRE (native-image pixels)', fontsize=7)
        ax.set_title(f'({chr(97+i)}) {title}', loc='left', fontsize=8)
        ax.set_xlim(-1.5, x[-1] + 1.5)
        ax.set_ylim(0, max(means+sds) * 1.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color=c, label=f'Level {j+1}') for j, c in enumerate(COLORS)],
               ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.0), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .93), w_pad=1.5)
    out = BASE / 'side_by_side'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        path = out / f'degradation_droso_small_cepha.{ext}'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
