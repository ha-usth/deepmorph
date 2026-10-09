"""Manuscript previews from existing E2 runs; original test splits retained."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('results/landmark_manuscript')
DATASETS = [('droso_small','Droso-small',15), ('droso_big','Droso-big',15),
            ('fly','Fly',8), ('bactro','Bactro',10), ('diacha','Diacha',15),
            ('tsetse','Tsetse',15), ('droso-281','Droso-281',15),
            ('sea_bass','Sea-bass',20), ('cepha','Cepha',20)]
# Explicit manuscript selection; do not relabel the underlying experiment runs.
PROTOCOLS = {d: ('PB' if d in ('droso_small', 'tsetse') else 'PA') for d, _, _ in DATASETS}
PROTOCOLS['cepha'] = 'PC'
SEEDS = {d: list(range(5 if d == 'cepha' else 10)) for d, _, _ in DATASETS}
PROTOCOL_NOTE = 'Droso-small / Tsetse: PB, 10 seeds; Cepha: PC, 5 seeds (0–4); others: PA, 10 seeds.'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, label, shots in DATASETS:
        protocol = PROTOCOLS[dataset]
        source = Path('results') / f'e2_{dataset}' / 'results.json'
        runs = [r for r in json.loads(source.read_text())['runs']
                if r['protocol']==protocol and r.get('head','heatmap')=='heatmap'
                and r['n_shots']==shots and r['seed'] in SEEDS[dataset]]
        assert sorted(r['seed'] for r in runs)==SEEDS[dataset], dataset
        values = np.array([r['mre_per_lm'] for r in runs])
        assert values.ndim==2 and np.isfinite(values).all()
        assert all(np.isclose(np.mean(r['mre_per_lm']),r['mre'],atol=1e-4) for r in runs)
        records.append(dict(dataset=dataset,label=label,shots=shots,protocol=protocol,
                            seeds=sorted(r['seed'] for r in runs),
                            test_images=sorted(set(r['n_images'] for r in runs)),
                            mean=values.mean(0).tolist(),sd=values.std(0,ddof=1).tolist(),
                            source=str(source)))
    ref=json.loads(Path('results/imorph_fig5_chog.json').read_text())
    imorph=np.array([ref['landmark_mre_px_approx'][str(k)] for k in range(1,16)])
    ours=records[0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11.8,5.2))
    x=np.arange(1,16)
    ax.bar(x-.19,imorph,.38,label='iMorph C-HOG (digitized reference)',color='#888888')
    ax.bar(x+.19,ours['mean'],.38,yerr=ours['sd'],capsize=3,
           label=f"Ours ({ours['protocol']}): mean ± SD, {len(ours['seeds'])} seeds",color='#2378A0',error_kw={'elinewidth':1})
    ax.set(xticks=x,xlabel='Landmark ID',ylabel='Mean radial error (original-image pixels)',
           ylim=(0,24),title='Droso-small · 15 annotated training images')
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=.18)
    ax.legend(frameon=False,loc='upper left')
    fig.text(.075,.025,'iMorph: Fig. 5, Nguyen et al. (2022), approximate values; no SD available.\n'
             'Ours: original 100-image test set, duplicates retained. Published-reference comparison; splits are not matched.',fontsize=9)
    fig.subplots_adjust(left=.075,right=.99,top=.89,bottom=.21)
    for ext in ('png','svg'):
        fig.savefig(OUT/f'droso_small_imorph_comparison.{ext}',dpi=220)
    plt.close(fig)
    # Two previews: compact mean-only and full mean ± seed SD.
    for full in (False,True):
        headers=['Dataset','n']+[f'L{k}' for k in range(1,20)]
        rows=[]
        for r in records:
            cells=[f'{m:.2f}\n± {s:.2f}' if full else f'{m:.2f}'
                   for m,s in zip(r['mean'],r['sd'])]
            rows.append([r['label'],str(r['shots'])]+cells+['']*(19-len(cells)))
        fig,ax=plt.subplots(figsize=(22,6.8 if full else 4.8))
        ax.axis('off')
        table=ax.table(cellText=rows,colLabels=headers,cellLoc='center',
                       colWidths=[.10,.032]+[.045]*19,bbox=[0,.13,1,.74])
        table.auto_set_font_size(False)
        table.set_fontsize(10 if full else 11)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#D8DEE3'); cell.set_linewidth(.45)
            if i==0:
                cell.set_facecolor('#E9EEF1');cell.set_text_props(weight='bold')
            elif j>=2 and j-2>=len(records[i-1]['mean']):
                cell.set_facecolor('#F2F3F4')
            else:
                cell.set_facecolor('white' if i%2 else '#F8FAFB')
        title='Per-landmark MRE (pixels) · '+('mean ± SD' if full else 'mean')
        ax.text(0,.95,title,fontsize=17,weight='bold',transform=ax.transAxes)
        ax.text(0,.035,'Source: E2 results, original test splits; duplicates retained. n = annotated training images.\n'
                'Blank cells: landmark absent. Native-pixel errors should be compared within each dataset.',
                fontsize=10,transform=ax.transAxes)
        fig.subplots_adjust(left=.018,right=.99,top=.98,bottom=.02)
        stem='all_datasets_landmark_mean_sd' if full else 'all_datasets_landmark_mean'
        for ext in ('png','svg'):
            fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
        plt.close(fig)
    (OUT/'source_data.json').write_text(json.dumps({'protocols':PROTOCOLS,'seeds':SEEDS,'head':'heatmap',
        'duplicates_retained':True,'datasets':records,'imorph':ref},indent=2))
    # Ready-to-paste LaTeX table, using booktabs and graphicx.
    lines=[r'\begin{table*}[t]',r'\centering',
           r'\caption{Landmark-wise MRE (native-image pixels), reported as mean $\pm$ SD. Original test splits are retained, including duplicate images. Blank cells indicate absent landmarks; $n$ denotes the number of annotated training images.}',
           r'\label{tab:landmark-errors}',r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{lr'+'r'*19+'}',r'\toprule',
           'Dataset & $n$ & '+' & '.join(f'L{k}' for k in range(1,20))+r' \\',r'\midrule']
    for r in records:
        cells=[f'${m:.2f} \\pm {s:.2f}$' for m,s in zip(r['mean'],r['sd'])]
        lines.append(' & '.join([r['label'],str(r['shots'])]+cells+['']*(19-len(cells)))+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}']
    (OUT/'all_datasets_landmark_mean_sd.tex').write_text('\n'.join(lines))
    print('Generated comparison and nine-dataset tables in',OUT)


if __name__=='__main__':
    main()
