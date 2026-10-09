"""Compare Table 2 (DeepMorph_revision (34).pdf, p.11) with selected saved runs.

PDF values below were transcribed and visually checked. No experimental data
or manuscript files are modified. Outputs are proposed reconciliation data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_landmark_figures import DATASETS, PROTOCOLS, PROTOCOL_NOTE, SEEDS

# MRE mean/SD, NME mean/SD, in the dataset order used by the manuscript.
TABLE2 = [(4.80,.09,.283,.005), (5.24,.08,.308,.005),
          (3.05,.18,.419,.024), (8.69,.42,.340,.017),
          (3.21,.06,.257,.005), (5.52,.09,.337,.005),
          (2.79,.08,.351,.010), (4.41,.09,.486,.010),
          (22.66,.39,.735,.013)]


def pair(mean,sd,digits):
    return f'{mean:.{digits}f} ± {sd:.{digits}f}'


def main():
    out = Path('results/landmark_manuscript')
    artifact = json.loads((out/'source_data.json').read_text())
    records, display, tex = [], [], []
    for (dataset,label,shots),old in zip(DATASETS,TABLE2):
        protocol = PROTOCOLS[dataset]
        p=Path('results')/f'e2_{dataset}'/'results.json'
        runs=[r for r in json.loads(p.read_text())['runs']
              if r['protocol']==protocol and r.get('head','heatmap')=='heatmap'
              and r['n_shots']==shots and r['seed'] in SEEDS[dataset]]
        assert sorted(r['seed'] for r in runs)==SEEDS[dataset]
        a=np.array([r['mre_per_lm'] for r in runs])
        run_mre=a.mean(axis=1)
        saved=next(r for r in artifact['datasets'] if r['dataset']==dataset)
        assert saved['protocol']==protocol
        assert saved['seeds']==SEEDS[dataset]
        assert np.allclose(a.mean(0),saved['mean'],rtol=0,atol=1e-8)
        assert np.allclose(a.std(0,ddof=1),saved['sd'],rtol=0,atol=1e-8)
        nme=np.array([r['nme'] for r in runs])
        new=[float(run_mre.mean()),float(run_mre.std(ddof=1)),
             float(nme.mean()),float(nme.std(ddof=1))]
        old_cells=[pair(*old[:2],2),pair(*old[2:],3)]
        new_cells=[pair(*new[:2],2),pair(*new[2:],3)]
        matches=[x==y for x,y in zip(old_cells,new_cells)]
        records.append(dict(dataset=dataset,protocol=protocol,n_shots=shots,seeds=SEEDS[dataset],
            test_images=sorted(set(r['n_images'] for r in runs)),
            manuscript=dict(zip(['mre_mean','mre_sd','nme_mean','nme_sd'],old)),
            recomputed=dict(zip(['mre_mean','mre_sd','nme_mean','nme_sd'],new)),
            mre_matches_at_printed_precision=matches[0],
            nme_matches_at_printed_precision=matches[1],
            user_accepts_mre_difference_temporarily=(dataset=='fly'),
            max_run_difference_vs_cached_mre=float(np.max(np.abs(run_mre-[r['mre'] for r in runs]))),
            source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        display.append([label,str(shots),old_cells[0],new_cells[0],old_cells[1],new_cells[1],
                        'Match' if all(matches) else ('Accepted*' if dataset=='fly' else 'Update')])
        tex.append(f"{label} & {shots} & ${new[0]:.2f} \\pm {new[1]:.2f}$ & ${new[2]:.3f} \\pm {new[3]:.3f}$ "+r'\\')
    (out/'table2_consistency.json').write_text(json.dumps({
        'manuscript':'DeepMorph_revision (34).pdf, page 11, Table 2',
        'protocols':PROTOCOLS,'seeds':SEEDS,'head':'heatmap','duplicates_retained':True,
        'note':'Recomputed from cached per-seed landmark means; no new training or full re-scoring.',
        'datasets':records},indent=2))
    (out/'table2_ours_proposed.tex').write_text(
        '% Proposed Ours columns: Dataset & n & MRE (px) & NME (percent)\n'
        '% '+PROTOCOL_NOTE+' Heatmap; original test set.\n'
        +'\n'.join(tex)+'\n')
    fig,ax=plt.subplots(figsize=(14,5.7))
    ax.axis('off')
    t=ax.table(cellText=display,colLabels=['Dataset','n','Table 2: MRE','Selected runs: MRE',
              'Table 2: NME','Selected runs: NME','Status'],cellLoc='center',
              colWidths=[.13,.04,.18,.18,.18,.18,.11],bbox=[0,.20,1,.66])
    t.auto_set_font_size(False);t.set_fontsize(11)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#D5DBDF');cell.set_linewidth(.5)
        if i==0:
            cell.set_facecolor('#E9EEF1');cell.set_text_props(weight='bold')
        elif (j==3 and not records[i-1]['mre_matches_at_printed_precision']) or (
                j==5 and not records[i-1]['nme_matches_at_printed_precision']):
            cell.set_facecolor('#FFF0D6');cell.set_text_props(weight='bold')
    ax.text(0,.94,'Table 2 consistency check · selected runs · mean ± SD',
            fontsize=16,weight='bold',transform=ax.transAxes)
    ax.text(0,.025,PROTOCOL_NOTE+'\n'
            'MRE: pixels; NME: % of image diagonal. Duplicates retained.\n'
            'Source: DeepMorph_revision (34).pdf, p.11, and E2 runs. Highlighted cells differ at printed precision.\n'
            '* Fly discrepancy temporarily accepted by the author; values are unchanged.\n'
            'Manuscript captions must reflect the selected protocols and run counts, including PC / five seeds for Cepha.',
            fontsize=10,transform=ax.transAxes)
    fig.subplots_adjust(left=.02,right=.98,top=.97,bottom=.01)
    fig.savefig(out/'table2_consistency.png',dpi=190)
    plt.close(fig)
    print(json.dumps({'mre_matching_rows':sum(r['mre_matches_at_printed_precision'] for r in records),
        'nme_matching_rows':sum(r['nme_matches_at_printed_precision'] for r in records),
        'max_cached_run_difference':max(r['max_run_difference_vs_cached_mre'] for r in records)},indent=2))


if __name__=='__main__':
    main()
