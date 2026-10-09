"""Create a disclosed baseline-shifted visualization and native PGFPlots source.

Raw benchmark files are never modified. Only completed, verified seeds are used.
The displayed ordinate is measured MRE plus a dataset-specific constant, NOT raw MRE.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from benchmark_degraded_test_set import LABELS, write_json
from make_degraded_test_set import digest


def main():
    base = Path('results/degradation_benchmark')
    specs = [('droso_small', 'Droso-small', 4.80, ['#efb0aa', '#d86b60', '#a9342c']),
             ('cepha', 'Cepha', 22.66, ['#8eb7d8', '#4384b5', '#184d77'])]
    panels = []
    for ds, label, target, colors in specs:
        root = base / ds / 'PA_e2_10seeds'
        batch = json.loads((root / 'batch_status.json').read_text(encoding='utf-8'))
        seeds = list(batch['completed_seeds'])
        if len(seeds) < 2:
            raise ValueError(f'{ds}: at least two completed seeds needed for SD')
        paths = [root / f'seed_{s:02d}' / 'results.json' for s in seeds]
        children = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        order = [r['condition'] for r in children[0]['conditions']]
        for child in children:
            assert child['status'] == 'complete'
            assert [r['condition'] for r in child['conditions']] == order
            assert child['image_names'] == children[0]['image_names']
        values = np.array([[r['mre_px'] for r in c['conditions']] for c in children])
        raw = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        offset = target - raw[0]
        shifted = raw + offset
        assert np.isclose(shifted[0], target)
        assert np.allclose(shifted - shifted[0], raw - raw[0])
        panels.append(dict(dataset=ds, label=label, colors=colors, seeds=seeds,
                           n_images=len(children[0]['image_names']), planned_seeds=batch['seeds'],
                           target_clean_px=target, measured_clean_px=float(raw[0]), additive_offset_px=float(offset),
                           conditions=order, corruptions=children[0]['run']['corruptions'],
                           raw_mean_px=raw.tolist(), displayed_mean_px=shifted.tolist(), sd_px=sd.tolist(),
                           source_files={str(p):digest(p) for p in paths}))
    assert panels[0]['conditions'] == panels[1]['conditions']
    out = base / 'combined_shifted' / f'droso_{len(panels[0]["seeds"]):02d}_cepha_{len(panels[1]["seeds"]):02d}'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'display_data.json', dict(
        kind='Baseline-shifted visualization, not measured MRE',
        formula='displayed MRE = measured MRE + additive_offset_px',
        sd_policy='SD unchanged by a fixed additive offset; sample SD across runs',
        created_utc=datetime.now(timezone.utc).isoformat(), panels=panels))
    names = panels[0]['corruptions']
    x = [0] + [2+4*g+l for g in range(len(names)) for l in range(3)]
    ticks = [0] + [3+4*g for g in range(len(names))]
    fig, axes = plt.subplots(2, 1, figsize=(14.6, 9.5))
    for i, (ax, panel) in enumerate(zip(axes, panels)):
        means, sd = np.array(panel['displayed_mean_px']), np.array(panel['sd_px'])
        bars = ax.bar(x, means, yerr=sd, capsize=2.5, width=.84,
                      color=['#888888'] + panel['colors']*len(names), error_kw={'elinewidth':1})
        ax.axhline(panel['target_clean_px'], color='#777777', linestyle='--', linewidth=1)
        pad = max(means+sd)*.009
        for bar, m, s in zip(bars, means, sd):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+pad, f'{m:.2f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(ticks)
        ax.set_xticklabels(['Clean']+[LABELS[n] for n in names], fontsize=9)
        ax.set_ylabel('Baseline-adjusted MRE (px)', fontsize=10)
        ax.set_title(f'({chr(97+i)}) {panel["label"]}', loc='left', fontsize=13, pad=10)
        ax.legend(handles=[Patch(color=c,label=f'Level {j+1}') for j,c in enumerate(panel['colors'])],
                  ncol=3, loc='upper left', frameon=False, fontsize=9)
        ax.set_ylim(0,max(means+sd)*1.22)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True,alpha=.15)
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=2)
    for ext in ('png','pdf'):
        fig.savefig(out / f'combined_degradation_shifted.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)

    # Standalone, editable vector figure: compile this file with pdflatex.
    latex = [r'\documentclass[tikz,border=3pt]{standalone}', r'\usepackage{pgfplots}',
             r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.18}']
    for i,panel in enumerate(panels):
        for j,color in enumerate(panel['colors']):
            latex.append(r'\definecolor{p%dlevel%d}{HTML}{%s}' % (i,j+1,color[1:]))
    latex += [r'\begin{document}',r'\begin{tikzpicture}',
              r'\begin{groupplot}[group style={group size=1 by 2,vertical sep=1.7cm},',
              r'width=18cm,height=6.2cm,xmin=-1.5,xmax=33.5,ymin=0,',
              r'axis x line*=bottom,axis y line*=left,ymajorgrids,grid style={gray!15},',
              'xtick={'+','.join(map(str,ticks))+'},',
              r'xticklabels={Clean,{Gaussian\\noise},{Gaussian\\blur},{Low\\contrast},Darken,Brighten,{Uneven\\illumination},{Low\\resolution},JPEG},',
              r'xticklabel style={align=center,font=\scriptsize},yticklabel style={font=\small},',
              r'ylabel={Baseline-adjusted MRE (px)},legend style={at={(0.01,0.99)},anchor=north west,draw=none,font=\scriptsize},legend columns=3]']
    for i,panel in enumerate(panels):
        maximum=max(m+s for m,s in zip(panel['displayed_mean_px'],panel['sd_px']))*1.22
        latex.append(r'\nextgroupplot[title={(%s) %s},title style={at={(0,1)},anchor=south west},ymax=%.8f]' % (chr(97+i),panel['label'],maximum))
        for series in range(4):
            indices = [0] if series==0 else list(range(series,len(x),3))
            color = 'gray' if series==0 else f'p{i}level{series}'
            opts = f'ybar,bar shift=0pt,bar width=4pt,draw=none,fill={color},mark=none'
            if series==0: opts += ',forget plot'
            opts += ',error bars/.cd,y dir=both,y explicit,error bar style={black,line width=0.4pt}'
            latex.append(r'\addplot+['+opts+r'] table[x=x,y=y,y error=sd] {')
            latex.append('x y sd')
            for k in indices:
                latex.append(f'{x[k]} {panel["displayed_mean_px"][k]:.10f} {panel["sd_px"][k]:.10f}')
            latex.append('};')
            if series: latex.append(r'\addlegendentry{Level %d}' % series)
        latex.append(r'\addplot[black!50,dashed,no marks,forget plot] coordinates {(-1.5,%.8f) (33.5,%.8f)};' % (panel['target_clean_px'],panel['target_clean_px']))
    latex += [r'\end{groupplot}',r'\end{tikzpicture}',r'\end{document}']
    (out/'combined_degradation_pgfplots.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    caption = ('Baseline-adjusted visualization of sensitivity to image degradation for '
               '(a) Droso-small and (b) Cepha. Bars and error bars show means and sample '
               f'standard deviations across {len(panels[0]["seeds"])} and {len(panels[1]["seeds"])} PA runs, respectively. '
               'For visualization only, dataset-specific constants '
               f'({panels[0]["additive_offset_px"]:+.6f} px and {panels[1]["additive_offset_px"]:+.6f} px, respectively) '
               'were added to all condition means to align the clean references to 4.80 and 22.66 px. '
               'These shifted values are not measured MRE; error bars and absolute differences from clean are unchanged.')
    snippet = '\n'.join([r'% Requires \usepackage{graphicx}',r'\begin{figure*}[t]',r'\centering',
                           r'\includegraphics[width=\textwidth]{combined_degradation_shifted.pdf}',
                           r'\caption{'+caption+'}',r'\label{fig:degradation-adjusted}',r'\end{figure*}'])
    (out/'figure_include.tex').write_text(snippet+'\n',encoding='utf-8')
    print(out)
    for p in panels:
        print(p['dataset'],len(p['seeds']),'seeds; raw clean',p['measured_clean_px'],'offset',p['additive_offset_px'])


if __name__ == '__main__':
    main()
