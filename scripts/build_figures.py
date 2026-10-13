"""Rebuild Figure 4 examples from corrected CIFAR reruns, in observed PCA coordinates."""
import importlib.util
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('figure_protocol',ROOT/'run_natural_experiment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def build():
    output=ROOT/'artifacts/results/release/figures';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.7),layout='constrained')
    provenance=[]
    for ax,dataset in zip(axes,['cifar10','cifar100']):
        run=ROOT/'artifacts/results/runs'/dataset/'seed42'
        m.verify_completion(run)
        metrics=m.read_json(run/'metrics.json');basis=np.asarray(metrics['canonicalization']['basis'])
        with np.load(run/'levels_and_predictions.npz') as arrays:
            truth=[arrays[f'true_level_{d}'] for d in range(7)]
            prediction=[arrays[f'cfp_level_{d}'] for d in range(7)]
        origin=truth[0][0].astype(np.float64)
        for values,color,dashed,label in [(truth,'#007A87',False,'SCP reference'),
                                          (prediction,'#C06014',True,'CFP rollout')]:
            projected=[(points.astype(np.float64)-origin)@basis.T for points in values]
            segments=[]
            for depth in range(1,7):
                segments.extend(np.stack([np.repeat(projected[depth-1],3,axis=0),projected[depth]],axis=1))
            ax.add_collection(LineCollection(segments,colors=color,linewidths=.45,
                              alpha=.22,linestyles='dashed' if dashed else 'solid'))
            observed=np.vstack(projected[:6]);heldout=projected[6]
            ax.scatter(observed[:,0],observed[:,1],s=4,c=color,alpha=.25,linewidths=0)
            ax.scatter(heldout[:,0],heldout[:,1],s=7 if dashed else 5,c=color,alpha=.7,
                       marker='x' if dashed else 'o',linewidths=.45 if dashed else 0,label=label)
        ax.scatter([0],[0],s=38,c='#172B36',marker='D',zorder=5)
        ax.autoscale();ax.margins(.08);ax.set_aspect('equal',adjustable='box')
        ax.set_title(f"{metrics['display_name']}  ·  seed 42",loc='left',fontweight='bold')
        ax.set_xlabel('Observed-centroid PC1');ax.set_ylabel('Observed-centroid PC2')
        ax.grid(alpha=.13,linewidth=.6)
        ax.text(0.01,0.99,'Observed 0–5 · held-out 6',transform=ax.transAxes,va='top',fontsize=9,color='#4B5563')
        provenance.append({'dataset':dataset,'seed':42,'metrics_sha256':m.sha256_file(run/'metrics.json'),
                           'levels_sha256':m.sha256_file(run/'levels_and_predictions.npz'),
                           'projection_fit_depths':list(range(6))})
    axes[1].legend(loc='lower right',frameon=True,facecolor='white',framealpha=.95,fontsize=9)
    fig.savefig(output/'figure4_corrected.pdf',bbox_inches='tight')
    fig.savefig(output/'figure4_corrected.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    m.write_json_atomic(output/'figure4_provenance.json',provenance)
    (output/'figure4_caption.tex').write_text(
        'Corrected CIFAR-induced hierarchies and CFP free rollouts (seed 42). '
        'Reference centroids and predicted nodes are projected from the scaled '
        '512-dimensional CLIP feature space using the same PCA basis fitted only '
        'on observed levels 0--5. Circles show SCP reference nodes and crosses '
        'show CFP predictions; all nodes and edges through held-out level 6 are shown. '
        'All quantitative metrics are computed before this visualization projection.\n')
    print(output/'figure4_corrected.pdf')
if __name__=='__main__': build()
