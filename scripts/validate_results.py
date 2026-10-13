#!/usr/bin/env python3
"""Inspect all frozen runs, recompute every Table 2 metric, and publish local release files."""
import datetime as dt
import importlib.util
import json
import pickle
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paper_release',ROOT/'run_natural_experiment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
DATASETS=['cifar10','cifar100','inaturalist','aircraft','food101']

def finalize():
    out=ROOT/'artifacts/results';records=[];failures=[]
    for dataset in DATASETS:
        cfg=m.dataset_config(dataset)
        for seed in m.dataset_seeds(dataset):
            run=out/'runs'/dataset/f'seed{seed}'
            record={'dataset':dataset,'seed':seed,'status':'missing'}
            try:
                if not (run/'complete.json').is_file():
                    task=ROOT/'artifacts/task_status'/f'{dataset}_seed{seed}.json'
                    if task.is_file(): record.update(m.read_json(task))
                    records.append(record);continue
                marker=m.verify_completion(run)
                payload=m.read_json(run/'metrics.json')
                if marker['run_identity']['script_sha256']!=m.sha256_file(ROOT/'run_natural_experiment.py'):
                    raise ValueError('implementation mismatch')
                if payload['heldout_depths']!=cfg['heldout_depths']: raise ValueError('wrong horizon')
                if payload['max_depth']!=cfg['max_depth']: raise ValueError('wrong maximum depth')
                if payload['canonicalization']['fit_depths']!=list(range(cfg['observed_depth']+1)):
                    raise ValueError('PCA observed-prefix mismatch')
                if payload['canonicalization']['displacement_centered'] is not False:
                    raise ValueError('PCA displacements were centered')
                if payload['genealogy_audit']['changed_edges']!=0: raise ValueError('changed genealogy')
                if set(payload['by_depth'])!=set(map(str,cfg['heldout_depths'])): raise ValueError('wrong metric depth keys')
                with np.load(run/'levels_and_predictions.npz',allow_pickle=False) as arrays:
                    truth=[arrays[f'true_level_{d}'] for d in range(cfg['max_depth']+1)]
                    predictions={method:[arrays[f'{slug}_level_{d}'] for d in range(cfg['max_depth']+1)]
                                 for method,slug in m.METHOD_SLUGS.items()}
                for depth,values in enumerate(truth):
                    if values.shape!=(3**depth,512) or not np.isfinite(values).all(): raise ValueError('invalid truth shape/values')
                with (run/'scp_tree.pkl').open('rb') as handle: tree=pickle.load(handle)
                snapshot=m.genealogy_snapshot(tree)
                if m.sha256_json(snapshot)!=payload['genealogy_audit']['snapshot_sha256']:
                    raise ValueError('saved-tree genealogy differs from audited identity')
                levels_from_tree=[[] for _ in truth]
                def inspect_node(node,depth):
                    levels_from_tree[depth].append(node['centroid'])
                    children=node.get('children') or []
                    if depth<cfg['max_depth']:
                        if len(children)!=3: raise ValueError('incomplete saved tree')
                        memberships=[set(map(int,c['indices'])) for c in children]
                        if not all(memberships) or set.union(*memberships)!=set(map(int,node['indices'])):
                            raise ValueError('invalid support membership partition')
                        if sum(map(len,memberships))!=len(set.union(*memberships)):
                            raise ValueError('overlapping child support membership')
                        displacement=np.asarray([c['centroid'] for c in children],dtype=np.float64)-np.asarray(node['centroid'],dtype=np.float64)
                        components=np.asarray(payload['canonicalization']['basis'])
                        projected=displacement@components.T
                        angles=np.arctan2(projected[:,1],projected[:,0])
                        keys=list(zip(angles,np.linalg.norm(displacement,axis=1)))
                        if keys!=sorted(keys): raise ValueError('saved sibling slots violate the angular rule')
                        for child in children: inspect_node(child,depth+1)
                    elif children: raise ValueError('saved tree exceeds paper maximum depth')
                inspect_node(tree,0)
                for depth,points in enumerate(levels_from_tree):
                    np.testing.assert_array_equal(np.stack(points),truth[depth])
                refit=m.fit_centroid_pca(truth,cfg['observed_depth'])
                np.testing.assert_allclose(refit.components_,payload['canonicalization']['basis'],rtol=1e-6,atol=1e-7)
                recomputed=m.evaluate_levels(truth,predictions,heldout_depths=cfg['heldout_depths'],branching=3)
                for depth in map(str,cfg['heldout_depths']):
                    for method in m.METHOD_SLUGS:
                        for metric in m.METRICS:
                            actual=recomputed['by_depth'][depth][method][metric]
                            saved=payload['by_depth'][depth][method][metric]
                            if not np.isfinite(saved) or not np.isclose(saved,actual,rtol=5e-4,atol=2e-6):
                                raise ValueError(f'metric mismatch {depth}/{method}/{metric}: {saved} vs {actual}')
                for method in m.METHOD_SLUGS:
                    for metric in m.METRICS:
                        expected=np.mean([payload['by_depth'][str(d)][method][metric] for d in cfg['heldout_depths']])
                        if not np.isclose(payload['heldout_average'][method][metric],expected,rtol=1e-12,atol=1e-12):
                            raise ValueError('per-seed horizon averaging mismatch')
                record.update(status='validated',heldout_average=payload['heldout_average'])
            except Exception as exc:
                record.update(status='invalid',error=str(exc));failures.append(record)
            records.append(record)
    complete=sum(row['status']=='validated' for row in records)
    status={'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'complete' if complete==25 else 'incomplete','validated_runs':complete,
            'expected_runs':25,'failures':failures,'runs':records}
    m.write_json_atomic(ROOT/'artifacts/audit/result_validation.json',status)
    print(json.dumps({k:v for k,v in status.items() if k!='runs'},indent=2),flush=True)
    if complete!=25: return 1
    m.aggregate(output_root=out,datasets=DATASETS)
    release=out/'release';table={}
    for dataset in DATASETS:
        selected=[r for r in records if r['dataset']==dataset]
        table[dataset]={method:{metric:{'mean':float(np.mean([r['heldout_average'][method][metric] for r in selected])*1000),
            'std':float(np.std([r['heldout_average'][method][metric] for r in selected],ddof=1)*1000)}
            for metric in m.METRICS} for method in m.METHOD_SLUGS}
    m.write_json_atomic(release/'table2_x1000.json',table)
    lines=[]
    for metric in m.METRICS:
        lines.append('% '+metric+'; units 10^{-3}; sample standard deviation across five per-seed horizon means')
        labels={'CFP':'CFP (Ours)','Linear Res Gen':'Affine Residual','Constant':'Learnable Const','Avg Residual':'Avg Residual'}
        for method in ['CFP','Linear Res Gen','Constant','Avg Residual']:
            lines.append(labels[method]+' & '+' & '.join(f"{table[d][method][metric]['mean']:.2f} $\\pm$ {table[d][method][metric]['std']:.2f}" for d in DATASETS)+r' \\')
    (release/'table2_replacement.tex').write_text('\n'.join(lines)+'\n')
    wins={d:{metric:all(table[d]['CFP'][metric]['mean']<table[d][b][metric]['mean'] for b in m.BASELINES)
             for metric in m.METRICS} for d in DATASETS}
    m.write_json_atomic(release/'claim_check.json',{'cfp_lower_mean_than_every_baseline':wins,
        'all_original_table2_mean_rank_claims_hold':all(all(row.values()) for row in wins.values()),
        'scope':'descriptive means over the fixed five seeds; no statistical significance claim'})
    return 0
if __name__=='__main__': raise SystemExit(finalize())
