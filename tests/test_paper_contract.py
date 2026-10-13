"""Behavioral checks for genealogy, observed-only fitting and complete experiment outputs."""
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
current = load('paper_natural', ROOT / 'run_natural_experiment.py')
legacy = load('legacy_natural', ROOT / 'legacy/run_natural_experiment.py')

def fixture_tree(k=3, depth=3, dimension=4):
    features = np.random.default_rng(17).normal(size=(k ** depth, dimension)).astype(np.float32)
    def build(indices, address):
        return dict(indices=indices, centroid=features[indices].mean(0), depth=len(address),
                    address=list(address), node_id=str(address),
                    children=[] if len(address)==depth else [build(part,address+(i,))
                    for i,part in enumerate(np.array_split(indices,k))])
    return build(np.arange(len(features)), ())

def edges(levels, k):
    return {(tuple(levels[d-1][i//k]), tuple(child))
            for d in range(1,len(levels)) for i,child in enumerate(levels[d])}

class PaperContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_seven_node_legacy_changes_four_edges_fixed_changes_zero(self):
        levels = [np.array([[0.,0.]]),np.array([[0.,1.],[0.,-1.]]),
                  np.array([[-1.,2.],[1.,2.],[-1.,-2.],[1.,-2.]])]
        basis=types.SimpleNamespace(components_=np.eye(2),mean_=np.zeros(2),transform=lambda x:x)
        old=legacy._cifar_canonicalize_levels_pca(levels,K=2,pca_model=basis)
        new=current._cifar_canonicalize_levels_pca(levels,K=2,pca_model=basis)
        self.assertEqual(len(edges(levels,2)-edges(old,2)),4)
        self.assertEqual(edges(levels,2),edges(new,2))
        self.assertFalse(np.array_equal(levels[1],new[1]))

    def test_deep_binary_and_ternary_edge_identity(self):
        for k,depth in ((2,5),(3,4)):
            rng=np.random.default_rng(81)
            levels=[rng.normal(size=(k**d,4)) for d in range(depth+1)]
            basis=current.fit_centroid_pca(levels,2)
            before=[x.copy() for x in levels]
            ordered=current._cifar_canonicalize_levels_pca(levels,K=k,pca_model=basis)
            self.assertEqual(edges(levels,k),edges(ordered,k))
            for a,b in zip(levels,before): np.testing.assert_array_equal(a,b)

    def test_pca_fits_observed_centroids_and_ignores_heldout(self):
        tree=fixture_tree();levels=current.tree_to_levels(tree,max_depth=3)
        actual=current.fit_observed_basis(tree,observed_depth=2)
        expected=PCA(n_components=2,svd_solver='full').fit(np.vstack(levels[:3]).astype(np.float64))
        np.testing.assert_allclose(actual['basis'].T@actual['basis'],expected.components_.T@expected.components_,atol=1e-12)
        np.testing.assert_allclose(actual['mean'],np.vstack(levels[:3]).mean(0),atol=1e-7)
        self.assertEqual(actual['fit_centroid_count'],13)
        for node in current.iter_nodes(tree):
            if node['depth']==3: node['centroid']+=100000
        changed=current.fit_observed_basis(tree,observed_depth=2)
        np.testing.assert_array_equal(actual['basis'],changed['basis'])
        self.assertEqual(actual['fit_depths'],[0,1,2])

    def test_direct_displacement_projection_does_not_use_pca_mean(self):
        levels=[np.array([[0.,0.]]),np.array([[1.,1.],[1.,-1.],[1.,0.]])]
        first=types.SimpleNamespace(components_=np.eye(2),mean_=np.array([100.,100.]))
        second=types.SimpleNamespace(components_=np.eye(2),mean_=np.array([-100.,-100.]))
        a=current._cifar_canonicalize_levels_pca(levels,pca_model=first)
        b=current._cifar_canonicalize_levels_pca(levels,pca_model=second)
        np.testing.assert_array_equal(a[1],[[1.,-1.],[1.,0.],[1.,1.]])
        np.testing.assert_array_equal(a[1],b[1])

    def test_tree_and_array_canonicalization_and_distances_agree(self):
        tree=fixture_tree();before=current.genealogy_snapshot(tree)
        levels=current.tree_to_levels(tree,max_depth=3)
        pca=current.fit_centroid_pca(levels,2)
        ordered,perms=current._cifar_canonicalize_levels_pca(levels,pca_model=pca,return_permutations=True)
        current.canonicalize(tree,basis=pca.components_,mean=pca.mean_)
        audit=current.validate_genealogy(before,current.genealogy_snapshot(tree))
        self.assertEqual(audit['changed_edges'],0)
        actual=current.tree_to_levels(tree,max_depth=3)
        for a,b in zip(actual,ordered): np.testing.assert_array_equal(a,b)
        old_dist=current.dpt_matrix(levels,upto_depth=3,branching=3)
        offsets=np.cumsum([0]+[len(v) for v in levels[:-1]])
        order=np.concatenate([p+o for p,o in zip(perms,offsets)])
        np.testing.assert_allclose(current.dpt_matrix(ordered,upto_depth=3,branching=3),
                                   old_dist[np.ix_(order,order)],atol=1e-6)

    def test_mean_cannot_change_tree_order(self):
        a=fixture_tree();b=copy.deepcopy(a);basis=current.fit_observed_basis(a,observed_depth=2)['basis']
        current.canonicalize(a,basis=basis,mean=np.full(4,9999.))
        current.canonicalize(b,basis=basis,mean=np.full(4,-9999.))
        for x,y in zip(current.tree_to_levels(a,max_depth=3),current.tree_to_levels(b,max_depth=3)):
            np.testing.assert_array_equal(x,y)

    def test_paper_horizons_and_observed_depths(self):
        expected={'cifar10':(6,5,[6]),'cifar100':(6,5,[6]),'food101':(5,4,[5]),
                  'aircraft':(5,3,[4,5]),'inaturalist':(6,4,[5,6])}
        for dataset,(maximum,observed,heldout) in expected.items():
            cfg=current.dataset_config(dataset)
            self.assertEqual((cfg['max_depth'],cfg['observed_depth'],cfg['heldout_depths']),
                             (maximum,observed,heldout))
        with self.assertRaises(ValueError):
            current._cifar_canonicalize_levels_pca([np.zeros((1,2)),np.zeros((3,2))])

    def test_real_training_rollout_metrics_and_completion_for_both_adapters(self):
        for dataset,seed in [('aircraft',0),('cifar10',42),('cifar100',42)]:
            with self.subTest(dataset=dataset),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);source=path/'features.npz';source.write_bytes(b'test-input')
                tree=fixture_tree();cfg=current.dataset_config(dataset)
                cfg.update(max_depth=3,observed_depth=2,heldout_depths=[3],cfp_epochs=2,
                           cfp_width=8,cfp_depth=4,cfp_print_every=2,support_count=27)
                features=current.DatasetFeatures(view_q=np.ones((27,4),dtype=np.float32),
                    view_k=np.ones((27,4),dtype=np.float32),deterministic=np.ones((27,4),dtype=np.float32),
                    ids=np.arange(27),labels={},source_path=source,source_sha256=current.sha256_file(source),scaling_audit={})
                with patch.object(current,'dataset_config',return_value=cfg),\
                     patch.object(current,'load_features',return_value=features),\
                     patch.object(current,'load_cifar_features',return_value=features),\
                     patch.object(current,'build_hierarchy',return_value=(copy.deepcopy(tree),[])),\
                     patch.object(current,'_cifar_build_recursive_scp_hierarchy',return_value=copy.deepcopy(tree)),\
                     patch.object(current,'_restore_cifar_rng'):
                    output=current.run_experiment(dataset,seed,require_cuda=False,device='cpu',
                                                  feature_path=source,output_root=path)
                    marker=current.verify_completion(output)
                    metrics=current.read_json(output/'metrics.json')
                    self.assertEqual(metrics['heldout_depths'],[3])
                    self.assertEqual(metrics['genealogy_audit']['changed_edges'],0)
                    self.assertEqual(set(metrics['heldout_average']),set(current.PROTOCOL['methods']))
                    for values in metrics['heldout_average'].values():
                        self.assertTrue(all(np.isfinite(list(values.values()))))
                    current.aggregate(output_root=path,datasets=[dataset],seed_map={dataset:[seed]})
                    (output/'scp_tree.pkl').write_bytes(b'corrupt')
                    with self.assertRaises(ValueError): current.verify_completion(output)
                    with self.assertRaises(ValueError):
                        current.aggregate(output_root=path,datasets=[dataset],seed_map={dataset:[seed]})

if __name__=='__main__': unittest.main()
