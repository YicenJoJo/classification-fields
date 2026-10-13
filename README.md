# Classification Fields

**Arbitrarily Fine Recursive Hierarchical Clustering From Few Examples**

[Paper](https://openreview.net/forum?id=dLTzQWpUx2)

A classification field predictor (CFP) learns a shared refinement rule from a
finite observed hierarchy and generates finer levels by recursive rollout:

$$
x_{\alpha k}=x_\alpha+s^{|\alpha|+1}\hat r(x_\alpha)_k.
$$

Experiments cover matched CFG hierarchies, affine and nonlinear IFSs, and
image-induced hierarchies built with frozen CLIP ViT-B/32 features and recursive
Soft Contrastive Partition (SCP). Baselines are affine, learnable-constant and
average-residual predictors. Evaluation uses ordered MSE, permutation-invariant
Chamfer distance (PI-CD) and normalized tree-path distortion.

## Setup

Use Python 3.12 and PyTorch/torchvision compatible with your CUDA environment.

```bash
git clone https://github.com/YicenJoJo/classification-fields.git
cd classification-fields
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Synthetic experiments

```bash
python run_experiment.py cfg --output results/cfg
python run_experiment.py affine --output results/affine
python run_experiment.py nonlinear --output results/nonlinear
```

CFG uses nine trials and nine unseen levels after observing levels 0–2.
IFS uses five trials and evaluates levels 3–5. Ablations are available as
`depth`, `scale`, `branching` and `ordering`.

## Image experiments

All image hierarchies are ternary. Child ordering uses PCA fitted on observed
centroids and angular projections of parent–child displacements, preserving
complete subtrees. Metrics use featurewise min–max-scaled CLIP features.

| Dataset | Support images | Seeds | Observed levels | Held-out levels |
|---|---:|---|---|---|
| `cifar10` | 50,000 | 42–46 | 0–5 | 6 |
| `cifar100` | 50,000 | 42–46 | 0–5 | 6 |
| `inaturalist` | 24,570 | 0–4 | 0–4 | 5–6 |
| `aircraft` | 6,667 | 0–4 | 0–3 | 4–5 |
| `food101` | 75,750 | 0–4 | 0–4 | 5 |

Download datasets separately. CIFAR and Food-101 use training images;
Aircraft uses train+validation with bounding-box crops. iNat819 uses the
supplied [split manifest](configs/inaturalist_split_manifest.json).

```bash
python run_natural_experiment.py config --dataset all
python run_experiment.py cifar --cifar-root /data/cifar --output artifacts/results
python run_natural_experiment.py run --dataset food101 --seed all --data-root /data/food-101 --output-root artifacts/results
python scripts/validate_results.py
python scripts/build_figures.py
```

For Aircraft or iNat819, select its dataset and data root; iNat819 also requires
`--manifest configs/inaturalist_split_manifest.json`. Existing feature caches
can be supplied with `--input-root PATH` or `--features FILE`.

## Outputs

Runs save metrics, checkpoints, trees and predictions under `artifacts/results/`.
Validation recomputes all metrics and exports tables. Held-out levels are averaged
within each seed before calculating the mean and sample standard deviation across
five seeds. CSVs retain raw metric units; JSON/LaTeX tables use units of $10^{-3}$.

[Results](results/table2/comparison_table.md) ·
[Per-seed metrics](results/table2/per_seed_metrics.csv) ·
[Figures](results/table2/figures/figure4_corrected.pdf) ·
[Tested environments](protocol/tested_environments.json)

The image pipeline uses [CLIP](https://github.com/openai/CLIP) and SCP from
*Keep it Light! Simplifying Image Clustering via Text-Free Adapters*.
