# Classification Fields

**Arbitrarily Fine Recursive Hierarchical Clustering From Few Examples**

[Paper](https://openreview.net/forum?id=dLTzQWpUx2)

A classification field predictor (CFP) learns a local refinement rule from an observed hierarchy and generates deeper levels by recursive rollout:

$$
x_{\alpha k} = x_\alpha + s^{|\alpha|+1}\hat r(x_\alpha)_k.
$$

Experiments cover matched CFG hierarchies, affine and nonlinear IFSs, and image-induced hierarchies built with frozen CLIP ViT-B/32 features and recursive SCP.

## Setup

Use Python 3.11 with PyTorch and torchvision matching your CUDA installation.

```bash
git clone https://github.com/YicenJoJo/classification-fields.git
cd classification-fields
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Synthetic experiments

```bash
python run_experiment.py cfg --output results/cfg
python run_experiment.py affine --output results/affine
python run_experiment.py nonlinear --output results/nonlinear
```

`matched_cfg.py` preserves the matched CFG experiment previously bundled with CIFAR. `synthetic_benchmarks.py` and `synthetic_benchmarks_nonlinear_ifs.py` contain the IFS suites. IFS runs observe levels 0–2 and evaluate levels 3–5 over five trials; matched CFG runs use nine trials. The synthetic CFP uses three ReLU hidden layers of width 64, with 2,000 epochs for IFS and 3,000 for CFG.

For ablations, replace the experiment name with `depth`, `scale`, `branching`, or `ordering`.

## Image experiments

`run_natural_experiment.py` provides `config`, `extract`, `run`, and `summarize` commands for all five datasets. Each run compares CFP with affine, learnable-constant, and average-residual predictors.

| Dataset argument | Support images | Seeds | Observed levels | Script evaluation | Table 2 evaluation |
|---|---:|---|---|---|---|
| `cifar10` | 50,000 | 42–46 | 0–5 | 6–7* | 6 |
| `cifar100` | 50,000 | 42–46 | 0–5 | 6–7* | 6 |
| `inaturalist` | 24,570 | 0–4 | 0–4 | 5–6 | 5–6 |
| `aircraft` | 6,667 | 0–4 | 0–3 | 4–5 | 4–5 |
| `food101` | 75,750 | 0–4 | 0–4 | 5–6 | 5 |

*CIFAR evaluates the available complete prefix; construction stops at small or empty splits. The script requests depth 7 for CIFAR and depth 6 for Food-101, one level beyond the paper's stated construction depths.

### Data and execution

Download datasets separately. CIFAR expects the extracted torchvision dataset folders; Food-101 expects `meta/` and `images/`; Aircraft expects the official `data/` directory, including bounding boxes. iNat819 uses `train_mini.json`, `val.json`, and their images, together with the supplied [split manifest](configs/inaturalist_split_manifest.json). Only support images enter hierarchy construction and predictor training; class labels are used for purity diagnostics.

```bash
python run_natural_experiment.py config --dataset all
python run_experiment.py cifar --cifar-root /data/cifar --output results/cifar
python run_natural_experiment.py run --dataset food101 --seed all \
  --data-root /data/food-101 --output-root results/natural
python run_natural_experiment.py run --dataset aircraft --seed all \
  --data-root /data/fgvc-aircraft-2013b/data --output-root results/natural
python run_natural_experiment.py run --dataset inaturalist --seed all \
  --data-root /data/inaturalist --manifest configs/inaturalist_split_manifest.json \
  --output-root results/natural
```

`run --data-root` extracts features before training. CLIP weights download on first use. Extraction writes caches to `<output-root>/inputs/<dataset>/tree_seed<seed>/features.npz`; use `--input-root <output-root>/inputs` to train from these caches, or `--features FILE` for one dataset and seed. `--trial 0` maps to seed 42 for CIFAR and seed 0 otherwise. CUDA is the default; `--allow-cpu` enables small diagnostic runs.

### Outputs and protocol notes

```bash
python run_natural_experiment.py summarize --dataset cifar10 --seed all --output-root results/cifar
python run_natural_experiment.py summarize --dataset food101 --seed all --output-root results/natural
```

Runs save metrics, checkpoints, centroids, predictions, and resolved settings under `runs/<dataset>/seed<seed>/`. Summaries are written under `release/` and replaced on each `summarize` call. Use `--dataset all` when all five datasets share an output root.

`per_seed_metrics.csv` and `summary_metrics.csv` contain per-depth results. `comparison_table.csv` averages **all script held-out depths** within each seed before computing mean and sample standard deviation across seeds. For Table 2, select the depths above, average them within each seed, and multiply values by 1,000. Metrics are coordinate-averaged MSE, symmetric squared Chamfer distance (PI-CD), and diameter-normalized tree-path distortion.

The supplied numerical implementation is retained. Its PCA ordering differs from Appendix F.3.4: CIFAR fits all constructed centroid levels; the other datasets fit observed parent–child displacements. Both center displacements before projection. CIFAR also reorders levels without propagating the permutation to descendant blocks, which can change parent–child relationships and path distances. These differences require reconciliation before claiming exact paper reproduction.

## Citation

```bibtex
@misc{classificationfields2026,
  title = {Classification Fields: Arbitrarily Fine Recursive Hierarchical Clustering From Few Examples},
  year  = {2026},
  url   = {https://openreview.net/forum?id=dLTzQWpUx2}
}
```

The image pipeline uses [CLIP](https://github.com/openai/CLIP) and SCP from *Keep it Light! Simplifying Image Clustering via Text-Free Adapters*.
