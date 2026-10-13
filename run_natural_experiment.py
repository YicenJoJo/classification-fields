#!/usr/bin/env python3
"""CLIP-SCP-CFP experiments on CIFAR-10/100, Food-101, Aircraft, and iNat819.

Use config, extract, run, or summarize; see README.md for dataset preparation
and the comparison with the paper protocol.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import pickle
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader, Dataset, TensorDataset

BUNDLE_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = BUNDLE_ROOT / "artifacts" / "inputs"
DEFAULT_OUTPUT_ROOT = BUNDLE_ROOT / "artifacts" / "results"
PROTOCOL_JSON = r'''{
  "schema_version": 1,
  "template": "../anonymous_cifar100_clip.py",
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "encoder": {
    "name": "OpenAI CLIP ViT-B/32",
    "dimension": 512,
    "pre_scp_normalization": "independent_featurewise_minmax",
    "epsilon": 1e-08
  },
  "canonicalization": {
    "method": "observed_prefix_displacement_pca_angle",
    "components": 2
  },
  "metrics": [
    "ordered_mse",
    "symmetric_squared_chamfer",
    "normalized_dpt_distortion"
  ],
  "methods": [
    "CFP",
    "Avg Residual",
    "Linear Res Gen",
    "Constant"
  ],
  "datasets": {
    "inaturalist": {
      "display_name": "iNaturalist 2021 mini (balanced 819 species)",
      "input_pattern": "artifacts/inputs/inaturalist/tree_seed{seed}/features.npz",
      "label_manifest": "artifacts/inputs/inaturalist/split_manifest.json",
      "support_count": 24570,
      "branching": 3,
      "max_depth": 6,
      "observed_depth": 4,
      "heldout_depths": [
        5,
        6
      ],
      "scp_epochs_per_node": 30,
      "scp_learning_rate": 0.001,
      "scp_batch_size": 512,
      "scp_min_parent_size": 3,
      "scp_max_retries": 5,
      "cfp_width": 512,
      "cfp_depth": 4,
      "cfp_learning_rate": 0.001,
      "cfp_epochs": 2500,
      "cfp_weight_decay": 0.0,
      "scale_clip": [
        0.5,
        0.98
      ],
      "purity_labels": [
        "species",
        "genus"
      ]
    },
    "aircraft": {
      "display_name": "FGVC-Aircraft 2013b",
      "input_pattern": "artifacts/inputs/aircraft/tree_seed{seed}/features.npz",
      "data_root": "../data/fgvc-aircraft/fgvc-aircraft-2013b/data",
      "support_count": 6667,
      "branching": 3,
      "max_depth": 5,
      "observed_depth": 3,
      "heldout_depths": [
        4,
        5
      ],
      "scp_epochs_per_node": 40,
      "scp_learning_rate": 0.001,
      "scp_batch_size": 256,
      "scp_min_parent_size": 3,
      "scp_max_retries": 7,
      "cfp_width": 384,
      "cfp_depth": 4,
      "cfp_learning_rate": 0.001,
      "cfp_epochs": 2500,
      "cfp_weight_decay": 0.0,
      "scale_clip": [
        0.5,
        0.98
      ],
      "purity_labels": [
        "variant",
        "family"
      ]
    },
    "food101": {
      "display_name": "Food-101",
      "input_pattern": "artifacts/inputs/food101/tree_seed{seed}/features.npz",
      "data_root": "data/food101",
      "support_count": 75750,
      "evaluation_count": 25250,
      "branching": 3,
      "max_depth": 6,
      "observed_depth": 4,
      "heldout_depths": [
        5,
        6
      ],
      "scp_epochs_per_node": 30,
      "scp_learning_rate": 0.001,
      "scp_batch_size": 512,
      "scp_min_parent_size": 3,
      "scp_max_retries": 7,
      "cfp_width": 512,
      "cfp_depth": 4,
      "cfp_learning_rate": 0.001,
      "cfp_epochs": 2500,
      "cfp_weight_decay": 0.0,
      "scale_clip": [
        0.5,
        0.98
      ],
      "purity_labels": [
        "class"
      ]
    }
  }
}
'''
PROTOCOL = json.loads(PROTOCOL_JSON)
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_JSON.encode()).hexdigest()


# CIFAR presets
CIFAR_PRESETS = json.loads(r'''{
  "cifar10": {
    "display_name": "CIFAR-10",
    "protocol": "original_cifar_scp",
    "source": "../anonymous_cifar100_clip.py",
    "seeds": [
      42,
      43,
      44,
      45,
      46
    ],
    "input_pattern": "artifacts/inputs/cifar10/tree_seed{seed}/features.npz",
    "support_count": 50000,
    "evaluation_count": 0,
    "classes": 10,
    "branching": 3,
    "max_depth": 7,
    "observed_depth": 5,
    "heldout_depths": [
      6,
      7
    ],
    "scp_epochs_per_node": 30,
    "scp_learning_rate": 0.001,
    "scp_batch_size": 512,
    "scp_min_parent_size": 50,
    "scp_max_retries": 0,
    "cfp_width": 1024,
    "cfp_depth": 7,
    "cfp_activation": "gelu",
    "cfp_layernorm": false,
    "cfp_learning_rate": 0.001,
    "cfp_epochs": 3000,
    "cfp_weight_decay": 0.0,
    "cfp_print_every": 300,
    "scale": 0.9,
    "scale_estimation": "fixed",
    "clip_batch_size": 256,
    "clip_num_workers": 4,
    "clip_l2_normalize": false,
    "feature_dtype": "float32",
    "augmentation": "CLIP preprocess, then randomly select padded RandomCrop or GaussianBlur independently per view",
    "canonicalization": "original_levelwise_pca_angle_all_centroids",
    "model_rng": "continuous stream after CLIP extraction, without per-node or per-model reseeding",
    "incomplete_tree": "use maximum full depth, clamp heldout depths; fail if no unseen depth remains",
    "purity_labels": [
      "fine"
    ]
  },
  "cifar100": {
    "display_name": "CIFAR-100",
    "protocol": "original_cifar_scp",
    "source": "../anonymous_cifar100_clip.py",
    "seeds": [
      42,
      43,
      44,
      45,
      46
    ],
    "input_pattern": "artifacts/inputs/cifar100/tree_seed{seed}/features.npz",
    "support_count": 50000,
    "evaluation_count": 0,
    "classes": 100,
    "branching": 3,
    "max_depth": 7,
    "observed_depth": 5,
    "heldout_depths": [
      6,
      7
    ],
    "scp_epochs_per_node": 30,
    "scp_learning_rate": 0.001,
    "scp_batch_size": 512,
    "scp_min_parent_size": 50,
    "scp_max_retries": 0,
    "cfp_width": 1024,
    "cfp_depth": 7,
    "cfp_activation": "gelu",
    "cfp_layernorm": false,
    "cfp_learning_rate": 0.001,
    "cfp_epochs": 3000,
    "cfp_weight_decay": 0.0,
    "cfp_print_every": 300,
    "scale": 0.9,
    "scale_estimation": "fixed",
    "clip_batch_size": 256,
    "clip_num_workers": 4,
    "clip_l2_normalize": false,
    "feature_dtype": "float32",
    "augmentation": "CLIP preprocess, then randomly select padded RandomCrop or GaussianBlur independently per view",
    "canonicalization": "original_levelwise_pca_angle_all_centroids",
    "model_rng": "continuous stream after CLIP extraction, without per-node or per-model reseeding",
    "incomplete_tree": "use maximum full depth, clamp heldout depths; fail if no unseen depth remains",
    "purity_labels": [
      "fine",
      "coarse"
    ]
  }
}
''')
PROTOCOL["datasets"].update(CIFAR_PRESETS)

# Appendix F.3.4 and Table 2, supplied camera-ready draft.
PROTOCOL["schema_version"] = 2
PROTOCOL["revision"] = "paper_centroid_pca_genealogy_v1"
PROTOCOL["canonicalization"] = {
    "method": "observed_centroid_pca_direct_displacement_angle",
    "components": 2, "svd_solver": "full",
    "component_sign": "largest_absolute_loading_positive",
    "projection": "displacement @ components.T; no subtraction of PCA mean",
    "tie_breaker": "Euclidean displacement norm, then stable input slot",
}
for _dataset in ("cifar10", "cifar100"):
    PROTOCOL["datasets"][_dataset].update(max_depth=6, heldout_depths=[6],
        canonicalization=PROTOCOL["canonicalization"]["method"],
        incomplete_tree="fail unless all paper evaluation depths are complete")
PROTOCOL["datasets"]["food101"].update(max_depth=5, heldout_depths=[5])
PROTOCOL_JSON = json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":"))
PROTOCOL_SHA256 = hashlib.sha256(PROTOCOL_JSON.encode()).hexdigest()



def dataset_seeds(dataset: str) -> list[int]:
    return list(PROTOCOL["datasets"][dataset].get("seeds", PROTOCOL["seeds"]))


def resolve_seeds(dataset: str, seed: str | None = None, trial: int | None = None,
                  *, all_by_default: bool = False) -> list[int]:
    available = dataset_seeds(dataset)
    if trial is not None:
        if not 0 <= trial < len(available):
            raise ValueError(f"Invalid trial {trial} for {dataset}")
        return [available[trial]]
    if seed == "all" or (seed is None and all_by_default):
        return available
    selected = available[0] if seed is None else int(seed)
    if selected not in available:
        raise ValueError(f"{dataset} seeds are {available}, got {selected}; use --trial to select by index")
    return [selected]


def load_protocol() -> dict[str, Any]:
    return copy.deepcopy(PROTOCOL)


def dataset_config(dataset: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(PROTOCOL["datasets"][dataset])
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset}") from exc


# --- io ---

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json_atomic(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# --- data ---

@dataclass
class DatasetFeatures:
    view_q: np.ndarray
    view_k: np.ndarray
    deterministic: np.ndarray | None
    ids: np.ndarray
    labels: dict[str, np.ndarray]
    source_path: Path
    source_sha256: str
    scaling_audit: dict[str, Any]


def _featurewise_minmax(values: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float32)
    minima = array.min(axis=0, keepdims=True)
    maxima = array.max(axis=0, keepdims=True)
    span = maxima - minima
    scaled = (array - minima) / (span + epsilon)
    norms_before = np.linalg.norm(array, axis=1)
    norms_after = np.linalg.norm(scaled, axis=1)
    audit = {
        "rows": int(len(array)),
        "dimension": int(array.shape[1]),
        "constant_dimensions": int(np.sum(span.reshape(-1) <= epsilon)),
        "norm_before_mean": float(norms_before.mean()),
        "norm_before_min": float(norms_before.min()),
        "norm_before_max": float(norms_before.max()),
        "norm_after_mean": float(norms_after.mean()),
        "norm_after_min": float(norms_after.min()),
        "norm_after_max": float(norms_after.max()),
        "value_after_min": float(scaled.min()),
        "value_after_max": float(scaled.max()),
    }
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite values after feature-wise min-max scaling")
    return scaled.astype(np.float32), audit


def _inat_labels(ids: np.ndarray, manifest_path: Path) -> dict[str, np.ndarray]:
    manifest = json.loads(manifest_path.read_text())
    image_to_species: dict[int, int] = {}
    image_to_genus: dict[int, int] = {}
    validate_inat_manifest(manifest)
    for species_index, row in enumerate(manifest["rows"]):
        for image_id in row["support_ids"]:
            image_to_species[int(image_id)] = species_index
            image_to_genus[int(image_id)] = int(row["genus_label"])
    missing = [int(value) for value in ids if int(value) not in image_to_species]
    if missing or set(map(int, ids)) != set(image_to_species):
        raise ValueError("iNaturalist support IDs differ from the frozen split manifest")
    return {
        "species": np.asarray([image_to_species[int(value)] for value in ids], dtype=np.int64),
        "genus": np.asarray([image_to_genus[int(value)] for value in ids], dtype=np.int64),
    }


def load_features(
    dataset: str, seed: int, *, feature_path: Path | None = None,
    input_root: Path = DEFAULT_INPUT_ROOT, manifest_path: Path | None = None,
) -> DatasetFeatures:
    if dataset in CIFAR_PRESETS:
        return load_cifar_features(dataset, seed, feature_path=feature_path, input_root=input_root)
    protocol = load_protocol()
    config = dataset_config(dataset)
    source = resolve_feature_path(dataset, seed, feature_path, input_root)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        required = {"support_view_q", "support_view_k", "support_deterministic", "support_ids"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{source}: missing arrays {sorted(missing)}")
        raw_q = np.asarray(payload["support_view_q"], dtype=np.float32)
        raw_k = np.asarray(payload["support_view_k"], dtype=np.float32)
        deterministic = np.asarray(payload["support_deterministic"], dtype=np.float32)
        ids = np.asarray(payload["support_ids"]).copy()
        if dataset == "aircraft":
            labels = {
                "variant": np.asarray(payload["support_variant_labels"]).copy(),
                "family": np.asarray(payload["support_family_labels"]).copy(),
            }
        elif dataset == "food101":
            labels = {
                "class": np.asarray(payload["support_class_labels"]).copy(),
            }
        else:
            labels = {
                name: np.asarray(payload[f"support_{name}_labels"]).copy()
                for name in ("species", "genus")
                if f"support_{name}_labels" in payload
            }
    expected = int(config["support_count"])
    if raw_q.shape != (expected, int(protocol["encoder"]["dimension"])):
        raise ValueError(f"{source}: q shape {raw_q.shape} != expected")
    if raw_k.shape != raw_q.shape or deterministic.shape != raw_q.shape:
        raise ValueError(f"{source}: feature view shapes differ")
    if ids.ndim != 1 or len(ids) != expected or len(np.unique(ids)) != expected:
        raise ValueError(f"{source}: support IDs are not unique and complete")
    for name, values in (("q", raw_q), ("k", raw_k), ("deterministic", deterministic)):
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {name} contains non-finite values")
    if dataset == "inaturalist":
        if manifest_path is not None or set(labels) != {"species", "genus"}:
            manifest_path = manifest_path or input_root / "inaturalist" / "split_manifest.json"
            labels = _inat_labels(ids, manifest_path)
        _, species_counts = np.unique(labels["species"], return_counts=True)
        if len(species_counts) != 819 or np.any(species_counts != 30):
            raise ValueError("Expected 819 iNaturalist species with 30 support images each")
        if len(np.unique(labels["genus"])) != 273:
            raise ValueError("Expected 273 iNaturalist genera")
    for label_name in config["purity_labels"]:
        if label_name not in labels or labels[label_name].shape != (expected,):
            raise ValueError(f"{source}: invalid {label_name} label coverage")
    epsilon = float(protocol["encoder"]["epsilon"])
    scaled_q, q_audit = _featurewise_minmax(raw_q, epsilon)
    scaled_k, k_audit = _featurewise_minmax(raw_k, epsilon)
    cosine = np.sum(
        raw_q / np.linalg.norm(raw_q, axis=1, keepdims=True).clip(min=1e-12)
        * raw_k / np.linalg.norm(raw_k, axis=1, keepdims=True).clip(min=1e-12),
        axis=1,
    )
    return DatasetFeatures(
        view_q=scaled_q,
        view_k=scaled_k,
        deterministic=deterministic,
        ids=ids,
        labels=labels,
        source_path=source,
        source_sha256=sha256_file(source),
        scaling_audit={
            "transform": "independent_featurewise_minmax",
            "q": q_audit,
            "k": k_audit,
            "raw_qk_cosine_mean": float(cosine.mean()),
            "raw_qk_cosine_min": float(cosine.min()),
        },
    )


def resolve_feature_path(
    dataset: str, seed: int, feature_path: Path | None, input_root: Path,
) -> Path:
    return (feature_path or input_root / dataset / f"tree_seed{seed}" / "features.npz").resolve()


def validate_inat_manifest(manifest: dict[str, Any]) -> None:
    rows = manifest["rows"]
    if len(rows) != 819 or len({r["category_id"] for r in rows}) != 819:
        raise ValueError("iNaturalist manifest must contain 819 distinct species")
    genera, counts = np.unique([r["genus_label"] for r in rows], return_counts=True)
    if len(genera) != 273 or np.any(counts != 3):
        raise ValueError("iNaturalist manifest must contain 273 genera with 3 species each")
    seen: set[int] = set()
    for row in rows:
        for role, expected in (("support", 30), ("cluster", 20), ("evaluation", 10)):
            ids = list(map(int, row[f"{role}_ids"]))
            if len(ids) != expected or len(set(ids)) != expected or seen.intersection(ids):
                raise ValueError(f"Invalid or overlapping {role} IDs for {row['category_id']}")
            seen.update(ids)


# --- scp ---

class TopologyFailure(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        address: tuple[int, ...],
        attempt_records: list[dict[str, Any]],
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.address = address
        self.attempt_records = list(attempt_records)


class SCPClusterHead(nn.Module):
    """SCP MLP with softmax cluster assignments."""

    def __init__(self, dimension: int, clusters: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 1024),
            nn.ReLU(),
            nn.Linear(1024, 786),
            nn.BatchNorm1d(786),
            nn.ReLU(),
            nn.Linear(786, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, clusters),
            nn.Softmax(dim=1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)


def stable_seed(
    dataset: str, tree_seed: int, address: tuple[int, ...], retry: int
) -> int:
    key = f"{dataset}|{tree_seed}|{','.join(map(str, address))}|{retry}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_assignment(
    view_q: np.ndarray,
    view_k: np.ndarray,
    indices: np.ndarray,
    *,
    clusters: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(indices) < clusters:
        raise ValueError(f"Cannot split {len(indices)} rows into {clusters} clusters")
    _seed_torch(seed)
    q_tensor = torch.from_numpy(np.asarray(view_q[indices], dtype=np.float32))
    k_tensor = torch.from_numpy(np.asarray(view_k[indices], dtype=np.float32))
    effective_batch = min(int(batch_size), len(indices))
    drop_last = len(indices) > effective_batch and len(indices) % effective_batch == 1
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(q_tensor, k_tensor),
        batch_size=effective_batch,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
    )
    model = SCPClusterHead(q_tensor.shape[1], clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    final_loss = float("nan")
    updates = 0
    for _ in range(epochs):
        model.train()
        for first, second in loader:
            if len(first) < 2:
                continue
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            p = model(first)
            q = model(second)
            soft_cross_entropy = nn.functional.cross_entropy(p, q)
            consistency = -torch.log(torch.sum(p * q, dim=1).clamp_min(1e-8)).mean()
            p_bar = p.mean(0)
            q_bar = q.mean(0)
            balance_entropy = -torch.sum(p_bar * torch.log(p_bar.clamp_min(1e-8)))
            balance_entropy -= torch.sum(q_bar * torch.log(q_bar.clamp_min(1e-8)))
            loss = soft_cross_entropy + consistency - balance_entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SCP loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            updates += 1
        scheduler.step()
    if updates == 0:
        raise RuntimeError("SCP made no optimizer updates")
    model.eval()
    assignments = []
    probabilities = []
    eval_loader = DataLoader(
        TensorDataset(q_tensor, k_tensor), batch_size=effective_batch, shuffle=False
    )
    with torch.inference_mode():
        for first, _ in eval_loader:
            probability = model(first.to(device, non_blocking=True))
            probabilities.append(probability.cpu())
            assignments.append(probability.argmax(dim=1).cpu())
    probability = torch.cat(probabilities).numpy()
    assignment = torch.cat(assignments).numpy().astype(np.int64)
    return assignment, {
        "updates": updates,
        "final_loss": final_loss,
        "mean_max_probability": float(probability.max(axis=1).mean()),
        "assignment_entropy": float(
            -np.sum(
                np.bincount(assignment, minlength=clusters) / len(assignment)
                * np.log(
                    np.bincount(assignment, minlength=clusters)
                    / len(assignment)
                    + 1e-12
                )
            )
        ),
    }


def build_hierarchy(
    view_q: np.ndarray,
    view_k: np.ndarray,
    *,
    dataset: str,
    tree_seed: int,
    clusters: int,
    max_depth: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    min_parent_size: int,
    max_retries: int,
    device: str | torch.device | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if view_q.shape != view_k.shape:
        raise ValueError("SCP views have different shapes")
    if min_parent_size < clusters:
        raise ValueError("min_parent_size must be at least the branching factor")
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    attempt_records: list[dict[str, Any]] = []

    def build(indices: np.ndarray, address: tuple[int, ...]) -> dict[str, Any]:
        node_id = hashlib.sha256(
            f"{dataset}|{tree_seed}|{','.join(map(str, address))}".encode()
        ).hexdigest()[:24]
        node: dict[str, Any] = {
            "node_id": node_id,
            "address": list(address),
            "depth": len(address),
            "indices": np.asarray(indices, dtype=np.int64),
            "centroid": np.asarray(view_q[indices].mean(axis=0), dtype=np.float32),
            "children": [],
        }
        if len(address) == max_depth:
            return node
        if len(indices) < min_parent_size:
            raise TopologyFailure(
                f"node {address} has {len(indices)} < {min_parent_size}",
                address=address,
                attempt_records=attempt_records,
            )
        accepted: np.ndarray | None = None
        for retry in range(max_retries + 1):
            seed = stable_seed(dataset, tree_seed, address, retry)
            assignments, audit = train_assignment(
                view_q,
                view_k,
                indices,
                clusters=clusters,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                device=resolved_device,
            )
            counts = np.bincount(assignments, minlength=clusters)
            valid = bool(np.all(counts > 0))
            attempt_records.append(
                {
                    "address": list(address),
                    "depth": len(address),
                    "parent_size": int(len(indices)),
                    "retry": retry,
                    "seed": seed,
                    "counts": counts.astype(int).tolist(),
                    "accepted": valid,
                    **audit,
                }
            )
            if valid:
                accepted = assignments
                break
            print(
                f"SCP_RETRY dataset={dataset} tree_seed={tree_seed} "
                f"address={address} retry={retry} counts={counts.tolist()}",
                flush=True,
            )
        if accepted is None:
            raise TopologyFailure(
                f"topology retries exhausted at {address}",
                address=address,
                attempt_records=attempt_records,
            )
        node["children"] = [
            build(indices[accepted == slot], (*address, slot))
            for slot in range(clusters)
        ]
        return node

    root = build(np.arange(len(view_q), dtype=np.int64), ())
    return root, attempt_records


# --- hierarchy ---

Tree = dict[str, Any]


def iter_nodes(root: Tree) -> Iterator[Tree]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.get("children", [])))


def nodes_by_depth(root: Tree) -> dict[int, list[Tree]]:
    grouped: dict[int, list[Tree]] = defaultdict(list)
    for node in iter_nodes(root):
        grouped[int(node["depth"])].append(node)
    for depth in grouped:
        grouped[depth].sort(key=lambda row: tuple(row["address"]))
    return dict(grouped)


def audit_topology(root: Tree, *, branching: int, max_depth: int) -> dict[str, Any]:
    failures: list[str] = []
    grouped = nodes_by_depth(root)
    root_indices = set(map(int, root["indices"]))
    split_minima = []
    for node in iter_nodes(root):
        depth = int(node["depth"])
        indices = set(map(int, node["indices"]))
        centroid = np.asarray(node["centroid"])
        if centroid.ndim != 1 or not np.isfinite(centroid).all():
            failures.append(f"invalid centroid at {node['address']}")
        children = node.get("children", [])
        if depth < max_depth:
            if len(children) != branching:
                failures.append(f"branching={len(children)} at {node['address']}")
                continue
            child_sets = [set(map(int, child["indices"])) for child in children]
            if any(not child for child in child_sets):
                failures.append(f"empty child at {node['address']}")
            union = set().union(*child_sets)
            if union != indices or sum(map(len, child_sets)) != len(union):
                failures.append(f"coverage/disjointness at {node['address']}")
            split_minima.append(min(map(len, child_sets)))
        elif children:
            failures.append(f"children below max depth at {node['address']}")
    expected = {depth: branching**depth for depth in range(max_depth + 1)}
    actual = {depth: len(grouped.get(depth, [])) for depth in expected}
    if actual != expected:
        failures.append(f"node counts {actual} != {expected}")
    leaves = grouped.get(max_depth, [])
    leaf_union = set().union(*(set(map(int, node["indices"])) for node in leaves))
    if leaf_union != root_indices:
        failures.append("leaf coverage differs from root")
    leaf_sizes = [len(node["indices"]) for node in leaves]
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "node_counts_by_depth": actual,
        "root_support_count": len(root_indices),
        "leaf_size_min": min(leaf_sizes, default=0),
        "leaf_size_median": float(np.median(leaf_sizes)) if leaf_sizes else 0.0,
        "leaf_size_max": max(leaf_sizes, default=0),
        "accepted_split_min_child_size": min(split_minima, default=0),
    }


def fit_centroid_pca(levels, observed_depth):
    """Fit on H_0..H_Ltrain; the PCA mean is never subtracted from a displacement."""
    if not 1 <= observed_depth < len(levels):
        raise ValueError("PCA requires a nonempty observed prefix")
    matrix = np.vstack(levels[:observed_depth + 1]).astype(np.float64)
    if matrix.shape[1] < 2:
        raise ValueError("Angular canonicalization requires at least two dimensions")
    model = PCA(n_components=2, svd_solver="full").fit(matrix)
    for component in model.components_:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    return model


def genealogy_snapshot(root):
    """Intrinsic identities use support membership, independent of ordered addresses."""
    nodes, edges = [], []
    def visit(node):
        membership = np.sort(np.asarray(node["indices"], dtype=np.int64))
        identity = hashlib.sha256(membership.tobytes()).hexdigest()
        coordinates = hashlib.sha256(np.asarray(node["centroid"]).tobytes()).hexdigest()
        nodes.append((identity, coordinates, int(node["depth"])))
        for child in node.get("children") or []:
            edges.append((identity, visit(child)))
        return identity
    visit(root)
    return {"nodes": sorted(nodes), "edges": sorted(edges)}


def validate_genealogy(before, after):
    if before != after:
        raise RuntimeError("Canonicalization changed coordinates, membership, or parent-child edges")
    return {"status": "passed", "node_count": len(before["nodes"]),
            "edge_count": len(before["edges"]), "changed_edges": 0,
            "snapshot_sha256": sha256_json(before)}


def verify_completion(output, identity=None):
    output = Path(output)
    marker = read_json(output / "complete.json")
    if marker.get("status") != "complete":
        raise ValueError(f"Invalid completion status at {output}")
    if identity is not None and marker.get("run_identity") != identity:
        raise ValueError(f"Run identity mismatch at {output}")
    for name, key in (("metrics.json", "metrics_sha256"),
                      ("levels_and_predictions.npz", "levels_sha256"),
                      ("scp_tree.pkl", "tree_sha256")):
        if marker.get(key) != sha256_file(output / name):
            raise ValueError(f"Artifact checksum mismatch: {output / name}")
    return marker


def fit_observed_basis(root: Tree, *, observed_depth: int, components: int = 2) -> dict[str, Any]:
    if components != 2:
        raise ValueError("The paper specifies two principal directions")
    grouped = nodes_by_depth(root)
    levels = [np.stack([node["centroid"] for node in grouped[d]])
              for d in range(observed_depth + 1)]
    pca = fit_centroid_pca(levels, observed_depth)
    return {"method": PROTOCOL["canonicalization"]["method"],
            "basis": pca.components_, "mean": pca.mean_,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "fit_depths": list(range(observed_depth + 1)),
            "fit_centroid_count": sum(map(len, levels)),
            "component_sign": PROTOCOL["canonicalization"]["component_sign"],
            "displacement_centered": False}


def canonicalize(root: Tree, *, basis: np.ndarray, mean: np.ndarray) -> None:
    basis = np.asarray(basis, dtype=np.float64)
    root.setdefault("address", [])

    def visit(parent: Tree) -> None:
        children = parent.get("children") or []
        if not children:
            return
        origin = np.asarray(parent["centroid"], dtype=np.float64)
        keys = []
        for child in children:
            displacement = np.asarray(child["centroid"], dtype=np.float64) - origin
            projection = displacement @ basis.T
            angle = float(np.arctan2(projection[1], projection[0]))
            distance = float(np.linalg.norm(displacement))
            keys.append((angle, distance))
        parent["children"] = [
            children[index] for index in sorted(range(len(children)), key=keys.__getitem__)
        ]
        for slot, child in enumerate(parent["children"]):
            child["address"] = list(parent["address"]) + [slot]
            visit(child)

    visit(root)


def tree_to_levels(root: Tree, *, max_depth: int) -> list[np.ndarray]:
    grouped = nodes_by_depth(root)
    levels = []
    for depth in range(max_depth + 1):
        nodes = grouped.get(depth, [])
        if len(nodes) == 0:
            raise ValueError(f"Missing hierarchy depth {depth}")
        levels.append(
            np.stack([np.asarray(node["centroid"], dtype=np.float32) for node in nodes])
        )
    return levels


def cluster_purity(
    root: Tree, labels: np.ndarray, depth: int
) -> dict[str, float | int]:
    nodes = nodes_by_depth(root).get(depth, [])
    purities = []
    weights = []
    for node in nodes:
        values = np.asarray(labels)[np.asarray(node["indices"], dtype=np.int64)]
        _, counts = np.unique(values, return_counts=True)
        purities.append(float(counts.max() / len(values)))
        weights.append(len(values))
    return {
        "clusters": len(nodes),
        "macro_purity": float(np.mean(purities)),
        "weighted_purity": float(np.average(purities, weights=weights)),
    }


# --- cfp ---

class BottleneckCFP(nn.Module):
    ACT_MAP = {
        "relu": lambda: nn.ReLU(),
        "gelu": lambda: nn.GELU(),
        "silu": lambda: nn.SiLU(),
    }

    def __init__(
        self,
        *,
        dimension: int,
        branching: int,
        width: int,
        depth: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("Bottleneck CFP requires depth >= 2")
        make_activation = self.ACT_MAP[activation]
        layers: list[nn.Module] = [nn.Linear(dimension, width), make_activation()]
        for _ in range(depth - 2):
            layers.extend((nn.Linear(width, width), make_activation()))
        layers.extend(
            (
                nn.Linear(width, dimension),
                make_activation(),
                nn.Linear(dimension, dimension * branching),
            )
        )
        self.net = nn.Sequential(*layers)
        self.dimension = dimension
        self.branching = branching
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).reshape(-1, self.branching, self.dimension)


class LinearResidualGenerator(nn.Module):
    def __init__(self, dimension: int, branching: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.branching = branching
        self.linear = nn.Linear(dimension, dimension * branching)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values).reshape(
            -1, self.branching, self.dimension
        )


class ConstantResidualGenerator(nn.Module):
    def __init__(self, dimension: int, branching: int) -> None:
        super().__init__()
        self.residuals = nn.Parameter(torch.randn(branching, dimension) * 0.1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.residuals.unsqueeze(0).expand(len(values), -1, -1)


def hierarchy_to_training_data(
    levels: list[np.ndarray], *, branching: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parents = []
    targets = []
    scales = []
    for parent_depth in range(len(levels) - 1):
        parent = levels[parent_depth]
        children = levels[parent_depth + 1]
        if len(children) != len(parent) * branching:
            raise ValueError("Hierarchy is not full K-ary")
        parents.append(parent)
        targets.append(children.reshape(len(parent), branching, parent.shape[1]))
        scales.append(
            np.full(
                (len(parent), 1, 1),
                scale ** (parent_depth + 1),
                dtype=np.float32,
            )
        )
    return tuple(
        torch.as_tensor(np.vstack(values), dtype=torch.float32)
        for values in (parents, targets, scales)
    )


def estimate_scale(
    levels: list[np.ndarray],
    *,
    observed_depth: int,
    branching: int,
    clip_range: tuple[float, float],
) -> dict[str, Any]:
    medians = []
    for edge_depth in range(1, observed_depth + 1):
        parent = levels[edge_depth - 1]
        children = levels[edge_depth].reshape(len(parent), branching, -1)
        distances = np.linalg.norm(children - parent[:, None, :], axis=-1)
        medians.append(float(np.median(distances)))
    ratios = np.asarray(medians[1:]) / np.asarray(medians[:-1]).clip(min=1e-12)
    raw = float(np.exp(np.median(np.log(ratios.clip(min=1e-12)))))
    scale = float(np.clip(raw, clip_range[0], clip_range[1]))
    return {
        "scale": scale,
        "raw_scale": raw,
        "edge_displacement_medians": medians,
        "successive_ratios": ratios.tolist(),
        "clip_range": list(clip_range),
    }


def train_generator(
    model: nn.Module,
    levels: list[np.ndarray],
    *,
    observed_depth: int,
    branching: int,
    scale: float,
    seed: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    device: str | torch.device,
    log_every: int = 100,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    train_levels = levels[: observed_depth + 1]
    x, y, factors = hierarchy_to_training_data(
        train_levels, branching=branching, scale=scale
    )
    x = x.to(resolved)
    y = y.to(resolved)
    factors = factors.to(resolved)
    model = model.to(resolved)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = x[:, None, :] + factors * model(x)
        loss = ((prediction - y) ** 2).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CFP/baseline loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            mse = float(((prediction.detach() - y) ** 2).mean())
            history.append(
                {
                    "epoch": epoch,
                    "loss_sum": float(loss.detach()),
                    "train_mse": mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    model.eval()
    return model, {
        "epochs": epochs,
        "training_parent_count": int(len(x)),
        "parameter_count": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "final_train_mse": history[-1]["train_mse"],
        "history": history,
    }


@torch.inference_mode()
def rollout_model(
    model: nn.Module,
    root: np.ndarray,
    *,
    max_depth: int,
    branching: int,
    scale: float,
) -> list[np.ndarray]:
    device = next(model.parameters()).device
    levels = [np.asarray(root, dtype=np.float32).reshape(1, -1)]
    for edge_depth in range(1, max_depth + 1):
        parent = levels[-1]
        tensor = torch.as_tensor(parent, dtype=torch.float32, device=device)
        residual = model(tensor).cpu().numpy()
        children = parent[:, None, :] + scale**edge_depth * residual
        levels.append(children.reshape(-1, parent.shape[1]).astype(np.float32))
    return levels


def estimate_average_residuals(
    levels: list[np.ndarray],
    *,
    observed_depth: int,
    branching: int,
    scale: float,
) -> np.ndarray:
    residual_sum = None
    count = 0
    for edge_depth in range(1, observed_depth + 1):
        parent = levels[edge_depth - 1]
        children = levels[edge_depth].reshape(len(parent), branching, -1)
        residual = (children - parent[:, None, :]) / scale**edge_depth
        residual_sum = (
            residual.sum(axis=0)
            if residual_sum is None
            else residual_sum + residual.sum(axis=0)
        )
        count += len(parent)
    return np.asarray(residual_sum / count, dtype=np.float32)


def rollout_average_residual(
    residual: np.ndarray,
    root: np.ndarray,
    *,
    max_depth: int,
    scale: float,
) -> list[np.ndarray]:
    levels = [np.asarray(root, dtype=np.float32).reshape(1, -1)]
    for edge_depth in range(1, max_depth + 1):
        parent = levels[-1]
        children = parent[:, None, :] + scale**edge_depth * residual[None, :, :]
        levels.append(children.reshape(-1, parent.shape[1]).astype(np.float32))
    return levels


# --- metrics ---

def ordered_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    if predicted.shape != target.shape:
        raise ValueError(f"Shape mismatch: {predicted.shape} vs {target.shape}")
    return float(np.mean((np.asarray(predicted) - np.asarray(target)) ** 2))


def symmetric_squared_chamfer(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    distance = (
        np.sum(predicted**2, axis=1)[:, None]
        + np.sum(target**2, axis=1)[None, :]
        - 2.0 * predicted @ target.T
    )
    np.maximum(distance, 0.0, out=distance)
    return float(distance.min(axis=1).mean() + distance.min(axis=0).mean())


def _flatten_levels(
    levels: list[np.ndarray], *, upto_depth: int, branching: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.vstack(levels[: upto_depth + 1]).astype(np.float64)
    depths = np.concatenate(
        [
            np.full(len(levels[depth]), depth, dtype=np.int16)
            for depth in range(upto_depth + 1)
        ]
    )
    parent = np.full(len(points), -1, dtype=np.int64)
    paths = np.full((len(points), upto_depth), -1, dtype=np.int16)
    offsets = []
    offset = 0
    for depth in range(upto_depth + 1):
        offsets.append(offset)
        if depth > 0:
            count = len(levels[depth])
            parent[offset : offset + count] = (
                offsets[depth - 1] + np.arange(count) // branching
            )
            for local in range(count):
                value = local
                digits = [0] * depth
                for position in range(depth - 1, -1, -1):
                    digits[position] = value % branching
                    value //= branching
                paths[offset + local, :depth] = digits
        offset += len(levels[depth])
    return points, parent, depths, paths


def dpt_matrix(
    levels: list[np.ndarray], *, upto_depth: int, branching: int
) -> np.ndarray:
    points, parent, depths, paths = _flatten_levels(
        levels, upto_depth=upto_depth, branching=branching
    )
    root_distance = np.zeros(len(points), dtype=np.float64)
    for index in range(1, len(points)):
        ancestor = parent[index]
        root_distance[index] = root_distance[ancestor] + np.linalg.norm(
            points[index] - points[ancestor]
        )
    common = np.zeros((len(points), len(points)), dtype=np.int16)
    active = np.ones_like(common, dtype=bool)
    for position in range(upto_depth):
        valid = (
            (depths[:, None] > position)
            & (depths[None, :] > position)
        )
        equal = paths[:, position][:, None] == paths[:, position][None, :]
        active &= valid & equal
        common += active
    offsets = np.asarray(
        [
            (branching**depth - 1) // (branching - 1)
            for depth in range(upto_depth + 1)
        ],
        dtype=np.int64,
    )
    ancestors = np.zeros((len(points), upto_depth + 1), dtype=np.int64)
    for depth in range(1, upto_depth + 1):
        prefix = np.zeros(len(points), dtype=np.int64)
        for position in range(depth):
            prefix = prefix * branching + np.maximum(paths[:, position], 0)
        ancestors[:, depth] = offsets[depth] + prefix
    lca = ancestors[np.arange(len(points))[:, None], common]
    distance = (
        root_distance[:, None]
        + root_distance[None, :]
        - 2.0 * root_distance[lca]
    )
    np.fill_diagonal(distance, 0.0)
    return distance


def evaluate_levels(
    truth: list[np.ndarray],
    predictions: dict[str, list[np.ndarray]],
    *,
    heldout_depths: list[int],
    branching: int,
) -> dict[str, Any]:
    max_depth = max(heldout_depths)
    true_dpt = dpt_matrix(truth, upto_depth=max_depth, branching=branching)
    predicted_dpt = {
        method: dpt_matrix(levels, upto_depth=max_depth, branching=branching)
        for method, levels in predictions.items()
    }
    by_depth: dict[str, Any] = {}
    for depth in heldout_depths:
        node_count = (branching ** (depth + 1) - 1) // (branching - 1)
        upper = np.triu_indices(node_count, k=1)
        denominator = float(np.max(true_dpt[:node_count, :node_count][upper]) + 1e-30)
        rows = {}
        for method, levels in predictions.items():
            rows[method] = {
                "ordered_mse": ordered_mse(levels[depth], truth[depth]),
                "symmetric_squared_chamfer": symmetric_squared_chamfer(
                    levels[depth], truth[depth]
                ),
                "normalized_dpt_distortion": float(
                    np.mean(
                        np.abs(
                            predicted_dpt[method][:node_count, :node_count][upper]
                            - true_dpt[:node_count, :node_count][upper]
                        )
                    )
                    / denominator
                ),
            }
        by_depth[str(depth)] = rows
    heldout_average = {}
    for method in predictions:
        heldout_average[method] = {
            metric: float(
                np.mean(
                    [
                        by_depth[str(depth)][method][metric]
                        for depth in heldout_depths
                    ]
                )
            )
            for metric in (
                "ordered_mse",
                "symmetric_squared_chamfer",
                "normalized_dpt_distortion",
            )
        }
    return {"by_depth": by_depth, "heldout_average": heldout_average}


# --- experiment ---

METHOD_SLUGS = {
    "CFP": "cfp",
    "Avg Residual": "avg_residual",
    "Linear Res Gen": "linear_res_gen",
    "Constant": "constant",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_basis(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **payload,
        "basis": np.asarray(payload["basis"]).tolist(),
        "mean": np.asarray(payload["mean"]).tolist(),
    }


def _save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, path)


def run_experiment(
    dataset: str,
    seed: int,
    *,
    device: str | None = None,
    force: bool = False,
    require_cuda: bool = True,
    feature_path: Path | None = None,
    input_root: Path = DEFAULT_INPUT_ROOT,
    manifest_path: Path | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    if dataset in CIFAR_PRESETS:
        return run_cifar_experiment(
            dataset, seed, device=device, force=force, require_cuda=require_cuda,
            feature_path=feature_path, input_root=input_root, output_root=output_root,
        )
    protocol = load_protocol()
    config = dataset_config(dataset)
    if seed not in protocol["seeds"]:
        raise ValueError(f"Seed {seed} is outside the frozen protocol")
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if require_cuda and resolved_device.type != "cuda":
        raise RuntimeError("Formal SCP→CFP runs require CUDA")
    source = resolve_feature_path(dataset, seed, feature_path, input_root)
    if dataset == "inaturalist" and manifest_path is None:
        candidate = input_root / "inaturalist" / "split_manifest.json"
        manifest_path = candidate if candidate.is_file() else None
    identity = {
        "dataset": dataset,
        "seed": seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "feature_sha256": sha256_file(source),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "script_sha256": sha256_file(__file__),
    }
    output = output_root / "runs" / dataset / f"seed{seed}"
    complete_path = output / "complete.json"
    metrics_path = output / "metrics.json"
    if complete_path.is_file() and metrics_path.is_file() and not force:
        complete = json.loads(complete_path.read_text())
        if (
            complete.get("run_identity") == identity
            and complete.get("metrics_sha256") == sha256_file(metrics_path)
        ):
            verify_completion(output, identity)
            print(json.dumps({"status": "already_complete", "path": str(output)}))
            return output
        raise RuntimeError(f"Invalid completion marker at {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    seed_everything(seed)
    features = load_features(
        dataset, seed, feature_path=source, input_root=input_root, manifest_path=manifest_path
    )
    resolved = {
        "dataset": dataset,
        "display_name": config["display_name"],
        "seed": seed,
        "device": str(resolved_device),
        "protocol_sha256": PROTOCOL_SHA256,
        "feature_path": str(features.source_path),
        "feature_sha256": features.source_sha256,
        "config": copy.deepcopy(config),
        "scaling_audit": features.scaling_audit,
        "run_identity": identity,
    }
    write_json_atomic(output / "resolved_config.json", resolved)
    try:
        tree, attempts = build_hierarchy(
            features.view_q,
            features.view_k,
            dataset=dataset,
            tree_seed=seed,
            clusters=int(config["branching"]),
            max_depth=int(config["max_depth"]),
            epochs=int(config["scp_epochs_per_node"]),
            learning_rate=float(config["scp_learning_rate"]),
            batch_size=int(config["scp_batch_size"]),
            min_parent_size=int(config["scp_min_parent_size"]),
            max_retries=int(config["scp_max_retries"]),
            device=resolved_device,
        )
    except TopologyFailure as exc:
        write_json_atomic(
            output / "topology_failure.json",
            {
                "dataset": dataset,
                "seed": seed,
                "reason": exc.reason,
                "address": list(exc.address),
                "attempt_records": exc.attempt_records,
            },
        )
        raise
    write_json_atomic(output / "scp_attempts.json", attempts)
    topology = audit_topology(
        tree,
        branching=int(config["branching"]),
        max_depth=int(config["max_depth"]),
    )
    if topology["status"] != "passed":
        raise RuntimeError(f"Topology audit failed: {topology['failures']}")
    basis = fit_observed_basis(
        tree,
        observed_depth=int(config["observed_depth"]),
        components=int(protocol["canonicalization"]["components"]),
    )
    before_ordering = genealogy_snapshot(tree)
    canonicalize(tree, basis=basis["basis"], mean=basis["mean"])
    genealogy_audit = validate_genealogy(before_ordering, genealogy_snapshot(tree))
    write_json_atomic(output / "genealogy_audit.json", genealogy_audit)
    topology_after = audit_topology(
        tree,
        branching=int(config["branching"]),
        max_depth=int(config["max_depth"]),
    )
    if topology_after["status"] != "passed":
        raise RuntimeError("Canonicalization changed topology validity")
    _atomic_pickle(output / "scp_tree.pkl", tree)
    levels = tree_to_levels(tree, max_depth=int(config["max_depth"]))
    scale_audit = estimate_scale(
        levels,
        observed_depth=int(config["observed_depth"]),
        branching=int(config["branching"]),
        clip_range=tuple(map(float, config["scale_clip"])),
    )
    scale = float(scale_audit["scale"])
    model_seed = seed
    common_training = {
        "levels": levels,
        "observed_depth": int(config["observed_depth"]),
        "branching": int(config["branching"]),
        "scale": scale,
        "seed": model_seed,
        "learning_rate": float(config["cfp_learning_rate"]),
        "epochs": int(config["cfp_epochs"]),
        "weight_decay": float(config["cfp_weight_decay"]),
        "device": resolved_device,
    }
    predictions: dict[str, list[np.ndarray]] = {}
    training_audit: dict[str, Any] = {}
    model_specs = (
        (
            "CFP",
            lambda: BottleneckCFP(
                dimension=levels[0].shape[1],
                branching=int(config["branching"]),
                width=int(config["cfp_width"]),
                depth=int(config["cfp_depth"]),
            ),
        ),
        (
            "Linear Res Gen",
            lambda: LinearResidualGenerator(
                levels[0].shape[1], int(config["branching"])
            ),
        ),
        (
            "Constant",
            lambda: ConstantResidualGenerator(
                levels[0].shape[1], int(config["branching"])
            ),
        ),
    )
    for method, factory in model_specs:
        seed_everything(model_seed)
        model = factory()
        model, audit = train_generator(model, **common_training)
        predictions[method] = rollout_model(
            model,
            levels[0][0],
            max_depth=int(config["max_depth"]),
            branching=int(config["branching"]),
            scale=scale,
        )
        training_audit[method] = audit
        _save_checkpoint(
            output / "checkpoints" / f"{METHOD_SLUGS[method]}.pt",
            model,
            {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "scale": scale,
                "protocol_sha256": resolved["protocol_sha256"],
                **audit,
            },
        )
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    average_residual = estimate_average_residuals(
        levels,
        observed_depth=int(config["observed_depth"]),
        branching=int(config["branching"]),
        scale=scale,
    )
    predictions["Avg Residual"] = rollout_average_residual(
        average_residual,
        levels[0][0],
        max_depth=int(config["max_depth"]),
        scale=scale,
    )
    training_audit["Avg Residual"] = {
        "training_parent_count": int(
            sum(int(config["branching"]) ** depth for depth in range(int(config["observed_depth"])))
        ),
        "parameter_count": 0,
        "analytic": True,
    }
    metric_payload = evaluate_levels(
        levels,
        predictions,
        heldout_depths=list(map(int, config["heldout_depths"])),
        branching=int(config["branching"]),
    )
    purity = {
        label_name: {
            str(depth): cluster_purity(tree, label_values, depth)
            for depth in range(int(config["max_depth"]) + 1)
        }
        for label_name, label_values in features.labels.items()
    }
    arrays: dict[str, np.ndarray] = {
        f"true_level_{depth}": values for depth, values in enumerate(levels)
    }
    for method, method_levels in predictions.items():
        slug = METHOD_SLUGS[method]
        arrays.update(
            {
                f"{slug}_level_{depth}": values
                for depth, values in enumerate(method_levels)
            }
        )
    _atomic_npz(output / "levels_and_predictions.npz", arrays)
    metrics = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": dataset,
        "display_name": config["display_name"],
        "seed": seed,
        "protocol_sha256": resolved["protocol_sha256"],
        "feature_sha256": features.source_sha256,
        "branching": int(config["branching"]),
        "max_depth": int(config["max_depth"]),
        "observed_depth": int(config["observed_depth"]),
        "heldout_depths": list(map(int, config["heldout_depths"])),
        "topology_audit": topology_after,
        "genealogy_audit": genealogy_audit,
        "canonicalization": _json_basis(basis),
        "scale_estimation": scale_audit,
        "training": training_audit,
        "purity": purity,
        **metric_payload,
    }
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(
        complete_path,
        {
            "status": "complete",
            "run_identity": identity,
            "metrics_sha256": sha256_file(metrics_path),
            "levels_sha256": sha256_file(output / "levels_and_predictions.npz"),
            "tree_sha256": sha256_file(output / "scp_tree.pkl"),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": dataset,
                "seed": seed,
                "metrics": str(metrics_path),
                "heldout_average": metrics["heldout_average"],
            },
            indent=2,
        )
    )
    return output


# --- aggregate ---

METRICS = (
    "ordered_mse",
    "symmetric_squared_chamfer",
    "normalized_dpt_distortion",
)


BASELINES = ("Avg Residual", "Linear Res Gen", "Constant")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    *, require_complete: bool = True, output_root: Path = DEFAULT_OUTPUT_ROOT,
    datasets: list[str] | None = None, seeds: list[int] | None = None,
    seed_map: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    protocol = load_protocol()
    datasets = list(protocol["datasets"]) if datasets is None else datasets
    expected_seeds = {d: dataset_seeds(d) if seeds is None else list(seeds) for d in datasets}
    if seed_map is not None:
        expected_seeds = {d: list(seed_map[d]) for d in datasets}
    for dataset, selected in expected_seeds.items():
        if not selected or not set(selected).issubset(dataset_seeds(dataset)):
            raise ValueError(f"Invalid seeds for {dataset}: {selected}")
    payloads = []
    missing = []
    for dataset in datasets:
        for seed in expected_seeds[dataset]:
            path = output_root / "runs" / dataset / f"seed{seed}" / "metrics.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            complete = verify_completion(path.parent)
            expected_config = dataset_config(dataset)
            expected_protocol = sha256_json(expected_config) if dataset in CIFAR_PRESETS else PROTOCOL_SHA256
            if complete["run_identity"].get("script_sha256") != sha256_file(__file__):
                raise ValueError(f"Different implementation revision in {path}")
            payload = json.loads(path.read_text())
            if payload["protocol_sha256"] != expected_protocol:
                raise ValueError(f"Different protocol in {path}")
            if payload["heldout_depths"] != expected_config["heldout_depths"]:
                raise ValueError(f"Incorrect Table 2 horizon in {path}")
            if payload.get("genealogy_audit", {}).get("status") != "passed":
                raise ValueError(f"Missing genealogy audit in {path}")
            if payload["dataset"] != dataset or payload["seed"] != seed:
                raise ValueError(f"Dataset/seed mismatch in {path}")
            if set(payload["by_depth"]) != set(map(str, expected_config["heldout_depths"])):
                raise ValueError(f"Unexpected evaluated depth keys in {path}")
            for method in protocol["methods"]:
                for metric in METRICS:
                    values = [payload["by_depth"][str(d)][method][metric]
                              for d in expected_config["heldout_depths"]]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite metric in {path}")
                    if not np.isclose(payload["heldout_average"][method][metric],
                                      np.mean(values), rtol=1e-12, atol=1e-12):
                        raise ValueError(f"Incorrect horizon average in {path}")
            payloads.append(payload)
    if missing and require_complete:
        raise FileNotFoundError(f"Missing {len(missing)} formal metrics files: {missing}")
    per_seed_rows = []
    for payload in payloads:
        for depth, methods in payload["by_depth"].items():
            for method, metrics in methods.items():
                per_seed_rows.append(
                    {
                        "dataset": payload["dataset"],
                        "seed": payload["seed"],
                        "level": depth,
                        "method": method,
                        **metrics,
                    }
                )
        for method, metrics in payload["heldout_average"].items():
            per_seed_rows.append(
                {
                    "dataset": payload["dataset"],
                    "seed": payload["seed"],
                    "level": "heldout_mean",
                    "method": method,
                    **metrics,
                }
            )
    groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    for row in per_seed_rows:
        key = (str(row["dataset"]), str(row["level"]), str(row["method"]))
        for metric in METRICS:
            groups[key][metric].append(float(row[metric]))
    summary_rows = []
    for (dataset, level, method), values in sorted(groups.items()):
        row: dict[str, Any] = {
            "dataset": dataset,
            "level": level,
            "method": method,
            "n_seeds": len(next(iter(values.values()))),
        }
        for metric, observations in values.items():
            array = np.asarray(observations, dtype=np.float64)
            row[f"{metric}_mean"] = float(array.mean())
            row[f"{metric}_std"] = float(array.std(ddof=1 if len(array) > 1 else 0))
        summary_rows.append(row)
    heldout = [row for row in summary_rows if row["level"] == "heldout_mean"]
    comparison_rows = []
    for dataset in datasets:
        by_method = {row["method"]: row for row in heldout if row["dataset"] == dataset}
        if not by_method:
            continue
        for method in protocol["methods"]:
            row = {
                "dataset": dataset,
                "method": method,
                "n_seeds": by_method[method]["n_seeds"],
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = by_method[method][f"{metric}_mean"]
                row[f"{metric}_std"] = by_method[method][f"{metric}_std"]
                best_baseline = min(
                    by_method[name][f"{metric}_mean"] for name in BASELINES
                )
                row[f"{metric}_vs_best_baseline_pct"] = (
                    100.0
                    * (best_baseline - by_method[method][f"{metric}_mean"])
                    / max(abs(best_baseline), 1e-30)
                    if method == "CFP"
                    else None
                )
            comparison_rows.append(row)
    release = output_root / "release"
    _write_csv(
        release / "per_seed_metrics.csv",
        per_seed_rows,
        ["dataset", "seed", "level", "method", *METRICS],
    )
    _write_csv(
        release / "summary_metrics.csv",
        summary_rows,
        [
            "dataset",
            "level",
            "method",
            "n_seeds",
            *[
                field
                for metric in METRICS
                for field in (f"{metric}_mean", f"{metric}_std")
            ],
        ],
    )
    comparison_fields = [
        "dataset",
        "method",
        "n_seeds",
        *[
            field
            for metric in METRICS
            for field in (
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_vs_best_baseline_pct",
            )
        ],
    ]
    _write_csv(release / "comparison_table.csv", comparison_rows, comparison_fields)
    lines = [
        "# SCP→CFP held-out comparison",
        "",
        "All values are mean ± sample standard deviation across hierarchy/model seeds; lower is better.",
        "",
        "| Dataset | Method | Ordered MSE | Symmetric squared Chamfer | Normalized d_pt distortion |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['ordered_mse_mean']:.6e} ± {row['ordered_mse_std']:.2e} | "
            f"{row['symmetric_squared_chamfer_mean']:.6e} ± "
            f"{row['symmetric_squared_chamfer_std']:.2e} | "
            f"{row['normalized_dpt_distortion_mean']:.6e} ± "
            f"{row['normalized_dpt_distortion_std']:.2e} |"
        )
    (release / "comparison_table.md").write_text("\n".join(lines) + "\n")
    result = {
        "schema_version": 1,
        "expected_seeds": expected_seeds,
        "completed_runs": len(payloads),
        "missing_runs": missing,
        "per_seed_rows": len(per_seed_rows),
        "summary_rows": len(summary_rows),
        "comparison_rows": comparison_rows,
    }
    write_json_atomic(release / "aggregate.json", result)
    return result


# --- Raw-image adapters and shared CLIP extraction ---

@dataclass
class ImageRecord:
    path: Path
    image_id: str | int
    box: tuple[int, int, int, int] | None = None


class ImageViews(Dataset):
    def __init__(self, records, deterministic, stochastic=None):
        self.records = records
        self.deterministic = deterministic
        self.stochastic = stochastic

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.path) as handle:
            image = handle.convert("RGB")
            if record.box is not None:
                x_min, y_min, x_max, y_max = record.box
                image = image.crop((
                    max(x_min - 1, 0), max(y_min - 1, 0),
                    min(x_max, image.width), min(y_max, image.height),
                ))
            deterministic = self.deterministic(image)
            if self.stochastic is None:
                return deterministic, record.image_id
            return (
                self.stochastic(image), self.stochastic(image),
                deterministic, record.image_id,
            )


def image_roles(
    dataset: str, data_root: Path, manifest_path: Path | None = None,
) -> tuple[dict[str, list[ImageRecord]], dict[str, np.ndarray], list[Path]]:
    """Return images, diagnostic labels, and annotations; pair views only for support images."""
    roles: dict[str, list[ImageRecord]] = {}
    labels: dict[str, np.ndarray] = {}
    sources: list[Path] = []
    if dataset == "food101":
        base = data_root if (data_root / "meta").is_dir() else data_root / "food-101"
        vocabulary = None
        for role, split, per_class in (("support", "train", 750), ("evaluation", "test", 250)):
            source = base / "meta" / f"{split}.json"
            sources.append(source)
            metadata = json.loads(source.read_text())
            classes = sorted(metadata)
            if len(classes) != 101 or any(len(metadata[c]) != per_class for c in classes):
                raise ValueError(f"Invalid Food-101 {split} class counts")
            if vocabulary is not None and classes != vocabulary:
                raise ValueError("Food-101 train/test class vocabularies differ")
            vocabulary = classes
            roles[role] = [
                ImageRecord(base / "images" / f"{image_id}.jpg", image_id)
                for c in classes for image_id in metadata[c]
            ]
            labels[f"{role}_class_labels"] = np.asarray([
                index for index, c in enumerate(classes) for _ in metadata[c]
            ], dtype=np.int64)
    elif dataset == "aircraft":
        source = data_root / "images_box.txt"
        sources.append(source)
        boxes = {}
        for line in source.read_text().splitlines():
            image_id, *coordinates = line.split()
            if len(coordinates) != 4:
                raise ValueError(f"Invalid Aircraft box: {line}")
            boxes[image_id] = tuple(map(int, coordinates))
        for role, split in (("support", "trainval"), ("evaluation", "test")):
            annotations = {}
            for name in ("family", "variant"):
                source = data_root / f"images_{name}_{split}.txt"
                sources.append(source)
                annotations[name] = dict(line.split(" ", 1) for line in source.read_text().splitlines())
            if annotations["family"].keys() != annotations["variant"].keys():
                raise ValueError(f"Aircraft {split} family/variant IDs differ")
            ids = list(annotations["family"])
            roles[role] = [ImageRecord(data_root / "images" / f"{i}.jpg", i, boxes[i]) for i in ids]
            for name, values in annotations.items():
                labels[f"{role}_{name}_labels"] = np.asarray([values[i] for i in ids])
    elif dataset == "inaturalist":
        if manifest_path is None:
            raise ValueError("iNaturalist extraction requires --manifest (the frozen 819-species split)")
        manifest = json.loads(manifest_path.read_text())
        validate_inat_manifest(manifest)
        sources.append(manifest_path)
        image_paths = {}
        split_ids = {}
        image_categories = {}
        for name in ("train_mini.json", "val.json"):
            source = data_root / name
            sources.append(source)
            metadata = json.loads(source.read_text())
            split_ids[name] = {int(row["id"]) for row in metadata["images"]}
            image_paths.update({int(row["id"]): data_root / row["file_name"] for row in metadata["images"]})
            image_categories.update({int(row["image_id"]): int(row["category_id"]) for row in metadata["annotations"]})
        for role in ("support", "cluster", "evaluation"):
            by_id = {
                int(i): (index, int(row["genus_label"]), int(row["category_id"]))
                for index, row in enumerate(manifest["rows"]) for i in row[f"{role}_ids"]
            }
            ids = sorted(by_id)
            source_split = "val.json" if role == "evaluation" else "train_mini.json"
            if not set(ids).issubset(split_ids[source_split]):
                raise ValueError(f"iNaturalist {role} IDs are outside {source_split}")
            if any(image_categories[i] != by_id[i][2] for i in ids):
                raise ValueError(f"iNaturalist {role} species annotations differ from manifest")
            roles[role] = [ImageRecord(image_paths[i], i) for i in ids]
            labels[f"{role}_species_labels"] = np.asarray([by_id[i][0] for i in ids], dtype=np.int64)
            labels[f"{role}_genus_labels"] = np.asarray([by_id[i][1] for i in ids], dtype=np.int64)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    expected_counts = {
        "food101": {"support": 75750, "evaluation": 25250},
        "aircraft": {"support": 6667, "evaluation": 3333},
        "inaturalist": {"support": 24570, "cluster": 16380, "evaluation": 8190},
    }[dataset]
    seen = set()
    for role, records in roles.items():
        ids = {r.image_id for r in records}
        if len(records) != expected_counts[role] or len(ids) != len(records) or seen.intersection(ids):
            raise ValueError(f"Invalid {dataset} {role} count or overlapping IDs")
        seen.update(ids)
    return roles, labels, sources


def extract_features(
    dataset: str, seed: int, *, data_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path | None = None, device: str | None = None,
    batch_size: int = 256, num_workers: int | None = None,
    clip_cache: Path = Path.home() / ".cache" / "clip", force: bool = False,
    require_cuda: bool = True,
) -> Path:
    """Extract frozen CLIP features using the dataset preset."""
    if dataset in CIFAR_PRESETS:
        return extract_cifar_features(
            dataset, seed, data_root=data_root, output_root=output_root,
            device=device, batch_size=batch_size, num_workers=num_workers,
            clip_cache=clip_cache, force=force, require_cuda=require_cuda,
        )
    if seed not in PROTOCOL["seeds"]:
        raise ValueError(f"Seed {seed} is outside the frozen protocol")
    if batch_size < 1 or (num_workers is not None and num_workers < 0):
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    num_workers = (8 if dataset == "food101" else 4) if num_workers is None else num_workers
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Formal CLIP extraction requires CUDA")
    roles, arrays, sources = image_roles(dataset, data_root, manifest_path)
    identity = {
        "dataset": dataset, "seed": seed, "data_root": str(data_root.resolve()),
        "annotation_sha256": {str(p.resolve()): sha256_file(p) for p in sources},
        "batch_size": batch_size, "num_workers": num_workers,
        "script_sha256": sha256_file(__file__), "protocol_sha256": PROTOCOL_SHA256,
    }
    output = output_root / "inputs" / dataset / f"tree_seed{seed}"
    feature_path = output / "features.npz"
    complete_path = output / "complete.json"
    if complete_path.is_file() and feature_path.is_file() and not force:
        complete = json.loads(complete_path.read_text())
        if complete.get("extraction_identity") == identity and complete.get("feature_sha256") == sha256_file(feature_path):
            print(json.dumps({"status": "cache_valid", "path": str(feature_path)}))
            return feature_path
        raise RuntimeError(f"Cache/config mismatch at {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    import clip
    import torchvision

    seed_everything(seed)
    model, deterministic = clip.load("ViT-B/32", device=device, download_root=str(clip_cache))
    model.eval()
    stochastic = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        torchvision.transforms.ToTensor(),
        deterministic.transforms[-1],
    ])
    for role, records in roles.items():
        paired = role == "support"
        options = {} if dataset == "inaturalist" else {"generator": torch.Generator().manual_seed(seed)}
        loader = DataLoader(
            ImageViews(records, deterministic, stochastic if paired else None),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, **options,
        )
        rows = [[], [], []] if paired else [[]]
        actual_ids = []
        with torch.inference_mode():
            for batch in loader:
                *views, ids = batch
                actual_ids.extend(ids.tolist() if torch.is_tensor(ids) else list(ids))
                for destination, values in zip(rows, views):
                    encoded = nn.functional.normalize(
                        model.encode_image(values.to(device, non_blocking=True)).float(), dim=-1
                    )
                    destination.append(encoded.cpu())
        if actual_ids != [r.image_id for r in records]:
            raise RuntimeError(f"{dataset} {role}: image order changed")
        keys = ["support_view_q", "support_view_k", "support_deterministic"] if paired else [f"{role}_deterministic"]
        for key, values in zip(keys, rows):
            vectors = torch.cat(values).numpy().astype(np.float32)
            if vectors.shape != (len(records), 512) or not np.isfinite(vectors).all():
                raise RuntimeError(f"Invalid CLIP features: {key}")
            if np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0)) > 2e-5:
                raise RuntimeError(f"Invalid CLIP normalization: {key}")
            arrays[key] = vectors.astype(np.float16)
        arrays[f"{role}_ids"] = np.asarray(actual_ids)
        print(f"ENCODED dataset={dataset} seed={seed} role={role} rows={len(records)}", flush=True)
    _atomic_npz(feature_path, arrays)
    checksum = sha256_file(feature_path)
    write_json_atomic(output / "manifest.json", {
        **identity, "encoder": "OpenAI CLIP ViT-B/32", "dimension": 512,
        "dtype": "float16", "normalization": "l2_before_formal_featurewise_minmax",
        "augmentation": "independent RandomResizedCrop(224, scale=(0.65,1.0))+flip+ColorJitter(0.2,0.2,0.2,0.05)",
        "crop": "official one-indexed inclusive bounding box" if dataset == "aircraft" else None,
        "counts": {role: len(rows) for role, rows in roles.items()},
        "feature_sha256": checksum, "device": str(device),
    })
    write_json_atomic(complete_path, {
        "status": "complete", "feature_sha256": checksum, "extraction_identity": identity,
    })
    return feature_path


# --- CIFAR numerical routines ---

def _cifar_hierarchy_to_training_data(levels, K=3, scale=1.0):
    """Return parents (N, d), children (N, K, d), and child-depth factors s**(l+1)."""
    if not (0 < scale <= 1):
        raise ValueError(f"scale must satisfy 0 < scale <= 1, but got {scale}.")

    X_list = []
    Y_list = []
    S_list = []

    for l in range(len(levels) - 1):
        parents = levels[l]        # shape (n_l, d)
        children = levels[l + 1]   # shape (n_{l+1}, d)

        n_parents, d = parents.shape

        if children.shape[0] != n_parents * K:
            raise ValueError(
                f"Level {l+1} should have {n_parents*K} nodes, "
                f"but got {children.shape[0]}."
            )

        children_grouped = children.reshape(n_parents, K, d)

        X_list.append(parents)
        Y_list.append(children_grouped)

        level_scales = np.full((n_parents, 1, 1), scale ** (l + 1), dtype=np.float32)
        S_list.append(level_scales)

    X = np.vstack(X_list)                  # (N, d)
    Y = np.vstack(Y_list)                  # (N, K, d)
    S = np.vstack(S_list)                  # (N, 1, 1)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(S, dtype=torch.float32),
    )


class _cifar_BottleneckHRGMLP(nn.Module):
    """Residual MLP: d -> wide -> ... -> d -> K*d, with `depth` hidden layers."""
    ACT_MAP = {
        "relu":       lambda: nn.ReLU(),
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "gelu":       lambda: nn.GELU(),
        "silu":       lambda: nn.SiLU(),
    }

    def __init__(self, d=512, K=3, wide=1024, depth=5,
                 act_type="gelu", use_layernorm=False):
        super().__init__()
        assert depth >= 2, "Need at least 2 hidden layers for bottleneck"
        make_act = self.ACT_MAP.get(act_type, self.ACT_MAP["gelu"])
        self.act_type = act_type
        layers = []
        layers.append(nn.Linear(d, wide))
        if use_layernorm:
            layers.append(nn.LayerNorm(wide))
        layers.append(make_act())
        for _ in range(depth - 2):
            layers.append(nn.Linear(wide, wide))
            if use_layernorm:
                layers.append(nn.LayerNorm(wide))
            layers.append(make_act())
        layers.append(nn.Linear(wide, d))
        if use_layernorm:
            layers.append(nn.LayerNorm(d))
        layers.append(make_act())
        layers.append(nn.Linear(d, d * K))
        self.net = nn.Sequential(*layers)
        self.d = d
        self.K = K
        self._init_weights()

    def _init_weights(self):
        nonlin = "leaky_relu" if self.act_type in ("leaky_relu", "gelu", "silu") else "relu"
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        last_linear = self.net[-1]
        nn.init.xavier_uniform_(last_linear.weight)

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.K, self.d)


class _cifar_ConstantHRG(nn.Module):
    """Learn one location-independent residual per child slot."""
    def __init__(self, d=2, K=3):
        super().__init__()
        self.d = d
        self.K = K
        self.residuals = nn.Parameter(torch.randn(K, d) * 0.1)

    def forward(self, x):
        return self.residuals.unsqueeze(0).expand(x.shape[0], -1, -1)


class _cifar_LinearHRG(nn.Module):
    """Map each parent to K residuals with an affine layer."""
    def __init__(self, d=2, K=3):
        super().__init__()
        self.d = d
        self.K = K
        self.linear = nn.Linear(d, d * K)

    def forward(self, x):
        return self.linear(x).view(-1, self.K, self.d)


def _cifar_train_hrg_model(
    model,
    levels,
    x0=None,
    L_train=None,
    n_extra_levels=1,
    K=3,
    scale=1.0,
    lr=1e-3,
    epochs=2000,
    print_every=200,
    device=None,
    grad_clip=1.0,
    use_scheduler=True,
    weight_decay=0.0,
):
    """Fit a residual generator on observed parent-child transitions."""
    if L_train is None:
        L_train = len(levels) - 1
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_level = L_train + n_extra_levels
    can_test = (
        x0 is not None and n_extra_levels >= 1
        and target_level < len(levels)
    )

    train_levels = levels[:L_train + 1]
    X, Y, S = _cifar_hierarchy_to_training_data(train_levels, K=K, scale=scale)
    X, Y, S = X.to(device), Y.to(device), S.to(device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = _cifar_hierarchy_loss(model, X, Y, S)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        is_log_epoch = (epoch % print_every == 0 or epoch == 1 or epoch == epochs)
        if is_log_epoch:
            with torch.no_grad():
                model.eval()
                pred_train = X.unsqueeze(1) + S * model(X)
                train_mse = ((pred_train - Y) ** 2).mean().item()
                test_str = ""
                if can_test:
                    lh = _cifar_build_hierarchy_from_model(
                        model, x0, max_level=target_level, scale=scale,
                    )
                    test_mse = np.mean((lh[target_level] - levels[target_level]) ** 2)
                    test_str = f" | Test MSE@L{target_level} = {test_mse:.8f}"
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | Loss = {loss.item():.8f} | "
                      f"Train MSE = {train_mse:.8f}{test_str} | lr={cur_lr:.2e}")

    model.eval()
    return model


def _cifar_hierarchy_loss(model, X, Y, S):
    """Sum squared child-coordinate errors using the supplied depth factors."""
    pred_residuals = model(X)                     # (N, K, d)
    pred_children = X.unsqueeze(1) + S * pred_residuals
    return ((pred_children - Y) ** 2).sum()


@torch.no_grad()
def _cifar_build_hierarchy_from_model(model, x0, max_level=2, scale=1.0):
    """Roll out child = parent + s**child_depth * residual(parent)."""
    if not (0 < scale <= 1):
        raise ValueError(f"scale must satisfy 0 < scale <= 1, but got {scale}.")

    device = next(model.parameters()).device
    x0 = np.asarray(x0, dtype=np.float32).reshape(-1)

    levels_hat = [x0[None, :]]

    for l in range(1, max_level + 1):
        parents = levels_hat[-1]                                   # (n_parents, d)
        level_scale = scale ** l

        parents_t = torch.tensor(parents, dtype=torch.float32, device=device)
        residuals = model(parents_t).cpu().numpy()                 # (n_parents, K, d)
        children = parents[:, None, :] + level_scale * residuals   # (n_parents, K, d)
        children = children.reshape(-1, parents.shape[1])          # (n_parents*K, d)

        levels_hat.append(children)

    return levels_hat


def _cifar_estimate_average_residuals(levels, L_train, K=3, scale=1.0):
    """Average (child - parent) / s**child_depth over observed parents, by slot."""
    if L_train < 1:
        raise ValueError('L_train must be at least 1.')

    residual_sums = None
    count = 0

    for l in range(1, L_train + 1):
        parents = levels[l - 1]                  # shape (n_parents, d)
        children = levels[l]                     # shape (n_parents*K, d)
        n_parents, d = parents.shape
        children_grouped = children.reshape(n_parents, K, d)

        level_residuals = (children_grouped - parents[:, None, :]) / (scale ** l)

        if residual_sums is None:
            residual_sums = level_residuals.sum(axis=0)   # (K, d)
        else:
            residual_sums += level_residuals.sum(axis=0)
        count += n_parents

    avg_residuals = residual_sums / count
    return avg_residuals


def _cifar_build_hierarchy_from_average_residuals(avg_residuals, x0, max_level=2, scale=1.0):
    """Roll out the slotwise average residuals from the root."""
    if not (0 < scale <= 1):
        raise ValueError(f'scale must satisfy 0 < scale <= 1, but got {scale}.')

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    avg_residuals = np.asarray(avg_residuals, dtype=float)

    levels_avg = [x0[None, :]]

    for l in range(1, max_level + 1):
        parents = levels_avg[-1]
        children_list = []
        level_scale = scale ** l

        for x in parents:
            children = x[None, :] + level_scale * avg_residuals
            children_list.append(children)

        levels_avg.append(np.vstack(children_list))

    return levels_avg


_CIFAR100_COARSE_MAP = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def _cifar_tree_to_levels(tree, target_depth=None):
    """Collect centroid arrays of shape (K**depth, d) through target_depth."""
    if target_depth is None:
        target_depth = _cifar_tree_max_full_depth(tree)

    levels = []
    current_nodes = [tree]

    for depth in range(target_depth + 1):
        centroids = np.array([n["centroid"] for n in current_nodes])
        levels.append(centroids)

        if depth < target_depth:
            next_nodes = []
            for n in current_nodes:
                if n["children"] is None or len(n["children"]) == 0:
                    raise ValueError(
                        f"Node at depth {depth} is a leaf but target_depth "
                        f"requires children.  Reduce target_depth or ensure "
                        f"the tree is full up to the desired depth."
                    )
                next_nodes.extend(n["children"])
            current_nodes = next_nodes

    return levels


def _cifar_tree_max_full_depth(tree, K=None):
    """Return the deepest complete K-ary prefix; infer K from the root if omitted."""
    if tree["children"] is None or len(tree["children"]) == 0:
        return 0
    if K is not None and len(tree["children"]) != K:
        return 0
    actual_k = len(tree["children"])
    child_depths = [_cifar_tree_max_full_depth(c, K=actual_k) for c in tree["children"]]
    return 1 + min(child_depths)


def _cifar_canonicalize_levels_pca(levels, K=3, pca_model=None, *,
                                    observed_depth=None, return_permutations=False):
    """Sort sibling slots and move complete descendant blocks with each sibling."""
    if pca_model is None:
        if observed_depth is None:
            raise ValueError("Specify observed_depth to prevent fitting held-out levels")
        pca_model = fit_centroid_pca(levels, observed_depth)
    levels_canon = [np.asarray(lev).copy() for lev in levels]
    permutations = [np.arange(len(lev)) for lev in levels]
    for depth, level in enumerate(levels_canon):
        if len(level) != K ** depth:
            raise ValueError(f"Incomplete K-ary level {depth}")
    for depth in range(1, len(levels_canon)):
        n_parents = len(levels_canon[depth - 1])
        new_order = np.empty(n_parents * K, dtype=int)
        for parent_index, parent in enumerate(levels_canon[depth - 1]):
            start = parent_index * K
            displacement = levels_canon[depth][start:start + K].astype(np.float64) - parent.astype(np.float64)
            projection = displacement @ pca_model.components_[:2].T
            angles = np.arctan2(projection[:, 1], projection[:, 0])
            distances = np.linalg.norm(displacement, axis=1)
            new_order[start:start + K] = start + np.lexsort((distances, angles))
        for descendant_depth in range(depth, len(levels_canon)):
            block = K ** (descendant_depth - depth)
            order = (new_order[:, None] * block + np.arange(block)[None, :]).ravel()
            levels_canon[descendant_depth] = levels_canon[descendant_depth][order]
            permutations[descendant_depth] = permutations[descendant_depth][order]
    for depth in range(1, len(levels_canon)):
        actual_parents = permutations[depth] // K
        expected_parents = np.repeat(permutations[depth - 1], K)
        if not np.array_equal(actual_parents, expected_parents):
            raise RuntimeError("Descendant-block permutation changed a parent-child edge")
    return (levels_canon, permutations) if return_permutations else levels_canon


def _cifar_chamfer_distance(pred, true, squared=True):
    """Symmetric Chamfer distance between two point sets.

    CD = mean_{x in pred} min_{y in true} ||x-y||^2
       + mean_{y in true} min_{x in pred} ||x-y||^2
    """
    diff = pred[:, None, :] - true[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    if not squared:
        dist2 = np.sqrt(dist2 + 1e-30)
    fwd = np.mean(np.min(dist2, axis=1))
    bwd = np.mean(np.min(dist2, axis=0))
    return float(fwd + bwd)


def _cifar_flatten_levels_upto(levels, upto_level, K):
    """Flatten hierarchy levels 0..upto_level into arrays."""
    points, parent_global, level_ids = [], [], []
    offset = 0
    offsets = []
    for l in range(upto_level + 1):
        offsets.append(offset)
        n_l = levels[l].shape[0]
        points.append(levels[l])
        level_ids.append(np.full(n_l, l, dtype=int))
        if l == 0:
            parent_global.append(np.array([-1], dtype=int))
        else:
            parent_local = np.arange(n_l, dtype=int) // K
            parent_global.append(offsets[l - 1] + parent_local)
        offset += n_l
    return (
        np.vstack(points),
        np.concatenate(parent_global),
        np.concatenate(level_ids),
    )


def _cifar_compute_dpt_matrix(levels, K, upto_level):
    """Pairwise tree-shortest-path distance matrix using Euclidean edge weights."""
    points, parent_global, level_ids = _cifar_flatten_levels_upto(
        levels, upto_level, K,
    )
    N = len(points)
    dist_root = np.zeros(N)
    for i in range(N):
        p = parent_global[i]
        if p >= 0:
            dist_root[i] = dist_root[p] + np.linalg.norm(points[i] - points[p])
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            anc_i = set()
            cur = i
            while cur >= 0:
                anc_i.add(cur)
                cur = parent_global[cur]
            lca = j
            while lca not in anc_i:
                lca = parent_global[lca]
            d_ij = (dist_root[i] - dist_root[lca]) + \
                   (dist_root[j] - dist_root[lca])
            D[i, j] = d_ij
            D[j, i] = d_ij
    return D


def _cifar_dpt_distortion(pred_levels, true_levels, K, upto_level,
                   normalize="diameter"):
    """Structure-aware d_pt-distortion between predicted and true hierarchies."""
    D_pred = _cifar_compute_dpt_matrix(pred_levels, K, upto_level)
    D_true = _cifar_compute_dpt_matrix(true_levels, K, upto_level)
    N = D_pred.shape[0]
    iu = np.triu_indices(N, k=1)
    diff = np.abs(D_pred[iu] - D_true[iu])
    raw = np.mean(diff)
    if normalize == "diameter":
        diam = np.max(D_true[iu]) + 1e-30
        return float(raw / diam)
    elif normalize == "mean":
        mu = np.mean(D_true[iu]) + 1e-30
        return float(raw / mu)
    return float(raw)


def _cifar_compute_cluster_purity(tree, labels, depth):
    """Return (cluster size, majority-label fraction, majority label) at a depth."""
    nodes = _cifar_collect_nodes_at_depth(tree, depth)
    results = []
    for node in nodes:
        cluster_labels = labels[node["indices"]]
        if len(cluster_labels) == 0:
            continue
        values, counts = np.unique(cluster_labels, return_counts=True)
        dominant = values[np.argmax(counts)]
        purity = counts.max() / len(cluster_labels)
        results.append((len(cluster_labels), purity, dominant))
    return results


def _cifar_collect_nodes_at_depth(tree, target_depth):
    """BFS to collect all nodes at a given depth."""
    if tree["depth"] == target_depth:
        return [tree]
    if tree["children"] is None:
        return []
    nodes = []
    for c in tree["children"]:
        nodes.extend(_cifar_collect_nodes_at_depth(c, target_depth))
    return nodes


class _cifar_RandomSelectTransform:
    """Apply CLIP preprocessing, then randomly pick one augmentation."""
    def __init__(self, preprocess, *augmentations):
        self.preprocess = preprocess
        self.augmentations = augmentations

    def __call__(self, x):
        x = self.preprocess(x)
        aug = random.choice(self.augmentations)
        return aug(x)


class _cifar_TwoCropsTransform:
    """Produce two independent augmented views of one image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class _cifar_SCPClusterHead(nn.Module):
    """SCP MLP with softmax cluster assignments."""
    def __init__(self, input_dim, cluster_num=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 786),
            nn.BatchNorm1d(786),
            nn.ReLU(),
            nn.Linear(786, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, cluster_num),
            nn.Softmax(dim=1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        return self.encoder(x)


def _cifar_scp_confidence_loss(p, q):
    return -torch.mean(torch.log(torch.sum(p * q, dim=1) + 1e-8))


def _cifar_scp_balance_loss(p, q):
    return (
        -torch.sum(torch.mean(p, 0) * torch.log(torch.mean(p, 0) + 1e-8))
        - torch.sum(torch.mean(q, 0) * torch.log(torch.mean(q, 0) + 1e-8))
    )


def _cifar_train_scp_on_subset(
    feat_q, feat_k, indices,
    cluster_num=2, epochs=30, lr=0.001, batch_size=512,
    device=None,
):
    """Train an SCP head and return one hard assignment per input row."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_q = torch.tensor(feat_q[indices], dtype=torch.float32)
    sub_k = torch.tensor(feat_k[indices], dtype=torch.float32)
    n_samples = len(sub_q)
    ds = TensorDataset(sub_q, sub_k)

    drop = (n_samples > batch_size) and (n_samples % batch_size == 1)
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=drop,
    )
    eval_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False,
    )

    input_dim = sub_q.shape[1]
    head = _cifar_SCPClusterHead(input_dim, cluster_num=cluster_num).to(device)
    opt = optim.Adam(head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for _ in range(epochs):
        head.train()
        for bq, bk in train_loader:
            if bq.size(0) < 2:
                continue
            bq, bk = bq.to(device), bk.to(device)
            opt.zero_grad()
            pq = head(bq)
            pk = head(bk)
            loss = (
                _cifar_cross_entropy_loss(pq, pk)
                + _cifar_scp_confidence_loss(pq, pk)
                - _cifar_scp_balance_loss(pq, pk)
            )
            loss.backward()
            opt.step()
        scheduler.step()

    head.eval()
    all_preds = []
    with torch.no_grad():
        for bq, _ in eval_loader:
            pq = head(bq.to(device))
            all_preds.append(pq.argmax(dim=1).cpu().numpy())

    return np.concatenate(all_preds)


def _cifar_build_recursive_scp_hierarchy(
    feat_q, feat_k,
    cluster_num=2,
    max_depth=5,
    min_cluster_size=50,
    scp_epochs=30,
    scp_lr=0.001,
    scp_batch_size=512,
    sort_children=True,
    device=None,
):
    """Build a centroid tree by recursively fitting SCP heads."""
    def _build(indices, depth):
        centroid = feat_q[indices].mean(axis=0)
        node = {
            "centroid": centroid,
            "indices": indices,
            "depth": depth,
            "children": None,
        }

        if depth >= max_depth or len(indices) < min_cluster_size:
            return node

        labels = _cifar_train_scp_on_subset(
            feat_q, feat_k, indices,
            cluster_num=cluster_num, epochs=scp_epochs, lr=scp_lr,
            batch_size=scp_batch_size, device=device,
        )

        child_indices = [indices[labels == c] for c in range(cluster_num)]
        if any(len(ci) == 0 for ci in child_indices):
            return node

        children = [_build(ci, depth + 1) for ci in child_indices]

        if sort_children:
            children.sort(key=lambda ch: ch["centroid"][0])

        node["children"] = children
        return node

    print(f"Building SCP hierarchy (cluster_num={cluster_num}, "
          f"max_depth={max_depth}, scp_epochs={scp_epochs})...")
    root_indices = np.arange(len(feat_q))
    tree = _build(root_indices, depth=0)
    print("SCP hierarchy construction complete.")
    return tree


_cifar_cross_entropy_loss = nn.CrossEntropyLoss()

# --- CIFAR features and experiment ---

CIFAR_FEATURE_PROTOCOL = "original_cifar_raw_clip_minmax_float32_v1"


class CifarPairedViews(Dataset):
    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        from PIL import Image
        image, label = self.base[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        q, k = self.transform(image)
        return q, k, label


def cifar_raw_dataset(dataset: str, data_root: Path):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    return cls(root=str(data_root), train=True, download=False)


def _capture_cifar_rng() -> dict[str, np.ndarray]:
    """Persist the post-extraction RNG stream used by the original CIFAR code."""
    numpy_state = np.random.get_state()
    state = {
        "rng_python": np.asarray(json.dumps(random.getstate())),
        "rng_numpy_name": np.asarray(numpy_state[0]),
        "rng_numpy_keys": numpy_state[1],
        "rng_numpy_position": np.asarray(numpy_state[2]),
        "rng_numpy_has_gauss": np.asarray(numpy_state[3]),
        "rng_numpy_cached_gaussian": np.asarray(numpy_state[4]),
        "rng_torch_cpu": torch.get_rng_state().numpy(),
    }
    for index, values in enumerate(torch.cuda.get_rng_state_all()):
        state[f"rng_torch_cuda_{index}"] = values.cpu().numpy()
    return state


def _restore_cifar_rng(source: Path) -> None:
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, list) else value

    with np.load(source, allow_pickle=False) as arrays:
        random.setstate(tuples(json.loads(str(arrays["rng_python"].item()))))
        np.random.set_state((
            str(arrays["rng_numpy_name"].item()), arrays["rng_numpy_keys"],
            int(arrays["rng_numpy_position"]), int(arrays["rng_numpy_has_gauss"]),
            float(arrays["rng_numpy_cached_gaussian"]),
        ))
        torch.set_rng_state(torch.from_numpy(arrays["rng_torch_cpu"].copy()))
        for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            key = f"rng_torch_cuda_{index}"
            if key in arrays:
                torch.cuda.set_rng_state(torch.from_numpy(arrays[key].copy()), device=index)


def load_cifar_features(
    dataset: str, seed: int, *, feature_path: Path | None = None,
    input_root: Path = DEFAULT_INPUT_ROOT,
) -> DatasetFeatures:
    config = dataset_config(dataset)
    source = resolve_feature_path(dataset, seed, feature_path, input_root)
    required = {"support_view_q", "support_view_k", "support_ids", "support_fine_labels",
                "feature_protocol", "tree_seed", "rng_python", "rng_numpy_name", "rng_numpy_keys",
                "rng_numpy_position", "rng_numpy_has_gauss", "rng_numpy_cached_gaussian", "rng_torch_cpu"}
    with np.load(source, allow_pickle=False) as arrays:
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{source}: missing CIFAR arrays {sorted(missing)}; regenerate with extract")
        if str(arrays["feature_protocol"].item()) != CIFAR_FEATURE_PROTOCOL:
            raise ValueError("CIFAR requires original augmentation and raw-CLIP min-max features")
        if int(arrays["tree_seed"]) != seed:
            raise ValueError("CIFAR feature seed does not match the requested experiment seed")
        q, k = arrays["support_view_q"].copy(), arrays["support_view_k"].copy()
        ids = arrays["support_ids"].copy()
        labels = {name: arrays[f"support_{name}_labels"].copy() for name in config["purity_labels"]}
    expected = int(config["support_count"])
    for name, values in (("q", q), ("k", k)):
        if values.shape != (expected, 512) or values.dtype != np.float32:
            raise ValueError(f"CIFAR {name} must be float32 with shape ({expected}, 512)")
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError(f"CIFAR {name} is not finite min-max-scaled data")
    if not np.array_equal(ids, np.arange(expected)):
        raise ValueError("CIFAR cache must retain all official training images in original order")
    for name, values in labels.items():
        if values.shape != (expected,):
            raise ValueError(f"Invalid CIFAR {name} labels")
    unique, counts = np.unique(labels["fine"], return_counts=True)
    if not np.array_equal(unique, np.arange(config["classes"])) or np.any(counts != expected // config["classes"]):
        raise ValueError("CIFAR training class counts differ from the official split")
    if dataset == "cifar100" and not np.array_equal(labels["coarse"], np.asarray(_CIFAR100_COARSE_MAP)[labels["fine"]]):
        raise ValueError("CIFAR-100 coarse labels differ from the original fine-to-coarse mapping")
    return DatasetFeatures(
        view_q=q, view_k=k, deterministic=None, ids=ids, labels=labels,
        source_path=source, source_sha256=sha256_file(source),
        scaling_audit={"transform": "independent_featurewise_minmax", "l2_before_minmax": False,
                       "already_scaled": True, "dtype": "float32", "rows": expected, "dimension": 512},
    )


def extract_cifar_features(
    dataset: str, seed: int, *, data_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
    device: str | None = None, batch_size: int = 256, num_workers: int | None = None,
    clip_cache: Path = Path.home() / ".cache" / "clip", force: bool = False,
    require_cuda: bool = True,
) -> Path:
    torch.set_num_threads(1)
    if seed not in dataset_seeds(dataset):
        raise ValueError(f"{dataset} seeds are {dataset_seeds(dataset)}")
    num_workers = 4 if num_workers is None else num_workers
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Formal CLIP extraction requires CUDA")
    identity = {
        "dataset": dataset, "seed": seed, "data_root": str(data_root.resolve()),
        "feature_protocol": CIFAR_FEATURE_PROTOCOL, "batch_size": batch_size,
        "num_workers": num_workers, "script_sha256": sha256_file(__file__),
        "protocol_sha256": sha256_json(dataset_config(dataset)),
    }
    output = output_root / "inputs" / dataset / f"tree_seed{seed}"
    feature_path, complete_path = output / "features.npz", output / "complete.json"
    if complete_path.is_file() and feature_path.is_file() and not force:
        complete = json.loads(complete_path.read_text())
        if complete.get("extraction_identity") == identity and complete.get("feature_sha256") == sha256_file(feature_path):
            print(json.dumps({"status": "cache_valid", "path": str(feature_path)}))
            return feature_path
        raise RuntimeError(f"Cache/config mismatch at {output}; use --force")
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    import clip
    import torchvision

    seed_everything(seed)
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(clip_cache))
    model.eval()
    transform = _cifar_TwoCropsTransform(_cifar_RandomSelectTransform(
        preprocess, torchvision.transforms.RandomCrop(224, padding=4),
        torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    ))
    raw = cifar_raw_dataset(dataset, data_root)
    labels = np.asarray(raw.targets, dtype=int)
    if len(raw) != dataset_config(dataset)["support_count"]:
        raise ValueError("Expected the complete CIFAR training split")
    loader = DataLoader(CifarPairedViews(raw, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    all_q, all_k = [], []
    with torch.no_grad():
        for q, k, _ in loader:
            all_q.append(model.encode_image(q.to(device)).float().cpu())
            all_k.append(model.encode_image(k.to(device)).float().cpu())
    arrays = {
        "support_ids": np.arange(len(raw)), "support_fine_labels": labels,
        "feature_protocol": np.asarray(CIFAR_FEATURE_PROTOCOL), "tree_seed": np.asarray(seed),
    }
    # CIFAR uses float32 min-max scaling of raw CLIP features.
    for name, values in (("q", all_q), ("k", all_k)):
        features = torch.cat(values)
        minima, maxima = features.min(0, keepdim=True).values, features.max(0, keepdim=True).values
        arrays[f"support_view_{name}"] = ((features - minima) / (maxima - minima + 1e-8)).float().numpy()
    if dataset == "cifar100":
        arrays["support_coarse_labels"] = np.asarray(_CIFAR100_COARSE_MAP)[labels]
    arrays.update(_capture_cifar_rng())
    _atomic_npz(feature_path, arrays)
    checksum = sha256_file(feature_path)
    write_json_atomic(output / "manifest.json", {
        **identity, "encoder": "OpenAI CLIP ViT-B/32", "dimension": 512,
        "dtype": "float32", "normalization": "raw_clip_independent_featurewise_minmax",
        "support_count": len(raw), "rng_state": "after_extraction_before_scp",
    })
    write_json_atomic(complete_path, {"feature_sha256": checksum, "extraction_identity": identity,
                                     "status": "complete"})
    return feature_path


def run_cifar_experiment(
    dataset: str, seed: int, *, device: str | None = None, force: bool = False,
    require_cuda: bool = True, feature_path: Path | None = None,
    input_root: Path = DEFAULT_INPUT_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Run the CIFAR preset and save metrics, predictions, and checkpoints."""
    config = dataset_config(dataset)
    torch.set_num_threads(1)
    if seed not in dataset_seeds(dataset):
        raise ValueError(f"{dataset} seeds are {dataset_seeds(dataset)}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Formal SCP→CFP runs require CUDA")
    source = resolve_feature_path(dataset, seed, feature_path, input_root)
    identity = {"dataset": dataset, "seed": seed, "protocol_sha256": sha256_json(config),
                "feature_sha256": sha256_file(source), "script_sha256": sha256_file(__file__)}
    output = output_root / "runs" / dataset / f"seed{seed}"
    complete_path, metrics_path = output / "complete.json", output / "metrics.json"
    if complete_path.is_file() and metrics_path.is_file() and not force:
        complete = json.loads(complete_path.read_text())
        if complete.get("run_identity") == identity and complete.get("metrics_sha256") == sha256_file(metrics_path):
            verify_completion(output, identity)
            print(json.dumps({"status": "already_complete", "path": str(output)}))
            return output
        raise RuntimeError(f"Invalid completion marker at {output}; use --force")
    features = load_cifar_features(dataset, seed, feature_path=source, input_root=input_root)
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    write_json_atomic(output / "resolved_config.json", {
        **identity, "device": str(device), "config": config, "feature_path": str(source),
        "scaling_audit": features.scaling_audit,
    })
    seed_everything(seed)
    _restore_cifar_rng(source)
    tree = _cifar_build_recursive_scp_hierarchy(
        features.view_q, features.view_k, cluster_num=config["branching"],
        max_depth=config["max_depth"], min_cluster_size=config["scp_min_parent_size"],
        scp_epochs=config["scp_epochs_per_node"], scp_lr=config["scp_learning_rate"],
        scp_batch_size=config["scp_batch_size"], sort_children=True, device=device,
    )
    full_depth = min(_cifar_tree_max_full_depth(tree), config["max_depth"])
    if full_depth != config["max_depth"]:
        write_json_atomic(output / "topology_failure.json", {
            "actual_full_depth": full_depth, "observed_depth": config["observed_depth"],
            "reason": "Incomplete paper horizon; no automatic depth clamping permitted",
        })
        raise ValueError(f"Paper requires depth {config['max_depth']}, but full depth is {full_depth}")
    levels = _cifar_tree_to_levels(tree, target_depth=full_depth)
    basis = fit_centroid_pca(levels, config["observed_depth"])
    array_order, permutations = _cifar_canonicalize_levels_pca(
        levels, K=config["branching"], pca_model=basis, return_permutations=True)
    before_ordering = genealogy_snapshot(tree)
    canonicalize(tree, basis=basis.components_, mean=basis.mean_)
    genealogy_audit = validate_genealogy(before_ordering, genealogy_snapshot(tree))
    levels = _cifar_tree_to_levels(tree, target_depth=full_depth)
    for left, right in zip(array_order, levels):
        if not np.array_equal(left, right):
            raise RuntimeError("Saved tree and canonical centroid arrays disagree")
    write_json_atomic(output / "genealogy_audit.json", genealogy_audit)
    _atomic_npz(output / "canonical_permutations.npz",
                {f"level_{d}": p for d, p in enumerate(permutations)})
    heldout = list(config["heldout_depths"])
    scale, branching = config["scale"], config["branching"]
    predictions, training = {}, {}
    factories = (
        ("CFP", lambda: _cifar_BottleneckHRGMLP(
            d=levels[0].shape[1], K=branching, wide=config["cfp_width"], depth=config["cfp_depth"],
            act_type=config["cfp_activation"], use_layernorm=config["cfp_layernorm"])),
        ("Linear Res Gen", lambda: _cifar_LinearHRG(d=levels[0].shape[1], K=branching)),
        ("Constant", lambda: _cifar_ConstantHRG(d=levels[0].shape[1], K=branching)),
    )
    for method, factory in factories:
        # Continue the post-SCP random stream across models.
        model = _cifar_train_hrg_model(
            factory(), levels, x0=levels[0][0], L_train=config["observed_depth"],
            n_extra_levels=len(heldout), K=branching, scale=scale,
            lr=config["cfp_learning_rate"], epochs=config["cfp_epochs"],
            print_every=config["cfp_print_every"], weight_decay=config["cfp_weight_decay"], device=device,
        )
        predictions[method] = _cifar_build_hierarchy_from_model(model, levels[0][0], max_level=full_depth, scale=scale)
        training[method] = {"epochs": config["cfp_epochs"], "parameter_count": sum(p.numel() for p in model.parameters()),
                            "training_parent_count": sum(branching ** d for d in range(config["observed_depth"]))}
        _save_checkpoint(output / "checkpoints" / f"{METHOD_SLUGS[method]}.pt", model,
                         {**identity, "method": method, "scale": scale})
        if method == "CFP":
            residual = _cifar_estimate_average_residuals(levels, L_train=config["observed_depth"], K=branching, scale=scale)
            predictions["Avg Residual"] = _cifar_build_hierarchy_from_average_residuals(
                residual, levels[0][0], max_level=full_depth, scale=scale)
            training["Avg Residual"] = {"analytic": True, "parameter_count": 0}
        del model
    by_depth = {}
    for depth in heldout:
        by_depth[str(depth)] = {
            method: {
                "ordered_mse": float(np.mean((predicted[depth] - levels[depth]) ** 2)),
                "symmetric_squared_chamfer": _cifar_chamfer_distance(predicted[depth], levels[depth]),
                "normalized_dpt_distortion": _cifar_dpt_distortion(predicted, levels, branching, upto_level=depth),
            } for method, predicted in predictions.items()
        }
    averages = {method: {metric: float(np.mean([by_depth[str(d)][method][metric] for d in heldout]))
                         for metric in METRICS} for method in predictions}
    purity = {}
    for name, labels in features.labels.items():
        purity[name] = {}
        for depth in range(full_depth + 1):
            rows = _cifar_compute_cluster_purity(tree, labels, depth)
            purity[name][str(depth)] = {
                "clusters": len(rows), "macro_purity": float(np.mean([row[1] for row in rows])),
                "weighted_purity": float(np.average([row[1] for row in rows], weights=[row[0] for row in rows])),
            }
    arrays = {f"true_level_{d}": points for d, points in enumerate(levels)}
    for method, predicted in predictions.items():
        arrays.update({f"{METHOD_SLUGS[method]}_level_{d}": points for d, points in enumerate(predicted)})
    _atomic_npz(output / "levels_and_predictions.npz", arrays)
    _atomic_pickle(output / "scp_tree.pkl", tree)
    metrics = {
        "schema_version": 1, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": dataset, "display_name": config["display_name"], "seed": seed,
        "protocol_sha256": identity["protocol_sha256"], "feature_sha256": features.source_sha256,
        "branching": branching, "max_depth": full_depth, "requested_max_depth": config["max_depth"],
        "observed_depth": config["observed_depth"], "heldout_depths": heldout,
        "canonicalization": {"method": config["canonicalization"], "basis": basis.components_.tolist(),
                             "mean": basis.mean_.tolist(), "fit_depths": list(range(config["observed_depth"] + 1)),
                             "displacement_centered": False},
        "genealogy_audit": genealogy_audit,
        "scale_estimation": {"method": "fixed", "scale": scale},
        "training": training, "purity": purity, "by_depth": by_depth, "heldout_average": averages,
    }
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(complete_path, {
        "status": "complete", "run_identity": identity, "metrics_sha256": sha256_file(metrics_path),
        "levels_sha256": sha256_file(output / "levels_and_predictions.npz"),
        "tree_sha256": sha256_file(output / "scp_tree.pkl"),
    })
    print(json.dumps({"status": "complete", "dataset": dataset, "seed": seed,
                      "metrics": str(metrics_path), "heldout_average": averages}, indent=2))
    return output


# --- Command-line interface ---

DATASET_ALIASES = {"inat": "inaturalist", "nat": "inaturalist", "fgvc-aircraft": "aircraft",
                   "cifar-10": "cifar10", "cifar-100": "cifar100"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("config", "run", "extract", "summarize"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--dataset", choices=[*PROTOCOL["datasets"], *DATASET_ALIASES, "all"], required=True)
        selection = sub.add_mutually_exclusive_group()
        selection.add_argument("--seed", choices=[*(str(i) for i in range(5)), *(str(i) for i in range(42, 47)), "all"],
                               help="Actual seed: CIFAR=42–46, other datasets=0–4; all uses each dataset's seeds")
        selection.add_argument("--trial", type=int, choices=range(5),
                               help="Trial index shared across datasets, e.g. 0 maps to CIFAR seed 42 and other seed 0")
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        if command in ("run", "extract"):
            sub.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                             help="Cache root with DATASET/tree_seedN/features.npz")
            sub.add_argument("--manifest", type=Path, help="Frozen iNaturalist split_manifest.json")
            sub.add_argument("--device", default=None)
            sub.add_argument("--allow-cpu", action="store_true", help="Allow debugging without CUDA")
            sub.add_argument("--force", action="store_true")
            sub.add_argument("--data-root", type=Path, required=command == "extract",
                             help="Raw images/metadata for one dataset; on run, extract features first")
            sub.add_argument("--batch-size", type=int, default=256, help="CLIP extraction batch size")
            sub.add_argument("--num-workers", type=int, help="CLIP workers: Food-101=8, others=4")
            sub.add_argument("--clip-cache", type=Path, default=Path.home() / ".cache" / "clip")
        if command == "run":
            sub.add_argument("--features", type=Path, help="Explicit NPZ for exactly one dataset and seed")
            sub.add_argument("--cifar-root", type=Path,
                             help="Shared raw CIFAR root; extract missing CIFAR caches when running all datasets")
        if command == "summarize":
            sub.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    datasets = list(PROTOCOL["datasets"]) if dataset == "all" else [dataset]
    try:
        seed_map = {d: resolve_seeds(d, args.seed, args.trial, all_by_default=args.command in ("config", "summarize"))
                    for d in datasets}
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "config":
        print(json.dumps({**load_protocol(), "selected_seeds_by_dataset": seed_map,
                          "datasets": {d: dataset_config(d) for d in datasets}}, indent=2))
        return
    if args.command == "summarize":
        print(json.dumps(aggregate(require_complete=not args.allow_partial,
                                   output_root=args.output_root, datasets=datasets, seed_map=seed_map), indent=2))
        return
    if getattr(args, "features", None) is not None and (len(datasets) != 1 or len(seed_map[datasets[0]]) != 1):
        parser.error("--features requires exactly one dataset and seed")
    if args.data_root is not None and len(datasets) != 1:
        parser.error("--data-root requires exactly one dataset; use --cifar-root for shared CIFAR images")
    if args.data_root is not None and getattr(args, "features", None) is not None:
        parser.error("Use either --data-root or --features")
    if args.manifest is not None and "inaturalist" not in datasets:
        parser.error("--manifest applies only to iNaturalist")
    if getattr(args, "cifar_root", None) is not None and not any(d in CIFAR_PRESETS for d in datasets):
        parser.error("--cifar-root requires a CIFAR dataset")
    for dataset in datasets:
        manifest = args.manifest if dataset == "inaturalist" else None
        if dataset == "inaturalist" and manifest is None:
            candidate = args.input_root / "inaturalist" / "split_manifest.json"
            manifest = candidate if candidate.is_file() else None
        for seed in seed_map[dataset]:
            feature_path = getattr(args, "features", None)
            raw_root = args.data_root
            if dataset in CIFAR_PRESETS and raw_root is None and feature_path is None:
                source = resolve_feature_path(dataset, seed, None, args.input_root)
                if not source.is_file():
                    # Reuse features extracted into the output directory.
                    cached = args.output_root / "inputs" / dataset / f"tree_seed{seed}" / "features.npz"
                    if cached.is_file():
                        feature_path = cached
                    elif getattr(args, "cifar_root", None) is not None:
                        raw_root = args.cifar_root
                    else:
                        parser.error(f"Missing CIFAR cache {source}; provide --cifar-root or --data-root to extract it")
            if raw_root is not None:
                feature_path = extract_features(
                    dataset, seed, data_root=raw_root, output_root=args.output_root,
                    manifest_path=manifest, device=args.device, batch_size=args.batch_size,
                    num_workers=args.num_workers, clip_cache=args.clip_cache,
                    force=args.force, require_cuda=not args.allow_cpu,
                )
            if args.command == "run":
                run_experiment(
                    dataset, seed, device=args.device, force=args.force,
                    require_cuda=not args.allow_cpu, feature_path=feature_path,
                    input_root=args.input_root, manifest_path=manifest, output_root=args.output_root,
                )
            else:
                print(json.dumps({"status": "extracted", "path": str(feature_path)}))


if __name__ == "__main__":
    main()
