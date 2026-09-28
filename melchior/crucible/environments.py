"""Diverse ML task and dataset generator for Crucible self-play."""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import (
    make_classification,
    make_regression,
    load_wine,
    load_breast_cancer,
    load_diabetes,
    load_digits,
)
from sklearn.model_selection import train_test_split


@dataclass
class TaskProfile:
    task_id: str
    task_type: str  # binary_classification | multiclass_classification | regression
    metric: str     # roc_auc | accuracy | r2 | f1
    direction: str  # maximize | minimize
    description: str
    n_train: int
    n_val: int
    n_features: int
    split: str = "train"
    dataset_id: str = ""
    data_hash: str = ""


def generate_task(seed: int, split: str = "train") -> tuple[TaskProfile, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates an empirical ML problem with stratified or random train/val split."""
    if split not in {"train", "eval"} or seed < 0:
        raise ValueError("split must be train/eval and seed non-negative")
    task_seed = seed
    task_variant = seed % 6
    # Separate random dataset realizations even when callers reuse a seed.
    seed = int.from_bytes(hashlib.sha256(f"tasks-v2:{split}:{seed}".encode()).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    dataset_id = f"synthetic:{split}:{task_seed}:{task_variant}"

    if task_variant == 0:
        # High-dimensional binary classification with redundant features
        n_samples = rng.randint(800, 2500)
        n_features = rng.randint(30, 80)
        n_informative = int(n_features * 0.4)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=int(n_features * 0.3),
            flip_y=0.05,
            random_state=seed,
        )
        task_type = "binary_classification"
        metric = "roc_auc"
        desc = (
            f"Binary classification with {n_samples} rows, {n_features} features "
            f"({n_informative} informative, rest redundant/noise), 5% label flip noise."
        )

    elif task_variant == 1:
        # Imbalanced binary classification (1:15 ratio)
        n_samples = rng.randint(1200, 3000)
        n_features = rng.randint(15, 45)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            weights=[0.93, 0.07],
            flip_y=0.02,
            random_state=seed,
        )
        task_type = "binary_classification"
        metric = "f1"
        desc = (
            f"Imbalanced binary classification with {n_samples} rows, {n_features} features, "
            f"class ratio 93:7 (rare positive class)."
        )

    elif task_variant == 2:
        # Linear regression with noisy target
        n_samples = rng.randint(600, 2000)
        n_features = rng.randint(10, 40)
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.6),
            noise=12.0,
            random_state=seed,
        )
        task_type = "regression"
        metric = "r2"
        desc = (
            f"Regression with {n_samples} rows, {n_features} continuous features, "
            f"non-zero Gaussian target noise (std=12.0)."
        )

    elif task_variant == 3:
        # Multiclass classification
        n_samples = rng.randint(900, 2200)
        n_features = rng.randint(16, 40)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=4,
            n_informative=int(n_features * 0.5),
            random_state=seed,
        )
        task_type = "multiclass_classification"
        metric = "accuracy"
        desc = (
            f"4-class multiclass classification with {n_samples} rows and {n_features} features."
        )

    elif task_variant == 4:
        # Whole built-in datasets are reserved for one partition, not resplit
        # into overlapping train/eval task pools.
        ds_loader = random.Random(seed).choice([load_breast_cancer, load_wine]) if split == "train" else load_diabetes
        dataset_id = ds_loader.__name__
        data = ds_loader()
        X, y = data.data, data.target
        task_type = "regression" if ds_loader == load_diabetes else (
            "multiclass_classification" if ds_loader == load_wine else "binary_classification"
        )
        metric = "r2" if task_type == "regression" else ("accuracy" if task_type == "multiclass_classification" else "roc_auc")
        desc = f"Empirical tabular benchmark ({data.get('filename', 'standard_ds')}) with {X.shape[0]} rows, {X.shape[1]} features."

    else:
        if split == "eval":
            data = load_digits()
            X, y = data.data, data.target
            dataset_id = "load_digits"
            desc = f"Handwritten digits classification: {len(y)} rows, 64 pixel features."
        else:
            X, y = make_classification(n_samples=1600, n_features=64, n_informative=24,
                                       n_classes=10, random_state=seed)
            desc = "Synthetic 10-class classification: 1600 rows, 64 features."
        task_type = "multiclass_classification"
        metric = "accuracy"

    stratify = y if task_type in ["binary_classification", "multiclass_classification"] else None
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=stratify
        )
    except ValueError:
        # Fallback if class has too few instances
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=seed
        )

    digest = hashlib.sha256()
    for array in (X_train, y_train, X_val, y_val):
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    profile = TaskProfile(
        task_id=f"crucible_{split}_{task_seed:08d}",
        task_type=task_type,
        metric=metric,
        direction="maximize",
        description=desc,
        n_train=len(X_train),
        n_val=len(X_val),
        n_features=X_train.shape[1],
        split=split, dataset_id=dataset_id, data_hash=digest.hexdigest(),
    )

    return profile, X_train, y_train, X_val, y_val
