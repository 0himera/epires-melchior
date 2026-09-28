"""Diverse ML task and dataset generator for Crucible self-play."""

from __future__ import annotations

import random
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


def generate_task(seed: int) -> tuple[TaskProfile, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates an empirical ML problem with stratified or random train/val split."""
    rng = np.random.RandomState(seed)
    task_variant = seed % 6

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
        # Non-linear regression with noisy target
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
        # Real-world benchmark: Wine or Breast Cancer with random feature mask
        ds_loader = random.choice([load_breast_cancer, load_wine, load_diabetes])
        data = ds_loader()
        X, y = data.data, data.target
        task_type = "regression" if ds_loader == load_diabetes else (
            "multiclass_classification" if ds_loader == load_wine else "binary_classification"
        )
        metric = "r2" if task_type == "regression" else ("accuracy" if task_type == "multiclass_classification" else "roc_auc")
        desc = f"Empirical tabular benchmark ({data.get('filename', 'standard_ds')}) with {X.shape[0]} rows, {X.shape[1]} features."

    else:
        # Digits multiclass
        data = load_digits()
        X, y = data.data, data.target
        task_type = "multiclass_classification"
        metric = "accuracy"
        desc = f"Handwritten digits image feature classification with {X.shape[0]} rows, 64 pixel features (8x8)."

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

    profile = TaskProfile(
        task_id=f"crucible_task_{seed:06d}",
        task_type=task_type,
        metric=metric,
        direction="maximize",
        description=desc,
        n_train=len(X_train),
        n_val=len(X_val),
        n_features=X_train.shape[1],
    )

    return profile, X_train, y_train, X_val, y_val
