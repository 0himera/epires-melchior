"""Trusted scoring and paired uncertainty estimates, outside generated code."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score

CONTRACT_VERSION = "predictions-v2"


def score_predictions(metric: str, y_true, predictions, *, classes=None) -> float:
    y = np.asarray(y_true)
    pred = np.asarray(predictions)
    if pred.shape != y.shape or pred.ndim != 1 or not len(pred):
        raise ValueError(f"Expected {y.shape} predictions; received {pred.shape}")
    if pred.dtype.kind not in "biuf" or not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite numeric values")
    if metric in {"accuracy", "f1"}:
        allowed = np.unique(y) if classes is None else np.asarray(classes)
        if not np.isin(pred, allowed).all():
            raise ValueError("Class predictions contain unknown labels")
    functions = {"accuracy": accuracy_score, "f1": f1_score,
                 "r2": r2_score, "roc_auc": roc_auc_score}
    if metric not in functions:
        raise ValueError(f"Unsupported metric: {metric}")
    kwargs = {"average": "binary", "pos_label": 1, "zero_division": 0} if metric == "f1" else {}
    value = float(functions[metric](y, pred, **kwargs))
    if not np.isfinite(value):
        raise ValueError("Computed metric is not finite")
    return value


def paired_interval(metric, y_true, a, b, *, seed=42, resamples=200):
    """95% percentile interval for score(B)-score(A) on paired test rows.

    Conditional uncertainty on this test sample, not uncertainty across datasets
    or model fits. Classification resamples preserve the observed class counts.
    """
    y, a, b = map(np.asarray, (y_true, a, b))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)] if metric != "r2" else [np.arange(len(y))]
    deltas = []
    for _ in range(resamples):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        deltas.append(score_predictions(metric, y[ix], b[ix]) - score_predictions(metric, y[ix], a[ix]))
    return [float(x) for x in np.quantile(deltas, [0.025, 0.975])]
