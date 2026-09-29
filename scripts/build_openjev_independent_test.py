"""Freeze an out-of-development algorithm-comparison benchmark before NLI scoring.

This benchmark deliberately uses hand-written, fixed sklearn candidates. It tests
transfer to a different candidate source, not the exact Qwen self-play distribution.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.client import CandidatePair
from melchior.crucible.environments import TaskProfile, generate_task
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.task_pack import TaskPack


SYNTHETIC_SEEDS = tuple(range(80000, 80096))  # 64 seeds in variants 0..3
REAL_DATASETS = (
    (31, "credit-g", "binary_classification", "roc_auc", None),
    (36, "segment", "multiclass_classification", "accuracy", None),
    (44, "spambase", "binary_classification", "roc_auc", None),
    (1590, "adult", "binary_classification", "roc_auc", None),
    (43257, "wine_quality", "regression", "r2", "quality"),
    (45026, "heloc", "binary_classification", "roc_auc", None),
)
REAL_SPLITS = (1729, 1730, 1731)
PAIR_GRAPHS = (
    (("linear", "hgb"), ("forest", "extra"), ("knn", "svm")),
    (("linear", "forest"), ("hgb", "knn"), ("svm", "extra")),
    (("linear", "extra"), ("forest", "knn"), ("hgb", "svm")),
)
ALGO_DESCRIPTIONS = {
    "linear": "A regularized linear model with standardized features.",
    "hgb": "A histogram-based gradient boosting model with limited leaves.",
    "forest": "A random forest of moderately deep trees.",
    "extra": "An extremely randomized tree ensemble with moderate depth.",
    "knn": "A distance-based nearest-neighbor model with standardized features.",
    "svm": "An RBF-kernel support-vector model with standardized features.",
}


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def code_for(algo: str, metric: str) -> str:
    regression = metric == "r2"
    if algo == "linear":
        imp = "Ridge" if regression else "LogisticRegression"
        ctor = "Ridge(alpha=10.0)" if regression else "LogisticRegression(C=1.0, max_iter=400)"
        model = f"make_pipeline(StandardScaler(), {ctor})"
        imports = f"from sklearn.linear_model import {imp}\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler"
    elif algo in ("forest", "extra"):
        cls = ("RandomForest" if algo == "forest" else "ExtraTrees") + ("Regressor" if regression else "Classifier")
        imports = f"from sklearn.ensemble import {cls}"
        model = f"{cls}(n_estimators=60, max_depth=10, n_jobs=1, random_state=42)"
    elif algo == "hgb":
        cls = "HistGradientBoostingRegressor" if regression else "HistGradientBoostingClassifier"
        imports = f"from sklearn.ensemble import {cls}"
        model = f"{cls}(max_iter=60, max_leaf_nodes=15, random_state=42)"
    elif algo == "knn":
        cls = "KNeighborsRegressor" if regression else "KNeighborsClassifier"
        imports = f"from sklearn.neighbors import {cls}\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler"
        model = f"make_pipeline(StandardScaler(), {cls}(n_neighbors=7))"
    elif algo == "svm":
        cls = "SVR" if regression else "SVC"
        imports = f"from sklearn.svm import {cls}\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler"
        ctor = "SVR(C=10.0, epsilon=0.1)" if regression else "SVC(C=3.0, gamma='scale')"
        model = f"make_pipeline(StandardScaler(), {ctor})"
    else:
        raise ValueError(algo)
    prediction = "predict_proba(X_test)[:, 1]" if metric == "roc_auc" and algo != "svm" else (
        "decision_function(X_test)" if metric == "roc_auc" else "predict(X_test)"
    )
    return f"{imports}\n\ndef fit_predict(X_train, y_train, X_test):\n    model = {model}\n    model.fit(X_train, y_train)\n    return model.{prediction}\n"


def real_task(spec, split_seed: int):
    dataset_id, name, task_type, metric, target_column = spec
    kwargs = {"data_id": dataset_id, "as_frame": True, "parser": "auto"}
    if target_column:
        kwargs["target_column"] = target_column
    source = fetch_openml(**kwargs)
    frame = source.data.copy()
    # Continuous measurements can be unique on every row. Cardinality alone
    # cannot identify an ID column, and must not discard numeric features.
    dropped = [col for col in frame if col.lower() == "id"]
    frame = frame.drop(columns=dropped)
    target = np.asarray(source.target.astype(float) if task_type == "regression" else LabelEncoder().fit_transform(source.target))
    # Bound runtime without using the hidden test labels during preprocessing.
    if len(frame) > 2500:
        ids, _ = train_test_split(np.arange(len(frame)), train_size=2500, random_state=split_seed,
                                  stratify=target if task_type != "regression" else None)
        frame, target = frame.iloc[ids].reset_index(drop=True), target[ids]
    train_ids, test_ids = train_test_split(np.arange(len(frame)), test_size=.25, random_state=split_seed,
                                            stratify=target if task_type != "regression" else None)
    train, test = frame.iloc[train_ids], frame.iloc[test_ids]
    numeric = list(frame.select_dtypes(include="number").columns)
    categorical = [c for c in frame if c not in numeric]
    x_train, x_test = [], []
    if numeric:
        a, b = train[numeric].to_numpy(dtype=float), test[numeric].to_numpy(dtype=float)
        med = np.nanmedian(a, axis=0)
        med = np.where(np.isfinite(med), med, 0)
        x_train.append(np.where(np.isfinite(a), a, med))
        x_test.append(np.where(np.isfinite(b), b, med))
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        a = train[categorical].astype("string").fillna("__missing__").astype(str)
        b = test[categorical].astype("string").fillna("__missing__").astype(str)
        x_train.append(encoder.fit_transform(a))
        x_test.append(encoder.transform(b))
    X_train, X_test = (np.concatenate(parts, axis=1).astype("float32") for parts in (x_train, x_test))
    y_train, y_test = target[train_ids], target[test_ids]
    profile = TaskProfile(
        task_id=f"independent_openml_{dataset_id}_{split_seed}", task_type=task_type,
        metric=metric, direction="maximize", split="eval",
        description=f"OpenML {name} (dataset {dataset_id}): {task_type}; train-only feature preprocessing.",
        n_train=len(y_train), n_val=len(y_test), n_features=X_train.shape[1],
        dataset_id=f"openml:{dataset_id}:{split_seed}",
        data_hash=digest_arrays(X_train, y_train, X_test, y_test),
    )
    return profile, X_train, y_train, X_test, y_test, {"source": "OpenML", "openml_id": dataset_id,
        "source_name": name, "split_seed": split_seed, "dropped_columns": dropped,
        "numeric_columns": numeric, "categorical_columns": categorical}


async def run_task(index, task, output: Path, semaphore, sandbox, arbiter):
    profile, X_train, y_train, X_test, y_test, preprocessing = task
    np.savez_compressed(output / "arrays" / f"{profile.task_id}.npz", X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
    algorithms = tuple(ALGO_DESCRIPTIONS)

    async def run_algo(algo):
        async with semaphore:
            return await sandbox.execute(code_for(algo, profile.metric), X_train, y_train, X_test, y_test,
                                         metric=profile.metric)

    results = dict(zip(algorithms, await asyncio.gather(*(run_algo(a) for a in algorithms))))
    candidate_rows = [{"schema": "independent-candidate-v1", "task_id": profile.task_id,
                       "algorithm": algo, "code": code_for(algo, profile.metric),
                       "hypothesis": ALGO_DESCRIPTIONS[algo], "result": asdict(result)}
                      for algo, result in results.items()]
    pair_rows = []
    for ordinal, (left, right) in enumerate(PAIR_GRAPHS[index % len(PAIR_GRAPHS)]):
        pair_seed = int.from_bytes(hashlib.sha256(f"independent:{profile.task_id}:{ordinal}".encode()).digest()[:4], "little")
        if pair_seed % 2:
            left, right = right, left
        pair = CandidatePair(ALGO_DESCRIPTIONS[left], code_for(left, profile.metric),
                             ALGO_DESCRIPTIONS[right], code_for(right, profile.metric),
                             operator=f"portfolio_{left}_vs_{right}",
                             generation={"source": "predeclared_sklearn_portfolio", "role_references_normalized": True})
        outcome = arbiter.evaluate(profile, pair, results[left], results[right], y_true=y_test, seed=pair_seed)
        pair_rows.append({"schema": "predictions-v2", "seed": pair_seed, "task_index": index,
                          "task_seed": profile.task_id, "profile": asdict(profile), "pair": asdict(pair),
                          "candidate_ids": {"A": left, "B": right},
                          "candidate_a": asdict(results[left]), "candidate_b": asdict(results[right]),
                          "outcome": asdict(outcome), "preprocessing": preprocessing,
                          "generation_swapped": False})
    return candidate_rows, pair_rows


async def main(args):
    pack = TaskPack(args.task_pack, split='eval') if args.task_pack else None
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "arrays").mkdir()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {"schema": "independent-test-protocol-v1", "source_sha256": source_hash,
                "synthetic_seeds": [] if pack else list(SYNTHETIC_SEEDS),
                "real_datasets": [] if pack else REAL_DATASETS,
                "real_split_seeds": [] if pack else REAL_SPLITS, "pair_graphs": PAIR_GRAPHS,
                "algorithms": list(ALGO_DESCRIPTIONS), "min_delta": .005,
                "paired_bootstrap_resamples": 200,
                "candidate_source": "fixed sklearn portfolio; not Qwen-generated",
                "selection": "All decisive pairs, no model-dependent filtering",
                "independence": "No model scores consulted before corpus freeze"}
    if pack:
        protocol['task_pack_sha256'] = pack.sha256
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    tasks = []
    if pack:
        tasks = [(*pack.load(seed), pack.metadata(seed)['preprocessing']) for seed in pack.seeds]
    else:
        for seed in SYNTHETIC_SEEDS:
            if seed % 6 < 4:
                p, a, b, c, d = generate_task(seed, split="eval")
                tasks.append((p, a, b, c, d, {"source": "crucible_synthetic", "seed": seed}))
        for spec in REAL_DATASETS:
            for split_seed in REAL_SPLITS:
                tasks.append(real_task(spec, split_seed))
    print(f"Frozen protocol; executing {len(tasks)} tasks, {len(tasks)*3} pairs", flush=True)
    semaphore = asyncio.Semaphore(args.workers)
    sandbox = AsyncSandbox(timeout_s=90)
    arbiter = CrucibleArbiter(min_delta=.005)
    all_candidates, all_pairs = [], []
    # Limit in-flight tasks while still sharing the process semaphore globally.
    for start in range(0, len(tasks), 8):
        batch = await asyncio.gather(*(run_task(i, tasks[i], output, semaphore, sandbox, arbiter)
                                      for i in range(start, min(start + 8, len(tasks)))))
        for candidates, pairs in batch:
            all_candidates.extend(candidates)
            all_pairs.extend(pairs)
        print(f"Labeled {len(all_pairs)}/{len(tasks)*3}; decisive "
              f"{sum(r['outcome']['winner'] in ('A','B') for r in all_pairs)}", flush=True)
    def write_jsonl(name, rows):
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_jsonl("candidates.jsonl", all_candidates)
    write_jsonl("evaluations.jsonl", all_pairs)
    decisive = [r for r in all_pairs if r["outcome"]["winner"] in ("A", "B")]
    write_jsonl("holdout.jsonl", decisive)
    manifest = {"schema": "independent-test-manifest-v1", "tasks": len(tasks),
                "candidate_executions": len(all_candidates), "pairs": len(all_pairs), "decisive": len(decisive),
                "outcomes": dict(Counter(r["outcome"]["winner"] for r in all_pairs)),
                "decisive_by_source": dict(Counter(r["preprocessing"]["source"] for r in decisive)),
                "files": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("tasks", "pairs", "decisive", "outcomes", "decisive_by_source")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--task-pack', type=Path, help='Use an existing frozen eval pack instead of built-in datasets')
    asyncio.run(main(parser.parse_args()))
