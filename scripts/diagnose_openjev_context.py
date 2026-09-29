"""Post-hoc context ablations on frozen comparisons; never a new acceptance test.

CV uses only the archived outer training arrays. Outer test outcomes are used
only to report accuracy after inference. All inputs, fold scores and responses
are saved so the diagnosis can be reproduced without generating new candidates.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from melchior.crucible.decisions import comparison, symmetric_choice
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.storage import atomic_json


COHORTS = {"run": "run/holdout.jsonl", "qwen": "extras/qwen/holdout.jsonl",
           "portfolio": "extras/portfolio/holdout.jsonl"}
MODES = ("full", "no_rationale", "no_code", "anonymous_task")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def requests_for(row, mode, pilot=None):
    """Construct model inputs from the allowed fields, excluding outcome/labels."""
    profile, pair = deepcopy(row["profile"]), deepcopy(row["pair"])
    if mode == "no_rationale":
        pair["hypothesis_a"] = pair["hypothesis_b"] = ""
    elif mode == "no_code":
        pair["code_a"] = pair["code_b"] = "[Implementation withheld]"
    elif mode == "anonymous_task":
        profile["description"] = f"A tabular {profile['task_type']} task; data distribution not supplied."
    elif mode not in ("full", "train_cv"):
        raise ValueError(mode)
    payloads = []
    for swapped in (False, True):
        payload = comparison(profile, pair, swapped=swapped,
                             generation_swapped=row.get("generation_swapped", False))
        if mode == "train_cv":
            if pilot is None:
                raise ValueError("train_cv requires observations")
            a, b = ("b", "a") if swapped else ("a", "b")
            observations = {"Candidate A": pilot[a], "Candidate B": pilot[b]}
            payload["state"] += ("\nObserved three-fold validation within the outer TRAIN partition only. "
                "These are measurements, not candidate claims; higher is better. "
                "No outer test labels or outer test performance are available.\n"
                + json.dumps(observations, sort_keys=True, allow_nan=False))
        payloads.append(payload)
    return payloads


def distribution_diagnostics(scored):
    """Descriptive binary metrics; these are not fitted calibration parameters."""
    n = len(scored)
    if not n:
        return {"n": 0}
    p = np.array([r["probabilities"]["A"] for r in scored])
    y = np.array([r["winner"] == "A" for r in scored], dtype=float)
    confidence = np.maximum(p, 1-p)
    correct = np.array([r["choice"] == r["winner"] for r in scored])
    bins = []
    for low, high in ((.5, .7), (.7, .9), (.9, 1.00001)):
        ix = (confidence >= low) & (confidence < high)
        if ix.any():
            bins.append({"range": [low, min(high, 1.)], "n": int(ix.sum()),
                         "mean_score": float(confidence[ix].mean()),
                         "accuracy": float(correct[ix].mean())})
    return {"n": n, "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "binary_brier": float(np.mean((p-y)**2)),
            "binary_log_loss": float(-np.mean(y*np.log(np.clip(p, 1e-8, 1))
                + (1-y)*np.log(np.clip(1-p, 1e-8, 1)))),
            "mean_relative_entailment_confidence": float(confidence.mean()), "bins": bins}


async def main(args):
    if args.workers < 1 or args.timeout <= 0:
        raise ValueError("Workers and timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = [(cohort, row) for cohort, relative in COHORTS.items()
              for row in rows(args.archive / relative)]
    expected = json.loads((args.archive / "base_rescore.json").read_text())["data_sha256"]
    hashes = {name: sha(args.archive / relative) for name, relative in COHORTS.items()}
    for name, relative in COHORTS.items():
        if expected["/experiment/" + relative] != hashes[name]:
            raise ValueError(f"Frozen corpus mismatch: {name}")
    training = {}
    if args.training:
        for split in ("train", "dev"):
            path = args.training / f"ml_{split}.jsonl"
            rs = rows(path)
            training[split] = {"sha256": sha(path), "groups": len(rs),
                "group_prefix": dict(Counter(r["group"].split(":")[0] for r in rs)),
                "metrics": dict(Counter(r["metric"] for r in rs)),
                "operators": dict(Counter(r["operator"] for r in rs)),
                "winners": dict(Counter(r["winner"] for r in rs))}
    atomic_json(args.output / "protocol.json", {"scope": "Post-hoc diagnosis on already inspected frozen tests",
        "script_sha256": sha(Path(__file__)), "corpus_sha256": hashes, "modes": MODES,
        "cv_cohorts": ["qwen", "portfolio"], "cv_folds": 3, "cv_random_state": 171414,
        "workers": args.workers, "candidate_timeout_s": args.timeout,
        "adapter_sha256": args.adapter_sha256, "training": training,
        "limitation": "Frozen train-only feature preprocessing is not refitted within inner CV folds; "
            "no outer test arrays/labels are loaded for CV. CV uncertainty is not a new independent test."})
    manifest = json.loads((args.archive / "extra_task_pack/manifest.json").read_text())
    task_by_id = {r["profile"]["task_id"]: r for r in manifest["tasks"]}
    semaphore = asyncio.Semaphore(args.workers)
    sandbox = AsyncSandbox(timeout_s=args.timeout)
    cache = {}
    fold_log = (args.output / "cv_folds.jsonl").open("w")

    async def pilot_for(row, side):
        task = task_by_id[row["profile"]["task_id"]]
        path = args.archive / "extra_task_pack" / task["arrays"]
        if sha(path) != task["arrays_sha256"]:
            raise ValueError("Training arrays checksum mismatch")
        code = row["pair"]["code_" + side]
        key = (task["arrays_sha256"], hashlib.sha256(code.encode()).hexdigest(), row["profile"]["metric"])
        if key not in cache:
            async def execute():
                with np.load(path, allow_pickle=False) as data:
                    X, y = data["X_train"], data["y_train"]
                splitter = (KFold if row["profile"]["metric"] == "r2" else StratifiedKFold)(
                    n_splits=3, shuffle=True, random_state=171414)
                outputs = []
                for fold, (train, val) in enumerate(splitter.split(X, y)):
                    async with semaphore:
                        result = await sandbox.execute(code, X[train], y[train], X[val], y[val],
                                                       metric=row["profile"]["metric"])
                    record = {"task_id": task["profile"]["task_id"], "arrays_sha256": key[0],
                        "code_sha256": key[1], "metric": key[2], "fold": fold,
                        "train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                        "validation_indices_sha256": hashlib.sha256(val.tobytes()).hexdigest(),
                        "status": result.status, "score": result.metric, "wall_time_s": result.wall_time_s,
                        "stderr": result.stderr}
                    fold_log.write(json.dumps(record, allow_nan=False) + "\n"); fold_log.flush()
                    outputs.append(record)
                scores = [r["score"] for r in outputs if r["status"] == "success"]
                return {"metric": key[2], "fold_scores": scores, "successful_folds": len(scores),
                    "mean": float(np.mean(scores)) if len(scores) == 3 else None,
                    "fold_std": float(np.std(scores, ddof=1)) if len(scores) == 3 else None,
                    "total_fit_seconds": sum(r["wall_time_s"] for r in outputs)}
            cache[key] = asyncio.create_task(execute())
        return await cache[key]

    async with httpx.AsyncClient(timeout=120.) as client:
        health = (await client.get(args.url + "/health")); health.raise_for_status()
        identity = health.json()["model_identity"]
        if identity["adapter_sha256"] != args.adapter_sha256:
            raise ValueError("Serving model identity mismatch")
        pilots = {}
        if args.pilot:
            jobs = [(cohort, row) for cohort, row in corpus if cohort != "run"]
            async def one_pilot(cohort, row):
                a, b = await asyncio.gather(pilot_for(row, "a"), pilot_for(row, "b"))
                pilots[(cohort, row["seed"])] = {"a": a, "b": b}
            await asyncio.gather(*(one_pilot(c, r) for c, r in jobs))
            print(f"CV complete: {len(cache)} unique candidates; {len(pilots)} pairs", flush=True)
        fold_log.close()
        atomic_json(args.output / "cv_pairs.json", [{"cohort": c, "seed": seed, **p}
                                                  for (c, seed), p in pilots.items()])
        results, errors = [], []
        with (args.output / "inference.jsonl").open("w") as log:
            for cohort, row in corpus:
                modes = list(MODES)
                pilot = pilots.get((cohort, row["seed"]))
                if pilot and all(pilot[s]["mean"] is not None for s in ("a", "b")):
                    modes.append("train_cv")
                for mode in modes:
                    record = {"cohort": cohort, "seed": row["seed"], "dataset_id": row["profile"]["dataset_id"],
                        "metric": row["profile"]["metric"], "winner": row["outcome"]["winner"], "mode": mode}
                    try:
                        payloads = requests_for(row, mode, pilot)
                        raw = []
                        for payload in payloads:
                            response = await client.post(args.url + "/v1/systemone", json=payload)
                            response.raise_for_status(); body = response.json()
                            if body["model_identity"] != identity:
                                raise ValueError("Model changed during diagnosis")
                            raw.append(body)
                        p, q = (r["answers"]["winner"]["probabilities"]["A"] for r in raw)
                        record.update(symmetric_choice(p, q))
                        record.update(raw_responses=raw, requests=payloads)
                        if mode == "train_cv":
                            record["direct_cv_choice"] = ("A" if pilot["a"]["mean"] > pilot["b"]["mean"] else
                                "B" if pilot["b"]["mean"] > pilot["a"]["mean"] else "UNCERTAIN")
                        results.append(record)
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        errors.append(record)
                    log.write(json.dumps(record, allow_nan=False) + "\n"); log.flush()
                if len(results) % 20 < len(modes):
                    print(f"Scored: {len(results)}; errors: {len(errors)}", flush=True)
    summary = {"scope": "Post-hoc diagnosis, no training or checkpoint selection", "errors": errors,
               "training": training, "results": {}}
    for cohort in ("run", "qwen", "portfolio", "real", "all"):
        summary["results"][cohort] = {}
        for mode in (*MODES, "train_cv"):
            selected = [r for r in results if r["mode"] == mode and
                (cohort == "all" or r["cohort"] == cohort or
                 (cohort == "real" and r["dataset_id"].startswith("openml:")))]
            summary["results"][cohort][mode] = distribution_diagnostics(selected)
            if mode == "train_cv" and selected:
                summary["results"][cohort][mode]["direct_cv_correct"] = sum(
                    r["direct_cv_choice"] == r["winner"] for r in selected)
    baseline = {(r["cohort"], r["seed"]): r for r in results if r["mode"] == "full"}
    summary["changed_from_full"] = {}
    for mode in (*MODES[1:], "train_cv"):
        selected = [r for r in results if r["mode"] == mode]
        comparable = [r for r in selected if (r["cohort"], r["seed"]) in baseline]
        summary["changed_from_full"][mode] = {"n": len(comparable),
            "flips": sum(r["choice"] != baseline[(r["cohort"], r["seed"])]["choice"] for r in comparable),
            "corrected": sum(r["choice"] == r["winner"] and
                baseline[(r["cohort"], r["seed"])]["choice"] != r["winner"] for r in comparable),
            "regressed": sum(r["choice"] != r["winner"] and
                baseline[(r["cohort"], r["seed"])]["choice"] == r["winner"] for r in comparable)}
    summary["errors_full"] = [{k: r[k] for k in ("cohort", "seed", "dataset_id", "winner", "choice", "probabilities")}
                              for r in results if r["mode"] == "full" and r["choice"] != r["winner"]]
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"errors": len(errors), "results": {c: {m: {k: v for k, v in s.items()
        if k in ("n", "correct", "direct_cv_correct")} for m, s in ms.items()}
        for c, ms in summary["results"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--training", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--adapter-sha256", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.)
    p.add_argument("--pilot", action="store_true")
    asyncio.run(main(p.parse_args()))
