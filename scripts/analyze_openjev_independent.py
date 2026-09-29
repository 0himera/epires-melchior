"""Summarize frozen independent NLI decisions with task-cluster uncertainty."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


MODELS = ("base", "s42_c00", "s42_c01", "s43_c00", "s43_c01")


def cluster_interval(differences, groups, seed=2141):
    by_group = defaultdict(list)
    for difference, group in zip(differences, groups):
        by_group[group].append(difference)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10000):
        sampled = rng.choice(keys, len(keys), replace=True)
        draw = [x for key in sampled for x in by_group[key]]
        means.append(float(np.mean(draw)))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def main(directory):
    corpus = directory / "holdout.jsonl"
    frozen = [json.loads(line) for line in corpus.read_text().splitlines()]
    scores = {name: json.loads((directory / "scores" / f"{name}.json").read_text()) for name in MODELS}
    digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
    for name, report in scores.items():
        assert report["data_sha256"]["/benchmark/holdout.jsonl"] == digest, name
        assert len(report["results"]) == len(frozen), name
        for row, output in zip(frozen, report["results"]):
            assert row["seed"] == output["seed"] and row["outcome"]["winner"] == output["winner"], name
    subsets = {"all": list(range(len(frozen))),
               "synthetic": [i for i, row in enumerate(frozen) if row["preprocessing"]["source"] == "crucible_synthetic"],
               "openml": [i for i, row in enumerate(frozen) if row["preprocessing"]["source"] == "OpenML"]}
    summary = {"scope": "Independent fixed sklearn portfolio; candidates differ from Qwen self-play distribution",
               "corpus_sha256": digest, "subsets": {}}
    for subset, indices in subsets.items():
        groups = [frozen[i]["profile"]["dataset_id"].rsplit(":", 1)[0]
                  if frozen[i]["preprocessing"]["source"] == "OpenML"
                  else frozen[i]["profile"]["dataset_id"] for i in indices]
        value = {"n": len(indices), "clusters": len(set(groups)),
                 "winner_counts": dict(Counter(frozen[i]["outcome"]["winner"] for i in indices)), "models": {}}
        for name in MODELS:
            rows = [scores[name]["results"][i] for i in indices]
            value["models"][name] = {"correct": sum(r["symmetric_correct"] for r in rows),
                "normal_correct": sum(r["normal_correct"] for r in rows),
                "swapped_correct": sum(r["swapped_correct"] for r in rows),
                "swap_consistent": sum(r["swap_consistent"] for r in rows),
                "mean_order_disagreement": float(np.mean([r["symmetric"]["order_disagreement"] for r in rows]))}
            if name != "base":
                base = [scores["base"]["results"][i]["symmetric_correct"] for i in indices]
                adapted = [r["symmetric_correct"] for r in rows]
                gains = sum(a and not b for a, b in zip(adapted, base))
                losses = sum(b and not a for a, b in zip(adapted, base))
                delta = [int(a)-int(b) for a, b in zip(adapted, base)]
                value["models"][name]["vs_base"] = {
                    "corrected": gains, "regressed": losses,
                    "accuracy_delta": float(np.mean(delta)),
                    "task_cluster_bootstrap_95pct": cluster_interval(delta, groups),
                    "mcnemar_exact_p_exploratory": float(binomtest(gains, gains+losses, .5).pvalue) if gains+losses else 1.0}
        summary["subsets"][subset] = value
    (directory / "scores" / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["subsets"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    main(p.parse_args().directory)
