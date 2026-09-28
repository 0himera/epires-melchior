#!/usr/bin/env python3
"""Evaluates any OpenJev / System 1 model against the frozen Holdout Validation Benchmark."""

import argparse
import json
import sys
from pathlib import Path
import httpx


def evaluate_benchmark(
    endpoint: str = "http://localhost:8080/v1/systemone",
    benchmark_path: str = "data/crucible_canary/holdout_validation_benchmark.jsonl",
) -> None:
    bench_file = Path(benchmark_path)
    if not bench_file.exists():
        print(f"[!] Benchmark file not found: {bench_file.resolve()}")
        sys.exit(1)

    with open(bench_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print("=" * 70)
    print("      EVALUATING MODEL AGAINST FROZEN HOLDOUT BENCHMARK")
    print("=" * 70)
    print(f"Target Endpoint:  {endpoint}")
    print(f"Holdout Records:  {len(records)} decisive tasks")
    print("Baseline prior:   55.0% (11/20 correct on default OpenJev v5)\n")

    correct_count = 0
    evaluated = 0

    client = httpx.Client(timeout=15.0)

    for i, rec in enumerate(records, 1):
        task_id = rec["task_id"]
        premise = f"Task: {rec['description']} Objective: Maximize {rec['metric']}."
        gt_winner = rec["ground_truth_winner"]
        delta = rec["empirical_delta"]

        payload = {
            "state": premise,
            "questions": {
                "winner": {
                    "type": "choice",
                    "instructions": f"Which candidate model will achieve higher {rec['metric']} on held-out validation data?",
                    "criteria": {
                        "A": f"Candidate approach A exploring alternative architectural or preprocessing bias for {rec['metric']}.",
                        "B": f"Candidate approach B with contrasting modeling hypothesis targeting {rec['metric']}.",
                    },
                }
            },
        }

        try:
            resp = client.post(endpoint, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get("winner", {})
                choice = str(ans.get("choice", "")).strip().upper()
                conf = float(ans.get("confidence", 0.5))
                is_correct = (choice == gt_winner)
                if is_correct:
                    correct_count += 1
                evaluated += 1

                res_icon = "✓" if is_correct else "✗"
                print(f"[{i:02d}/{len(records):02d}] {res_icon} {task_id} | GT Winner: {gt_winner} (Δ={delta:.3f}) | Model: {choice} (conf={conf:.2f})")
            else:
                print(f"[{i:02d}/{len(records):02d}] ! {task_id} | HTTP error {resp.status_code}")
        except Exception as e:
            print(f"[{i:02d}/{len(records):02d}] ! {task_id} | Request failed: {e}")

    acc = (correct_count / evaluated * 100.0) if evaluated > 0 else 0.0
    baseline_acc = 55.0
    lift = acc - baseline_acc

    print("\n" + "=" * 70)
    print("                    BENCHMARK VERDICT")
    print("=" * 70)
    print(f"Evaluated pairs:       {evaluated} / {len(records)}")
    print(f"Correct predictions:   {correct_count} / {evaluated}")
    print(f"New Model Accuracy:    {acc:.1f}%")
    print(f"Baseline (Prior):      {baseline_acc:.1f}%")
    sign = "+" if lift >= 0 else ""
    print(f"Empirical Gain / Lift: {sign}{lift:.1f}%")

    if acc >= 75.0:
        print("[✓] HYPOTHESIS CONFIRMED: Model demonstrates strong execution-grounded reasoning ability!")
    else:
        print("[•] Model shows marginal or neutral alignment with empirical execution.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenJev against frozen holdout benchmark")
    parser.add_argument("--endpoint", "-e", default="http://localhost:8080/v1/systemone", help="OpenJev API endpoint")
    parser.add_argument("--benchmark", "-b", default="data/crucible_canary/holdout_validation_benchmark.jsonl", help="Holdout JSONL path")
    args = parser.parse_args()
    evaluate_benchmark(args.endpoint, args.benchmark)
