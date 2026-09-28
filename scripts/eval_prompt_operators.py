#!/usr/bin/env python3
"""Evaluates and compares the 5 Crucible research prompt operators.

Tests:
1. inductive_bias
2. data_centric
3. hyperparameter_frontier
4. ensemble_diversity
5. pathology_defense

Measures execution pass rate and empirical metric differences on matched tasks.
Code tags are heuristic descriptions, not measures of reasoning quality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import asdict

from melchior.crucible.environments import generate_task
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.client import CrucibleLLMClient
from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.prompts import OPERATOR_KEYS


def analyze_code_patterns(operator: str, code_a: str, code_b: str) -> list[str]:
    """Detects specific algorithmic patterns promoted by each operator."""
    tags = []
    combined = f"{code_a}\n{code_b}".lower()

    if "for " in combined and (" in " in combined):
        tags.append("cv_or_grid_loop")
    if "gridsearchcv" in combined or "randomizedsearchcv" in combined:
        tags.append("sklearn_search")
    if "quantiletransformer" in combined or "powertransformer" in combined:
        tags.append("quantile_power_transform")
    if "polynomialfeatures" in combined or "interactions" in combined:
        tags.append("feature_interactions")
    if "votingclassifier" in combined or "votingregressor" in combined:
        tags.append("voting_ensemble")
    if "stackingclassifier" in combined or "stackingregressor" in combined:
        tags.append("stacking_ensemble")
    if "class_weight" in combined or "balanced" in combined:
        tags.append("imbalance_defense")
    if "robustscaler" in combined or "huber" in combined:
        tags.append("outlier_robust")
    if "histgradientboosting" in combined or "randomforest" in combined:
        tags.append("tree_boosting")
    if "ridge" in combined or "logisticregression" in combined or "elasticnet" in combined:
        tags.append("regularized_linear")

    return tags


async def evaluate_operator(
    operator: str,
    seeds: list[int],
    client: CrucibleLLMClient,
    sandbox: AsyncSandbox,
    arbiter: CrucibleArbiter,
) -> dict:
    print(f"\n--- Testing Operator: [{operator.upper()}] ({len(seeds)} task(s)) ---")
    results = []
    total_candidates = 0
    successful_candidates = 0
    deltas = []

    for seed in seeds:
        profile, X_tr, y_tr, X_va, y_va = generate_task(seed, split="eval")
        print(f"  [Task #{seed}] {profile.description} ({profile.task_type}, metric: {profile.metric})")

        t0 = time.time()
        try:
            pair = await client.generate_pair(profile, seed=seed, operator=operator)
            gen_time = time.time() - t0
            print(f"    -> Generated in {gen_time:.1f}s")
        except Exception as e:
            print(f"    [!] Generation failed: {e}")
            results.append({"seed": seed, "error": f"{type(e).__name__}: {e}", "patterns": []})
            continue

        total_candidates += 2

        # Dual sandbox execution
        t_exec = time.time()
        res_a, res_b = await asyncio.gather(
            sandbox.execute(pair.code_a, X_tr, y_tr, X_va, y_va, metric=profile.metric),
            sandbox.execute(pair.code_b, X_tr, y_tr, X_va, y_va, metric=profile.metric),
        )
        exec_time = time.time() - t_exec

        succ_a = (res_a.status == "success" and res_a.metric is not None)
        succ_b = (res_b.status == "success" and res_b.metric is not None)
        if succ_a:
            successful_candidates += 1
        if succ_b:
            successful_candidates += 1

        status_a = f"{res_a.metric:.4f}" if succ_a else f"FAIL ({res_a.status})"
        status_b = f"{res_b.metric:.4f}" if succ_b else f"FAIL ({res_b.status})"
        print(f"    -> Cand A: {status_a} | Cand B: {status_b} (Sandbox: {exec_time:.2f}s)")

        outcome = await asyncio.to_thread(arbiter.evaluate, profile, pair, res_a, res_b, y_true=y_va, seed=seed)
        if outcome.delta is not None:
            deltas.append(outcome.delta)
        patterns = analyze_code_patterns(operator, pair.code_a, pair.code_b)

        print(f"    -> Arbiter: Winner={outcome.winner} (Δ={outcome.delta}) | Patterns: {', '.join(patterns) if patterns else 'standard'}")

        results.append({
            "seed": seed,
            "profile": asdict(profile),
            "hypothesis_a": pair.hypothesis_a,
            "hypothesis_b": pair.hypothesis_b,
            "code_a": pair.code_a,
            "code_b": pair.code_b,
            "res_a": asdict(res_a),
            "res_b": asdict(res_b),
            "winner": outcome.winner,
            "interval": outcome.interval,
            "delta": outcome.delta,
            "patterns": patterns,
        })

    pass_rate = (successful_candidates / total_candidates * 100.0) if total_candidates > 0 else 0.0
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0

    return {
        "operator": operator,
        "total_tasks": len(seeds),
        "generation_failures": sum("error" in r for r in results),
        "total_candidates": total_candidates,
        "successful_candidates": successful_candidates,
        "pass_rate_pct": pass_rate,
        "mean_delta": mean_delta,
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate 5 Crucible Prompt Operators")
    parser.add_argument("--mode", default="opencode", choices=["opencode", "api", "mock", "auto"])
    parser.add_argument("--model", default="openai/gpt-6-luna-fast", help="LLM model name")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM URL if mode=api")
    parser.add_argument("--tasks-per-op", type=int, default=1, help="Number of tasks per operator (default 1 for fast eval)")
    parser.add_argument("--output", default="data/prompt_eval/prompt_eval_results.json")
    args = parser.parse_args()

    print("=" * 72)
    print("        MELCHIOR CRUCIBLE: 5-OPERATOR PROMPT EVALUATION")
    print("=" * 72)
    print(f"Mode:              {args.mode}")
    print(f"Model:             {args.model}")
    print(f"Tasks per operator: {args.tasks_per_op} (total tasks: {args.tasks_per_op * len(OPERATOR_KEYS)})")
    print("=" * 72)

    client = CrucibleLLMClient(
        base_url=args.url,
        model=args.model,
        mode=args.mode,
        timeout_s=60.0,
    )
    sandbox = AsyncSandbox(timeout_s=15.0)
    arbiter = CrucibleArbiter(min_delta=0.005)

    operator_summaries = []
    base_seed = 42

    try:
        for idx, op in enumerate(OPERATOR_KEYS):
            seeds = [base_seed + i for i in range(args.tasks_per_op)]
            summary = await evaluate_operator(op, seeds, client, sandbox, arbiter)
            operator_summaries.append(summary)
    finally:
        await client.close()

    # Aggregate metrics
    all_total_cand = sum(s["total_candidates"] for s in operator_summaries)
    all_succ_cand = sum(s["successful_candidates"] for s in operator_summaries)
    overall_pass_rate = (all_succ_cand / all_total_cand * 100.0) if all_total_cand > 0 else 0.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "mode": args.mode,
            "timestamp": time.time(),
            "overall_pass_rate_pct": overall_pass_rate,
            "total_candidates": all_total_cand,
            "successful_candidates": all_succ_cand,
            "operators": operator_summaries,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("                     EVALUATION MATRIX SUMMARY")
    print("=" * 72)
    print(f"{'Operator':<26} | {'Pass Rate':<10} | {'Mean Δ':<8} | Key Patterns Observed")
    print("-" * 72)
    for s in operator_summaries:
        all_pat = set()
        for r in s["results"]:
            all_pat.update(r["patterns"])
        pat_str = ", ".join(list(all_pat)[:3]) if all_pat else "none"
        print(f"{s['operator']:<26} | {s['pass_rate_pct']:5.1f}%     | {s['mean_delta']:6.4f} | {pat_str}")
    print("-" * 72)
    print(f"{'OVERALL (5 Operators)':<26} | {overall_pass_rate:5.1f}%     | Candidates: {all_succ_cand}/{all_total_cand} executed successfully")
    print(f"Results saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    asyncio.run(main())
