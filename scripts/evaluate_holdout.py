#!/usr/bin/env python3
"""Evaluate actual candidate pairs with order swaps and explicit API coverage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import httpx
import numpy as np
from melchior.crucible.decisions import comparison, symmetric_choice, answer_probability_a
from melchior.crucible.metrics import CONTRACT_VERSION
from melchior.crucible.runner import parse_answer


def evaluate_benchmark(endpoint='http://localhost:8080/v1/systemone',
                       benchmark_path='data/crucible_canary/holdout.jsonl', *, client=None):
    records = [json.loads(x) for x in Path(benchmark_path).read_text().splitlines() if x.strip()]
    seen = set()
    for r in records:
        if r.get('schema') != CONTRACT_VERSION or r.get('mode') == 'mock':
            raise ValueError('Require v2 empirical records with real candidates; legacy/mock benchmark unsupported')
        if r['profile']['split'] != 'eval' or r['outcome']['winner'] not in {'A', 'B'}:
            raise ValueError('Only decisive quality pairs from eval split are eligible')
        key = (r['profile']['data_hash'], r['pair']['operator'])
        if key in seen:
            raise ValueError('Duplicate task/operator in benchmark')
        seen.add(key)
        if any(not r['pair'].get(k) for k in ('code_a', 'code_b', 'hypothesis_a', 'hypothesis_b')):
            raise ValueError('Missing real candidate contents')
    own_client = client is None
    client = client or httpx.Client(timeout=30.)
    rows = []
    try:
        for r in records:
            row = {'seed': r['seed'], 'winner': r['outcome']['winner'], 'group': r['profile']['dataset_id']}
            try:
                answers = []
                for swapped in (False, True):
                    response = client.post(endpoint, json=comparison(r['profile'], r['pair'], swapped=swapped,
                                           generation_swapped=r.get('generation_swapped', False)))
                    response.raise_for_status()
                    answers.append(parse_answer(response.json()))
                a, b = answers
                symmetric = symmetric_choice(answer_probability_a(a), answer_probability_a(b))
                row.update(correct=a['choice'] == row['winner'],
                           swapped_correct=b['choice'] != row['winner'],
                           swap_consistent=a['choice'] != b['choice'], answers=answers,
                           symmetric=symmetric, symmetric_correct=symmetric['choice'] == row['winner'])
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
    finally:
        if own_client:
            client.close()
    valid = [r for r in rows if 'correct' in r]
    n = len(valid)
    groups = defaultdict(list)
    for r in valid:
        groups[r['group']].append(int(r['correct']))
    interval = None
    # Repeated splits of the same built-in corpus are a single cluster.
    if len(groups) > 1:
        rng = np.random.default_rng(42)
        clusters = list(groups.values())
        values = [np.mean([x for i in rng.integers(len(clusters), size=len(clusters)) for x in clusters[i]])
                  for _ in range(2000)]
        interval = np.quantile(values, [.025, .975]).tolist()
    summary = {'requested': len(rows), 'scored_both_orders': n, 'api_errors': len(rows)-n,
               'accuracy': sum(r['correct'] for r in valid)/n if n else None,
               'accuracy_cluster_bootstrap_95': interval, 'independent_dataset_clusters': len(groups),
               'swapped_accuracy': sum(r['swapped_correct'] for r in valid)/n if n else None,
               'swap_consistency': sum(r['swap_consistent'] for r in valid)/n if n else None,
               'symmetric_accuracy': sum(r['symmetric_correct'] for r in valid)/n if n else None,
               'symmetric_coverage': sum(r['symmetric']['choice'] != 'UNCERTAIN' for r in valid)/n if n else None,
               'always_a_accuracy': sum(r['winner'] == 'A' for r in valid)/n if n else None,
               'always_b_accuracy': sum(r['winner'] == 'B' for r in valid)/n if n else None,
               'uniform_random_expected_accuracy': .5 if n else None, 'results': rows}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', '-e', default='http://localhost:8080/v1/systemone')
    parser.add_argument('--benchmark', '-b', default='data/crucible_canary/holdout.jsonl')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    result = evaluate_benchmark(args.endpoint, args.benchmark)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
