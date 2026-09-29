"""Execute a frozen real-data portfolio and save scores from trained serving."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import httpx

from melchior.crucible.storage import atomic_json
from scripts.build_openjev_independent_test import main as build_portfolio


def main(args):
    asyncio.run(build_portfolio(argparse.Namespace(output=args.output, workers=args.workers,
                                                  task_pack=args.task_pack)))
    corpus = args.output / 'holdout.jsonl'
    frozen = [json.loads(line) for line in corpus.read_text().splitlines()]
    scores_dir = args.output / 'scores'
    scores_dir.mkdir()
    results, errors = [], []
    with httpx.Client(timeout=120.) as client, (scores_dir / 'live_responses.jsonl').open('w') as log:
        for row in frozen:
            try:
                response = client.post(args.jev_url, json={
                    'profile': row['profile'], 'pair': row['pair'],
                    'generation_swapped': row.get('generation_swapped', False)})
                response.raise_for_status()
                raw = response.json()
                if raw['model_identity']['adapter_sha256'] != args.adapter_sha256:
                    raise ValueError('Serving adapter identity mismatch')
                p, q = raw['normal_p_a'], raw['swapped_p_a']
                symmetric = {k: raw[k] for k in ('choice', 'probabilities', 'order_disagreement')}
                winner = row['outcome']['winner']
                record = {'cohort': 'benchmark', 'seed': row['seed'],
                    'dataset_id': row['profile']['dataset_id'], 'metric': row['profile']['metric'],
                    'operator': row['pair']['operator'], 'winner': winner,
                    'normal_p_a': p, 'swapped_p_a': q, 'symmetric': symmetric,
                    'normal_correct': ('A' if p >= .5 else 'B') == winner,
                    'swapped_correct': ('A' if q >= .5 else 'B') != winner,
                    'swap_consistent': (p >= .5) != (q >= .5),
                    'symmetric_correct': symmetric['choice'] == winner, 'raw_response': raw}
                results.append(record)
            except Exception as exc:
                record = {'seed': row['seed'], 'error': f'{type(exc).__name__}: {exc}'}
                errors.append(record)
            log.write(json.dumps(record, allow_nan=False) + '\n')
            log.flush()
            os.fsync(log.fileno())
            if (len(results) + len(errors)) % 10 == 0:
                print(f'Live scores: {len(results)}; errors: {len(errors)}', flush=True)
    summary = {'n': len(results), 'symmetric_correct': sum(r['symmetric_correct'] for r in results),
               'swap_consistent': sum(r['swap_consistent'] for r in results),
               'errors': len(errors), 'winner_counts': dict(Counter(r['winner'] for r in results))}
    report = {'scope': 'Fresh real-data fixed portfolio; selected trained model served online',
              'data_sha256': {'/benchmark/holdout.jsonl': hashlib.sha256(corpus.read_bytes()).hexdigest()},
              'adapter_sha256': args.adapter_sha256, 'aggregate': {'benchmark': summary},
              'results': results, 'errors': errors}
    atomic_json(scores_dir / 's42_c00_live.json', report)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-pack', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jev-url', default='http://127.0.0.1:8080/v1/compare')
    parser.add_argument('--adapter-sha256', required=True)
    main(parser.parse_args())
