"""Resumable paired experiments with bounded concurrency and trusted scoring."""
from __future__ import annotations
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
import httpx

from melchior.crucible.environments import generate_task
from melchior.crucible.client import CrucibleLLMClient, CandidatePair
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.decisions import comparison
from melchior.crucible.metrics import CONTRACT_VERSION
from melchior.crucible.storage import RunStore, fingerprint, atomic_json


def parse_answer(data):
    ans = data['answers']['winner']
    choice, confidence = ans['choice'], float(ans['confidence'])
    if choice not in {'A', 'B'} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid OpenJev choice or confidence')
    return {'choice': choice, 'confidence': confidence}


class CrucibleRunner:
    def __init__(self, concurrency=20, output_dir='data/crucible', llm_base_url=None,
                 llm_model='qwen', llm_mode='auto', sandbox_timeout_s=12., max_pairs=None,
                 *, split='train', seed_start=1000, resume=False, max_hours=None,
                 min_delta=0.005, jev_url=None, reasoning_effort='xhigh',
                 generation_timeout_s=600., generation_max_tokens=12288):
        if concurrency < 1 or (max_pairs is not None and max_pairs < 0) or seed_start < 0:
            raise ValueError('Invalid concurrency, pair budget or starting seed')
        if split not in {'train', 'eval'} or (max_hours is not None and max_hours <= 0):
            raise ValueError('Invalid split or time budget')
        self.concurrency, self.max_pairs = concurrency, max_pairs
        self.output_dir = Path(output_dir)
        self.split, self.seed_start, self.resume, self.max_hours = split, seed_start, resume, max_hours
        self.llm_mode, self.jev_url = llm_mode, jev_url
        self.sandbox = AsyncSandbox(timeout_s=sandbox_timeout_s)
        self.arbiter = CrucibleArbiter(min_delta)
        self.client = CrucibleLLMClient(base_url=llm_base_url, model=llm_model, mode=llm_mode,
                                        timeout_s=generation_timeout_s, reasoning_effort=reasoning_effort,
                                        max_tokens=generation_max_tokens)
        self.http_client = httpx.AsyncClient(timeout=20.)
        self.manifest = {'schema': CONTRACT_VERSION, 'split': split, 'seed_start': seed_start,
                         'llm_url': self.client.base_url, 'llm_model': llm_model, 'mode': llm_mode,
                         'sandbox_timeout_s': sandbox_timeout_s, 'min_delta': min_delta,
                         'jev_url': jev_url, 'reasoning_effort': reasoning_effort,
                         'enable_thinking': True, 'generation_timeout_s': generation_timeout_s,
                         'generation_max_tokens': generation_max_tokens, **fingerprint()}
        self._total_evaluated = 0

    async def _query_openjev(self, profile, pair, *, swapped=False):
        response = await self.http_client.post(self.jev_url, json=comparison(profile, pair, swapped=swapped))
        response.raise_for_status()
        return parse_answer(response.json())

    async def _process_task(self, seed):
        record = {'schema': CONTRACT_VERSION, 'seed': seed, 'mode': self.llm_mode}
        started = time.monotonic()
        try:
            profile, X_train, y_train, X_test, y_test = generate_task(seed, split=self.split)
            record['profile'] = asdict(profile)
            pair = await self.client.generate_pair(profile, seed)
            # Avoid a fixed association of algorithm family with displayed A/B.
            swapped = bool(hashlib.sha256(f'{self.split}:{seed}:order'.encode()).digest()[0] & 1)
            if swapped:
                pair = CandidatePair(pair.hypothesis_b, pair.code_b, pair.hypothesis_a, pair.code_a, pair.operator, generation=pair.generation)
            record.update(pair=asdict(pair), generation_swapped=swapped, resolved_model=self.client.model)
            # TaskGroup cancels and drains the sibling if an execution raises unexpectedly.
            async with asyncio.TaskGroup() as group:
                a = group.create_task(self.sandbox.execute(pair.code_a, X_train, y_train, X_test, y_test, metric=profile.metric))
                b = group.create_task(self.sandbox.execute(pair.code_b, X_train, y_train, X_test, y_test, metric=profile.metric))
            ra, rb = a.result(), b.result()
            record.update(res_a=asdict(ra), res_b=asdict(rb), test_labels=y_test.tolist())
            outcome = await asyncio.to_thread(self.arbiter.evaluate, profile, pair, ra, rb, y_true=y_test, seed=seed)
            record['outcome'] = asdict(outcome)
            if self.jev_url and outcome.winner in {'A', 'B'}:
                try:
                    normal = await self._query_openjev(profile, pair)
                    reverse = await self._query_openjev(profile, pair, swapped=True)
                    record['jev'] = {'normal': normal, 'swapped': reverse,
                                     'correct': normal['choice'] == outcome.winner,
                                     'swap_consistent': normal['choice'] != reverse['choice']}
                except Exception as exc:
                    record['jev_error'] = f'{type(exc).__name__}: {exc}'
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            if hasattr(exc, 'attempts'):
                record['generation_attempts'] = exc.attempts
        record['wall_time_s'] = time.monotonic() - started
        return record

    def _summary(self, store, status):
        records = store.records()
        evaluated = [r for r in records if 'outcome' in r]
        decisive = [r for r in evaluated if r['outcome']['winner'] in {'A', 'B'}]
        jev = [r['jev'] for r in decisive if 'jev' in r]
        success = sum(r[k]['status'] == 'success' for r in evaluated for k in ('res_a', 'res_b'))
        summary = {'status': status, 'split': self.split, 'mock': self.llm_mode == 'mock',
                   'attempts': len(records), 'evaluated': len(evaluated),
                   'generation_or_pipeline_errors': len(records)-len(evaluated),
                   'successful_candidates': success, 'total_candidates': 2*len(evaluated),
                   'decisive_quality_pairs': len(decisive),
                   'outcomes': dict(Counter(r['outcome']['winner'] for r in evaluated)),
                   'operators': dict(Counter(r['pair']['operator'] for r in evaluated)),
                   'jev_scored': len(jev), 'jev_errors': sum('jev_error' in r for r in decisive),
                   'jev_correct': sum(j['correct'] for j in jev),
                   'jev_swap_consistent': sum(j['swap_consistent'] for j in jev),
                   'session_elapsed_s': time.monotonic()-self._start_time}
        atomic_json(self.output_dir / 'summary.json', summary)
        return summary

    async def run(self):
        self._start_time = time.monotonic()
        store, workers = None, []
        status = 'failed'
        try:
            if self.llm_mode != 'mock':
                self.manifest['resolved_model'] = await self.client._resolve_model()
            store = RunStore(self.output_dir, self.manifest, resume=self.resume)
            done = store.completed()
            store.export()
            next_seed = self.seed_start
            deadline = self._start_time + self.max_hours*3600 if self.max_hours is not None else math.inf
            async def worker():
                nonlocal next_seed
                while time.monotonic() < deadline:
                    while next_seed in done:
                        next_seed += 1
                    if self.max_pairs is not None and next_seed >= self.seed_start+self.max_pairs:
                        return
                    seed = next_seed
                    next_seed += 1
                    record = await self._process_task(seed)
                    store.save(record)
                    done.add(seed)
                    self._total_evaluated += 'outcome' in record
                    if len(done) % 10 == 0 or len(done) == 1:
                        summary = self._summary(store, 'running')
                        store.export()
                        print(f"[{len(done)} attempts] valid candidates {summary['successful_candidates']}/{summary['total_candidates']}; decisive {summary['decisive_quality_pairs']}", flush=True)
            workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]
            try:
                if self.max_hours is None:
                    await asyncio.gather(*workers)
                else:
                    async with asyncio.timeout(max(0., deadline-time.monotonic())):
                        await asyncio.gather(*workers)
                status = 'complete'
            except TimeoutError:
                status = 'time_limit'
        except asyncio.CancelledError:
            status = 'cancelled'
            raise
        finally:
            for worker in workers:
                if not worker.done():
                    worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.client.close()
            await self.http_client.aclose()
            if store:
                try:
                    store.export()
                    summary = self._summary(store, status)
                    print(summary, flush=True)
                finally:
                    store.close()
        return summary
