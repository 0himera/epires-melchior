"""Real subprocesses, controlled HTTP and durable interruption/resume."""
import asyncio
import json
import sqlite3
import httpx
import numpy as np
import pytest
from melchior.crucible.canary import CanaryRunner
from melchior.crucible.environments import TaskProfile

PAIR = {f'candidate_{k}': {'hypothesis': f'Approach {k}', 'code': code} for k, code in [
    ('a', 'import numpy as np\ndef fit_predict(X_train,y_train,X_test): return np.zeros(len(X_test))'),
    ('b', 'def fit_predict(X_train,y_train,X_test): return X_test[:,0]')
]}


def tiny_task(seed, split='eval'):
    y = np.array([0, 1]*30)
    return (TaskProfile(f'task_{seed}', 'binary_classification', 'accuracy', 'maximize', f'Test seed={seed}',
                        60, 60, 1, split, f'synthetic:{seed}', f'hash{seed}'), y[:, None], y, y[:, None], y)


def reply(pair):
    return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(pair)}, 'finish_reason': 'stop'}]})


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x]


async def runner_with_http(tmp_path, monkeypatch, handler, num_tasks=3, **kwargs):
    monkeypatch.setattr('melchior.crucible.runner.generate_task', tiny_task)
    runner = CanaryRunner(num_tasks=num_tasks, concurrency=2, llm_model='test-model',
                          output_dir=tmp_path, sandbox_timeout_s=5, **kwargs)
    await runner.client._http_client.aclose()
    await runner.http_client.aclose()
    runner.client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    def judge(request):
        payload = json.loads(request.content)
        # The evaluator must provide the actual candidates, without measured outcomes.
        assert 'METRIC:' not in payload['state']
        assert 'independent test' in payload['questions']['winner']['instructions']
        a = payload['state'].split('Candidate B rationale:')[0]
        choice = 'A' if 'return X_test[:,0]' in a else 'B'
        return httpx.Response(200, json={'answers': {'winner': {'choice': choice, 'confidence': .8}}})
    runner.http_client = httpx.AsyncClient(transport=httpx.MockTransport(judge))
    return runner


@pytest.mark.asyncio
async def test_missing_candidate_isolated_and_actual_quality_scored(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        prompt = json.loads(request.content)['messages'][1]['content']; calls.append(prompt)
        return reply({'candidate_a': PAIR['candidate_a']} if 'seed=1001' in prompt else PAIR)
    runner = await runner_with_http(tmp_path, monkeypatch, handler)
    summary = await asyncio.wait_for(runner.run(), 15)
    assert summary['attempts'] == 3
    assert summary['generation_or_pipeline_errors'] == 1
    assert summary['evaluated'] == summary['decisive_quality_pairs'] == 2
    assert summary['jev_correct'] == summary['jev_swap_consistent'] == 2
    assert len(calls) == 4
    assert len(rows(tmp_path / 'holdout.jsonl')) == 2
    assert len(rows(tmp_path / 'errors.jsonl')) == 1
    assert rows(tmp_path / 'openjev_ml_nli.jsonl') == []  # Never train on eval data.
    assert runner.client._http_client.is_closed and runner.http_client.is_closed


@pytest.mark.asyncio
async def test_committed_results_survive_cancel_and_resume_without_duplicates(tmp_path, monkeypatch):
    blocked = asyncio.Event()
    async def handler(request):
        if 'seed=1001' in json.loads(request.content)['messages'][1]['content']:
            blocked.set(); await asyncio.Event().wait()
        return reply(PAIR)
    runner = await runner_with_http(tmp_path, monkeypatch, handler, num_tasks=2)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(blocked.wait(), 2)
        async def committed():
            while True:
                with sqlite3.connect(tmp_path / 'records.sqlite3') as db:
                    if db.execute('SELECT count(*) FROM records').fetchone()[0]:
                        return
                await asyncio.sleep(.02)
        await asyncio.wait_for(committed(), 8)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
    finally:
        task.cancel(); await asyncio.gather(task, return_exceptions=True)
    assert [r['seed'] for r in rows(tmp_path / 'evaluations.jsonl')] == [1000]
    assert json.loads((tmp_path / 'summary.json').read_text())['status'] == 'cancelled'
    calls = []
    def resumed(request):
        calls.append(request); return reply(PAIR)
    runner = await runner_with_http(tmp_path, monkeypatch, resumed, num_tasks=2, resume=True)
    summary = await runner.run()
    assert summary['attempts'] == 2 and len(calls) == 1
    assert [r['seed'] for r in rows(tmp_path / 'evaluations.jsonl')] == [1000, 1001]


@pytest.mark.asyncio
@pytest.mark.parametrize('num_tasks', [0, 3])
async def test_no_successes_produces_valid_summary(tmp_path, monkeypatch, num_tasks):
    runner = await runner_with_http(tmp_path, monkeypatch, lambda request: httpx.Response(503), num_tasks)
    summary = await asyncio.wait_for(runner.run(), 5)
    assert summary['attempts'] == summary['generation_or_pipeline_errors'] == num_tasks
    assert summary['total_candidates'] == summary['jev_scored'] == 0
    assert rows(tmp_path / 'holdout.jsonl') == []


@pytest.mark.asyncio
async def test_real_backend_training_export_contains_only_supported_preferences(tmp_path, monkeypatch):
    from melchior.crucible.runner import CrucibleRunner
    monkeypatch.setattr('melchior.crucible.runner.generate_task', tiny_task)
    runner = CrucibleRunner(concurrency=1, max_pairs=1, output_dir=tmp_path, llm_mode='api',
                            llm_model='test-model', split='train', sandbox_timeout_s=5)
    await runner.client._http_client.aclose()
    runner.client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: reply(PAIR)))
    summary = await runner.run()
    assert summary['decisive_quality_pairs'] == 1
    nli = rows(tmp_path / 'openjev_ml_nli.jsonl')
    assert len(nli) == 2 and {r['label'] for r in nli} == {0, 1}
    assert all(set(r) == {'premise', 'hypothesis', 'label', 'source', 'image'} for r in nli)
    dpo = rows(tmp_path / 'melchior_dpo_pairs.jsonl')
    assert len(dpo) == 1 and 'return X_test[:,0]' in dpo[0]['chosen']
    assert rows(tmp_path / 'holdout.jsonl') == []
