import json
from dataclasses import asdict
import httpx
import numpy as np
import pytest
from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.client import CandidatePair
from melchior.crucible.decisions import comparison, TEMPLATE
from melchior.crucible.environments import generate_task, TaskProfile
from melchior.crucible.metrics import score_predictions, paired_interval, CONTRACT_VERSION
from melchior.crucible.sandbox import SandboxResult
from melchior.crucible.storage import RunStore
from melchior.crucible.runner import CrucibleRunner
from scripts.evaluate_holdout import evaluate_benchmark


@pytest.mark.parametrize('seed', range(6))
def test_train_eval_tasks_do_not_share_data(seed):
    a, *_ = generate_task(seed, 'train')
    b, *_ = generate_task(seed, 'eval')
    assert a.dataset_id != b.dataset_id
    assert a.data_hash != b.data_hash


@pytest.mark.parametrize('metric,y,pred,expected', [
    ('f1', [0,0,0,1], [0,0,0,0], 0.),
    ('accuracy', [0,1,2,2], [0,1,2,1], .75),
    ('roc_auc', [0,1,0,1], [.1,.8,.2,.9], 1.),
    ('r2', [0,1,2], [0,1,2], 1.),
])
def test_metric_definitions(metric, y, pred, expected):
    assert score_predictions(metric, y, pred) == expected


def test_pairing_preserved_in_uncertainty():
    y = [0,1]*50
    pred = [0]*100
    assert paired_interval('accuracy', y, pred, pred) == [0., 0.]


def fixture():
    profile = TaskProfile('test', 'binary_classification', 'accuracy', 'maximize', 'Task description',
                          100, 100, 1, 'eval', 'dataset', 'hash')
    pair = CandidatePair('A rationale', 'A implementation', 'B rationale', 'B implementation')
    y = [0,1]*50
    a = SandboxResult('success', .5, 1., '', '', [0]*100)
    b = SandboxResult('success', 1., 1., '', '', y)
    return profile, pair, y, a, b


def test_empirical_choice_matches_serving_template_and_schema():
    profile, pair, y, a, b = fixture()
    outcome = CrucibleArbiter(.01).evaluate(profile, pair, a, b, y_true=y)
    assert outcome.winner == 'B'
    assert outcome.interval[0] > .01
    payload = comparison(profile, pair)
    q = payload['questions']['winner']
    for row, label in zip(outcome.nli_records, ['A','B']):
        assert set(row) == {'premise','hypothesis','label','source','image'}
        assert row['premise'] == payload['state']
        assert row['hypothesis'] == TEMPLATE.format(instr=q['instructions'], label=label, crit=q['criteria'][label])
        assert row['label'] == (label == 'B')
        assert 'A implementation' in row['premise'] and 'B implementation' in row['premise']
        assert '1.0' not in row['premise']


def test_failed_execution_and_uncertainty_are_not_quality_labels():
    profile, pair, y, a, b = fixture()
    arbiter = CrucibleArbiter()
    equal = arbiter.evaluate(profile, pair, a, a, y_true=y)
    assert equal.winner == 'UNCERTAIN' and equal.nli_records == []
    assert equal.dpo_record is None
    a.status = 'failed'; a.metric = None
    failed = arbiter.evaluate(profile, pair, a, b, y_true=y)
    assert failed.execution_winner == 'B'
    assert failed.delta is None and failed.nli_records == [] and failed.dpo_record is None


def test_resume_refuses_mixed_manifest_and_duplicate_seeds(tmp_path):
    store = RunStore(tmp_path, {'v': 1})
    store.save({'seed': 1, 'error': 'failed'})
    with pytest.raises(BlockingIOError):
        RunStore(tmp_path, {'v': 1}, resume=True)
    store.close()
    with pytest.raises(ValueError, match='mismatch'):
        RunStore(tmp_path, {'v': 2}, resume=True)
    resumed = RunStore(tmp_path, {'v': 1}, resume=True)
    assert resumed.completed() == {1}
    resumed.export()
    assert len((tmp_path / 'errors.jsonl').read_text().splitlines()) == 1
    resumed.close()


@pytest.mark.asyncio
async def test_time_budget_cancels_inflight_and_can_resume(tmp_path, monkeypatch):
    runner = CrucibleRunner(llm_mode='mock', output_dir=tmp_path, concurrency=1, max_hours=.0001)
    active = False
    async def block(profile, seed):
        nonlocal active
        active = True
        try:
            import asyncio
            await asyncio.Event().wait()
        finally:
            active = False
    monkeypatch.setattr(runner.client, 'generate_pair', block)
    summary = await runner.run()
    assert summary['status'] == 'time_limit' and summary['attempts'] == 0
    assert not active and runner.client._http_client.is_closed


def test_benchmark_real_candidates_swapped_and_api_errors(tmp_path):
    profile, pair, y, a, b = fixture()
    outcome = CrucibleArbiter().evaluate(profile, pair, a, b, y_true=y)
    r = {'schema': CONTRACT_VERSION, 'mode': 'api', 'seed': 1, 'profile': asdict(profile),
         'pair': asdict(pair), 'outcome': asdict(outcome)}
    path = tmp_path / 'holdout.jsonl'; path.write_text(json.dumps(r)+'\n')
    calls = []
    def handler(request):
        payload = json.loads(request.content); calls.append(payload)
        assert 'A implementation' in payload['state'] and 'B implementation' in payload['state']
        choice = 'B' if len(calls) == 1 else 'A'
        return httpx.Response(200, json={'answers': {'winner': {'choice': choice, 'confidence': .8}}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = evaluate_benchmark('http://test', path, client=client)
    assert summary['accuracy'] == summary['swap_consistency'] == 1.
    assert summary['always_a_accuracy'] == 0.
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as client:
        summary = evaluate_benchmark('http://test', path, client=client)
    assert summary['accuracy'] is None and summary['api_errors'] == 1
    path.write_text(json.dumps({'description':'legacy row'})+'\n')
    with pytest.raises(ValueError, match='legacy'):
        evaluate_benchmark('http://test', path)


def test_abrupt_process_death_preserves_committed_raw_record(tmp_path):
    import subprocess
    import sys
    script = """
from melchior.crucible.storage import RunStore
import os, sys
store = RunStore(sys.argv[1], {'version': 1})
store.save({'seed': 7, 'error': 'persisted'})
# No close/export/finally handlers, like an external process kill.
os._exit(9)
"""
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], timeout=5)
    assert result.returncode == 9
    store = RunStore(tmp_path, {'version': 1}, resume=True)
    try:
        assert store.completed() == {7}
        store.export()
        assert json.loads((tmp_path / 'errors.jsonl').read_text())['seed'] == 7
    finally:
        store.close()


@pytest.mark.asyncio
async def test_session_budget_also_bounds_model_discovery(tmp_path):
    import asyncio
    runner = CrucibleRunner(llm_mode='api', output_dir=tmp_path, max_hours=.00005)
    async def stalled(request):
        await asyncio.Event().wait()
    await runner.client._http_client.aclose()
    runner.client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(stalled))
    import time
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(runner.run(), 2)
    assert time.monotonic() - started < 1
    assert runner.client._http_client.is_closed and runner.http_client.is_closed
