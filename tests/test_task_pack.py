import asyncio
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from melchior.crucible.canary import CanaryRunner
from melchior.crucible.environments import TaskProfile
from melchior.crucible.task_pack import TaskPack
from scripts.build_openjev_independent_test import real_task


def pack(directory, seeds=(90000, 90100)):
    directory.mkdir()
    tasks = []
    y = np.array([0, 1] * 30)
    for seed in seeds:
        file = directory / f'{seed}.npz'
        np.savez_compressed(file, X_train=y[:, None], y_train=y, X_test=y[:, None], y_test=y)
        profile = TaskProfile(f'task_{seed}', 'binary_classification', 'accuracy', 'maximize',
                              'frozen test', 60, 60, 1, 'eval', f'frozen:{seed}', '')
        tasks.append({'seed': seed, 'profile': asdict(profile), 'arrays': file.name,
                      'arrays_sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                      'preprocessing': {'source': 'test'}})
    (directory / 'manifest.json').write_text(json.dumps({'schema': 'frozen-task-pack-v1', 'tasks': tasks}))
    return directory


def test_corrupt_arrays_are_rejected_before_generation(tmp_path):
    root = pack(tmp_path / 'pack')
    (root / '90000.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        TaskPack(root, split='eval')


def test_eval_pack_cannot_be_used_for_training(tmp_path):
    root = pack(tmp_path / 'pack')
    with pytest.raises(ValueError, match='split mismatch'):
        TaskPack(root, split='train')


def test_continuous_features_are_retained_and_id_is_removed(monkeypatch):
    frame = pd.DataFrame({'id': range(40), 'measurement': np.linspace(0, 1, 40),
                          'category': ['a', 'b'] * 20})
    monkeypatch.setattr('scripts.build_openjev_independent_test.fetch_openml',
                        lambda **kwargs: SimpleNamespace(data=frame, target=pd.Series(['no', 'yes'] * 20)))
    profile, X_train, y_train, X_test, y_test, meta = real_task(
        (1, 'fake', 'binary_classification', 'roc_auc', None), 2718)
    assert meta['dropped_columns'] == ['id']
    assert meta['numeric_columns'] == ['measurement']
    assert X_train.shape == (30, 3) and X_test.shape == (10, 3)
    assert np.allclose(sorted(np.r_[X_train[:, 0], X_test[:, 0]]), frame.measurement)


@pytest.mark.asyncio
async def test_nonconsecutive_frozen_tasks_save_inputs_and_pinned_judge_identity(tmp_path):
    root = pack(tmp_path / 'pack')
    identity = {'base_model': 'base', 'adapter': 's42_c00/epoch_2', 'adapter_sha256': 'expected'}
    runner = CanaryRunner(num_tasks=2, concurrency=2, output_dir=tmp_path / 'run',
                          llm_model='test-model', sandbox_timeout_s=5,
                          task_pack=root, jev_adapter_sha256='expected')
    await runner.client._http_client.aclose()
    await runner.http_client.aclose()
    pair = {f'candidate_{k}': {'hypothesis': f'Approach {k}', 'code': code} for k, code in [
        ('a', 'import numpy as np\ndef fit_predict(X_train,y_train,X_test): return np.zeros(len(X_test))'),
        ('b', 'def fit_predict(X_train,y_train,X_test): return X_test[:,0]')]}
    calls = []
    def generate(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(pair)}, 'finish_reason': 'stop'}]})
    def judge(request):
        if request.method == 'GET':
            return httpx.Response(200, json={'model_identity': identity})
        state = json.loads(request.content)['state']
        a = state.split('Candidate B rationale:')[0]
        choice = 'A' if 'return X_test[:,0]' in a else 'B'
        return httpx.Response(200, json={'answers': {'winner': {'choice': choice, 'confidence': .8}},
                                       'model_identity': identity})
    runner.client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(generate))
    runner.http_client = httpx.AsyncClient(transport=httpx.MockTransport(judge))
    summary = await asyncio.wait_for(runner.run(), 15)
    records = [json.loads(line) for line in (tmp_path / 'run/evaluations.jsonl').read_text().splitlines()]
    assert [r['seed'] for r in records] == [90000, 90100]
    assert summary['attempts'] == summary['jev_symmetric_correct'] == 2 and len(calls) == 2
    assert all(r['task_input']['arrays_sha256'] for r in records)
    assert all(r['jev']['normal']['model_identity'] == identity for r in records)
    assert (tmp_path / 'run/openjev_ml_nli.jsonl').read_text() == ''


@pytest.mark.asyncio
async def test_wrong_serving_adapter_aborts_before_any_generation(tmp_path):
    runner = CanaryRunner(num_tasks=1, output_dir=tmp_path / 'run', llm_model='test-model',
                          jev_adapter_sha256='expected')
    await runner.http_client.aclose()
    runner.http_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={'model_identity': {'adapter_sha256': 'wrong'}})))
    with pytest.raises(ValueError, match='startup adapter identity mismatch'):
        await runner.run()
    assert not (tmp_path / 'run/manifest.json').exists()
    assert runner.client._http_client.is_closed and runner.http_client.is_closed
