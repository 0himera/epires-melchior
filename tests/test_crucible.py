"""Tests for Melchior Crucible execution-grounded dataset generator."""

import json
import asyncio
import numpy as np
import pytest
from pathlib import Path
from melchior.crucible.environments import generate_task
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.client import CrucibleLLMClient, CandidatePair
from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.runner import CrucibleRunner


@pytest.mark.parametrize("seed", range(6))
def test_generate_task(seed):
    profile, X_tr, y_tr, X_va, y_va = generate_task(seed=seed)
    assert profile.n_train == len(X_tr)
    assert profile.n_val == len(X_va)
    assert X_tr.shape[1] == profile.n_features
    assert profile.metric in ["roc_auc", "accuracy", "r2", "f1"]
    repeated_profile, *repeated_data = generate_task(seed=seed)
    assert repeated_profile == profile
    for actual, repeated in zip((X_tr, y_tr, X_va, y_va), repeated_data):
        np.testing.assert_array_equal(actual, repeated)


@pytest.mark.asyncio
async def test_async_sandbox():
    sandbox = AsyncSandbox(timeout_s=5.0)
    X_tr = np.array([[0], [1], [8], [9]])
    y_tr = np.array([0, 0, 1, 1])
    X_va = np.array([[0.5], [8.5]])
    y_va = np.array([0, 1])

    code = """
from sklearn.tree import DecisionTreeClassifier
def fit_predict(X_train, y_train, X_test):
    return DecisionTreeClassifier(max_depth=1, random_state=42).fit(X_train, y_train).predict(X_test)
"""
    res = await sandbox.execute(code, X_tr, y_tr, X_va, y_va)
    assert res.status == "success"
    assert res.metric == 1.0
    assert res.wall_time_s > 0


@pytest.mark.asyncio
async def test_crucible_llm_client_mock():
    client = CrucibleLLMClient(mode="mock")
    profile, _, _, _, _ = generate_task(seed=100)
    pair = await client.generate_pair(profile, seed=100)

    assert "def fit_predict" in pair.code_a
    assert "def fit_predict" in pair.code_b
    assert len(pair.hypothesis_a) > 5
    assert len(pair.hypothesis_b) > 5
    await client.close()


@pytest.mark.asyncio
async def test_crucible_runner_end_to_end(tmp_path):
    out_dir = tmp_path / "crucible_data"
    runner = CrucibleRunner(
        concurrency=3,
        output_dir=out_dir,
        llm_mode="mock",
        sandbox_timeout_s=5.0,
        max_pairs=3,
    )
    await asyncio.wait_for(runner.run(), 30)
    assert runner._total_evaluated == 3

    nli_file = out_dir / "openjev_ml_nli.jsonl"
    dpo_file = out_dir / "melchior_dpo_pairs.jsonl"

    assert nli_file.exists()
    assert dpo_file.exists()

    with open(nli_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert lines == []  # Mock fixtures must never enter training exports.
    records = [json.loads(x) for x in (out_dir / 'evaluations.jsonl').read_text().splitlines()]
    assert {r['seed'] for r in records} == {1000, 1001, 1002}
    assert all('predictions' in r['res_a'] for r in records)
