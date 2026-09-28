"""Tests for Melchior Crucible execution-grounded dataset generator."""

import json
import pytest
from pathlib import Path
from melchior.crucible.environments import generate_task
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.client import CrucibleLLMClient, CandidatePair
from melchior.crucible.arbiter import CrucibleArbiter
from melchior.crucible.runner import CrucibleRunner


def test_generate_task():
    profile, X_tr, y_tr, X_va, y_va = generate_task(seed=42)
    assert profile.n_train == len(X_tr)
    assert profile.n_val == len(X_va)
    assert X_tr.shape[1] == profile.n_features
    assert profile.metric in ["roc_auc", "accuracy", "r2", "f1"]


@pytest.mark.asyncio
async def test_async_sandbox():
    sandbox = AsyncSandbox(timeout_s=5.0)
    profile, X_tr, y_tr, X_va, y_va = generate_task(seed=10)

    code = """
import numpy as np
data = np.load("data.npz")
print("METRIC:0.8950")
"""
    res = await sandbox.execute(code, X_tr, y_tr, X_va, y_va)
    assert res.status == "success"
    assert res.metric == 0.8950
    assert res.wall_time_s > 0


@pytest.mark.asyncio
async def test_crucible_llm_client_mock():
    client = CrucibleLLMClient(mode="mock")
    profile, _, _, _, _ = generate_task(seed=100)
    pair = await client.generate_pair(profile, seed=100)

    assert "import numpy as np" in pair.code_a
    assert "import numpy as np" in pair.code_b
    assert len(pair.hypothesis_a) > 5
    assert len(pair.hypothesis_b) > 5
    await client.close()


def test_arbiter_winner_and_nli_format():
    arbiter = CrucibleArbiter(min_delta=0.01)
    profile, _, _, _, _ = generate_task(seed=1)
    pair = CandidatePair(
        hypothesis_a="Model A",
        code_a="code a",
        hypothesis_b="Model B",
        code_b="code b",
    )

    from melchior.crucible.sandbox import SandboxResult

    res_a = SandboxResult(status="success", metric=0.82, wall_time_s=1.0, stdout="", stderr="")
    res_b = SandboxResult(status="success", metric=0.91, wall_time_s=1.2, stdout="", stderr="")

    outcome = arbiter.evaluate(profile, pair, res_a, res_b)
    assert outcome.winner == "B"
    assert outcome.delta == pytest.approx(0.09)
    assert len(outcome.nli_records) == 2

    # B won: Entailment (1)
    b_rec = [r for r in outcome.nli_records if r["hypothesis"] == "Model B"][0]
    assert b_rec["label"] == 1

    # A lost: Contradiction (0)
    a_rec = [r for r in outcome.nli_records if r["hypothesis"] == "Model A"][0]
    assert a_rec["label"] == 0

    assert outcome.dpo_record is not None
    assert "Model B" in outcome.dpo_record["chosen"]


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
    await runner.run()

    nli_file = out_dir / "openjev_ml_nli.jsonl"
    dpo_file = out_dir / "melchior_dpo_pairs.jsonl"

    assert nli_file.exists()
    assert dpo_file.exists()

    with open(nli_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) >= 3
    for row in lines:
        assert "premise" in row
        assert "hypothesis" in row
        assert row["label"] in [0, 1, 2]
        assert row["source"] == "crucible_ml_empirical"
