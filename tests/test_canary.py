"""Canary orchestration against controlled HTTP responses and real subprocesses."""

import asyncio
import json

import httpx
import numpy as np
import pytest

from melchior.crucible.canary import CanaryRunner
from melchior.crucible.environments import TaskProfile


PAIR = {
    "candidate_a": {"hypothesis": "Baseline A", "code": "print('METRIC:0.6')"},
    "candidate_b": {"hypothesis": "Improved B", "code": "print('METRIC:0.8')"},
}


def tiny_task(seed):
    return (
        TaskProfile(f"crucible_task_{seed:06d}", "binary_classification", "accuracy",
                    "maximize", f"Test seed={seed}", 4, 2, 1),
        np.zeros((4, 1)), np.zeros(4), np.zeros((2, 1)), np.zeros(2),
    )


def reply(pair):
    return httpx.Response(200, json={"choices": [{
        "message": {"content": json.dumps(pair)}, "finish_reason": "stop",
    }]})


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def runner_with_http(tmp_path, monkeypatch, handler, num_tasks=3):
    monkeypatch.setattr("melchior.crucible.canary.generate_task", tiny_task)
    runner = CanaryRunner(num_tasks=num_tasks, concurrency=2, llm_model="test-model",
                          output_dir=tmp_path, sandbox_timeout_s=5)
    await runner.client._http_client.aclose()
    await runner.http_client.aclose()
    runner.client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runner.http_client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={
            "answers": {"winner": {"choice": "B", "confidence": 0.8}},
        }),
    ))
    return runner


@pytest.mark.asyncio
async def test_missing_candidate_does_not_abort_other_tasks(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        prompt = json.loads(request.content)["messages"][1]["content"]
        calls.append(prompt)
        if "seed=1001" in prompt:
            return reply({"candidate_a": PAIR["candidate_a"]})
        return reply(PAIR)

    runner = await runner_with_http(tmp_path, monkeypatch, handler)
    summary = await asyncio.wait_for(runner.run(), 10)
    assert summary["status"] == "completed_with_errors"
    assert summary["requested_tasks"] == summary["completed_tasks"] == 3
    assert summary["failed_tasks"] == 1
    assert summary["total_tasks"] == 2
    assert summary["pass_rate_pct"] == 100
    assert summary["openjev_baseline_accuracy_pct"] == 100
    assert len(calls) == 4  # Two attempts for the incomplete pair, one per good task.
    evaluations = rows(runner.evaluations_file)
    assert {e["seed"] for e in evaluations} == {1002, 1003}
    assert all(e["winner"] == "B" and e["cand_b_metric"] == 0.8 for e in evaluations)
    holdout = rows(runner.holdout_file)
    assert all(e["code_b"] == PAIR["candidate_b"]["code"] for e in holdout)
    assert all(e["hypothesis_a"] == PAIR["candidate_a"]["hypothesis"] for e in holdout)
    errors = rows(runner.errors_file)
    assert len(errors) == 1
    assert errors[0]["seed"] == 1001
    assert "candidate_b" in errors[0]["error"]
    assert json.loads(runner.summary_file.read_text()) == summary
    assert runner.client._http_client.is_closed and runner.http_client.is_closed


@pytest.mark.asyncio
async def test_completed_result_is_persisted_before_other_tasks_finish(tmp_path, monkeypatch):
    blocked = asyncio.Event()

    async def handler(request):
        prompt = json.loads(request.content)["messages"][1]["content"]
        if "seed=1002" in prompt:
            blocked.set()
            await asyncio.Event().wait()
        return reply(PAIR)

    runner = await runner_with_http(tmp_path, monkeypatch, handler, num_tasks=2)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(blocked.wait(), 2)

        async def wait_for_checkpoint():
            while not runner.holdout_file.exists() or not runner.holdout_file.read_text():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_checkpoint(), 5)
        assert not task.done()
        assert [e["seed"] for e in rows(runner.evaluations_file)] == [1001]
        checkpoint = json.loads(runner.summary_file.read_text())
        assert checkpoint["status"] == "running"
        assert checkpoint["completed_tasks"] == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert rows(runner.holdout_file)[0]["code_b"] == PAIR["candidate_b"]["code"]
        assert json.loads(runner.summary_file.read_text())["status"] == "interrupted"
        assert runner.client._http_client.is_closed and runner.http_client.is_closed
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_tasks", [0, 3])
async def test_no_successes_still_produces_a_valid_summary(tmp_path, monkeypatch, num_tasks):
    runner = await runner_with_http(
        tmp_path, monkeypatch, lambda request: httpx.Response(503), num_tasks=num_tasks,
    )
    summary = await asyncio.wait_for(runner.run(), 5)
    assert summary["completed_tasks"] == summary["failed_tasks"] == num_tasks
    assert summary["total_candidates"] == 0
    assert summary["pass_rate_pct"] == 0
    assert summary["openjev_baseline_tested"] == 0
    assert len(rows(runner.errors_file)) == num_tasks
    assert rows(runner.holdout_file) == []
    assert runner.client._http_client.is_closed and runner.http_client.is_closed
