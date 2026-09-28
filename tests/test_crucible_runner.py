import asyncio
import json

import pytest

from melchior.crucible.runner import CrucibleRunner


@pytest.mark.asyncio
async def test_all_generation_failures_are_logged_and_respect_pair_limit(tmp_path, monkeypatch):
    runner = CrucibleRunner(concurrency=2, max_pairs=3, output_dir=tmp_path, llm_mode="mock")
    seeds = []

    async def fail(profile, seed):
        seeds.append(seed)
        raise ValueError("Missing candidate_b")

    monkeypatch.setattr(runner.client, "generate_pair", fail)
    await asyncio.wait_for(runner.run(), 5)
    assert sorted(seeds) == [1000, 1001, 1002]
    errors = [json.loads(line) for line in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert {e["seed"] for e in errors} == set(seeds)
    assert all("candidate_b" in e["error"] for e in errors)
    assert runner._total_evaluated == 0
    assert runner.client._http_client.is_closed


@pytest.mark.asyncio
async def test_runner_cancellation_stops_workers_before_closing_client(tmp_path, monkeypatch):
    runner = CrucibleRunner(concurrency=2, output_dir=tmp_path, llm_mode="mock")
    started = asyncio.Event()
    active = 0

    async def block(profile, seed):
        nonlocal active
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert not runner.client._http_client.is_closed
            active -= 1

    monkeypatch.setattr(runner.client, "generate_pair", block)
    task = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert active == 0
        assert runner.client._http_client.is_closed
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
