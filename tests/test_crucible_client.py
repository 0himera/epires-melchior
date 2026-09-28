"""Exercise the actual response parser and HTTP client without external models."""

import asyncio
import json

import httpx
import pytest

from melchior.crucible.client import CrucibleLLMClient, _parse_candidate_json
from melchior.crucible.environments import generate_task
from melchior.crucible.prompts import get_prompt_for_operator


VALID_PAIR = {
    "candidate_a": {"hypothesis": "Linear baseline", "code": "print('METRIC:0.6')"},
    "candidate_b": {"hypothesis": "Tree model", "code": "print('METRIC:0.8')"},
}


def response(content, finish_reason="stop"):
    return httpx.Response(200, json={"choices": [{
        "message": {"content": content}, "finish_reason": finish_reason,
    }]})


@pytest.mark.parametrize("content", [
    None, "", "[]", "not json", '{"candidate_a":',
    json.dumps({"candidate_a": VALID_PAIR["candidate_a"]}),
    json.dumps({**VALID_PAIR, "candidate_b": []}),
    json.dumps({**VALID_PAIR, "candidate_b": {"hypothesis": "ok", "code": 12}}),
    json.dumps({**VALID_PAIR, "candidate_b": {"hypothesis": " ", "code": "print(1)"}}),
])
def test_invalid_pair_is_rejected(content):
    with pytest.raises(ValueError):
        _parse_candidate_json(content)


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Result:\n{}\nDone."])
def test_json_wrappers_preserve_both_candidates(wrapper):
    pair = _parse_candidate_json(wrapper.format(json.dumps(VALID_PAIR)), "data_centric")
    assert pair.code_a == VALID_PAIR["candidate_a"]["code"]
    assert pair.hypothesis_b == VALID_PAIR["candidate_b"]["hypothesis"]
    assert pair.operator == "data_centric"


@pytest.mark.asyncio
async def test_incomplete_pair_is_retried_and_operator_reaches_http_prompt():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response(json.dumps({"candidate_a": VALID_PAIR["candidate_a"]}))
        return response(json.dumps(VALID_PAIR))

    client = CrucibleLLMClient(mode="api", model="test-model")
    await client._http_client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        profile, *_ = generate_task(42)
        pair = await client.generate_pair(profile, 42, operator="pathology_defense")
        assert pair.code_b == VALID_PAIR["candidate_b"]["code"]
        assert len(requests) == 2
        assert all(r["messages"][0]["content"] == get_prompt_for_operator("pathology_defense")
                   for r in requests)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["api", "auto"])
@pytest.mark.parametrize("failure", ["missing_b", "http_error", "truncated"])
async def test_failed_generation_has_bounded_retries_and_never_returns_mock(mode, failure, monkeypatch):
    monkeypatch.setattr("melchior.crucible.client.find_opencode_bin", lambda: None)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if failure == "http_error":
            return httpx.Response(503)
        if failure == "truncated":
            return response(json.dumps(VALID_PAIR), "length")
        return response(json.dumps({"candidate_a": VALID_PAIR["candidate_a"]}))

    client = CrucibleLLMClient(mode=mode, model="test-model", max_attempts=2)
    await client._http_client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="Candidate generation failed"):
            await client.generate_pair(generate_task(42)[0], 42)
        assert calls == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_generation_deadline_bounds_unresponsive_backend():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    client = CrucibleLLMClient(mode="api", model="test-model", timeout_s=0.05)
    await client._http_client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError, match="TimeoutError"):
            await asyncio.wait_for(client.generate_pair(generate_task(42)[0], 42), 2)
        assert calls == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancellation_is_not_retried():
    started = asyncio.Event()
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        started.set()
        await asyncio.Event().wait()

    client = CrucibleLLMClient(mode="api", model="test-model")
    await client._http_client.aclose()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    task = asyncio.create_task(client.generate_pair(generate_task(42)[0], 42))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert calls == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_b", "nonzero", "timeout"])
async def test_opencode_failures_are_explicit_and_processes_are_reaped(tmp_path, monkeypatch, failure):
    import os
    import subprocess
    import sys
    from melchior.crucible import process

    if os.name == "nt":
        pytest.skip("Executable test script uses a POSIX shebang")
    executable = tmp_path / "opencode"
    if failure == "missing_b":
        event = {"type": "text", "part": {"text": json.dumps({"candidate_a": VALID_PAIR["candidate_a"]})}}
        body = f"print({json.dumps(event)!r})"
    elif failure == "nonzero":
        body = "import sys; sys.stderr.write('backend unavailable'); sys.exit(7)"
    else:
        body = "import time; time.sleep(60)"
    executable.write_text(f"#!{sys.executable}\n{body}\n")
    executable.chmod(0o755)
    monkeypatch.setattr("melchior.crucible.client.find_opencode_bin", lambda: str(executable))
    children = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(process.subprocess, "Popen", spawn)
    client = CrucibleLLMClient(mode="opencode", timeout_s=0.2, max_attempts=2)
    try:
        with pytest.raises(RuntimeError, match="Candidate generation failed"):
            await asyncio.wait_for(client.generate_pair(generate_task(42)[0], 42), 3)
        assert len(children) == 2
        assert all(child.poll() is not None for child in children)
    finally:
        await client.close()
