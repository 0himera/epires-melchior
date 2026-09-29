import io
import json
import urllib.error

import pytest

from melchior.brain.jev_client import Choice, JevClient
from melchior.config import Settings


def test_local_uses_server_without_api_key(monkeypatch):
    def respond(request, timeout):
        assert request.full_url == 'http://localhost:8080/v1/systemone'
        assert request.get_header('Authorization') is None
        assert json.loads(request.data)['questions']['q']['type'] == 'choice'
        return io.BytesIO(json.dumps({'answers': {'q': {'choice': 'B', 'confidence': .93}}}).encode())
    monkeypatch.setattr('urllib.request.urlopen', respond)
    client = JevClient(mode='local')
    assert client.choice('test', 'pick', {'A': 'first', 'B': 'second'}) == ('B', .93)


@pytest.mark.parametrize('mode,key', [('local', None), ('api', 'test-key')])
def test_network_failure_does_not_silently_produce_mock_decision(monkeypatch, mode, key):
    def fail(*args, **kwargs):
        raise urllib.error.URLError('connection refused')
    monkeypatch.setattr('urllib.request.urlopen', fail)
    with pytest.raises(RuntimeError, match='connection refused'):
        JevClient(mode=mode, api_key=key).ask('test', {'q': Choice('pick', {'A': 'first'})})


def test_local_settings_default_to_local_server(monkeypatch):
    monkeypatch.setenv('MELCHIOR_JEV_MODE', 'local')
    monkeypatch.delenv('JEV_API_URL', raising=False)
    assert Settings().jev_api_url == 'http://localhost:8080/v1/systemone'


def test_api_still_requires_credentials():
    with pytest.raises(ValueError, match='JEV_API_KEY'):
        JevClient(mode='api').ask('test', {})
