from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import supervise_openjev_pilot as supervisor


def test_resume_preserves_existing_training_and_verification(monkeypatch):
    existing={'first','first-verify','openjev-lora-pilot-s42-c0'}
    created=[]
    monkeypatch.setattr(supervisor,'exists',lambda name:name in existing)
    monkeypatch.setattr(supervisor,'wait',lambda name:0)
    monkeypatch.setattr(supervisor,'create',lambda template,name,cmd,repo:created.append((name,cmd)))
    report=Mock();run=Mock()
    monkeypatch.setattr(supervisor,'report',report)
    monkeypatch.setattr(supervisor.subprocess,'run',run)
    args=SimpleNamespace(first='first',root=Path('/experiment'),repository=Path('/repo'),
                         micro_groups=8,accumulation=1,gradient_checkpointing=False)
    supervisor.main(args)
    training=[(name,cmd) for name,cmd in created if not name.endswith('-verify')]
    assert [name for name,_ in training]==['openjev-lora-pilot-s43-c1','openjev-lora-pilot-s43-c0']
    assert all('--no-gradient-checkpointing' in cmd and cmd[cmd.index('--micro-groups')+1]=='8' for _,cmd in training)
    assert len(created)==5
    assert report.call_args.args[1]=='complete'
    run.assert_called_once_with(['docker','start','vllm-qwen'],check=True)


def test_failed_existing_run_is_not_overwritten(monkeypatch):
    monkeypatch.setattr(supervisor,'exists',lambda name:True)
    monkeypatch.setattr(supervisor,'wait',lambda name:1)
    create=Mock();report=Mock();run=Mock()
    monkeypatch.setattr(supervisor,'create',create)
    monkeypatch.setattr(supervisor,'report',report)
    monkeypatch.setattr(supervisor.subprocess,'run',run)
    args=SimpleNamespace(first='first',root=Path('/experiment'),repository=Path('/repo'))
    with pytest.raises(RuntimeError,match='Training failed'):
        supervisor.main(args)
    create.assert_not_called()
    assert report.call_args.args[1]=='failed'
    run.assert_called_once_with(['docker','start','vllm-qwen'],check=True)
