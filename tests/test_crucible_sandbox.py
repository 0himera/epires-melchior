"""Real subprocess regressions, with an external watchdog for loop shutdown."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from melchior.crucible.sandbox import AsyncSandbox


@pytest.fixture
def data():
    return np.zeros((4, 2)), np.zeros(4), np.zeros((2, 2)), np.zeros(2)


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "status", "metric"), [
    ("print('METRIC:-1.2e-3')", "success", -0.0012),
    ("print('METRIC:0.0')", "success", 0.0),
    ("print('METRIC:0.9'); raise ValueError('bad model')", "failed", None),
    ("print('training complete')", "failed", None),
    ("print('METRIC:nan')", "failed", None),
    ("print('METRIC:inf')", "failed", None),
    ("print('METRIC:0.9oops')", "failed", None),
    ("print('METRIC:0.8'); print('METRIC:0.9')", "failed", None),
])
async def test_sandbox_validates_the_actual_process_output(data, code, status, metric):
    result = await AsyncSandbox(timeout_s=5).execute(code, *data)
    assert result.status == status
    assert result.metric == metric


@pytest.mark.asyncio
async def test_stderr_larger_than_pipe_buffer_does_not_deadlock(data):
    result = await AsyncSandbox(timeout_s=5).execute(
        "import sys\nsys.stderr.write('x' * 200000)\nprint('METRIC:0.75')", *data,
    )
    assert result.status == "success"
    assert result.metric == 0.75
    assert len(result.stderr) == 200000


@pytest.mark.asyncio
async def test_silent_process_times_out_and_temp_files_are_removed(data, tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    result = await AsyncSandbox(timeout_s=0.2).execute("import time; time.sleep(60)", *data)
    assert result.status == "timeout"
    assert result.wall_time_s < 3
    assert result.metric is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("scenario", ["cancel_at_spawn", "error_during_parallel_spawn"])
def test_asyncio_shutdown_reaps_children_without_hanging(tmp_path, scenario):
    # A timeout within the same event loop cannot detect a deadlock in
    # asyncio.run() shutdown. Run the reproduction in a separate interpreter.
    script = textwrap.dedent('''
        import asyncio, subprocess, sys, tempfile
        from pathlib import Path
        import numpy as np
        from melchior.crucible.sandbox import AsyncSandbox
        from melchior.crucible import process

        tempfile.tempdir = sys.argv[1]
        scenario = sys.argv[2]
        children = []
        original = subprocess.Popen
        def spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            if scenario == 'cancel_at_spawn':
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            return child
        process.subprocess.Popen = spawn
        data = (np.zeros((2, 1)), np.zeros(2), np.zeros((1, 1)), np.zeros(1))
        async def fail():
            await asyncio.sleep(0)
            raise RuntimeError("missing candidate_b")
        async def main():
            calls = [AsyncSandbox(timeout_s=60).execute('import time; time.sleep(60)', *data)
                     for _ in range(8)]
            if scenario == 'error_during_parallel_spawn':
                calls.append(fail())
            await asyncio.gather(*calls)
        try:
            asyncio.run(main())
        except (RuntimeError, asyncio.CancelledError):
            pass
        else:
            raise AssertionError('Expected failure or cancellation')
        assert len(children) == 8, len(children)
        assert all(child.poll() is not None for child in children), 'live child leaked'
        assert not list(Path(tempfile.tempdir).glob('crucible_run_*')), 'temp directories leaked'
        print('cleanup verified')
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), scenario],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cleanup verified" in result.stdout
    assert "Task was destroyed" not in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_timeout_kills_descendants(tmp_path):
    script = textwrap.dedent('''
        import asyncio, os, sys, time
        from pathlib import Path
        from melchior.crucible.process import run_process
        marker = Path(sys.argv[1])
        code = "import os, time\\nfrom pathlib import Path\\nchild = os.fork()\\nif child == 0:\\n    Path(%r + '.ready').write_text('started')\\n    time.sleep(1.5)\\n    Path(%r).write_text('leaked')\\n    os._exit(0)\\ntime.sleep(60)" % (str(marker), str(marker))
        async def main():
            try:
                await run_process([sys.executable, '-c', code], timeout_s=0.5)
            except TimeoutError:
                pass
            else:
                raise AssertionError('No timeout')
        asyncio.run(main())
        assert Path(str(marker) + '.ready').exists(), 'descendant never started'
        time.sleep(1.7)
        assert not marker.exists(), 'descendant survived timeout'
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "leaked")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
