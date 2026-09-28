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
@pytest.mark.parametrize(("body", "status", "metric"), [
    ("return np.zeros(len(X_test))", "success", 1.0),
    ("print('METRIC:999'); return np.zeros(len(X_test))", "success", 1.0),
    ("raise ValueError('bad model')", "failed", None),
    ("return np.array([float('nan')]*len(X_test))", "failed", None),
    ("return np.zeros((len(X_test), 2))", "failed", None),
    ("return np.zeros(len(X_test)+1)", "failed", None),
    ("return np.full(len(X_test), 777)", "failed", None),
])
async def test_sandbox_scores_predictions_in_parent(data, body, status, metric):
    code = "import numpy as np\ndef fit_predict(X_train, y_train, X_test):\n    " + body
    result = await AsyncSandbox(timeout_s=5).execute(code, *data)
    assert result.status == status
    assert result.metric == metric


@pytest.mark.asyncio
async def test_no_test_labels_in_child(data):
    code = """import numpy as np
def fit_predict(X_train, y_train, X_test):
    data = np.load('data.npz')
    assert set(data.files) == {'X_train', 'y_train', 'X_test'}
    return np.zeros(len(X_test))
"""
    result = await AsyncSandbox(timeout_s=5).execute(code, *data)
    assert result.status == 'success'


@pytest.mark.asyncio
async def test_printed_metric_without_predictions_is_rejected(data):
    result = await AsyncSandbox().execute("print('METRIC:1')", *data)
    assert result.status == 'failed'


@pytest.mark.asyncio
async def test_stderr_larger_than_pipe_buffer_does_not_deadlock(data):
    result = await AsyncSandbox(timeout_s=5).execute(
        "import sys, numpy as np\ndef fit_predict(X_train, y_train, X_test):\n    sys.stderr.write('x' * 200000)\n    return np.zeros(len(X_test))", *data,
    )
    assert result.status == "success"
    assert result.metric == 1
    assert len(result.stderr) == 16000


@pytest.mark.asyncio
async def test_silent_process_times_out_and_temp_files_are_removed(data, tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    result = await AsyncSandbox(timeout_s=0.2).execute("def fit_predict(a,b,c):\n    import time; time.sleep(60)", *data)
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
            calls = [AsyncSandbox(timeout_s=60).execute('def fit_predict(a,b,c): import time; time.sleep(60)', *data)
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
