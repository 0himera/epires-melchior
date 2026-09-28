"""Async candidate execution with time and memory limits."""

from __future__ import annotations

import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from melchior.crucible.process import run_process


@dataclass
class SandboxResult:
    status: str  # success | failed | timeout
    metric: float | None
    wall_time_s: float
    stdout: str
    stderr: str


# Apply limits in the new interpreter, avoiding preexec_fn in a threaded parent.
_LAUNCHER = """
import os, runpy, sys
if os.name != 'nt':
    import resource
    limit = 2 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
runpy.run_path(sys.argv[1], run_name='__main__')
"""


class AsyncSandbox:
    def __init__(self, timeout_s: float = 12.0, python_executable: str | None = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.python_executable = python_executable or sys.executable

    async def execute(
        self, code: str, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> SandboxResult:
        start = time.monotonic()
        stdout_data = stderr_data = ""
        status = "success"
        metric = None
        try:
            with tempfile.TemporaryDirectory(prefix="crucible_run_") as directory:
                workdir = Path(directory)
                np.savez_compressed(
                    workdir / "data.npz", X_tr=X_train, y_tr=y_train, X_va=X_val, y_va=y_val,
                )
                script = workdir / "solution.py"
                script.write_text(code, encoding="utf-8")
                env = os.environ.copy()
                env.update(dict.fromkeys([
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                ], "1"))
                remaining = self.timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError(f"Execution exceeded timeout of {self.timeout_s}s")
                result = await run_process(
                    [self.python_executable, "-u", "-c", _LAUNCHER, str(script)],
                    timeout_s=remaining, cwd=workdir, env=env,
                )
                stdout_data = result.stdout.decode("utf-8", errors="replace")
                stderr_data = result.stderr.decode("utf-8", errors="replace")
                if result.returncode != 0:
                    status = "failed"
                else:
                    matches = re.findall(r"^METRIC:\s*(\S+)\s*$", stdout_data, re.MULTILINE)
                    if len(matches) != 1:
                        raise ValueError("Expected exactly one METRIC:<finite float> output")
                    metric = float(matches[0])
                    if not math.isfinite(metric):
                        metric = None
                        raise ValueError("METRIC must be finite")
        except TimeoutError as exc:
            status = "timeout"
            stderr_data += f"\n{exc}"
        except Exception as exc:
            status = "failed"
            metric = None
            stderr_data += f"\n{type(exc).__name__}: {exc}"
        # CancelledError deliberately propagates after process and directory cleanup.
        return SandboxResult(
            status, metric, round(time.monotonic() - start, 3), stdout_data, stderr_data,
        )
