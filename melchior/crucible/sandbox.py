"""Async isolated sandbox for high-throughput candidate execution."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SandboxResult:
    status: str  # success | failed | timeout
    metric: float | None
    wall_time_s: float
    stdout: str
    stderr: str


def _set_limits():
    """Sets memory limit to 2GB per subprocess to protect server RAM."""
    try:
        import resource
        limit_bytes = 2 * 1024 * 1024 * 1024  # 2 GB
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ImportError, ValueError, OSError):
        pass


class AsyncSandbox:
    def __init__(self, timeout_s: float = 12.0, python_executable: str | None = None):
        self.timeout_s = timeout_s
        self.python_executable = python_executable or sys.executable

    async def execute(
        self,
        code: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> SandboxResult:
        """Saves data.npz and code in a temporary directory, runs asynchronously, and measures metric."""
        temp_dir = Path(tempfile.mkdtemp(prefix="crucible_run_"))
        data_path = temp_dir / "data.npz"
        np.savez_compressed(data_path, X_tr=X_train, y_tr=y_train, X_va=X_val, y_va=y_val)

        script_path = temp_dir / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.time()
        stdout_data = ""
        stderr_data = ""
        status = "success"
        metric = None

        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_executable,
                "-u",
                str(script_path),
                cwd=str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_set_limits if os.name != "nt" else None,
            )

            try:
                raw_out, raw_err = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_s
                )
                stdout_data = raw_out.decode("utf-8", errors="replace")
                stderr_data = raw_err.decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    status = "failed"
                else:
                    # Extract METRIC:<float>
                    m = re.search(r"METRIC:([0-9\.\-]+)", stdout_data)
                    if m:
                        metric = float(m.group(1))
                    else:
                        status = "failed"
                        stderr_data += "\nNo METRIC:<val> output detected."

            except asyncio.TimeoutError:
                status = "timeout"
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                stderr_data = f"Execution exceeded timeout of {self.timeout_s}s"

        except Exception as e:
            status = "failed"
            stderr_data = str(e)
        finally:
            # Cleanup temp folder
            try:
                if data_path.exists():
                    data_path.unlink()
                if script_path.exists():
                    script_path.unlink()
                temp_dir.rmdir()
            except OSError:
                pass

        wall_time_s = round(time.time() - start, 3)
        return SandboxResult(
            status=status,
            metric=metric,
            wall_time_s=wall_time_s,
            stdout=stdout_data,
            stderr=stderr_data,
        )
