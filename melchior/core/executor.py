"""Executor — sandboxed subprocess execution with dense feedback and Jev early-stopping."""

from __future__ import annotations

import os
import re
import sys
import time
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from melchior.brain.jev_client import JevClient
from melchior.brain.decisions import check_early_stop, classify_error


@dataclass
class ExecutionResult:
    status: str  # success | failed | early_stopped | timeout
    metric: float | None
    epoch_metrics: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None
    auto_fixable: bool = False


class Executor:
    def __init__(self, jev: JevClient, python_executable: str | None = None):
        self.jev = jev
        self.python_executable = python_executable or sys.executable

    def run(
        self,
        code: str,
        best_so_far: float = 0.0,
        timeout_seconds: int = 180,
        workdir: Path | None = None,
    ) -> ExecutionResult:
        """Run candidate Python code in isolated subprocess and stream dense feedback."""
        start_time = time.time()
        workdir = workdir or Path(tempfile.mkdtemp(prefix="melchior_trial_"))
        workdir.mkdir(parents=True, exist_ok=True)
        script_path = workdir / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        epoch_metrics: list[float] = []
        final_metric: float | None = None
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        status = "success"
        error_type: str | None = None
        auto_fixable = False

        try:
            proc = subprocess.Popen(
                [self.python_executable, "-u", str(script_path)],
                cwd=str(workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            deadline = start_time + timeout_seconds
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if time.time() > deadline:
                        proc.kill()
                        status = "timeout"
                        break

                    stdout_lines.append(line)
                    epoch_match = re.search(r"EPOCH_METRIC:\d+:([0-9\.\-]+)", line)
                    if epoch_match:
                        val = float(epoch_match.group(1))
                        epoch_metrics.append(val)
                        if len(epoch_metrics) >= 3:
                            decision = check_early_stop(self.jev, epoch_metrics, best_so_far)
                            if decision.should_stop:
                                proc.terminate()
                                status = "early_stopped"
                                break

                    final_match = re.search(r"FINAL_METRIC:([0-9\.\-]+)", line)
                    if final_match:
                        final_metric = float(final_match.group(1))

            rem_stderr = proc.stderr.read() if proc.stderr else ""
            if rem_stderr:
                stderr_lines.append(rem_stderr)

            proc.wait(timeout=5)
            if proc.returncode != 0 and status != "early_stopped" and status != "timeout":
                status = "failed"
                full_stderr = "".join(stderr_lines)
                full_stdout = "".join(stdout_lines)
                err_dec = classify_error(self.jev, full_stderr, full_stdout)
                error_type = err_dec.error_type
                auto_fixable = err_dec.auto_fixable

        except subprocess.TimeoutExpired:
            proc.kill()
            status = "timeout"
        except Exception as e:
            status = "failed"
            stderr_lines.append(str(e))

        wall_time_s = round(time.time() - start_time, 3)

        # If final_metric wasn't explicitly tagged but we have epoch metrics, use the last one
        if final_metric is None and epoch_metrics:
            final_metric = epoch_metrics[-1]

        return ExecutionResult(
            status=status,
            metric=final_metric,
            epoch_metrics=epoch_metrics,
            wall_time_s=wall_time_s,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
            error_type=error_type,
            auto_fixable=auto_fixable,
        )
