"""Execute prediction code; score in the parent against undisclosed labels."""
from __future__ import annotations

import ast
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from melchior.crucible.metrics import CONTRACT_VERSION, score_predictions
from melchior.crucible.process import run_process


@dataclass
class SandboxResult:
    status: str
    metric: float | None
    wall_time_s: float
    stdout: str
    stderr: str
    predictions: list[float] = field(default_factory=list)
    score_source: str = CONTRACT_VERSION


def validate_code(code: str) -> None:
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "fit_predict" for node in tree.body):
        raise ValueError("Candidate must define fit_predict(X_train, y_train, X_test)")
    # Contract checks prevent ordinary accidental dataset reconstruction. This
    # subprocess boundary is not a security sandbox for hostile Python code.
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            if node.module == "sklearn" and any(a.name == "datasets" for a in node.names):
                raise ValueError("Candidates must use supplied arrays, not sklearn.datasets")
        if any(m.startswith("sklearn.datasets") for m in modules):
            raise ValueError("Candidates must use supplied arrays, not sklearn.datasets")


class AsyncSandbox:
    def __init__(self, timeout_s: float = 12.0, python_executable: str | None = None):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.timeout_s = timeout_s
        self.python_executable = python_executable or sys.executable

    async def execute(self, code, X_train, y_train, X_test, y_test, *, metric="accuracy") -> SandboxResult:
        start = time.monotonic()
        stdout = stderr = ""
        status, score, predictions = "success", None, []
        try:
            validate_code(code)
            with tempfile.TemporaryDirectory(prefix="crucible_run_") as directory:
                path = Path(directory)
                # y_test is deliberately never serialized or passed to the child.
                np.savez_compressed(path / "data.npz", X_train=X_train, y_train=y_train, X_test=X_test)
                (path / "solution.py").write_text(code, encoding="utf-8")
                env = os.environ.copy()
                env.update(dict.fromkeys([
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                ], "1"))
                remaining = self.timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError("Sandbox deadline exceeded during preparation")
                result = await run_process([
                    self.python_executable, "-u", str(Path(__file__).with_name("predict_worker.py")),
                    str(path / "solution.py"), str(path / "data.npz"), str(path / "predictions.npy"),
                ], timeout_s=remaining, cwd=path, env=env)
                stdout = result.stdout.decode(errors="replace")[-16000:]
                stderr = result.stderr.decode(errors="replace")[-16000:]
                if result.returncode:
                    raise ValueError(f"Candidate exited with code {result.returncode}")
                output = path / "predictions.npy"
                if output.stat().st_size > len(y_test) * 16 + 1024:
                    raise ValueError("Prediction artifact exceeds expected size")
                pred = np.load(output, allow_pickle=False)
                score = score_predictions(metric, y_test, pred, classes=np.unique(y_train))
                predictions = pred.tolist()
        except TimeoutError as exc:
            status, stderr = "timeout", stderr + f"\n{exc}"
        except Exception as exc:
            status, stderr = "failed", stderr + f"\n{type(exc).__name__}: {exc}"
        return SandboxResult(status, score, round(time.monotonic() - start, 3), stdout, stderr, predictions)
