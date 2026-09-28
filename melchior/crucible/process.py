"""Subprocess ownership that remains deterministic during asyncio cancellation."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


async def run_process(
    args: list[str],
    *,
    timeout_s: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a process, killing its group and reaping it on every exit path.

    Use regular files for output and Popen without preexec_fn. Cancelling asyncio
    subprocess creation while its pipe transports are connecting can deadlock
    loop shutdown (including on Python 3.12). Here cancellation is delivered
    only after ownership of the process is established. No transport tasks are
    created, and output cannot block a child on a full pipe.
    """
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(
            args, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=os.name != "nt",
        )
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Execution exceeded timeout of {timeout_s}s")
                if proc.poll() is not None:
                    break
                await asyncio.sleep(min(0.02, remaining))
        finally:
            # Also remove descendants if the direct child has already exited.
            # Cleanup has no await points: repeated cancellation cannot abandon
            # the owned child between killing and reaping it.
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                elif proc.poll() is None:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)

        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(args, proc.returncode, stdout.read(), stderr.read())
