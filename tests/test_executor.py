"""Tests for Executor and dense feedback streaming."""

from melchior.brain.jev_client import JevClient
from melchior.core.executor import Executor


def test_executor_successful_run(tmp_path):
    jev = JevClient(mode="mock")
    executor = Executor(jev=jev)

    code = """
print("EPOCH_METRIC:1:0.75")
print("EPOCH_METRIC:2:0.88")
print("FINAL_METRIC:0.9250")
"""
    result = executor.run(code, best_so_far=0.8, timeout_seconds=10, workdir=tmp_path)
    assert result.status == "success"
    assert result.metric == 0.9250
    assert result.epoch_metrics == [0.75, 0.88]
    assert result.wall_time_s > 0


def test_executor_failure_classification(tmp_path):
    jev = JevClient(mode="mock")
    executor = Executor(jev=jev)

    code = """
import sys
raise ValueError("Dimension mismatch between [10, 4] and [10, 5]")
"""
    result = executor.run(code, best_so_far=0.8, timeout_seconds=10, workdir=tmp_path)
    assert result.status == "failed"
    assert result.metric is None
    assert result.error_type == "shape"
    assert result.auto_fixable is True
