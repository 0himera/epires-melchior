"""Core research loop modules for Melchior."""

from melchior.core.task import Task
from melchior.core.journal import Journal, TrialRecord
from melchior.core.strategist import Strategist
from melchior.core.filter import CandidateFilter
from melchior.core.executor import Executor, ExecutionResult
from melchior.core.report import generate_report

__all__ = [
    "Task",
    "Journal",
    "TrialRecord",
    "Strategist",
    "CandidateFilter",
    "Executor",
    "ExecutionResult",
    "generate_report",
]
