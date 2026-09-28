"""Tests for CandidateFilter (static analysis and pilot ranking)."""

from melchior.brain.jev_client import JevClient
from melchior.brain.llm_client import CandidateProposal
from melchior.core.executor import Executor
from melchior.core.filter import CandidateFilter


def test_static_check():
    jev = JevClient(mode="mock")
    executor = Executor(jev=jev)
    c_filter = CandidateFilter(jev=jev, executor=executor)

    valid_code = "import numpy as np\nx = np.zeros(10)"
    ok, msg = c_filter.static_check(valid_code)
    assert ok is True

    syntax_err = "def bad_func(: print(1)"
    ok, msg = c_filter.static_check(syntax_err)
    assert ok is False
    assert "SyntaxError" in msg

    prohibited = "import os\nos.system('rm -rf /')"
    ok, msg = c_filter.static_check(prohibited)
    assert ok is False
    assert "Prohibited" in msg


def test_filter_and_rank():
    jev = JevClient(mode="mock")
    executor = Executor(jev=jev)
    c_filter = CandidateFilter(jev=jev, executor=executor)

    candidates = [
        CandidateProposal(
            hypothesis="Candidate A",
            code="print('FINAL_METRIC:0.95')",
            predicted_metric=0.95,
            technique="ModelA",
        ),
        CandidateProposal(
            hypothesis="Candidate B with syntax error",
            code="invalid python code :::: syntax",
            predicted_metric=0.99,
            technique="ModelB",
        ),
    ]

    ranked = c_filter.filter_and_rank(candidates, best_so_far=0.90, pilot_timeout_s=5)
    # Candidate B should be filtered out by static analysis
    assert len(ranked) == 1
    assert ranked[0][0].technique == "ModelA"
    assert ranked[0][1] > 0.0
