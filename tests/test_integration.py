"""End-to-end integration test for Melchior research loop."""

from pathlib import Path
from melchior.cli import run_experiment_loop
from melchior.core.journal import Journal


def test_iris_autonomous_research_loop(tmp_path):
    task_yaml = Path(__file__).parent.parent / "examples" / "iris_classification.yaml"
    out_dir = tmp_path / "artifacts"

    # Run loop
    exit_code = run_experiment_loop(
        task_path=str(task_yaml),
        output_dir=str(out_dir),
        n_candidates=2,
    )
    assert exit_code == 0

    # Verify artifacts created
    journal_file = out_dir / "journal.jsonl"
    report_file = out_dir / "report.md"
    best_file = out_dir / "best_solution.py"

    assert journal_file.exists()
    assert report_file.exists()
    assert best_file.exists()

    # Verify journal content
    journal = Journal(journal_file)
    assert len(journal.records) >= 1
    best_record = journal.best_record()
    assert best_record is not None
    assert best_record.actual_metric is not None
    # On Iris, our ML pipeline reaches > 0.90 accuracy
    assert best_record.actual_metric >= 0.90

    # Verify calibration was tracked
    assert best_record.calibration_error is not None

    # Verify best_solution.py runs standalone and prints metric
    import subprocess
    import sys

    res = subprocess.run([sys.executable, str(best_file)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "FINAL_METRIC:" in res.stdout
