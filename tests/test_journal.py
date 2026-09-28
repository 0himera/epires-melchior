"""Tests for Journal append-only log and self-calibration tracking."""

from melchior.core.journal import Journal, TrialRecord
from melchior.core.task import TaskBudget


def test_journal_append_and_stats(tmp_path):
    log_file = tmp_path / "journal.jsonl"
    journal = Journal(filepath=log_file)

    assert len(journal.records) == 0

    # Add trial 1 (overestimated)
    journal.append(
        TrialRecord(
            trial_idx=1,
            strategy="explore",
            technique="LogisticRegression",
            hypothesis="Baseline model",
            code="print('hello')",
            predicted_metric=0.95,
            actual_metric=0.90,
            status="success",
            wall_time_s=2.5,
        )
    )

    # Add trial 2 (underestimated)
    journal.append(
        TrialRecord(
            trial_idx=2,
            strategy="exploit",
            technique="RandomForest",
            hypothesis="Non-linear model",
            code="print('rf')",
            predicted_metric=0.92,
            actual_metric=0.96,
            status="success",
            wall_time_s=3.0,
        )
    )

    assert len(journal.records) == 2
    best = journal.best_record()
    assert best is not None
    assert best.technique == "RandomForest"
    assert best.actual_metric == 0.96

    # Calibration error checks
    cal = journal.calibration_stats()
    assert cal["count"] == 2
    # Trial 1 error: 0.95 - 0.90 = +0.05
    # Trial 2 error: 0.92 - 0.96 = -0.04
    # Mean error: (+0.05 - 0.04) / 2 = 0.005
    assert abs(cal["mean_error"] - 0.005) < 1e-4

    # Reload from disk
    reloaded = Journal(filepath=log_file)
    assert len(reloaded.records) == 2
    assert reloaded.best_record().actual_metric == 0.96


def test_budget_remaining(tmp_path):
    journal = Journal(filepath=tmp_path / "journal.jsonl")
    budget = TaskBudget(max_trials=2, max_hours=1.0)

    assert journal.budget_remaining(budget) is True

    journal.append(
        TrialRecord(
            trial_idx=1,
            strategy="explore",
            technique="M1",
            hypothesis="H1",
            code="x=1",
            predicted_metric=0.8,
            actual_metric=0.85,
        )
    )
    assert journal.budget_remaining(budget) is True

    journal.append(
        TrialRecord(
            trial_idx=2,
            strategy="exploit",
            technique="M2",
            hypothesis="H2",
            code="x=2",
            predicted_metric=0.9,
            actual_metric=0.91,
        )
    )
    assert journal.budget_remaining(budget) is False
