"""Append-only experiment journal with self-calibration tracking."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from pydantic import BaseModel, Field
from melchior.core.task import TaskBudget


class TrialRecord(BaseModel):
    trial_idx: int
    strategy: str
    technique: str
    hypothesis: str
    code: str
    code_hash: str = ""
    predicted_metric: float
    actual_metric: float | None = None
    calibration_error: float | None = None
    status: str = "success"  # success | failed | early_stopped | pruned
    epoch_metrics: list[float] = Field(default_factory=list)
    wall_time_s: float = 0.0
    error_type: str | None = None
    error_message: str | None = None

    def model_post_init(self, __context) -> None:
        if not self.code_hash and self.code:
            self.code_hash = hashlib.sha256(self.code.encode("utf-8")).hexdigest()[:10]
        if self.actual_metric is not None and self.calibration_error is None:
            self.calibration_error = round(self.predicted_metric - self.actual_metric, 4)


class Journal:
    def __init__(self, filepath: str | Path = "journal.jsonl"):
        self.filepath = Path(filepath)
        self.records: list[TrialRecord] = []
        if self.filepath.exists():
            self._load()

    def _load(self) -> None:
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(TrialRecord(**json.loads(line)))

    def append(self, record: TrialRecord) -> None:
        self.records.append(record)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

    def best_record(self, direction: str = "maximize") -> TrialRecord | None:
        valid = [r for r in self.records if r.actual_metric is not None and r.status == "success"]
        if not valid:
            return None
        if direction == "maximize":
            return max(valid, key=lambda r: r.actual_metric or -float("inf"))
        else:
            return min(valid, key=lambda r: r.actual_metric or float("inf"))

    def best_metric(self, direction: str = "maximize") -> float:
        best = self.best_record(direction)
        if best and best.actual_metric is not None:
            return best.actual_metric
        return -float("inf") if direction == "maximize" else float("inf")

    def budget_remaining(self, budget: TaskBudget) -> bool:
        if len(self.records) >= budget.max_trials:
            return False
        total_time_hours = sum(r.wall_time_s for r in self.records) / 3600.0
        if total_time_hours >= budget.max_hours:
            return False
        return True

    def summary(self) -> str:
        """Concise summary for Jev decision context."""
        if not self.records:
            return "Initial state: 0 trials executed. No baseline yet."

        best = self.best_record()
        best_str = f"{best.actual_metric:.4f} ({best.technique})" if best and best.actual_metric else "None"
        recent = self.records[-3:]

        lines = [
            f"Trials completed: {len(self.records)}",
            f"Best metric: {best_str}",
            "Recent trials:",
        ]
        for r in recent:
            metric_str = f"{r.actual_metric:.4f}" if r.actual_metric is not None else "failed"
            lines.append(
                f"  Trial {r.trial_idx} ({r.strategy}): {r.technique} -> {metric_str} [{r.status}]"
            )

        cal_stats = self.calibration_stats()
        lines.append(f"Calibration error mean: {cal_stats.get('mean_error', 0.0):.4f}")
        return "\n".join(lines)

    def results_for_technique(self, technique: str) -> str:
        matches = [r for r in self.records if technique.lower() in r.technique.lower()]
        if not matches:
            return "No previous attempts."
        return ", ".join(
            f"Trial {m.trial_idx}: {m.actual_metric}" for m in matches if m.actual_metric is not None
        )

    def calibration_stats(self) -> dict[str, float]:
        errors = [r.calibration_error for r in self.records if r.calibration_error is not None]
        if not errors:
            return {"mean_error": 0.0, "mae": 0.0, "count": 0}
        return {
            "mean_error": round(sum(errors) / len(errors), 4),
            "mae": round(sum(abs(e) for e in errors) / len(errors), 4),
            "count": len(errors),
        }
