"""Task specification schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class TaskBudget(BaseModel):
    max_trials: int = Field(default=10, description="Maximum experiment iterations")
    max_hours: float = Field(default=1.0, description="Total execution time budget")
    pilot_timeout_s: int = Field(default=30, description="Per-pilot timeout in seconds")
    trial_timeout_s: int = Field(default=180, description="Per-trial timeout in seconds")


class Task(BaseModel):
    name: str = Field(..., description="Unique name of the ML task")
    goal: str = Field(..., description="Natural language objective")
    domain: str = Field(
        default="tabular_classification",
        description="ML problem domain (tabular_classification, tabular_regression, vision, etc.)",
    )
    metric: str = Field(default="val_accuracy", description="Primary target metric name")
    direction: Literal["maximize", "minimize"] = Field(
        default="maximize", description="Optimization direction"
    )
    target_metric: float | None = Field(
        default=None, description="Early completion target value"
    )
    data_path: str | None = Field(default=None, description="Path to input data")
    budget: TaskBudget = Field(default_factory=TaskBudget)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Task:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
