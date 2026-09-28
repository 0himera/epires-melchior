"""All Jev-powered System One decision points in Melchior.

Each function is a typed, deterministic unit that transforms state into a calibrated judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from melchior.brain.jev_client import JevClient, Choice, Noul, Score


@dataclass
class StrategyDecision:
    strategy: str  # explore | exploit | ablate | ensemble
    confidence: float


@dataclass
class ViabilityDecision:
    technique: str
    viable: bool
    probability: float


@dataclass
class PilotRanking:
    candidate_idx: int
    score: float


@dataclass
class EarlyStopDecision:
    should_stop: bool
    probability: float


@dataclass
class ErrorClassification:
    error_type: str  # oom | shape | import | data | logic | unknown
    confidence: float
    auto_fixable: bool


@dataclass
class BudgetDecision:
    should_continue: bool
    probability: float


def decide_strategy(jev: JevClient, journal_summary: str) -> StrategyDecision:
    """① Choose research strategy for next trial."""
    resp = jev.ask(
        state=journal_summary,
        questions={
            "strategy": Choice(
                instructions="Given research progress, select optimal next strategy",
                criteria={
                    "explore": "Try a fundamentally different approach or model family",
                    "exploit": "Refine the current best-performing solution",
                    "ablate": "Systematically remove components to isolate what helps",
                    "ensemble": "Combine the top 2-3 solutions",
                },
            )
        },
    )
    return StrategyDecision(
        strategy=resp["strategy"]["choice"],
        confidence=resp["strategy"]["confidence"],
    )


def assess_technique(
    jev: JevClient, technique: str, domain: str, past_results: str
) -> ViabilityDecision:
    """② Quick viability check before expensive LLM code generation."""
    resp = jev.ask(
        state=f"Domain: {domain}\nTechnique: {technique}\nPast results: {past_results}",
        questions={
            "viable": Noul(
                f"Is {technique} likely to improve results in this context?"
            )
        },
    )
    prob = resp["viable"]["noul"]
    return ViabilityDecision(technique=technique, viable=prob > 0.5, probability=prob)


def rank_pilot(jev: JevClient, pilot_summary: str) -> float:
    """⑥ Score a pilot run's quality and promise."""
    resp = jev.ask(
        state=pilot_summary,
        questions={
            "quality": Score(
                instructions="Rate the quality and promise of this pilot training run "
                "(learning curve shape, initial metrics, stability)"
            )
        },
    )
    return resp["quality"]["score"]


def check_early_stop(
    jev: JevClient, epoch_metrics: list[float], best_so_far: float
) -> EarlyStopDecision:
    """⑦ Should we stop training early?"""
    trend = "improving"
    if len(epoch_metrics) >= 3 and epoch_metrics[-1] <= epoch_metrics[-2] <= epoch_metrics[-3]:
        trend = "plateau/declining"

    resp = jev.ask(
        state=(
            f"Last 5 epoch metrics: {epoch_metrics[-5:]}\n"
            f"Best metric overall: {best_so_far}\n"
            f"Trend: {trend}"
        ),
        questions={
            "stop": Noul(
                "Has training plateaued or started overfitting? Should we stop early?"
            )
        },
    )
    prob = resp["stop"]["noul"]
    return EarlyStopDecision(should_stop=prob > 0.7, probability=prob)


def classify_error(
    jev: JevClient, traceback: str, last_stdout: str
) -> ErrorClassification:
    """⑧ Classify runtime error for potential auto-fix."""
    resp = jev.ask(
        state=f"Traceback:\n{traceback[-1000:]}\nStdout:\n{last_stdout[-500:]}",
        questions={
            "error_type": Choice(
                instructions="Classify this ML training error",
                criteria={
                    "oom": "Out of memory (GPU or CPU)",
                    "shape": "Tensor shape mismatch",
                    "import": "Missing package or import error",
                    "data": "Data loading or format issue",
                    "logic": "Logic error in training loop",
                    "unknown": "Cannot determine from traceback",
                },
            ),
            "fixable": Noul(
                "Can this error be auto-fixed by adjusting batch size, shapes, or imports?"
            ),
        },
    )
    return ErrorClassification(
        error_type=resp["error_type"]["choice"],
        confidence=resp["error_type"]["confidence"],
        auto_fixable=resp["fixable"]["noul"] > 0.7,
    )


def assess_budget(jev: JevClient, progress_summary: str) -> BudgetDecision:
    """⑨ Is continued search worthwhile given remaining budget?"""
    resp = jev.ask(
        state=progress_summary,
        questions={
            "continue": Noul(
                "Given the improvement trajectory and remaining budget, "
                "is continued experimentation likely to yield meaningful gains?"
            )
        },
    )
    prob = resp["continue"]["noul"]
    return BudgetDecision(should_continue=prob > 0.4, probability=prob)
