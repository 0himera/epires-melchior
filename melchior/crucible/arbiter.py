"""Objective Ground-Truth Arbiter comparing empirical metrics and producing NLI/DPO records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from melchior.crucible.environments import TaskProfile
from melchior.crucible.client import CandidatePair
from melchior.crucible.sandbox import SandboxResult


@dataclass
class ArbiterOutcome:
    winner: str  # "A" | "B" | "TIE" | "BOTH_FAILED"
    delta: float
    nli_records: list[dict[str, Any]]
    dpo_record: dict[str, Any] | None


class CrucibleArbiter:
    def __init__(self, min_delta: float = 0.005):
        self.min_delta = min_delta

    def evaluate(
        self,
        profile: TaskProfile,
        pair: CandidatePair,
        res_a: SandboxResult,
        res_b: SandboxResult,
    ) -> ArbiterOutcome:
        """Determines the objective winner and formats OpenJev NLI and DPO records."""
        met_a = res_a.metric if res_a.status == "success" else None
        met_b = res_b.metric if res_b.status == "success" else None

        premise = (
            f"Task: {profile.description} "
            f"Objective: Maximize {profile.metric}. "
            f"Train samples: {profile.n_train}, Val samples: {profile.n_val}."
        )

        operator = getattr(pair, "operator", "unknown")
        nli_records: list[dict[str, Any]] = []
        dpo_record = None
        winner = "TIE"
        delta = 0.0

        if met_a is not None and met_b is not None:
            diff = met_b - met_a
            delta = abs(diff)

            if diff > self.min_delta:
                winner = "B"
                # B is Entailment (1), A is Contradiction (0)
                nli_records.append({
                    "premise": premise,
                    "hypothesis": pair.hypothesis_b,
                    "label": 1,
                    "source": "crucible_ml_empirical",
                    "metadata": {"task_id": profile.task_id, "metric": met_b, "delta": delta, "operator": operator},
                })
                nli_records.append({
                    "premise": premise,
                    "hypothesis": pair.hypothesis_a,
                    "label": 0,
                    "source": "crucible_ml_empirical",
                    "metadata": {"task_id": profile.task_id, "metric": met_a, "delta": -delta, "operator": operator},
                })
                dpo_record = {
                    "prompt": premise,
                    "chosen": f"# Hypothesis: {pair.hypothesis_b}\n{pair.code_b}",
                    "rejected": f"# Hypothesis: {pair.hypothesis_a}\n{pair.code_a}",
                    "margin": round(delta, 4),
                    "operator": operator,
                }

            elif diff < -self.min_delta:
                winner = "A"
                # A is Entailment (1), B is Contradiction (0)
                nli_records.append({
                    "premise": premise,
                    "hypothesis": pair.hypothesis_a,
                    "label": 1,
                    "source": "crucible_ml_empirical",
                    "metadata": {"task_id": profile.task_id, "metric": met_a, "delta": delta, "operator": operator},
                })
                nli_records.append({
                    "premise": premise,
                    "hypothesis": pair.hypothesis_b,
                    "label": 0,
                    "source": "crucible_ml_empirical",
                    "metadata": {"task_id": profile.task_id, "metric": met_b, "delta": -delta, "operator": operator},
                })
                dpo_record = {
                    "prompt": premise,
                    "chosen": f"# Hypothesis: {pair.hypothesis_a}\n{pair.code_a}",
                    "rejected": f"# Hypothesis: {pair.hypothesis_b}\n{pair.code_b}",
                    "margin": round(delta, 4),
                    "operator": operator,
                }

            else:
                winner = "TIE"
                # Both valid, neutral difference
                nli_records.append({
                    "premise": premise,
                    "hypothesis": pair.hypothesis_a,
                    "label": 2,  # Neutral
                    "source": "crucible_ml_empirical",
                    "metadata": {"task_id": profile.task_id, "metric": met_a, "delta": 0.0, "operator": operator},
                })

        elif met_a is not None and met_b is None:
            winner = "A"
            delta = 1.0
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_a,
                "label": 1,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "metric": met_a, "status": "success", "operator": operator},
            })
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_b,
                "label": 0,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "error": res_b.stderr[:150], "status": res_b.status, "operator": operator},
            })
            dpo_record = {
                "prompt": premise,
                "chosen": f"# Hypothesis: {pair.hypothesis_a}\n{pair.code_a}",
                "rejected": f"# Hypothesis: {pair.hypothesis_b}\n{pair.code_b}",
                "margin": 1.0,
                "operator": operator,
            }

        elif met_b is not None and met_a is None:
            winner = "B"
            delta = 1.0
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_b,
                "label": 1,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "metric": met_b, "status": "success", "operator": operator},
            })
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_a,
                "label": 0,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "error": res_a.stderr[:150], "status": res_a.status, "operator": operator},
            })
            dpo_record = {
                "prompt": premise,
                "chosen": f"# Hypothesis: {pair.hypothesis_b}\n{pair.code_b}",
                "rejected": f"# Hypothesis: {pair.hypothesis_a}\n{pair.code_a}",
                "margin": 1.0,
                "operator": operator,
            }

        else:
            winner = "BOTH_FAILED"
            delta = 0.0
            # Both failed: hard negative contradiction examples
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_a,
                "label": 0,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "error": res_a.stderr[:150], "operator": operator},
            })
            nli_records.append({
                "premise": premise,
                "hypothesis": pair.hypothesis_b,
                "label": 0,
                "source": "crucible_ml_empirical",
                "metadata": {"task_id": profile.task_id, "error": res_b.stderr[:150], "operator": operator},
            })

        return ArbiterOutcome(
            winner=winner,
            delta=delta,
            nli_records=nli_records,
            dpo_record=dpo_record,
        )
