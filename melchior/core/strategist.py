"""Strategist — dual-brain synthesis coordinating Jev (System 1) and LLM (System 2)."""

from __future__ import annotations

from dataclasses import dataclass
from melchior.brain.jev_client import JevClient
from melchior.brain.llm_client import LLMClient, CandidateProposal
from melchior.brain.decisions import decide_strategy, assess_technique
from melchior.core.task import Task
from melchior.core.journal import Journal


DOMAIN_TECHNIQUES: dict[str, list[str]] = {
    "tabular_classification": [
        "LogisticRegression + StandardScaler",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "VotingEnsemble",
        "Dropout + MLP",
        "Feature Selection + ExtraTrees",
    ],
    "tabular_regression": [
        "Ridge Regression",
        "GradientBoostingRegressor",
        "RandomForestRegressor",
        "PolynomialFeatures + ElasticNet",
    ],
    "vision": [
        "CNN + BatchNormalization",
        "ResNet Backbone",
        "Data Augmentation + AdamW",
    ],
}


@dataclass
class StrategyPlan:
    strategy: str
    viable_techniques: list[str]
    candidates: list[CandidateProposal]


class Strategist:
    def __init__(self, jev: JevClient, llm: LLMClient):
        self.jev = jev
        self.llm = llm

    def plan_next_trial(self, task: Task, journal: Journal, n_candidates: int = 1) -> StrategyPlan:
        """Execute dual-brain planning:
        1. Jev selects meta-strategy (explore/exploit/ablate/ensemble)
        2. Jev filters viable techniques (fast System 1 Noul gating)
        3. LLM generates code proposals for approved techniques
        """
        # Step 1: Jev decides high-level research strategy
        strat_dec = decide_strategy(self.jev, journal.summary())
        strategy = strat_dec.strategy

        # Step 2: Jev filters domain techniques
        all_techniques = DOMAIN_TECHNIQUES.get(
            task.domain,
            DOMAIN_TECHNIQUES["tabular_classification"],
        )
        viable_techniques: list[str] = []
        for tech in all_techniques:
            past_res = journal.results_for_technique(tech)
            v_dec = assess_technique(self.jev, tech, task.domain, past_res)
            if v_dec.viable:
                viable_techniques.append(tech)

        if not viable_techniques:
            viable_techniques = all_techniques[:2]

        # Step 3: LLM generates candidate code
        best_rec = journal.best_record(task.direction)
        best_code = best_rec.code if best_rec else None
        best_metric = best_rec.actual_metric if best_rec and best_rec.actual_metric is not None else 0.0

        candidates = self.llm.generate_candidates(
            task_domain=task.domain,
            goal=task.goal,
            metric_name=task.metric,
            strategy=strategy,
            viable_techniques=viable_techniques,
            journal_history=[r.model_dump() for r in journal.records],
            best_code=best_code,
            best_metric=best_metric,
            n_candidates=n_candidates,
        )

        return StrategyPlan(
            strategy=strategy,
            viable_techniques=viable_techniques,
            candidates=candidates,
        )
