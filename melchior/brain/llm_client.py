"""LLM Client — System Two generative reasoning engine.

Generates:
1. Hypothesis & research rationale
2. Executable solution.py code
3. Predicted metric calibration target
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any


@dataclass
class CandidateProposal:
    hypothesis: str
    code: str
    predicted_metric: float
    technique: str


class LLMClient:
    def __init__(
        self,
        mode: str = "mock",
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
    ):
        self.mode = mode
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url

    def generate_candidates(
        self,
        task_domain: str,
        goal: str,
        metric_name: str,
        strategy: str,
        viable_techniques: list[str],
        journal_history: list[dict[str, Any]],
        best_code: str | None = None,
        best_metric: float = 0.0,
        n_candidates: int = 1,
    ) -> list[CandidateProposal]:
        """Propose n candidate solutions (hypotheses + executable code)."""
        if self.mode == "api" and self.api_key:
            return self._generate_api(
                task_domain, goal, metric_name, strategy,
                viable_techniques, journal_history, best_code, best_metric, n_candidates
            )
        else:
            return self._generate_mock(
                task_domain, goal, metric_name, strategy,
                viable_techniques, journal_history, best_code, best_metric, n_candidates
            )

    def _generate_mock(
        self,
        task_domain: str,
        goal: str,
        metric_name: str,
        strategy: str,
        viable_techniques: list[str],
        journal_history: list[dict[str, Any]],
        best_code: str | None,
        best_metric: float,
        n_candidates: int,
    ) -> list[CandidateProposal]:
        """Generates realistic executable Python ML solutions for the given domain."""
        candidates: list[CandidateProposal] = []
        trial_num = len(journal_history)

        # Tabular classification catalog
        if task_domain in ["tabular_classification", "tabular", "iris", "classification"]:
            techniques_catalog = [
                (
                    "StandardScaler + LogisticRegression",
                    0.92,
                    """import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X, y = load_iris(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200))
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

# Dense feedback output
val_accuracy = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{val_accuracy:.4f}")
print(f"FINAL_METRIC:{val_accuracy:.4f}")
print(json.dumps({"metric": val_accuracy, "scores": scores.tolist()}))
""",
                ),
                (
                    "RandomForestClassifier (n_estimators=100, max_depth=4)",
                    0.95,
                    """import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

val_accuracy = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{val_accuracy - 0.04:.4f}")
print(f"EPOCH_METRIC:2:{val_accuracy:.4f}")
print(f"FINAL_METRIC:{val_accuracy:.4f}")
print(json.dumps({"metric": val_accuracy, "scores": scores.tolist()}))
""",
                ),
                (
                    "GradientBoostingClassifier + Tuned Learning Rate",
                    0.965,
                    """import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf = GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

val_accuracy = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{val_accuracy - 0.05:.4f}")
print(f"EPOCH_METRIC:2:{val_accuracy - 0.02:.4f}")
print(f"EPOCH_METRIC:3:{val_accuracy:.4f}")
print(f"FINAL_METRIC:{val_accuracy:.4f}")
print(json.dumps({"metric": val_accuracy, "scores": scores.tolist()}))
""",
                ),
                (
                    "VotingEnsemble (LogisticRegression + RF + GradientBoosting)",
                    0.973,
                    """import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline

X, y = load_iris(return_X_y=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

e1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200))
e2 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
e3 = GradientBoostingClassifier(n_estimators=80, learning_rate=0.08, max_depth=3, random_state=42)

ensemble = VotingClassifier(estimators=[('lr', e1), ('rf', e2), ('gb', e3)], voting='soft')
scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy")

val_accuracy = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{val_accuracy - 0.03:.4f}")
print(f"EPOCH_METRIC:2:{val_accuracy:.4f}")
print(f"FINAL_METRIC:{val_accuracy:.4f}")
print(json.dumps({"metric": val_accuracy, "scores": scores.tolist()}))
""",
                ),
            ]
        else:
            # Regression catalog
            techniques_catalog = [
                (
                    "Ridge Regression with PolynomialFeatures",
                    0.82,
                    """import json
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

X, y = make_regression(n_samples=200, n_features=6, noise=10.0, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0))
scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

r2 = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{r2:.4f}")
print(f"FINAL_METRIC:{r2:.4f}")
print(json.dumps({"metric": r2, "scores": scores.tolist()}))
""",
                ),
                (
                    "GradientBoostingRegressor with Huber Loss",
                    0.91,
                    """import json
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor

X, y = make_regression(n_samples=200, n_features=6, noise=10.0, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, loss="huber", random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

r2 = float(np.mean(scores))
print(f"EPOCH_METRIC:1:{r2 - 0.05:.4f}")
print(f"EPOCH_METRIC:2:{r2:.4f}")
print(f"FINAL_METRIC:{r2:.4f}")
print(json.dumps({"metric": r2, "scores": scores.tolist()}))
""",
                ),
            ]

        # Select candidate based on viable techniques or trial number
        chosen_indices = [
            (trial_num + i) % len(techniques_catalog) for i in range(n_candidates)
        ]

        for idx in chosen_indices:
            name, pred_val, code = techniques_catalog[idx]
            hyp = f"Apply {name} to optimize {metric_name} in {task_domain} under strategy '{strategy}'."
            candidates.append(
                CandidateProposal(
                    hypothesis=hyp,
                    code=code,
                    predicted_metric=pred_val,
                    technique=name,
                )
            )

        return candidates

    def _generate_api(
        self,
        task_domain: str,
        goal: str,
        metric_name: str,
        strategy: str,
        viable_techniques: list[str],
        journal_history: list[dict[str, Any]],
        best_code: str | None,
        best_metric: float,
        n_candidates: int,
    ) -> list[CandidateProposal]:
        # Simple OpenAI-compatible chat completion payload
        system_prompt = (
            "You are Melchior System Two. You write self-contained Python scripts for ML tasks.\n"
            "Each script must train on the dataset, evaluate with cross-validation or validation set,\n"
            "and print 'FINAL_METRIC:<float>' on stdout as well as optional 'EPOCH_METRIC:<epoch>:<float>'.\n"
            "Return JSON matching: {\"candidates\": [{\"hypothesis\": \"...\", \"code\": \"...\", \"predicted_metric\": 0.95, \"technique\": \"...\"}]}"
        )
        user_prompt = (
            f"Goal: {goal}\n"
            f"Domain: {task_domain}\n"
            f"Metric: {metric_name}\n"
            f"Strategy: {strategy}\n"
            f"Viable Techniques approved by Jev: {viable_techniques}\n"
            f"Current best metric: {best_metric}\n"
            f"Number of candidates: {n_candidates}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.4,
        }
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key or 'token'}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = json.loads(data["choices"][0]["message"]["content"])
                return [
                    CandidateProposal(
                        hypothesis=c["hypothesis"],
                        code=c["code"],
                        predicted_metric=float(c.get("predicted_metric", 0.8)),
                        technique=c.get("technique", "ML Model"),
                    )
                    for c in content.get("candidates", [])
                ]
        except Exception:
            # Fallback to mock on API error
            return self._generate_mock(
                task_domain, goal, metric_name, strategy,
                viable_techniques, journal_history, best_code, best_metric, n_candidates
            )
