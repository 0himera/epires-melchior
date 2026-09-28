"""Async client for generating competing ML solutions via Qwen 27B / vLLM."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
import httpx
from melchior.crucible.environments import TaskProfile


@dataclass
class CandidatePair:
    hypothesis_a: str
    code_a: str
    hypothesis_b: str
    code_b: str


SYSTEM_PROMPT = """You are an automated ML researcher. Given a dataset profile and target metric, propose TWO competing solutions (Candidate A and Candidate B) exploring different modeling approaches.
Your output MUST be valid JSON with this exact structure:
{
  "candidate_a": {
    "hypothesis": "<1-2 sentences on why approach A should work>",
    "code": "<executable python code>"
  },
  "candidate_b": {
    "hypothesis": "<1-2 sentences on why approach B should work>",
    "code": "<executable python code>"
  }
}

Code Requirements:
1. Load data using:
   import numpy as np
   data = np.load("data.npz")
   X_train, y_train, X_val, y_val = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]
2. Train model on X_train, y_train and evaluate on X_val, y_val.
3. Compute the specified metric and print:
   print(f"METRIC:{metric_value:.4f}")
4. Self-contained (use scikit-learn, numpy, scipy, pandas). No network calls.
"""


class CrucibleLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen",
        mode: str = "auto",  # auto | api | mock
        timeout_s: float = 35.0,
    ):
        self.base_url = base_url or os.getenv("CRUCIBLE_LLM_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("CRUCIBLE_LLM_MODEL", "qwen")
        self.timeout_s = timeout_s
        self.mode = mode
        self._http_client = httpx.AsyncClient(timeout=timeout_s)
        self._model_resolved = False

    async def _resolve_model(self) -> str:
        """Auto-discovers the model ID loaded in vLLM if model is 'qwen' or 'auto'."""
        if self._model_resolved:
            return self.model
        if self.model and self.model not in ["qwen", "auto"]:
            self._model_resolved = True
            return self.model

        try:
            resp = await self._http_client.get(f"{self.base_url.rstrip('/')}/models")
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    self.model = data["data"][0]["id"]
                    self._model_resolved = True
                    return self.model
        except Exception:
            pass

        self._model_resolved = True
        return self.model

    async def generate_pair(self, profile: TaskProfile, seed: int) -> CandidatePair:
        """Asynchronously requests two competing solutions from the LLM endpoint or mock generator."""
        if self.mode == "mock":
            return self._generate_mock(profile, seed)

        try:
            return await self._call_vllm(profile)
        except Exception as e:
            if self.mode == "api":
                raise RuntimeError(f"vLLM API call failed: {e}") from e
            # Graceful fallback to generator in auto mode
            return self._generate_mock(profile, seed)

    async def _call_vllm(self, profile: TaskProfile) -> CandidatePair:
        model_name = await self._resolve_model()
        user_prompt = (
            f"Task: {profile.description}\n"
            f"Type: {profile.task_type}\n"
            f"Target Metric to maximize: {profile.metric}\n"
            f"Features: {profile.n_features}, Train rows: {profile.n_train}, Val rows: {profile.n_val}\n"
            f"Propose Candidate A and Candidate B with contrasting architectures or preprocessing."
        )

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        resp = await self._http_client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        raw_content = data["choices"][0]["message"]["content"].strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.MULTILINE).strip()
        content = json.loads(raw_content)

        return CandidatePair(
            hypothesis_a=content["candidate_a"]["hypothesis"],
            code_a=content["candidate_a"]["code"],
            hypothesis_b=content["candidate_b"]["hypothesis"],
            code_b=content["candidate_b"]["code"],
        )

    def _generate_mock(self, profile: TaskProfile, seed: int) -> CandidatePair:
        """High-variety offline code synthesizer for testing and local generation."""
        rng = random.Random(seed)

        if profile.task_type in ["binary_classification", "multiclass_classification"]:
            models = [
                (
                    "StandardScaler + LogisticRegression (regularized)",
                    """import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import {metric_func}

data = np.load("data.npz")
X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

clf = LogisticRegression(C=0.8, max_iter=250, random_state=42)
clf.fit(X_tr_s, y_tr)
preds = clf.predict_proba(X_va_s)[:, 1] if "{metric}" in ["roc_auc"] else clf.predict(X_va_s)
score = {metric_eval}
print(f"METRIC:{{score:.4f}}")
""",
                ),
                (
                    "RandomForestClassifier (n_estimators=100, max_features='sqrt')",
                    """import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import {metric_func}

data = np.load("data.npz")
X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

clf = RandomForestClassifier(n_estimators=100, max_depth=8, max_features="sqrt", random_state=42)
clf.fit(X_tr, y_tr)
preds = clf.predict_proba(X_va)[:, 1] if "{metric}" in ["roc_auc"] else clf.predict(X_va)
score = {metric_eval}
print(f"METRIC:{{score:.4f}}")
""",
                ),
                (
                    "HistGradientBoostingClassifier + Early Stopping",
                    """import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import {metric_func}

data = np.load("data.npz")
X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42)
clf.fit(X_tr, y_tr)
preds = clf.predict_proba(X_va)[:, 1] if "{metric}" in ["roc_auc"] else clf.predict(X_va)
score = {metric_eval}
print(f"METRIC:{{score:.4f}}")
""",
                ),
            ]

            if profile.metric == "roc_auc":
                metric_func = "roc_auc_score"
                metric_eval = "roc_auc_score(y_va, preds)"
            elif profile.metric == "f1":
                metric_func = "f1_score"
                metric_eval = "f1_score(y_va, preds, average='weighted')"
            else:
                metric_func = "accuracy_score"
                metric_eval = "accuracy_score(y_va, preds)"

        else:
            models = [
                (
                    "StandardScaler + Ridge Regression",
                    """import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

data = np.load("data.npz")
X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

model = Ridge(alpha=1.5)
model.fit(X_tr_s, y_tr)
preds = model.predict(X_va_s)
score = r2_score(y_va, preds)
print(f"METRIC:{{score:.4f}}")
""",
                ),
                (
                    "HistGradientBoostingRegressor with L2 Regularization",
                    """import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

data = np.load("data.npz")
X_tr, y_tr, X_va, y_va = data["X_tr"], data["y_tr"], data["X_va"], data["y_va"]

model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, l2_regularization=0.5, random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_va)
score = r2_score(y_va, preds)
print(f"METRIC:{{score:.4f}}")
""",
                ),
            ]
            metric_func = "r2_score"
            metric_eval = "r2_score(y_va, preds)"

        chosen = rng.sample(models, 2)
        cand_a_name, cand_a_tmpl = chosen[0]
        cand_b_name, cand_b_tmpl = chosen[1]

        code_a = cand_a_tmpl.format(metric_func=metric_func, metric=profile.metric, metric_eval=metric_eval)
        code_b = cand_b_tmpl.format(metric_func=metric_func, metric=profile.metric, metric_eval=metric_eval)

        return CandidatePair(
            hypothesis_a=f"Using {cand_a_name} will maximize {profile.metric} on {profile.task_type}.",
            code_a=code_a,
            hypothesis_b=f"Using {cand_b_name} will provide better generalization for {profile.metric}.",
            code_b=code_b,
        )

    async def close(self):
        await self._http_client.aclose()
