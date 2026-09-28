"""Async client for generating competing ML solutions via Qwen 27B / vLLM."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
import httpx
from melchior.crucible.environments import TaskProfile
from melchior.crucible.prompts import (
    OPERATOR_KEYS,
    get_prompt_for_operator,
    select_operator_by_seed,
)


@dataclass
class CandidatePair:
    hypothesis_a: str
    code_a: str
    hypothesis_b: str
    code_b: str
    operator: str = "inductive_bias"


def _parse_candidate_json(raw_text: str, operator: str = "default") -> CandidatePair:
    """Parses raw model output into CandidatePair with fallback extraction."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
    m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if m_block:
        raw_text = m_block.group(1).strip()

    try:
        content = json.loads(raw_text)
    except Exception:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = json.loads(raw_text[start : end + 1])
        else:
            raise ValueError(f"Failed to find JSON object in LLM output: {raw_text[:200]}")

    cand_a = content.get("candidate_a") or content.get("Candidate_A") or {}
    cand_b = content.get("candidate_b") or content.get("Candidate_B") or {}

    hyp_a = cand_a.get("hypothesis", "")
    code_a = cand_a.get("code", "")
    hyp_b = cand_b.get("hypothesis", "")
    code_b = cand_b.get("code", "")

    if not code_a or not code_b:
        raise ValueError("Missing code_a or code_b in LLM response")

    return CandidatePair(
        hypothesis_a=hyp_a,
        code_a=code_a,
        hypothesis_b=hyp_b,
        code_b=code_b,
        operator=operator,
    )


import asyncio
import shutil
import re


class CrucibleLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen",
        mode: str = "auto",  # auto | api | opencode | mock
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

    async def generate_pair(
        self, profile: TaskProfile, seed: int, operator: str | None = None
    ) -> CandidatePair:
        """Asynchronously requests two competing solutions from the LLM endpoint, OpenCode, or mock."""
        selected_op = operator or select_operator_by_seed(seed)

        if self.mode == "opencode":
            try:
                return await self._call_opencode(profile, selected_op)
            except Exception:
                return self._generate_mock(profile, seed, selected_op)
        elif self.mode == "mock":
            return self._generate_mock(profile, seed, selected_op)
        elif self.mode == "api":
            try:
                return await self._call_vllm(profile, selected_op)
            except Exception:
                return self._generate_mock(profile, seed, selected_op)

        # auto mode
        try:
            return await self._call_vllm(profile, selected_op)
        except Exception:
            if shutil.which("opencode"):
                try:
                    return await self._call_opencode(profile, selected_op)
                except Exception:
                    pass
            return self._generate_mock(profile, seed, selected_op)

    async def _call_opencode(
        self, profile: TaskProfile, operator: str = "inductive_bias"
    ) -> CandidatePair:
        """Generates candidate pair via OpenCode CLI."""
        from melchior.config import find_opencode_bin

        user_prompt = (
            f"Dataset Profile:\n"
            f"- Description: {profile.description}\n"
            f"- Task Type: {profile.task_type}\n"
            f"- Target Metric to maximize: {profile.metric}\n"
            f"- Features: {profile.n_features}, Train samples: {profile.n_train}, Validation samples: {profile.n_val}\n\n"
            f"Apply the research operator guidelines above to formulate Candidate A and Candidate B.\n"
            f"Output strictly valid raw JSON with keys 'candidate_a' and 'candidate_b', without markdown code fences."
        )
        system_prompt = get_prompt_for_operator(operator)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        opencode_bin = find_opencode_bin() or "opencode"

        cmd = [opencode_bin, "run", "--pure", "--format", "json"]
        if self.model and self.model not in ["qwen", "auto", "default", "mock"]:
            cmd.extend(["-m", self.model])
        cmd.append(full_prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        text_parts = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "text" and "part" in data and "text" in data["part"]:
                    text_parts.append(data["part"]["text"])
            except Exception:
                pass

        full_text = "".join(text_parts)
        return _parse_candidate_json(full_text, operator=operator)

    async def _call_vllm(
        self, profile: TaskProfile, operator: str = "inductive_bias"
    ) -> CandidatePair:
        model_name = await self._resolve_model()
        user_prompt = (
            f"Dataset Profile:\n"
            f"- Description: {profile.description}\n"
            f"- Task Type: {profile.task_type}\n"
            f"- Target Metric to maximize: {profile.metric}\n"
            f"- Features: {profile.n_features}, Train samples: {profile.n_train}, Validation samples: {profile.n_val}\n\n"
            f"Apply the research operator guidelines to formulate Candidate A and Candidate B."
        )
        system_prompt = get_prompt_for_operator(operator)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        resp = await self._http_client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        raw_content = data["choices"][0]["message"]["content"].strip()
        return _parse_candidate_json(raw_content, operator=operator)

    def _generate_mock(self, profile: TaskProfile, seed: int, operator: str = "inductive_bias") -> CandidatePair:
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
            operator=operator,
        )

    async def close(self):
        await self._http_client.aclose()
