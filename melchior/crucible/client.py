"""Async client for generating competing ML solutions via Qwen 27B / vLLM."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
import httpx
from melchior.config import find_opencode_bin
from melchior.crucible.process import run_process
from melchior.crucible.environments import TaskProfile
from melchior.crucible.prompts import (
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
    generation: dict = field(default_factory=dict)


def _parse_candidate_json(raw_text: str, operator: str = "default") -> CandidatePair:
    """Parses raw model output into CandidatePair with fallback extraction."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("LLM response content must be a non-empty string")
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()
    m_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if m_block:
        raw_text = m_block.group(1).strip()

    try:
        content = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = json.loads(raw_text[start : end + 1])
        else:
            raise ValueError(f"Failed to find JSON object in LLM output: {raw_text[:200]}")

    if not isinstance(content, dict):
        raise ValueError("LLM response must be a JSON object")
    candidates = []
    for key, alias in [("candidate_a", "Candidate_A"), ("candidate_b", "Candidate_B")]:
        candidate = content.get(key, content.get(alias))
        if not isinstance(candidate, dict):
            raise ValueError(f"Missing or invalid {key} in LLM response")
        for field in ("hypothesis", "code"):
            if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                raise ValueError(f"{key}.{field} must be a non-empty string")
        candidates.append(candidate)
    cand_a, cand_b = candidates

    return CandidatePair(
        hypothesis_a=cand_a["hypothesis"],
        code_a=cand_a["code"],
        hypothesis_b=cand_b["hypothesis"],
        code_b=cand_b["code"],
        operator=operator,
    )


class CrucibleLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen",
        mode: str = "auto",  # auto | api | opencode | mock
        timeout_s: float = 35.0,
        max_attempts: int = 2,
        reasoning_effort: str = "xhigh",
        max_tokens: int = 12288,
    ):
        if mode not in {"auto", "api", "opencode", "mock"}:
            raise ValueError(f"Unknown LLM mode: {mode}")
        if max_attempts < 1 or timeout_s <= 0:
            raise ValueError("max_attempts and timeout_s must be positive")
        if reasoning_effort not in {"low", "medium", "xhigh"} or max_tokens < 1:
            raise ValueError("Invalid reasoning effort or token budget")
        self.reasoning_effort, self.max_tokens = reasoning_effort, max_tokens
        self.base_url = base_url or os.getenv("CRUCIBLE_LLM_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("CRUCIBLE_LLM_MODEL", "qwen")
        self.timeout_s = timeout_s
        self.mode = mode
        self.max_attempts = max_attempts
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

        if self.mode == "mock":
            return self._generate_mock(profile, seed, selected_op)

        backends = [self.mode] if self.mode != "auto" else ["api"]
        if self.mode == "auto" and find_opencode_bin():
            backends.append("opencode")
        errors = []
        attempts = []
        for backend in backends:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    call = self._call_opencode if backend == "opencode" else self._call_vllm
                    return await asyncio.wait_for(call(profile, selected_op), self.timeout_s)
                except Exception as exc:
                    errors.append(f"{backend} attempt {attempt}: {type(exc).__name__}: {exc}")
                    attempts.append({"backend": backend, "attempt": attempt, "error": str(exc),
                                     "response_text": getattr(exc, "response_text", None)})
                    last_error = exc
        failure = RuntimeError("Candidate generation failed; " + "; ".join(errors))
        failure.attempts = attempts
        raise failure from last_error

    async def _call_opencode(
        self, profile: TaskProfile, operator: str = "inductive_bias"
    ) -> CandidatePair:
        """Generates candidate pair via OpenCode CLI."""
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

        result = await run_process(cmd, timeout_s=self.timeout_s)
        if result.returncode != 0:
            raise RuntimeError(f"OpenCode exited with code {result.returncode}: "
                               f"{result.stderr.decode(errors='replace')[-500:]}")
        stdout = result.stdout
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
            # This vLLM build fails its grammar FSM with reasoning + MTP.
            # Request ordinary final text; validate the complete pair below.
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": self.reasoning_effort},
            "reasoning_effort": self.reasoning_effort,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "max_tokens": self.max_tokens,
        }

        resp = await self._http_client.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        raw_content = choice["message"]["content"]
        try:
            if choice.get("finish_reason") == "length":
                raise ValueError("LLM response was truncated by the token limit")
            pair = _parse_candidate_json(raw_content, operator=operator)
            pair.generation = {
                "reasoning": choice["message"].get("reasoning") or choice["message"].get("reasoning_content"),
                "reasoning_effort": self.reasoning_effort, "enable_thinking": True,
                "usage": data.get("usage"), "finish_reason": choice.get("finish_reason"),
                "max_tokens": self.max_tokens, "model": data.get("model", model_name),
            }
            return pair
        except ValueError as exc:
            exc.response_text = raw_content
            raise

    def _generate_mock(self, profile: TaskProfile, seed: int, operator: str = "inductive_bias") -> CandidatePair:
        """Deterministic executable fixtures; mock output is never a research result."""
        classification = profile.task_type != "regression"
        linear = "LogisticRegression(max_iter=200, random_state=42)" if classification else "Ridge(alpha=1.5)"
        tree = "HistGradientBoostingClassifier" if classification else "HistGradientBoostingRegressor"
        expressions = [f"make_pipeline(StandardScaler(), {linear})", f"{tree}(max_iter=60, random_state=42)"]
        random.Random(seed).shuffle(expressions)
        prediction = "model.predict_proba(X_test)[:, 1]" if profile.metric == "roc_auc" else "model.predict(X_test)"
        def code(expression):
            return ("from sklearn.pipeline import make_pipeline\n"
                    "from sklearn.preprocessing import StandardScaler\n"
                    "from sklearn.linear_model import LogisticRegression, Ridge\n"
                    "from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor\n"
                    "def fit_predict(X_train, y_train, X_test):\n"
                    f"    model = {expression}\n"
                    "    model.fit(X_train, y_train)\n"
                    f"    return {prediction}\n")
        return CandidatePair(expressions[0], code(expressions[0]), expressions[1], code(expressions[1]), operator)

    async def close(self):
        await self._http_client.aclose()
