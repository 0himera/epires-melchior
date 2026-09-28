"""Melchior Crucible Canary Run & Baseline Benchmark."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx

from melchior.crucible.environments import generate_task, TaskProfile
from melchior.crucible.client import CrucibleLLMClient, CandidatePair
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.arbiter import CrucibleArbiter


@dataclass
class CanaryEvaluation:
    seed: int
    task_id: str
    task_type: str
    metric_name: str
    description: str
    cand_a_status: str
    cand_a_metric: float | None
    cand_a_time: float
    cand_b_status: str
    cand_b_metric: float | None
    cand_b_time: float
    winner: str
    delta: float
    is_decisive: bool
    hypothesis_a: str
    hypothesis_b: str
    code_a: str
    code_b: str
    operator: str = "unknown"
    cand_a_stderr: str = ""
    cand_b_stderr: str = ""
    jev_choice: str | None = None
    jev_confidence: float | None = None
    jev_correct: bool | None = None


class CanaryRunner:
    def __init__(
        self,
        num_tasks: int = 50,
        concurrency: int = 10,
        llm_url: str = "http://localhost:8000/v1",
        llm_model: str = "qwen",
        jev_url: str = "http://localhost:8080/v1/systemone",
        output_dir: str | Path = "data/crucible_canary",
        sandbox_timeout_s: float = 14.0,
        min_delta: float = 0.01,
    ):
        if num_tasks < 0 or concurrency < 1:
            raise ValueError("num_tasks must be non-negative and concurrency positive")
        self.num_tasks = num_tasks
        self.concurrency = concurrency
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.jev_url = jev_url
        self.output_dir = Path(output_dir)
        self.sandbox_timeout_s = sandbox_timeout_s
        self.min_delta = min_delta

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_file = self.output_dir / "holdout_validation_50.jsonl"
        self.summary_file = self.output_dir / "canary_summary.json"
        self.evaluations_file = self.output_dir / "canary_evaluations.jsonl"
        self.errors_file = self.output_dir / "canary_errors.jsonl"

        self.arbiter = CrucibleArbiter(min_delta=self.min_delta)
        self.sandbox = AsyncSandbox(timeout_s=self.sandbox_timeout_s)
        self.client = CrucibleLLMClient(
            base_url=self.llm_url,
            model=self.llm_model,
            mode="api",
            timeout_s=45.0,
        )
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.evaluations: list[CanaryEvaluation] = []
        self.errors: list[dict] = []
        self._start_time = 0.0

    async def _query_openjev(
        self, profile: TaskProfile, pair: CandidatePair
    ) -> tuple[str, float] | None:
        """Asks OpenJev System 1 to choose which candidate hypothesis is superior."""
        premise = f"Task: {profile.description} Objective: Maximize {profile.metric}."
        payload = {
            "state": premise,
            "questions": {
                "winner": {
                    "type": "choice",
                    "instructions": "Which candidate model will achieve higher metric on held-out validation data?",
                    "criteria": {
                        "A": pair.hypothesis_a,
                        "B": pair.hypothesis_b,
                    },
                }
            },
        }
        try:
            resp = await self.http_client.post(self.jev_url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answers", {}).get("winner", {})
                choice = str(ans.get("choice", "")).strip().upper()
                conf = float(ans.get("confidence", 0.5))
                return choice, conf
        except Exception:
            pass
        return None

    async def _process_task(self, seed: int) -> CanaryEvaluation:
        # 1. Environment task & split
        profile, X_tr, y_tr, X_va, y_va = generate_task(seed)

        # 2. Candidate generation from LLM
        pair = await self.client.generate_pair(profile, seed)

        # 3. Dual sandbox execution
        res_a, res_b = await asyncio.gather(
            self.sandbox.execute(pair.code_a, X_tr, y_tr, X_va, y_va),
            self.sandbox.execute(pair.code_b, X_tr, y_tr, X_va, y_va),
        )

        # 4. Objective evaluation
        outcome = self.arbiter.evaluate(profile, pair, res_a, res_b)
        is_decisive = outcome.winner in ["A", "B"] and outcome.delta >= self.min_delta

        jev_choice = None
        jev_conf = None
        jev_correct = None

        # 5. OpenJev Baseline Scoring on decisive pairs
        if is_decisive:
            jev_res = await self._query_openjev(profile, pair)
            if jev_res:
                jev_choice, jev_conf = jev_res
                jev_correct = (jev_choice == outcome.winner)

        return CanaryEvaluation(
            seed=seed,
            task_id=profile.task_id,
            task_type=profile.task_type,
            metric_name=profile.metric,
            description=profile.description,
            cand_a_status=res_a.status,
            cand_a_metric=res_a.metric,
            cand_a_time=res_a.wall_time_s,
            cand_b_status=res_b.status,
            cand_b_metric=res_b.metric,
            cand_b_time=res_b.wall_time_s,
            winner=outcome.winner,
            delta=outcome.delta,
            is_decisive=is_decisive,
            hypothesis_a=pair.hypothesis_a,
            hypothesis_b=pair.hypothesis_b,
            code_a=pair.code_a,
            code_b=pair.code_b,
            operator=pair.operator,
            cand_a_stderr=res_a.stderr,
            cand_b_stderr=res_b.stderr,
            jev_choice=jev_choice,
            jev_confidence=jev_conf,
            jev_correct=jev_correct,
        )

    @staticmethod
    def _append(path: Path, record: dict) -> None:
        # Closing after every row preserves completed work even if the process
        # subsequently crashes; no await occurs between recording and checkpointing.
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_summary(self, status: str) -> dict:
        total = len(self.evaluations)
        candidates = total * 2
        successful = sum(
            e.cand_a_status == "success" for e in self.evaluations
        ) + sum(e.cand_b_status == "success" for e in self.evaluations)
        timeouts = sum(
            e.cand_a_status == "timeout" for e in self.evaluations
        ) + sum(e.cand_b_status == "timeout" for e in self.evaluations)
        decisive = [e for e in self.evaluations if e.is_decisive]
        judged = [e for e in decisive if e.jev_correct is not None]
        correct = sum(e.jev_correct is True for e in judged)

        def percent(n, d):
            return round(100 * n / d, 2) if d else 0.0

        summary = {
            "status": status,
            "requested_tasks": self.num_tasks,
            "completed_tasks": total + len(self.errors),
            "failed_tasks": len(self.errors),
            "total_tasks": total,
            "total_candidates": candidates,
            "elapsed_seconds": round(time.monotonic() - self._start_time, 2),
            "pass_rate_pct": percent(successful, candidates),
            "timeout_rate_pct": percent(timeouts, candidates),
            "avg_wall_time_s": round(sum(
                e.cand_a_time + e.cand_b_time for e in self.evaluations
            ) / candidates, 2) if candidates else 0.0,
            "decisive_rate_pct": percent(len(decisive), total),
            "tie_rate_pct": percent(sum(e.winner == "TIE" for e in self.evaluations), total),
            "both_failed_rate_pct": percent(
                sum(e.winner == "BOTH_FAILED" for e in self.evaluations), total,
            ),
            "avg_winning_margin": round(sum(e.delta for e in decisive) / len(decisive), 4)
            if decisive else 0.0,
            "openjev_baseline_tested": len(judged),
            "openjev_baseline_correct": correct,
            "openjev_baseline_accuracy_pct": percent(correct, len(judged)),
            "holdout_file": str(self.holdout_file),
            "evaluations_file": str(self.evaluations_file),
            "errors_file": str(self.errors_file),
        }
        temporary = self.summary_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(self.summary_file)
        return summary

    async def run(self) -> dict:
        print(f"[*] Canary: {self.num_tasks} tasks, {self.concurrency} workers")
        print(f"[*] LLM: {self.llm_url}; Jev: {self.jev_url}")
        self._start_time = time.monotonic()
        self.evaluations.clear()
        self.errors.clear()
        tasks = []
        status = "failed"
        sem = asyncio.Semaphore(self.concurrency)

        async def evaluate(seed: int):
            async with sem:
                try:
                    result = await self._process_task(seed)
                except Exception as exc:
                    error = {
                        "seed": seed, "task_id": f"crucible_task_{seed:06d}",
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
                    self._append(self.errors_file, error)
                    self.errors.append(error)
                    print(f"[!] {error['task_id']}: {error['error_type']}: {exc}", flush=True)
                else:
                    record = asdict(result)
                    self._append(self.evaluations_file, record)
                    if result.is_decisive:
                        self._append(self.holdout_file, record)
                    self.evaluations.append(result)
                    print(f"[+] {result.task_id}: {result.winner} (Δ={result.delta:.3f})", flush=True)
                self._write_summary("running")

        try:
            for path in (self.evaluations_file, self.errors_file, self.holdout_file):
                path.write_text("", encoding="utf-8")
            self._write_summary("running")
            tasks = [asyncio.create_task(evaluate(seed))
                     for seed in range(1001, 1001 + self.num_tasks)]
            await asyncio.gather(*tasks)
            status = "completed_with_errors" if self.errors else "completed"
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.gather(self.client.close(), self.http_client.aclose())
            finally:
                summary = self._write_summary(status)
        print(f"[*] Canary {status}: {summary['total_tasks']} evaluated, "
              f"{summary['failed_tasks']} errors. Saved {self.summary_file}")
        return summary
