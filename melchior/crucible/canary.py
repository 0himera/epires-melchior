"""Melchior Crucible Canary Run & Baseline Benchmark."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import httpx

from melchior.crucible.environments import generate_task, TaskProfile
from melchior.crucible.client import CrucibleLLMClient, CandidatePair
from melchior.crucible.sandbox import AsyncSandbox, SandboxResult
from melchior.crucible.arbiter import CrucibleArbiter, ArbiterOutcome


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
            jev_choice=jev_choice,
            jev_confidence=jev_conf,
            jev_correct=jev_correct,
        )

    async def run(self):
        print("=" * 70)
        print("          MELCHIOR CRUCIBLE CANARY RUN & BASELINE BENCHMARK")
        print("=" * 70)
        print(f"[*] Tasks to evaluate:    {self.num_tasks}")
        print(f"[*] Concurrency workers:  {self.concurrency}")
        print(f"[*] LLM Endpoint (Sys 2): {self.llm_url} (model: {self.llm_model})")
        print(f"[*] Jev Endpoint (Sys 1): {self.jev_url}")
        print(f"[*] Sandbox Timeout:      {self.sandbox_timeout_s}s")
        print(f"[*] Output directory:     {self.output_dir.resolve()}\n")

        start_time = time.time()
        sem = asyncio.Semaphore(self.concurrency)

        async def _bounded_task(seed: int, idx: int):
            async with sem:
                res = await self._process_task(seed)
                status_icon = "✓" if res.is_decisive else "•"
                winner_str = f"Winner: {res.winner} (Δ={res.delta:.3f})" if res.is_decisive else f"Result: {res.winner}"
                jev_str = f"| Jev: {res.jev_choice} (corr={res.jev_correct})" if res.jev_choice else ""
                print(f"[{idx+1:02d}/{self.num_tasks:02d}] {status_icon} Task {res.task_id} | {winner_str} {jev_str}")
                return res

        tasks = [_bounded_task(seed, i) for i, seed in enumerate(range(1001, 1001 + self.num_tasks))]
        self.evaluations = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time
        await self.client.close()
        await self.http_client.aclose()

        # ==================== ANALYSIS & REPORT ====================
        total_tasks = len(self.evaluations)
        total_candidates = total_tasks * 2

        # 1. Pass Rate
        success_a = sum(1 for e in self.evaluations if e.cand_a_status == "success")
        success_b = sum(1 for e in self.evaluations if e.cand_b_status == "success")
        total_success = success_a + success_b
        pass_rate = (total_success / total_candidates) * 100.0

        timeouts_a = sum(1 for e in self.evaluations if e.cand_a_status == "timeout")
        timeouts_b = sum(1 for e in self.evaluations if e.cand_b_status == "timeout")
        total_timeouts = timeouts_a + timeouts_b
        timeout_rate = (total_timeouts / total_candidates) * 100.0

        all_times = [e.cand_a_time for e in self.evaluations] + [e.cand_b_time for e in self.evaluations]
        avg_time = sum(all_times) / len(all_times) if all_times else 0.0

        # 2. Decisive Rate & Margins
        decisive_evals = [e for e in self.evaluations if e.is_decisive]
        decisive_count = len(decisive_evals)
        decisive_rate = (decisive_count / total_tasks) * 100.0

        ties = [e for e in self.evaluations if e.winner == "TIE"]
        both_failed = [e for e in self.evaluations if e.winner == "BOTH_FAILED"]
        tie_rate = (len(ties) / total_tasks) * 100.0
        failed_rate = (len(both_failed) / total_tasks) * 100.0

        avg_margin = (
            sum(e.delta for e in decisive_evals) / decisive_count if decisive_count > 0 else 0.0
        )

        # 3. OpenJev Baseline Accuracy
        evaluated_jev = [e for e in decisive_evals if e.jev_correct is not None]
        correct_jev = sum(1 for e in evaluated_jev if e.jev_correct is True)
        jev_accuracy = (correct_jev / len(evaluated_jev) * 100.0) if evaluated_jev else 0.0

        # Save Holdout Set
        with open(self.holdout_file, "w", encoding="utf-8") as f:
            for e in decisive_evals:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

        summary = {
            "total_tasks": total_tasks,
            "elapsed_seconds": round(elapsed, 2),
            "pass_rate_pct": round(pass_rate, 2),
            "timeout_rate_pct": round(timeout_rate, 2),
            "avg_wall_time_s": round(avg_time, 2),
            "decisive_rate_pct": round(decisive_rate, 2),
            "tie_rate_pct": round(tie_rate, 2),
            "both_failed_rate_pct": round(failed_rate, 2),
            "avg_winning_margin": round(avg_margin, 4),
            "openjev_baseline_tested": len(evaluated_jev),
            "openjev_baseline_correct": correct_jev,
            "openjev_baseline_accuracy_pct": round(jev_accuracy, 2),
            "holdout_file": str(self.holdout_file),
        }
        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("                   CANARY RUN CALIBRATION RESULTS")
        print("=" * 70)
        print(f"Elapsed Time:                {elapsed:.1f}s ({(total_tasks / elapsed) * 60:.1f} pairs/min)")
        print("\n[A. ТЕХНИЧЕСКАЯ КАЛИБРОВКА (SANITY CHECK)]")
        print(f"  • Pass Rate генератора:    {pass_rate:.1f}% ({total_success}/{total_candidates} скриптов)")
        pass_grade = "ОТЛИЧНО (>75%)" if pass_rate >= 75 else ("НОРМАЛЬНО (50-75%)" if pass_rate >= 50 else "ТРЕБУЕТ ДОРАБОТКИ (<50%)")
        print(f"    Вердикт:                 {pass_grade}")
        print(f"  • Timeout Rate (14s):      {timeout_rate:.1f}% ({total_timeouts}/{total_candidates})")
        print(f"  • Среднее время обучения:  {avg_time:.2f}s на скрипт")

        print("\n[Б. СОДЕРЖАТЕЛЬНАЯ КАЛИБРОВКА (РАЗНООБРАЗИЕ И ДЕЛЬТА)]")
        print(f"  • Decisive Rate (Δ >= 0.01): {decisive_rate:.1f}% ({decisive_count}/{total_tasks} пар)")
        print(f"  • Ничьи (Δ < 0.01):        {tie_rate:.1f}% ({len(ties)}/{total_tasks})")
        print(f"  • Оба решения упали:       {failed_rate:.1f}% ({len(both_failed)}/{total_tasks})")
        print(f"  • Средняя маржа победителя: Δ = {avg_margin:.4f}")

        print("\n[В. BASELINE BENCHMARK ДЛЯ OPENJEV (ТОЧКА ОТСЧЁТА)]")
        print(f"  • Оценено пар в OpenJev:   {len(evaluated_jev)}")
        print(f"  • Угадано верно дефолтным: {correct_jev} / {len(evaluated_jev)}")
        print(f"  • Baseline Accuracy:       {jev_accuracy:.1f}% (чистый априорный рандом ~50%)")
        print(f"  • Замороженный Holdout:    {self.holdout_file}")
        print(f"  • JSON Сводка:             {self.summary_file}")
        print("=" * 70 + "\n")
        return summary
