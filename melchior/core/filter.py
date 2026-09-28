"""Candidate Filter — static safety analysis and cheap pilot ranking."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from melchior.brain.jev_client import JevClient
from melchior.brain.decisions import rank_pilot
from melchior.brain.llm_client import CandidateProposal
from melchior.core.executor import Executor


@dataclass
class FilterResult:
    passed: bool
    reason: str
    pilot_score: float = 0.0


class CandidateFilter:
    def __init__(self, jev: JevClient, executor: Executor):
        self.jev = jev
        self.executor = executor

    def static_check(self, code: str) -> tuple[bool, str]:
        """Validate syntax and basic safety rules via Python AST."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError on line {e.lineno}: {e.msg}"

        # Safety checks: disallow arbitrary destructive system calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in ["system", "popen", "spawn"]:
                    return False, f"Prohibited system execution call: {func.attr}"
                elif isinstance(func, ast.Name) and func.id in ["exec", "eval"]:
                    return False, f"Prohibited dynamic code execution: {func.id}"

        return True, "Static checks passed"

    def filter_and_rank(
        self,
        candidates: list[CandidateProposal],
        best_so_far: float = 0.0,
        pilot_timeout_s: int = 30,
    ) -> list[tuple[CandidateProposal, float]]:
        """Run static validation and pilot experiments, then rank candidates by Jev score."""
        scored_candidates: list[tuple[CandidateProposal, float]] = []

        for candidate in candidates:
            # Step 1: Static verification
            ok, msg = self.static_check(candidate.code)
            if not ok:
                continue

            # Step 2: Pilot execution (1 quick run or subset)
            exec_res = self.executor.run(
                code=candidate.code,
                best_so_far=best_so_far,
                timeout_seconds=pilot_timeout_s,
            )

            # Step 3: Jev score pilot quality
            pilot_summary = (
                f"Candidate: {candidate.technique}\n"
                f"Predicted: {candidate.predicted_metric}\n"
                f"Pilot status: {exec_res.status}\n"
                f"Pilot metric: {exec_res.metric}\n"
                f"Epochs seen: {exec_res.epoch_metrics}\n"
                f"Time: {exec_res.wall_time_s}s"
            )
            score = rank_pilot(self.jev, pilot_summary)

            # Boost score if pilot succeeded with good metric
            if exec_res.status == "success" and exec_res.metric is not None:
                score = (score + exec_res.metric) / 2.0
            elif exec_res.status != "success":
                score = 0.1

            scored_candidates.append((candidate, round(score, 4)))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates
