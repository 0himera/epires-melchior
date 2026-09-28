"""Paired empirical preferences, separate from execution failures."""
from dataclasses import dataclass
from typing import Any
from melchior.crucible.decisions import comparison, nli_rows
from melchior.crucible.metrics import paired_interval


@dataclass
class ArbiterOutcome:
    winner: str
    delta: float | None
    nli_records: list[dict[str, Any]]
    dpo_record: dict[str, Any] | None
    interval: list[float] | None = None
    execution_winner: str | None = None


class CrucibleArbiter:
    def __init__(self, min_delta=0.005):
        if min_delta < 0:
            raise ValueError('min_delta must be non-negative')
        self.min_delta = min_delta

    def evaluate(self, profile, pair, res_a, res_b, *, y_true=None, seed=42):
        valid = [r.status == 'success' and r.metric is not None for r in (res_a, res_b)]
        if not all(valid):
            execution = 'A' if valid[0] else 'B' if valid[1] else None
            return ArbiterOutcome('EXECUTION_FAILURE', None, [], None, execution_winner=execution)
        if y_true is None or not res_a.predictions or not res_b.predictions:
            raise ValueError('Paired test labels and predictions are required for quality arbitration')
        diff = res_b.metric - res_a.metric
        interval = paired_interval(profile.metric, y_true, res_a.predictions, res_b.predictions, seed=seed)
        winner = 'B' if interval[0] > self.min_delta else 'A' if interval[1] < -self.min_delta else 'UNCERTAIN'
        outcome = ArbiterOutcome(winner, abs(diff), [], None, interval)
        if winner in {'A', 'B'}:
            payload = comparison(profile, pair)
            outcome.nli_records = nli_rows(payload, winner)
            chosen, rejected = ('a', 'b') if winner == 'A' else ('b', 'a')
            outcome.dpo_record = {
                'prompt': payload['state'],
                'chosen': getattr(pair, 'code_'+chosen), 'rejected': getattr(pair, 'code_'+rejected),
                'margin': abs(diff), 'operator': pair.operator,
            }
        return outcome
