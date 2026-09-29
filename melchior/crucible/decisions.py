"""One typed-decision contract for OpenJev training and serving."""
from dataclasses import asdict
import math
import re

TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'


def normalize_pair(pair, *, generation_swapped=False):
    """Remove positional prose references using the original generation identity.

    Legacy records need generation_swapped; new records are normalized before
    the runner shuffles them. Python code is deliberately preserved byte for byte.
    """
    c = asdict(pair) if not isinstance(pair, dict) else dict(pair)
    c['generation'] = dict(c.get('generation', {}))
    if c['generation'].get('role_references_normalized'):
        return c
    for side in ('a', 'b'):
        original = ('b' if side == 'a' else 'a') if generation_swapped else side
        c['hypothesis_'+side] = re.sub(
            r'\bcandidate\s+([ab])\b',
            lambda m: 'this candidate' if m[1].lower() == original else 'the alternative candidate',
            c['hypothesis_'+side], flags=re.IGNORECASE)
    c['generation']['role_references_normalized'] = True
    return c


def symmetric_choice(p_a, reverse_p_a, *, tolerance=1e-8):
    """Map both presentations to original identities before averaging.

    Scores are relative entailment scores, not calibrated win probabilities.
    """
    if tolerance < 0 or not all(math.isfinite(p) and 0 <= p <= 1 for p in (p_a, reverse_p_a)):
        raise ValueError('Invalid symmetry scores or tolerance')
    # Antisymmetric difference avoids cancellation around 0.5.
    diff = p_a-reverse_p_a
    score = .5 + .5*diff
    return {'choice': 'UNCERTAIN' if abs(diff) <= 2*tolerance else ('A' if diff > 0 else 'B'),
            'probabilities': {'A': score, 'B': 1-score},
            'order_disagreement': abs(p_a-(1-reverse_p_a))}


def answer_probability_a(answer):
    if 'probabilities' in answer:
        return float(answer['probabilities']['A'])
    return float(answer['confidence']) if answer['choice'] == 'A' else 1-float(answer['confidence'])


def comparison(profile, pair, *, swapped=False, generation_swapped=False):
    p = asdict(profile) if not isinstance(profile, dict) else profile
    c = normalize_pair(pair, generation_swapped=generation_swapped)
    a, b = ('b', 'a') if swapped else ('a', 'b')
    state = (f"Task: {p['description']}\nObjective: maximize {p['metric']}.\n"
             f"Training examples: {p['n_train']}; test examples: {p['n_val']}; features: {p['n_features']}.\n"
             f"Candidate A rationale: {c['hypothesis_'+a]}\nCandidate A implementation:\n{c['code_'+a]}\n"
             f"Candidate B rationale: {c['hypothesis_'+b]}\nCandidate B implementation:\n{c['code_'+b]}")
    instructions = f"Which candidate will achieve higher {p['metric']} on independent test data?"
    criteria = {x: f"Candidate {x} achieves the higher test {p['metric']}, assuming both execute successfully."
                for x in ('A', 'B')}
    return {"state": state, "questions": {"winner": {
        "type": "choice", "instructions": instructions, "criteria": criteria}}}


def nli_rows(payload, winner):
    q = payload['questions']['winner']
    return [{"premise": payload['state'],
             "hypothesis": TEMPLATE.format(instr=q['instructions'], label=k, crit=v),
             "label": int(k == winner), "source": "crucible_ml_preference_v2", "image": ""}
            for k, v in q['criteria'].items()]
