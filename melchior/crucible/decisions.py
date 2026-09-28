"""One typed-decision contract for OpenJev training and serving."""
from dataclasses import asdict

TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'


def comparison(profile, pair, *, swapped=False):
    p = asdict(profile) if not isinstance(profile, dict) else profile
    c = asdict(pair) if not isinstance(pair, dict) else pair
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
