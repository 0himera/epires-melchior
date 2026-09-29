from dataclasses import asdict
import pytest
from melchior.crucible.client import CandidatePair
from melchior.crucible.decisions import comparison, normalize_pair, symmetric_choice
from melchior.crucible.environments import TaskProfile


def test_legacy_shuffled_references_follow_candidate_identity():
    code_a = "# Candidate B\nA = 'Candidate A'"
    legacy = CandidatePair('Candidate B beats Candidate A', code_a,
                           'Candidate A is simpler than Candidate B', 'B = 7')
    repaired = normalize_pair(legacy, generation_swapped=True)
    assert repaired['hypothesis_a'] == 'this candidate beats the alternative candidate'
    assert repaired['hypothesis_b'] == 'this candidate is simpler than the alternative candidate'
    assert repaired['code_a'] == code_a
    assert normalize_pair(repaired, generation_swapped=True) == repaired
    p = TaskProfile('test','regression','r2','maximize','test',20,10,2)
    normal = comparison(p, repaired)['state']
    reversed_state = comparison(p, repaired, swapped=True)['state']
    assert 'Candidate A rationale: this candidate beats' in normal
    assert 'Candidate B rationale: this candidate beats' in reversed_state


@pytest.mark.parametrize('p,q', [(0.1,.8),(.8,.1),(.7,.7),(.500000001,.5),(.95,.8)])
def test_symmetric_choice_equivariant_and_ties_abstain(p,q):
    normal = symmetric_choice(p,q)
    reverse = symmetric_choice(q,p)
    assert normal['probabilities']['A'] == pytest.approx(reverse['probabilities']['B'])
    assert normal['choice'] == {'A':'B','B':'A','UNCERTAIN':'UNCERTAIN'}[reverse['choice']]
    if abs(p-q) < 2e-8:
        assert normal['choice'] == 'UNCERTAIN'


def test_normalization_does_not_edit_standalone_python_letters():
    pair=CandidatePair('A regularized classifier', 'A = 1', 'B spline regression', 'B = 2')
    assert normalize_pair(pair)['hypothesis_a'] == pair.hypothesis_a
    assert normalize_pair(pair)['code_b'] == pair.code_b
