"""Check evidence provenance and identity mapping in the post-hoc diagnosis."""
import json

from scripts.diagnose_openjev_context import requests_for


def sample():
    return {"profile": {"description": "An ordinary dataset", "task_type": "regression",
                "metric": "r2", "n_train": 90, "n_val": 30, "n_features": 4},
            "pair": {"hypothesis_a": "Original A", "hypothesis_b": "Original B",
                     "code_a": "def fit_predict(): pass # first",
                     "code_b": "def fit_predict(): pass # second"},
            "outcome": {"winner": "NEVER_REVEAL_TEST_RESULT"},
            "test_labels": ["NEVER_REVEAL_TEST_LABEL"]}


def test_cv_observations_follow_candidate_when_order_is_reversed():
    pilot = {"a": {"mean": .234, "fold_scores": [.2, .234, .268]},
             "b": {"mean": .789, "fold_scores": [.7, .789, .878]}}
    normal, reversed_ = requests_for(sample(), "train_cv", pilot)
    normal_cv = json.loads(normal["state"].split("\n")[-1])
    reverse_cv = json.loads(reversed_["state"].split("\n")[-1])
    assert normal_cv["Candidate A"]["mean"] == .234
    assert reverse_cv["Candidate A"]["mean"] == .789
    assert "# second" in reversed_["state"].split("Candidate B implementation:")[0]


def test_judge_input_excludes_outcomes_and_test_labels_in_every_mode():
    row = sample()
    for mode in ("full", "no_rationale", "no_code", "anonymous_task", "train_cv"):
        payloads = requests_for(row, mode, {"a": {"mean": .1}, "b": {"mean": .2}})
        assert "NEVER_REVEAL" not in json.dumps(payloads)
    assert row["pair"]["hypothesis_a"] == "Original A"
