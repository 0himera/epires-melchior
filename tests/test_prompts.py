import pytest
from melchior.crucible.prompts import (
    PROMPTS,
    OPERATOR_KEYS,
    get_prompt_for_operator,
    select_operator_by_seed,
)
from melchior.crucible.environments import generate_task
from melchior.crucible.client import CrucibleLLMClient


def test_operator_prompts_complete():
    expected_operators = [
        "inductive_bias",
        "data_centric",
        "hyperparameter_frontier",
        "ensemble_diversity",
        "pathology_defense",
    ]
    assert OPERATOR_KEYS == expected_operators
    for op in expected_operators:
        prompt = get_prompt_for_operator(op)
        assert len(prompt) > 200
        assert "fit_predict(X_train, y_train, X_test)" in prompt
        assert "candidate_a and candidate_b" in prompt


def test_select_operator_by_seed():
    selected = [select_operator_by_seed(i) for i in range(10)]
    assert selected[:5] == OPERATOR_KEYS
    assert selected[5:10] == OPERATOR_KEYS


@pytest.mark.asyncio
async def test_client_operator_dispatch_mock():
    client = CrucibleLLMClient(mode="mock")
    profile, _, _, _, _ = generate_task(seed=42)

    for op in OPERATOR_KEYS:
        pair = await client.generate_pair(profile, seed=42, operator=op)
        assert pair.operator == op
        assert len(pair.code_a) > 0
        assert len(pair.code_b) > 0

    await client.close()
