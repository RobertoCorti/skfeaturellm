"""Tests for the LLMInterface class."""

from unittest.mock import Mock

import pytest

from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.schemas import FeatureEngineeringIdea


@pytest.fixture
def sample_features():
    """Provide feature descriptions for testing."""
    return [
        {"name": "age", "type": "int", "description": "Customer age"},
        {"name": "income", "type": "float", "description": "Annual income"},
    ]


def test_initialization(mocker):
    """Test initialization of LLMInterface."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    assert llm is not None  # Verify successful initialization


def test_format_features(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """Test formatting of feature descriptions via generate_prompt_context."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features, target_description="Test", max_features=1
    )
    formatted = prompt_context["feature_descriptions"]
    assert "- age (int): Customer age" in formatted
    assert "- income (float): Annual income" in formatted
    assert "\n" in formatted


def test_generate_prompt(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """Test generation of prompt context."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features,
        target_description="Predict churn",
        max_features=3,
    )
    assert isinstance(prompt_context, dict)
    assert "feature_descriptions" in prompt_context
    assert "target_description" in prompt_context
    assert prompt_context["target_description"] == "Predict churn"
    assert "Generate up to 3 features" in prompt_context["additional_context"]


def test_generate_prompt_no_target_description(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """Test prompt context generation without a target."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features,
        problem_type="classification",
        target_description=None,
        max_features=None,
    )

    assert (
        prompt_context["problem_type"] == "This is a supervised classification problem."
    )
    assert prompt_context["target_description"] == ("Not specified.")
    assert prompt_context["additional_context"] == ""


def test_generate_features(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """Test feature engineering generation with mock."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    mock_chain = Mock()
    llm.chain = mock_chain
    mock_chain.invoke.return_value = [
        FeatureEngineeringIdea(
            type="mul",
            feature_name="age_squared",
            left_column="age",
            right_constant=2.0,
            description="Double the age",
        )
    ]

    result = llm.generate_engineered_features(
        feature_descriptions=sample_features, target_description="Test", max_features=2
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].feature_name == "age_squared"
