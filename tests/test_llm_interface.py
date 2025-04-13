"""Tests for LLMInterface class."""

from unittest.mock import Mock

import pytest

from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.schemas import FeatureEngineeringIdea


@pytest.fixture
def sample_features():
    """Provide feature descriptions."""
    return [
        {"name": "age", "type": "int", "description": "Âge du client"},
        {"name": "income", "type": "float", "description": "Revenu annuel"}
    ]


def test_initialization(mocker):
    """Test LLMInterface initialization."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    assert llm is not None  # Vérifie que l'initialisation réussit


def test_format_features(mocker, sample_features):  # pylint: disable=redefined-outer-name
    """Test feature descriptions formatting via generate_prompt_context."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features,
        target_description="Test",
        max_features=1
    )
    formatted = prompt_context["feature_descriptions"]
    assert "- age (int): Âge du client" in formatted
    assert "- income (float): Revenu annuel" in formatted
    assert "\n" in formatted


def test_generate_prompt(mocker, sample_features):  # pylint: disable=redefined-outer-name
    """Test prompt context generation."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features,
        target_description="Prédire le churn",
        max_features=3
    )
    assert isinstance(prompt_context, dict)
    assert "feature_descriptions" in prompt_context
    assert "target_description" in prompt_context
    assert prompt_context["target_description"] == "Prédire le churn"
    assert "Generate up to 3 features" in prompt_context["additional_context"]


def test_generate_prompt_unsupervised(mocker, sample_features):  # pylint: disable=redefined-outer-name
    """Test prompt context generation without target."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    prompt_context = llm.generate_prompt_context(
        feature_descriptions=sample_features,
        target_description=None,
        max_features=None
    )
    assert prompt_context["target_description"] == (
        "This is an unsupervised feature engineering task."
    )
    assert prompt_context["additional_context"] == ""


def test_generate_features(mocker, sample_features):  # pylint: disable=redefined-outer-name
    """Test feature engineering generation with mock."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    mock_chain = Mock()
    llm.chain = mock_chain
    mock_chain.invoke.return_value = [
        FeatureEngineeringIdea(
            name="age_squared",
            formula="lambda x: x['age'] ** 2",
            description="Âge au carré"
        )
    ]

    result = llm.generate_engineered_features(
        feature_descriptions=sample_features,
        target_description="Test",
        max_features=2
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].name == "age_squared"