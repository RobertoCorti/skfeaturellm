"""Tests for LLMFeatureEngineer class."""

from unittest.mock import Mock

import pandas as pd
import pytest

from skfeaturellm.feature_engineer import LLMFeatureEngineer
from skfeaturellm.schemas import FeatureEngineeringIdea


@pytest.fixture
def sample_data_frame():
    """Provide a test DataFrame."""
    return pd.DataFrame({
        "age": [25, 30],
        "income": [50000, 60000],
        "city": ["Paris", "Lyon"]
    })


@pytest.fixture
def sample_features():
    """Provide feature descriptions."""
    return [
        {"name": "age", "type": "int", "description": "Âge du client"},
        {"name": "income", "type": "float", "description": "Revenu annuel"},
        {"name": "city", "type": "str", "description": "Ville de résidence"}
    ]


def test_initialization(mocker):
    """Test LLMFeatureEngineer initialization."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(
        model_name="gpt-4o",
        target_col="default",
        max_features=3,
        feature_prefix="test_",
        model_provider="openai"
    )
    assert engineer.target_col == "default"
    assert engineer.max_features == 3
    assert engineer.feature_prefix == "test_"
    assert not engineer.generated_features


def test_fit_no_features(mocker, sample_data_frame):  # pylint: disable=redefined-outer-name
    """Test fit without explicit feature descriptions."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="age_squared",
            formula="lambda x: x['age'] ** 2",
            description="Âge au carré"
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(model_name="gpt-4o", model_provider="openai")
    engineer.generated_features_ideas = mock_generate.return_value  # Simuler fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].name == "age_squared"
    mock_generate.assert_not_called()  # fit n'est pas appelé directement ici


def test_fit_with_features(mocker, sample_data_frame, sample_features):  # pylint: disable=redefined-outer-name
    """Test fit with provided feature descriptions."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="income_plus",
            formula="lambda x: x['income'] + 1",
            description="Revenu plus un"
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(model_name="gpt-4o", model_provider="openai")
    engineer.generated_features_ideas = mock_generate.return_value  # Simuler fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].name == "income_plus"
    mock_generate.assert_not_called()


def test_transform_without_fit(mocker, sample_data_frame):  # pylint: disable=redefined-outer-name
    """Test transform raises error if fit not called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(model_name="gpt-4o", model_provider="openai")
    with pytest.raises(ValueError, match="fit must be called before transform"):
        engineer.transform(sample_data_frame)


def test_transform_valid_feature(mocker, sample_data_frame):  # pylint: disable=redefined-outer-name
    """Test transform with a valid feature formula."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="age_double",
            formula="lambda x: x['age'] * 2",
            description="Double de l'âge"
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(model_name="gpt-4o", model_provider="openai")
    engineer.generated_features_ideas = mock_generate.return_value  # Simuler fit
    transformed_data = engineer.transform(sample_data_frame)

    assert "age_double" in transformed_data.columns
    assert transformed_data["age_double"].tolist() == [50, 60]


def test_transform_invalid_feature(mocker, sample_data_frame):  # pylint: disable=redefined-outer-name
    """Test transform with an invalid feature formula."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="invalid_feature",
            formula="lambda x: x['unknown_column']",
            description="Formule invalide"
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(model_name="gpt-4o", model_provider="openai")
    engineer.generated_features_ideas = mock_generate.return_value  # Simuler fit
    with pytest.raises(KeyError, match="unknown_column"):
        engineer.transform(sample_data_frame)