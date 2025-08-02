"""Tests for the LLMFeatureEngineer class."""

import warnings
from unittest.mock import Mock

import pandas as pd
import pytest

from skfeaturellm.feature_engineer import LLMFeatureEngineer
from skfeaturellm.schemas import FeatureEngineeringIdea
from skfeaturellm.types import ProblemType


@pytest.fixture
def sample_data_frame():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {"age": [25, 30], "income": [50000, 60000], "city": ["Paris", "Lyon"]}
    )


@pytest.fixture
def sample_features():
    """Provide feature descriptions for testing."""
    return [
        {"name": "age", "type": "int", "description": "Customer age"},
        {"name": "income", "type": "float", "description": "Annual income"},
        {"name": "city", "type": "str", "description": "City of residence"},
    ]


def test_initialization(mocker):
    """Test initialization of LLMFeatureEngineer."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        target_col="default",
        max_features=3,
        feature_prefix="test_",
        model_provider="openai",
    )
    assert engineer.problem_type == ProblemType.CLASSIFICATION
    assert engineer.target_col == "default"
    assert engineer.max_features == 3
    assert engineer.feature_prefix == "test_"
    assert not engineer.generated_features


def test_fit_no_features(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test fit method without explicit feature descriptions."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="age_squared",
            formula="lambda x: x['age'] ** 2",
            description="Age squared",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].name == "age_squared"
    mock_generate.assert_not_called()  # Fit not called directly here


def test_fit_with_features(
    mocker, sample_data_frame, sample_features
):  # pylint: disable=redefined-outer-name
    """Test fit method with provided feature descriptions."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="income_plus",
            formula="income + 1",
            description="Income plus one",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].name == "income_plus"
    mock_generate.assert_not_called()


def test_transform_without_fit(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that transform raises an error if fit is not called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    with pytest.raises(ValueError, match="fit must be called before transform"):
        engineer.transform(sample_data_frame)


def test_transform_valid_feature(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test transform with a valid feature formula."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="age_double",
            formula="age * 2",
            description="Double the age",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", feature_prefix="llm_feat_"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    transformed_data = engineer.transform(sample_data_frame)

    assert "llm_feat_age_double" in transformed_data.columns
    assert transformed_data["llm_feat_age_double"].tolist() == [50, 60]


def test_transform_invalid_feature(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test transform with an invalid feature formula."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            name="invalid_feature",
            formula="lambda x: x['unknown_column']",
            description="Invalid formula",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    # catch the warning with message "The formula lambda x: x['unknown_column'] is not a valid lambda function. Skipping feature invalid_feature."
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The formula lambda x: x['unknown_column'] is not a valid lambda function. Skipping feature invalid_feature.",
        )
        engineer.transform(sample_data_frame)
