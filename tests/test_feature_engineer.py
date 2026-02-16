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
            type="mul",
            feature_name="age_squared",
            columns=["age", "age"],
            description="Age squared",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].feature_name == "age_squared"
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
            type="add",
            feature_name="income_plus",
            columns=["income"],
            parameters={"constant": 1.0},
            description="Income plus one",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].feature_name == "income_plus"
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
    """Test transform with a valid feature transformation."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            type="mul",
            feature_name="age_double",
            columns=["age"],
            parameters={"constant": 2.0},
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
    assert transformed_data["llm_feat_age_double"].tolist() == [50.0, 60.0]


def test_transform_invalid_feature(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test transform with an invalid feature (missing column)."""
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features"
    )
    mock_generate.return_value = [
        FeatureEngineeringIdea(
            type="add",
            feature_name="invalid_feature",
            columns=["unknown_column"],
            parameters={"constant": 1.0},
            description="Invalid - column doesn't exist",
        )
    ]
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    engineer.generated_features_ideas = mock_generate.return_value  # Simulate fit

    # Transform should skip invalid features with a warning
    with pytest.warns(UserWarning):
        result = engineer.transform(sample_data_frame)

    # Invalid feature should not be in the result
    assert "llm_feat_invalid_feature" not in result.columns
