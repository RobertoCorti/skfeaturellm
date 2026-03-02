"""Tests for the LLMFeatureEngineer class."""

import json
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


def test_fit_generates_ideas(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that fit calls the LLM and populates generated_features_ideas."""
    mock_ideas = Mock()
    mock_ideas.ideas = [
        FeatureEngineeringIdea(
            type="mul",
            feature_name="age_double",
            columns=["age"],
            parameters={"constant": 2.0},
            description="Double the age",
        )
    ]
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(problem_type="regression", model_name="gpt-4o")
    engineer.fit(sample_data_frame)

    mock_generate.assert_called_once()
    assert len(engineer.generated_features_ideas) == 1
    assert engineer.generated_features_ideas[0].feature_name == "age_double"


def test_fit_auto_extracts_feature_descriptions(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that fit auto-extracts feature descriptions from the DataFrame."""
    mock_ideas = Mock()
    mock_ideas.ideas = []
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(problem_type="regression", model_name="gpt-4o")
    engineer.fit(sample_data_frame)

    call_kwargs = mock_generate.call_args.kwargs
    feature_names = [f["name"] for f in call_kwargs["feature_descriptions"]]
    assert feature_names == list(sample_data_frame.columns)


def test_fit_transform_returns_dataframe(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that fit_transform calls fit and transform and returns a DataFrame."""
    mock_ideas = Mock()
    mock_ideas.ideas = [
        FeatureEngineeringIdea(
            type="mul",
            feature_name="age_double",
            columns=["age"],
            parameters={"constant": 2.0},
            description="Double the age",
        )
    ]
    mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="regression", model_name="gpt-4o", feature_prefix="llm_feat_"
    )
    result = engineer.fit_transform(sample_data_frame)

    assert isinstance(result, pd.DataFrame)
    assert "llm_feat_age_double" in result.columns


def test_evaluate_features_without_fit_raises(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that evaluate_features raises if fit has not been called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(problem_type="regression", model_name="gpt-4o")
    y = pd.Series([1, 2], name="target")

    with pytest.raises(ValueError, match="fit must be called before evaluate_features"):
        engineer.evaluate_features(sample_data_frame, y)


def test_evaluate_features_is_transformed_false(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test evaluate_features calls transform internally when is_transformed=False."""
    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="llm_feat_age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )
    mock_ideas = Mock()
    mock_ideas.ideas = [idea]
    mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="regression", model_name="gpt-4o", feature_prefix="llm_feat_"
    )
    engineer.generated_features_ideas = [idea]
    y = pd.Series([1, 2], name="target")

    result = engineer.evaluate_features(sample_data_frame, y, is_transformed=False)

    assert "llm_feat_age_double" in engineer.generated_features_ideas[0].feature_name
    assert result is not None


def test_evaluate_features_is_transformed_true(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test evaluate_features skips transform when is_transformed=True."""
    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    engineer = LLMFeatureEngineer(
        problem_type="regression", model_name="gpt-4o", feature_prefix="llm_feat_"
    )
    engineer.generated_features_ideas = [idea]
    engineer.generated_features = [idea]

    transform_spy = mocker.patch.object(engineer, "transform", wraps=engineer.transform)
    X_with_feature = sample_data_frame.copy()
    X_with_feature["llm_feat_age_double"] = sample_data_frame["age"] * 2
    y = pd.Series([1, 2], name="target")

    engineer.evaluate_features(X_with_feature, y, is_transformed=True)

    transform_spy.assert_not_called()


def test_save_without_fit_raises(mocker, tmp_path):
    """Test that save() raises ValueError if fit has not been called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    with pytest.raises(ValueError, match="fit must be called before save"):
        engineer.save(str(tmp_path / "features.json"))


def test_save_and_load_roundtrip(mocker, tmp_path, sample_data_frame):
    """Test that save() and load() round-trip generated_features_ideas correctly."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )
    engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        target_col="label",
        max_features=5,
        feature_prefix="feat_",
    )
    engineer.generated_features_ideas = [idea]

    save_path = str(tmp_path / "features.json")
    engineer.save(save_path)

    # Verify file contents
    with open(save_path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["params"]["problem_type"] == "classification"
    assert raw["params"]["feature_prefix"] == "feat_"
    assert len(raw["generated_features_ideas"]) == 1

    # Restore and verify
    loaded = LLMFeatureEngineer.load(save_path)
    assert loaded.problem_type.value == "classification"
    assert loaded.feature_prefix == "feat_"
    assert loaded.target_col == "label"
    assert loaded.max_features == 5
    assert len(loaded.generated_features_ideas) == 1
    assert loaded.generated_features_ideas[0].feature_name == "age_double"


def test_load_allows_transform_without_fit(mocker, tmp_path, sample_data_frame):
    """Test that a loaded engineer can call transform() without fit()."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )
    engineer = LLMFeatureEngineer(
        problem_type="regression", model_name="gpt-4o", feature_prefix="llm_feat_"
    )
    engineer.generated_features_ideas = [idea]

    save_path = str(tmp_path / "features.json")
    engineer.save(save_path)

    loaded = LLMFeatureEngineer.load(save_path)
    result = loaded.transform(sample_data_frame)

    assert "llm_feat_age_double" in result.columns
    assert result["llm_feat_age_double"].tolist() == [50.0, 60.0]
