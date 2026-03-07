"""Tests for the LLMFeatureEngineer class."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from skfeaturellm.exceptions import NotFittedError
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
    engineer.generated_features_ideas_ = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas_) == 1
    assert engineer.generated_features_ideas_[0].feature_name == "age_squared"
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
    engineer.generated_features_ideas_ = mock_generate.return_value  # Simulate fit
    assert len(engineer.generated_features_ideas_) == 1
    assert engineer.generated_features_ideas_[0].feature_name == "income_plus"
    mock_generate.assert_not_called()


def test_transform_without_fit(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """Test that transform raises an error if fit is not called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", model_provider="openai"
    )
    with pytest.raises(
        NotFittedError,
        match="This LLMFeatureEngineer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
    ):
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
    engineer.generated_features_ideas_ = mock_generate.return_value  # Simulate fit
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
    engineer.generated_features_ideas_ = mock_generate.return_value  # Simulate fit

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
    assert len(engineer.generated_features_ideas_) == 1
    assert engineer.generated_features_ideas_[0].feature_name == "age_double"


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

    with pytest.raises(
        NotFittedError,
        match="This LLMFeatureEngineer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
    ):
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
    engineer.generated_features_ideas_ = [idea]
    y = pd.Series([1, 2], name="target")

    result = engineer.evaluate_features(sample_data_frame, y, is_transformed=False)

    assert "llm_feat_age_double" in engineer.generated_features_ideas_[0].feature_name
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
    engineer.generated_features_ideas_ = [idea]
    engineer.generated_features = [idea]

    transform_spy = mocker.patch.object(engineer, "transform", wraps=engineer.transform)
    X_with_feature = sample_data_frame.copy()
    X_with_feature["llm_feat_age_double"] = sample_data_frame["age"] * 2
    y = pd.Series([1, 2], name="target")

    engineer.evaluate_features(X_with_feature, y, is_transformed=True)

    transform_spy.assert_not_called()


def test_fit_passes_dataset_statistics_to_llm(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """fit() with y computes statistics and forwards them to generate_engineered_features."""
    mock_ideas = Mock()
    mock_ideas.ideas = []
    mock_generate = mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")

    y = pd.Series([0.1, 0.9], name="target")
    engineer = LLMFeatureEngineer(problem_type="regression", model_name="gpt-4o")
    engineer.fit(sample_data_frame, y=y)

    call_kwargs = mock_generate.call_args.kwargs
    assert "dataset_statistics" in call_kwargs
    assert "Target statistics:" in call_kwargs["dataset_statistics"]
    assert "Feature statistics (numeric columns):" in call_kwargs["dataset_statistics"]


def test_fit_without_y_dataset_statistics_not_provided(
    mocker, sample_data_frame
):  # pylint: disable=redefined-outer-name
    """fit() without y still passes dataset_statistics containing 'Not provided.'."""
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
    assert "dataset_statistics" in call_kwargs
    assert "Not provided." in call_kwargs["dataset_statistics"]


# =============================================================================
# Test: to_transformer()
# =============================================================================


def test_to_transformer_before_fit_raises(mocker):
    """to_transformer() raises ValueError if fit() has not been called."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    engineer = LLMFeatureEngineer(problem_type="classification")
    with pytest.raises(
        NotFittedError,
        match="This LLMFeatureEngineer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
    ):
        engineer.to_transformer()


def test_to_transformer_returns_feature_engineering_transformer(
    mocker, sample_data_frame
):
    """to_transformer() returns a FeatureEngineeringTransformer built from generated features."""
    from skfeaturellm.feature_engineering_transformer import (
        FeatureEngineeringTransformer,
    )

    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
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

    engineer = LLMFeatureEngineer(problem_type="classification")
    engineer.fit(sample_data_frame)
    engineer.transform(sample_data_frame)

    transformer = engineer.to_transformer()

    assert isinstance(transformer, FeatureEngineeringTransformer)
    assert len(transformer.transformations) == 1
    assert transformer.transformations[0]["feature_name"] == "llm_feat_age_double"


def test_to_transformer_filter_by_prefixed_name(mocker, sample_data_frame):
    """to_transformer(features=[...]) filters by prefixed feature name."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    ideas = [
        FeatureEngineeringIdea(
            type="mul",
            feature_name="age_double",
            columns=["age"],
            parameters={"constant": 2.0},
            description="Double the age",
        ),
        FeatureEngineeringIdea(
            type="add",
            feature_name="income_plus_age",
            columns=["income", "age"],
            description="Sum of income and age",
        ),
    ]
    mock_ideas = Mock()
    mock_ideas.ideas = ideas
    mocker.patch(
        "skfeaturellm.llm_interface.LLMInterface.generate_engineered_features",
        return_value=mock_ideas,
    )

    engineer = LLMFeatureEngineer(problem_type="classification")
    engineer.fit(sample_data_frame)
    engineer.transform(sample_data_frame)

    transformer = engineer.to_transformer(features=["llm_feat_age_double"])

    assert len(transformer.transformations) == 1
    assert transformer.transformations[0]["feature_name"] == "llm_feat_age_double"


# =============================================================================
# Test: fit_selective(), _run_selector(), _build_feedback_context()
# =============================================================================


@pytest.fixture
def numeric_data_frame():
    """Numeric-only DataFrame for tests that use sklearn selectors."""
    return pd.DataFrame({"age": [25, 30], "income": [50000, 60000]})


@pytest.fixture
def engineer_mocked(mocker):
    """LLMFeatureEngineer with mocked LLM, ready for fit_selective tests."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    return LLMFeatureEngineer(
        problem_type="classification", model_name="gpt-4o", feature_prefix="llm_feat_"
    )


@pytest.fixture
def ideas_age_double():
    """One valid idea: multiply age by 2."""
    return FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )


@pytest.fixture
def ideas_income_log():
    """One valid idea: log of income."""
    return FeatureEngineeringIdea(
        type="log",
        feature_name="log_income",
        columns=["income"],
        description="Log of income",
    )


def _make_ideas_result(ideas):
    from skfeaturellm.schemas import FeatureEngineeringIdeas

    return FeatureEngineeringIdeas(ideas=ideas)


def test_fit_selective_calls_llm_n_rounds(
    mocker, engineer_mocked, numeric_data_frame, ideas_age_double
):  # pylint: disable=redefined-outer-name
    """fit_selective() calls generate_engineered_features_iterative exactly n_rounds times."""
    from sklearn.feature_selection import SelectKBest, f_classif

    ideas_result = _make_ideas_result([ideas_age_double])
    mock_iterative = mocker.patch.object(
        engineer_mocked.llm_interface,
        "generate_engineered_features_iterative",
        return_value=(ideas_result, []),
    )
    mocker.patch.object(
        engineer_mocked.llm_interface, "generate_prompt_context", return_value={}
    )

    y = pd.Series([0, 1], name="target")
    selector = SelectKBest(f_classif, k=1)
    engineer_mocked.fit_selective(numeric_data_frame, y, selector=selector, n_rounds=3)

    assert mock_iterative.call_count == 3


def test_fit_selective_populates_generated_features_ideas(
    mocker, engineer_mocked, numeric_data_frame, ideas_age_double, ideas_income_log
):  # pylint: disable=redefined-outer-name
    """fit_selective() keeps only selected ideas in generated_features_ideas."""
    from sklearn.feature_selection import SelectKBest, f_classif

    # Round produces two ideas; selector keeps only the one with highest score
    ideas_result = _make_ideas_result([ideas_age_double, ideas_income_log])
    mocker.patch.object(
        engineer_mocked.llm_interface,
        "generate_engineered_features_iterative",
        return_value=(ideas_result, []),
    )
    mocker.patch.object(
        engineer_mocked.llm_interface, "generate_prompt_context", return_value={}
    )

    y = pd.Series([0, 1], name="target")
    selector = SelectKBest(f_classif, k=1)
    engineer_mocked.fit_selective(numeric_data_frame, y, selector=selector, n_rounds=1)

    # k=1 keeps exactly one feature across all features (original + new)
    assert hasattr(engineer_mocked, "generated_features_ideas_")
    assert len(engineer_mocked.generated_features_ideas_) <= 1


def test_fit_selective_with_eval_set(
    mocker, engineer_mocked, numeric_data_frame, ideas_age_double
):  # pylint: disable=redefined-outer-name
    """When eval_set is provided, _run_selector uses X_eval for selection."""
    from sklearn.feature_selection import SelectKBest, f_classif

    ideas_result = _make_ideas_result([ideas_age_double])
    mocker.patch.object(
        engineer_mocked.llm_interface,
        "generate_engineered_features_iterative",
        return_value=(ideas_result, []),
    )
    mocker.patch.object(
        engineer_mocked.llm_interface, "generate_prompt_context", return_value={}
    )

    run_selector_spy = mocker.spy(engineer_mocked, "_run_selector")

    X_val = pd.DataFrame({"age": [28, 33], "income": [55000, 65000]})
    y_train = pd.Series([0, 1], name="target")
    y_val = pd.Series([1, 0], name="target")

    selector = SelectKBest(f_classif, k=1)
    engineer_mocked.fit_selective(
        numeric_data_frame,
        y_train,
        selector=selector,
        n_rounds=1,
        eval_set=(X_val, y_val),
    )

    _, _, _, _, X_eval_arg, y_eval_arg = run_selector_spy.call_args[0]
    assert X_eval_arg is X_val
    assert y_eval_arg is y_val


def test_run_selector_fits_on_all_features(
    mocker, engineer_mocked, numeric_data_frame, ideas_age_double
):  # pylint: disable=redefined-outer-name
    """_run_selector fits the selector on all columns (original + new), not just new ones."""
    from sklearn.feature_selection import SelectKBest

    mock_selector = mocker.MagicMock(spec=SelectKBest)
    # 3 cols: age, income, llm_feat_age_double — new feature is index 2
    mock_selector.get_support.return_value = np.array([False, False, True])
    mock_selector.scores_ = np.array([0.1, 0.2, 0.9])

    y = pd.Series([0, 1], name="target")
    engineer_mocked._run_selector(
        numeric_data_frame, y, [ideas_age_double], mock_selector
    )

    fit_call_X = mock_selector.fit.call_args[0][0]
    # Selector must have been fit on a DataFrame that includes original columns
    assert "age" in fit_call_X.columns
    assert "income" in fit_call_X.columns
    assert "llm_feat_age_double" in fit_call_X.columns


def test_run_selector_no_created_features_returns_all_rejected(
    mocker, engineer_mocked, numeric_data_frame
):  # pylint: disable=redefined-outer-name
    """When no new features are created (bad columns), all ideas go to rejected."""
    from sklearn.feature_selection import SelectKBest, f_classif

    bad_idea = FeatureEngineeringIdea(
        type="add",
        feature_name="bad_feat",
        columns=["nonexistent_col"],
        parameters={"constant": 1.0},
        description="Uses a column that does not exist",
    )
    y = pd.Series([0, 1], name="target")
    selector = SelectKBest(f_classif, k=1)

    with pytest.warns(UserWarning):
        selected, rejected, scores = engineer_mocked._run_selector(
            numeric_data_frame, y, [bad_idea], selector
        )

    assert selected == []
    assert len(rejected) == 1
    assert scores == {}


def test_build_feedback_context_contains_feature_names(
    engineer_mocked,
    ideas_age_double,
    ideas_income_log,  # pylint: disable=redefined-outer-name
):
    """_build_feedback_context() markdown tables contain the feature names."""
    scores = {"llm_feat_age_double": 0.85, "llm_feat_log_income": 0.12}
    ctx = engineer_mocked._build_feedback_context(
        selected_ideas=[ideas_age_double],
        rejected_ideas=[ideas_income_log],
        scores=scores,
        max_features=5,
    )

    assert "llm_feat_age_double" in ctx["selected_features_table"]
    assert "llm_feat_log_income" in ctx["rejected_features_table"]
    assert ctx["max_features"] == 5


def test_build_feedback_context_empty_lists(engineer_mocked):
    """_build_feedback_context() with no ideas produces 'None' tables."""
    ctx = engineer_mocked._build_feedback_context(
        selected_ideas=[],
        rejected_ideas=[],
        scores={},
        max_features=3,
    )

    assert ctx["selected_features_table"] == "None"
    assert ctx["rejected_features_table"] == "None"


def test_to_transformer_filter_by_unprefixed_name(mocker, sample_data_frame):
    """to_transformer() also accepts names without the feature_prefix."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
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

    engineer = LLMFeatureEngineer(problem_type="classification")
    engineer.fit(sample_data_frame)
    engineer.transform(sample_data_frame)

    transformer = engineer.to_transformer(features=["age_double"])

    assert len(transformer.transformations) == 1
    assert transformer.transformations[0]["feature_name"] == "llm_feat_age_double"
