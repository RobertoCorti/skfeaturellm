"""Tests for the LLMInterface class."""

from unittest.mock import Mock

import pandas as pd
import pytest

from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.prompts import utils
from skfeaturellm.schemas import FeatureEngineeringIdea
from skfeaturellm.types import ProblemType


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
            columns=["age"],
            parameters={"constant": 2.0},
            description="Double the age",
        )
    ]

    result = llm.generate_engineered_features(
        feature_descriptions=sample_features, target_description="Test", max_features=2
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].feature_name == "age_squared"


# --- _format_dataset_statistics ---


@pytest.fixture
def regression_xy():
    """Small regression dataset with one non-numeric column."""
    X = pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0],
            "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0, 90_000.0],
            "city": ["Paris", "Lyon", "Marseille", "Paris", "Lyon"],
        }
    )
    y = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5], name="target")
    return X, y


@pytest.fixture
def classification_xy():
    """Small binary classification dataset."""
    X = pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            "income": [50_000.0, 60_000.0, 70_000.0, 80_000.0, 90_000.0, 100_000.0],
        }
    )
    y = pd.Series(["cat", "dog", "cat", "dog", "cat", "dog"], name="label")
    return X, y


def test_format_dataset_statistics_regression(
    regression_xy,
):  # pylint: disable=redefined-outer-name
    """Target stats are min/max/mean/std; feature table and pearson_corr section present."""
    X, y = regression_xy
    result = utils.format_dataset_statistics(X, y, ProblemType.REGRESSION)

    assert "Target statistics:" in result
    assert (
        "min=" in result and "max=" in result and "mean=" in result and "std=" in result
    )
    assert "Feature statistics (numeric columns):" in result
    assert "age" in result
    assert "skewness" in result
    assert "Feature statistics vs target:" in result
    assert "pearson_corr" in result


def test_format_dataset_statistics_classification(
    classification_xy,
):  # pylint: disable=redefined-outer-name
    """Target stats show class counts; vs-target shows class-mean columns, not pearson_corr."""
    X, y = classification_xy
    result = utils.format_dataset_statistics(X, y, ProblemType.CLASSIFICATION)

    assert "Target statistics:" in result
    assert "class 'cat'" in result
    assert "class 'dog'" in result
    assert "%" in result
    assert "Feature statistics (numeric columns):" in result
    assert "Feature statistics vs target:" in result
    assert "pearson_corr" not in result
    assert "cat" in result and "dog" in result


def test_format_dataset_statistics_no_target(
    regression_xy,
):  # pylint: disable=redefined-outer-name
    """When y is None, target block says 'Not provided.' and vs-target section is absent."""
    X, _ = regression_xy
    result = utils.format_dataset_statistics(X, None, ProblemType.REGRESSION)

    assert "Not provided." in result
    assert "Feature statistics (numeric columns):" in result
    assert "Feature statistics vs target:" not in result


def test_format_dataset_statistics_no_numeric_cols():
    """When X has no numeric columns, feature block says 'No numeric features.'."""
    X = pd.DataFrame({"city": ["Paris", "Lyon", "Marseille"]})
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    result = utils.format_dataset_statistics(X, y, ProblemType.REGRESSION)

    assert "No numeric features." in result
    assert "Feature statistics vs target:" not in result


# --- generate_prompt_context: dataset_statistics key ---


def test_generate_prompt_context_includes_dataset_statistics(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """dataset_statistics value is passed through to the prompt context dict."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    stats = "Target statistics:\n  min=1, max=10"
    ctx = llm.generate_prompt_context(
        feature_descriptions=sample_features, dataset_statistics=stats
    )

    assert "dataset_statistics" in ctx
    assert ctx["dataset_statistics"] == stats


def test_generate_prompt_context_default_dataset_statistics(
    mocker, sample_features
):  # pylint: disable=redefined-outer-name
    """When dataset_statistics is omitted, the key defaults to 'Not provided.'."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    ctx = llm.generate_prompt_context(feature_descriptions=sample_features)

    assert ctx["dataset_statistics"] == "Not provided."


# --- generate_engineered_features_iterative ---


@pytest.fixture
def llm_with_mock(mocker, sample_features):  # pylint: disable=redefined-outer-name
    """LLMInterface with mocked init_chat_model and a mock llm attribute."""
    mocker.patch("skfeaturellm.llm_interface.init_chat_model")
    llm = LLMInterface(model_name="gpt-4o", model_provider="openai")
    mock_llm = Mock()
    llm.llm = mock_llm
    return llm, sample_features


def _make_ideas():
    from skfeaturellm.schemas import FeatureEngineeringIdeas

    idea = FeatureEngineeringIdea(
        type="mul",
        feature_name="age_double",
        columns=["age"],
        parameters={"constant": 2.0},
        description="Double the age",
    )
    return FeatureEngineeringIdeas(ideas=[idea])


def test_iterative_first_round_uses_prompt_template(
    mocker, llm_with_mock, sample_features
):  # pylint: disable=redefined-outer-name
    """First round (empty history) formats messages from the prompt template."""
    llm, features = llm_with_mock
    ideas = _make_ideas()
    llm.llm.invoke.return_value = ideas

    prompt_context = llm.generate_prompt_context(feature_descriptions=features)
    result_ideas, history = llm.generate_engineered_features_iterative(
        prompt_context=prompt_context,
        conversation_history=[],
    )

    llm.llm.invoke.assert_called_once()
    assert result_ideas is ideas
    # history = formatted messages + AIMessage
    assert len(history) >= 2
    assert history[-1].content == ideas.model_dump_json()


def test_iterative_subsequent_round_appends_feedback_message(
    mocker, llm_with_mock, sample_features
):  # pylint: disable=redefined-outer-name
    """Subsequent rounds prepend existing history and append a HumanMessage feedback."""
    from langchain_core.messages import AIMessage, HumanMessage

    llm, features = llm_with_mock
    ideas = _make_ideas()
    llm.llm.invoke.return_value = ideas

    prompt_context = llm.generate_prompt_context(feature_descriptions=features)

    # Simulate history from round 1
    fake_history = [HumanMessage(content="round 1 prompt"), AIMessage(content="{}")]
    feedback_context = {
        "selected_features_table": (
            "| feature | type | score |\n|---|---|---|\n| llm_feat_age_double | mul | 0.9 |"
        ),
        "rejected_features_table": "None",
        "max_features": 5,
    }

    result_ideas, updated_history = llm.generate_engineered_features_iterative(
        prompt_context=prompt_context,
        conversation_history=fake_history,
        feedback_context=feedback_context,
    )

    llm.llm.invoke.assert_called_once()
    call_args = llm.llm.invoke.call_args[0][0]
    # First two messages are from fake_history
    assert call_args[0] is fake_history[0]
    assert call_args[1] is fake_history[1]
    # Third message is the feedback HumanMessage
    assert isinstance(call_args[2], HumanMessage)
    assert "selected" in call_args[2].content.lower()
    # Updated history ends with AIMessage
    assert updated_history[-1].content == ideas.model_dump_json()


def test_iterative_returns_ideas_and_updated_history(
    mocker, llm_with_mock, sample_features
):  # pylint: disable=redefined-outer-name
    """Return value is (FeatureEngineeringIdeas, list of messages with AI response appended)."""
    from skfeaturellm.schemas import FeatureEngineeringIdeas

    llm, features = llm_with_mock
    ideas = _make_ideas()
    llm.llm.invoke.return_value = ideas

    prompt_context = llm.generate_prompt_context(feature_descriptions=features)
    result, history = llm.generate_engineered_features_iterative(
        prompt_context=prompt_context,
        conversation_history=[],
    )

    assert isinstance(result, FeatureEngineeringIdeas)
    assert isinstance(history, list)
    assert history[-1].content == ideas.model_dump_json()
