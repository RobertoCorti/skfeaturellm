import numpy as np
import pandas as pd
import pytest

from skfeaturellm.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from skfeaturellm.types import ProblemType


@pytest.fixture
def sample_data_regression():
    X = pd.DataFrame(
        {
            "feature_strong": [1, 2, 3, 4, 5],  # Perfect correlation
            "feature_weak": [1, 5, 2, 4, 3],  # Weak correlation
            "feature_constant": [1, 1, 1, 1, 1],  # Constant
            "feature_missing": [1, np.nan, 3, np.nan, 5],  # Missing values
        }
    )
    y = pd.Series([10, 20, 30, 40, 50])
    return X, y


@pytest.fixture
def sample_data_classification():
    X = pd.DataFrame(
        {
            "feature_predictive": [1, 1, 2, 2, 3, 3],  # Correlates with classes
            "feature_random": [5, 1, 4, 1, 3, 0],  # Random
        }
    )
    y = pd.Series([0, 0, 1, 1, 2])
    return X, y


def test_summary_sorting():
    """Test that summary sorts by the primary metric."""
    df = pd.DataFrame(
        {"metric_a": [0.1, 0.9, 0.5], "metric_b": [1, 2, 3]},
        index=["feat1", "feat2", "feat3"],
    )
    result = FeatureEvaluationResult(df)
    summary = result.summary

    # Should be sorted by metric_a descending
    assert summary.index.tolist() == ["feat2", "feat3", "feat1"]
    assert summary.iloc[0]["metric_a"] == 0.9


def test_to_dict():
    """Test conversion to dictionary."""
    df = pd.DataFrame({"val": [1]}, index=["feat1"])
    result = FeatureEvaluationResult(df)
    res_dict = result.to_dict()
    assert "val" in res_dict
    assert res_dict["val"]["feat1"] == 1


def test_regression_metrics(sample_data_regression):
    X, y = sample_data_regression
    evaluator = FeatureEvaluator(problem_type=ProblemType.REGRESSION)
    features = ["feature_strong", "feature_weak"]

    result = evaluator.evaluate(X, y, features)
    summary = result.summary

    # Check columns exist
    assert "spearman_corr" in summary.columns
    assert "pearson_corr" in summary.columns
    assert "missing_pct" in summary.columns
    assert "variance" in summary.columns

    # Check values for strong feature
    strong_stats = summary.loc["feature_strong"]
    assert np.allclose(strong_stats["pearson_corr"], 1.0)
    assert np.allclose(strong_stats["spearman_corr"], 1.0)
    assert strong_stats["missing_pct"] == 0.0


def test_classification_metrics(sample_data_classification):
    X, y = sample_data_classification
    evaluator = FeatureEvaluator(problem_type=ProblemType.CLASSIFICATION)
    features = ["feature_predictive", "feature_random"]

    result = evaluator.evaluate(X, y, features)
    summary = result.summary

    assert "mutual_info" in summary.columns
    # Predictive feature should have higher MI than random
    assert (
        summary.loc["feature_predictive", "mutual_info"]
        > summary.loc["feature_random", "mutual_info"]
    )


def test_quality_metrics(sample_data_regression):
    X, y = sample_data_regression
    evaluator = FeatureEvaluator(problem_type=ProblemType.REGRESSION)
    features = ["feature_constant", "feature_missing"]

    result = evaluator.evaluate(X, y, features)
    summary = result.summary

    # Check constant feature
    assert summary.loc["feature_constant", "variance"] == 0.0
    assert summary.loc["feature_constant", "is_constant"]

    # Check missing feature
    assert summary.loc["feature_missing", "missing_pct"] == 0.4  # 2/5 missing
