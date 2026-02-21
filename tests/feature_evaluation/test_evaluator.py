import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from skfeaturellm.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from skfeaturellm.types import ProblemType


def test_summary_sorting_default():
    """Test that summary sorts by first column when no primary_metric given."""
    df = pd.DataFrame(
        {"metric_a": [0.1, 0.9, 0.5], "metric_b": [1, 2, 3]},
        index=["feat1", "feat2", "feat3"],
    )
    result = FeatureEvaluationResult(df)
    summary = result.summary

    # Should be sorted by metric_a descending (first column fallback)
    assert summary.index.tolist() == ["feat2", "feat3", "feat1"]
    assert summary.iloc[0]["metric_a"] == 0.9


def test_summary_sorting_primary_metric():
    """Test that summary sorts by the explicit primary_metric."""
    df = pd.DataFrame(
        {"metric_a": [0.1, 0.9, 0.5], "metric_b": [3, 1, 2]},
        index=["feat1", "feat2", "feat3"],
    )
    result = FeatureEvaluationResult(df, primary_metric="metric_b")
    summary = result.summary

    # Should be sorted by metric_b descending, not metric_a
    assert summary.index.tolist() == ["feat1", "feat3", "feat2"]
    assert summary.iloc[0]["metric_b"] == 3


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
    assert "mutual_info" in summary.columns
    assert "spearman_corr" in summary.columns
    assert "pearson_corr" in summary.columns
    assert "missing_pct" in summary.columns
    assert "variance" in summary.columns

    # Check values for strong feature
    strong_stats = summary.loc["feature_strong"]
    assert np.allclose(strong_stats["pearson_corr"], 1.0)
    assert np.allclose(strong_stats["spearman_corr"], 1.0)
    assert strong_stats["missing_pct"] == 0.0

    # Strong feature should have higher MI than weak
    assert (
        summary.loc["feature_strong", "mutual_info"]
        > summary.loc["feature_weak", "mutual_info"]
    )


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

    # Check missing feature (~40% missing)
    assert summary.loc["feature_missing", "missing_pct"] > 0.0


def test_evaluate_result_carries_context(sample_data_regression):
    """evaluate() passes X, y, problem_type to the result for later use."""
    X, y = sample_data_regression
    evaluator = FeatureEvaluator(problem_type=ProblemType.REGRESSION)
    features = ["feature_strong", "feature_weak"]

    result = evaluator.evaluate(X, y, features)

    assert result._X is not None
    assert result._y is not None
    assert result._problem_type == ProblemType.REGRESSION
    assert list(result._X.columns) == features


def test_plot_distributions_returns_one_figure_per_feature(sample_data_regression):
    X, y = sample_data_regression
    evaluator = FeatureEvaluator(problem_type=ProblemType.REGRESSION)
    features = ["feature_strong", "feature_weak"]

    figs = evaluator.plot_distributions(X, y, features)

    assert set(figs.keys()) == set(features)
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_distributions_classification(sample_data_classification):
    X, y = sample_data_classification
    evaluator = FeatureEvaluator(problem_type=ProblemType.CLASSIFICATION)
    features = ["feature_predictive", "feature_random"]

    figs = evaluator.plot_distributions(X, y, features)

    assert set(figs.keys()) == set(features)
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
