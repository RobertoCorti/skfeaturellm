import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # non-interactive backend, no GUI windows

from skfeaturellm.feature_evaluation.visualizations import plot_feature_vs_target
from skfeaturellm.types import ProblemType


def test_regression_returns_figure(sample_data_regression):
    X, y = sample_data_regression
    fig = plot_feature_vs_target(X["feature_strong"], y, ProblemType.REGRESSION)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_classification_returns_figure(sample_data_classification):
    X, y = sample_data_classification
    fig = plot_feature_vs_target(X["feature_predictive"], y, ProblemType.CLASSIFICATION)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_regression_axes_labels(sample_data_regression):
    X, y = sample_data_regression
    x = X["feature_strong"]
    y.name = "target"
    fig = plot_feature_vs_target(x, y, ProblemType.REGRESSION)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "feature_strong"
    assert ax.get_ylabel() == "target"
    plt.close(fig)


def test_classification_axes_labels(sample_data_classification):
    X, y = sample_data_classification
    x = X["feature_predictive"]
    y.name = "class"
    fig = plot_feature_vs_target(x, y, ProblemType.CLASSIFICATION)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "class"
    assert ax.get_ylabel() == "feature_predictive"
    plt.close(fig)


def test_title_contains_feature_and_target_names(sample_data_regression):
    X, y = sample_data_regression
    x = X["feature_strong"]
    y.name = "target"
    fig = plot_feature_vs_target(x, y, ProblemType.REGRESSION)
    ax = fig.axes[0]
    assert x.name in ax.get_title()
    assert y.name in ax.get_title()
    plt.close(fig)
