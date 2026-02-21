"""
Pure functions for visualizing feature vs target relationships.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from skfeaturellm.types import ProblemType


def plot_feature_vs_target(
    x: pd.Series, y: pd.Series, problem_type: ProblemType
) -> Figure:
    """
    Plot the relationship between a single feature and the target variable.

    For regression, produces a scatter plot (feature vs target).
    For classification, produces a box plot of the feature grouped by class.

    Parameters
    ----------
    x : pd.Series
        Feature values.
    y : pd.Series
        Target values.
    problem_type : ProblemType
        Problem type (classification or regression).

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    x_label = x.name if x.name is not None else "feature"
    y_label = y.name if y.name is not None else "target"

    fig, ax = plt.subplots()

    if problem_type == ProblemType.REGRESSION:
        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:
        classes = sorted(y.unique())
        data = [x[y == cls].dropna().values for cls in classes]
        ax.boxplot(data, tick_labels=classes)
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)

    ax.set_title(f"{x_label} vs {y_label}")
    plt.tight_layout()

    return fig
