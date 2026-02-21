"""
Module for evaluating the quality of generated features.
"""

from typing import List

import pandas as pd

from skfeaturellm.feature_evaluation import metrics
from skfeaturellm.feature_evaluation.result import FeatureEvaluationResult
from skfeaturellm.types import ProblemType


class FeatureEvaluator:  # pylint: disable=too-few-public-methods
    """Class for evaluating the quality of generated features."""

    def __init__(
        self,
        problem_type: ProblemType,
    ):
        self.problem_type = problem_type

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, features: List[str]
    ) -> FeatureEvaluationResult:
        """
        Evaluate features using various metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        features : List[str]
            List of features to evaluate

        Returns
        -------
        FeatureEvaluationResult
            Result object containing the evaluation metrics
        """
        X_subset = X[features]

        # 1. Quality Metrics (Computationally cheap)
        quality_metrics = self._compute_stability_metrics(X_subset)

        # 2. Relevance Metrics (Problem-type dependent)
        relevance_metrics = self._compute_relevance_metrics(X_subset, y)

        # 3. Combine everything
        full_results = pd.concat([relevance_metrics, quality_metrics], axis=1)

        return FeatureEvaluationResult(full_results)

    def _compute_stability_metrics(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute stability metrics."""
        return pd.DataFrame(
            {
                "missing_pct": metrics.missing_percentage(X),
                "variance": metrics.variance(X),
                "is_constant": metrics.is_constant(X),
            }
        )

    def _compute_relevance_metrics(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute relevance metrics."""
        relevance_metrics_dict = {}

        relevance_metrics_dict["mutual_info"] = metrics.mutual_information(
            X, y, problem_type=self.problem_type
        )

        if self.problem_type == ProblemType.REGRESSION:
            relevance_metrics_dict["spearman_corr"] = (
                metrics.absolute_spearman_correlation(X, y)
            )
            relevance_metrics_dict["pearson_corr"] = (
                metrics.absolute_pearson_correlation(X, y)
            )

        return pd.DataFrame.from_dict(relevance_metrics_dict)
