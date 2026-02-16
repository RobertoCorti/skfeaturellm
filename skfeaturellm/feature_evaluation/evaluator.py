"""
Module for evaluating the quality of generated features.
"""

import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

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
        quality_metrics = pd.DataFrame(index=features)
        quality_metrics["missing_pct"] = X_subset.isnull().mean()
        quality_metrics["variance"] = X_subset.var()
        # "Stability" check: is it effectively constant?
        quality_metrics["is_constant"] = quality_metrics["variance"] == 0

        # 2. Relevance Metrics (Problem-type dependent)
        if self.problem_type == ProblemType.REGRESSION:
            relevance = self._compute_regression_metrics(X_subset, y)
        elif self.problem_type == ProblemType.CLASSIFICATION:
            relevance = self._compute_classification_metrics(X_subset, y)
        else:
            relevance = pd.DataFrame(index=features)

        # Combine everything
        full_results = pd.concat([relevance, quality_metrics], axis=1)

        return FeatureEvaluationResult(full_results, primary_metric="mutual_info")

    def _compute_regression_metrics(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Compute regression-specific metrics."""
        return pd.DataFrame(
            {
                "mutual_info": self._compute_mutual_information(
                    X, y, mi_func=mutual_info_regression
                ),
                "spearman_corr": X.corrwith(y, method="spearman").abs(),
                "pearson_corr": X.corrwith(y, method="pearson").abs(),
            }
        )

    def _compute_classification_metrics(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Compute classification-specific metrics."""
        return pd.DataFrame(
            {
                "mutual_info": self._compute_mutual_information(
                    X, y, mi_func=mutual_info_classif
                ),
            }
        )

    def _compute_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        mi_func: callable = mutual_info_classif,
    ) -> pd.Series:
        """Compute mutual information between features and target.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        mi_func : callable
            Mutual information function (mutual_info_classif or mutual_info_regression)
        """
        X = X.copy()
        mi_scores = []

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) > 0:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

        for col in X.columns:
            xi = X[col]
            valid_mask = ~xi.isna() & ~y.isna()
            if valid_mask.sum() == 0:
                mi_scores.append(np.nan)
                continue

            xi_valid = xi[valid_mask].values.reshape(-1, 1)
            y_valid = y[valid_mask].values
            discrete = pd.api.types.is_integer_dtype(xi) or col in categorical_cols

            try:
                mi = mi_func(xi_valid, y_valid, discrete_features=discrete)
                mi_scores.append(mi[0])
            except ValueError as e:
                warnings.warn(
                    f"Error computing mutual information for feature {col}: {e}",
                    stacklevel=2,
                )
                mi_scores.append(np.nan)

        return pd.Series(mi_scores, index=X.columns, name="mutual_information")
