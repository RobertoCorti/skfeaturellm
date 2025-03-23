"""
Module for evaluating the quality of generated features.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeatureEvaluator:  # pylint: disable=too-few-public-methods
    """Class for evaluating the quality of generated features."""

    def evaluate(
        self,
        X: pd.DataFrame,
        feature_specs: List[Dict[str, Any]],  # pylint: disable=unused-argument
        y: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate features using various metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        feature_specs : List[Dict[str, Any]]
            Specifications of generated features
        y : Optional[pd.Series]
            Target variable for supervised evaluation
        metrics : Optional[List[str]]
            List of metrics to compute

        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation metrics
        """
        if metrics is None:
            metrics = ["correlation", "mutual_information", "variance"]

        results = {}

        if "correlation" in metrics:
            results["correlation"] = self._compute_correlations(X)

        if "mutual_information" in metrics and y is not None:
            results["mutual_information"] = self._compute_mutual_information(X, y)

        if "variance" in metrics:
            results["variance"] = self._compute_variance(X)

        return results

    def _compute_correlations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for features."""
        return X.corr()

    def _compute_mutual_information(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Compute mutual information between features and target."""
        if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
        return dict(zip(X.columns, mi_scores))

    def _compute_variance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Compute variance of features."""
        return X.var().to_dict()
