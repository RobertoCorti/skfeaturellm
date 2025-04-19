"""
Module for evaluating the quality of generated features.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from skfeaturellm.types import ProblemType


class FeatureEvaluator:  # pylint: disable=too-few-public-methods
    """Class for evaluating the quality of generated features."""

    def __init__(
        self,
        problem_type: ProblemType,
    ):
        self.problem_type = problem_type
        self.results = {}

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, features: List[str]
    ) -> pd.DataFrame:
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
        pd.DataFrame
            DataFrame with features as rows and metrics as columns
        """

        if self.problem_type == ProblemType.CLASSIFICATION:
            self.results["mutual_information"] = self._compute_mutual_information(
                X[features], y
            )
        elif self.problem_type == ProblemType.REGRESSION:
            self.results["correlation"] = self._compute_correlation(X[features], y)

        return self._format_results()

    def _format_results(self) -> pd.DataFrame:
        """
        Format the results into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with features as rows and metrics as columns
        """

        if self.results is None:
            raise ValueError("No results to format")

        return pd.DataFrame(self.results).sort_values(
            by=list(self.results.keys())[0], ascending=False
        )

    def _compute_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Compute mutual information between features and target."""
        ## compute mutual information for each feature and target, be NaN robust

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
                mi = mutual_info_classif(xi_valid, y_valid, discrete_features=discrete)
                mi_scores.append(mi[0])
            except Exception as e:
                print(f"Error computing mutual information for feature {col}: {e}")
                mi_scores.append(np.nan)

        return pd.Series(mi_scores, index=X.columns, name="mutual_information")

    def _compute_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """Compute correlation between features and target."""
        return X.corrwith(y)
