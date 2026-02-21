import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from skfeaturellm.types import ProblemType


def missing_percentage(X: pd.DataFrame) -> pd.Series:
    """Compute missing percentage for each feature."""
    return X.isnull().mean()


def variance(X: pd.DataFrame) -> pd.Series:
    """Compute variance for each feature."""
    return X.var()


def is_constant(X: pd.DataFrame) -> pd.Series:
    """Check if a feature is constant."""
    return X.var() == 0


def absolute_spearman_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute Spearman correlation between features and target."""
    return X.corrwith(y, method="spearman").abs()


def absolute_pearson_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute Pearson correlation between features and target."""
    return X.corrwith(y, method="pearson").abs()


def mutual_information(
    X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
) -> pd.Series:
    """Compute mutual information between features and target.

    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    problem_type : ProblemType
        Problem type (classification or regression)
    """

    if problem_type == ProblemType.CLASSIFICATION:
        mi_func = mutual_info_classif
    else:
        mi_func = mutual_info_regression

    X = X.copy()
    mi_scores = []

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
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
