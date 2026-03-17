from inspect import isclass

import pandas as pd

from skfeaturellm.exceptions import NotFittedError


def check_is_fitted(estimator, msg=None):
    if isclass(estimator):
        raise TypeError(f"{estimator} is a class, not an instance.")
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator} is not an estimator instance.")

    if not _is_fitted(estimator):
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def _is_fitted(estimator):
    fitted_attrs = [
        v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
    ]
    return len(fitted_attrs) > 0


def validate_data(
    X,
    y=None,
    *,
    estimator_name: str = "estimator",
) -> None:
    """Validate input data X and optional target y.

    Parameters
    ----------
    X : object
        Input features to validate.
    y : object, optional
        Target variable to validate.
    estimator_name : str
        Name of the estimator, used in error messages.

    Raises
    ------
    ValueError
        If X is not a non-empty DataFrame, or X and y have different lengths.
    TypeError
        If X is not a DataFrame or y is not a Series.
    """

    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            f"[{estimator_name}] X must be a pandas DataFrame, "
            f"got {type(X).__name__!r}"
        )
    if X.empty:
        raise ValueError(f"[{estimator_name}] X must not be empty.")
    if y is not None:
        if not isinstance(y, pd.Series):
            raise ValueError(
                f"[{estimator_name}] y must be a pandas Series or None, "
                f"got {type(y).__name__!r}"
            )
        if len(y) != len(X):
            raise ValueError(
                f"[{estimator_name}] X and y must have the same length, "
                f"got X={len(X)} and y={len(y)}"
            )