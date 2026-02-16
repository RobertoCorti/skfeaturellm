"""
Unary arithmetic transformations for feature engineering.
"""

from typing import Set

import numpy as np
import pandas as pd

from skfeaturellm.transformations.base import BaseTransformation, TransformationError
from skfeaturellm.transformations.executor import register_transformation


class InvalidValueError(TransformationError):
    """Raised when a transformation encounters invalid values (e.g., log of negative)."""

    pass


class UnaryTransformation(BaseTransformation):
    """
    Base class for unary transformations (single column operations).

    Parameters
    ----------
    feature_name : str
        Name for the resulting feature
    column : str
        Name of the column to transform
    """

    def __init__(self, feature_name: str, column: str):
        self._feature_name = feature_name
        self._column = column

    @property
    def feature_name(self) -> str:
        return self._feature_name

    def get_required_columns(self) -> Set[str]:
        return {self._column}

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Execute the transformation."""
        self.validate_columns(df)
        values = df[self._column]
        result = self._apply_operation(values)
        result.name = self._feature_name
        return result

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        """Apply the specific unary operation. Must be implemented by subclasses."""
        raise NotImplementedError


@register_transformation("log")
class LogTransformation(UnaryTransformation):
    """
    Natural logarithm transformation: log(column).

    Raises InvalidValueError if any values are <= 0.

    Examples
    --------
    >>> t = LogTransformation("log_income", "income")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Natural logarithm (log(column)) - useful for right-skewed distributions"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        if (values <= 0).any():
            raise InvalidValueError(
                f"Log transformation requires all values > 0 in column '{self._column}'"
            )
        return np.log(values)


@register_transformation("log1p")
class Log1pTransformation(UnaryTransformation):
    """
    Log(1+x) transformation: log(1 + column).

    Useful for data with zeros. Raises InvalidValueError if any values are < 0.

    Examples
    --------
    >>> t = Log1pTransformation("log1p_count", "count")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Log(1+x) transformation (log(1 + column)) - handles zero values"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        if (values < 0).any():
            raise InvalidValueError(
                f"Log1p transformation requires all values >= 0 in column '{self._column}'"
            )
        return np.log1p(values)


@register_transformation("sqrt")
class SqrtTransformation(UnaryTransformation):
    """
    Square root transformation: sqrt(column).

    Raises InvalidValueError if any values are < 0.

    Examples
    --------
    >>> t = SqrtTransformation("sqrt_area", "area")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Square root (sqrt(column)) - reduces right skew"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        if (values < 0).any():
            raise InvalidValueError(
                f"Sqrt transformation requires all values >= 0 in column '{self._column}'"
            )
        return np.sqrt(values)


@register_transformation("abs")
class AbsTransformation(UnaryTransformation):
    """
    Absolute value transformation: abs(column).

    Examples
    --------
    >>> t = AbsTransformation("abs_diff", "difference")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Absolute value (abs(column)) - magnitude regardless of sign"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        return np.abs(values)


@register_transformation("exp")
class ExpTransformation(UnaryTransformation):
    """
    Exponential transformation: exp(column).

    Examples
    --------
    >>> t = ExpTransformation("exp_log_price", "log_price")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Exponential (exp(column)) - inverse of log"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        return np.exp(values)


@register_transformation("square")
class SquareTransformation(UnaryTransformation):
    """
    Square transformation: column ** 2.

    Examples
    --------
    >>> t = SquareTransformation("age_squared", "age")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Square (column ** 2) - captures non-linear relationships"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        return values**2


@register_transformation("cube")
class CubeTransformation(UnaryTransformation):
    """
    Cube transformation: column ** 3.

    Examples
    --------
    >>> t = CubeTransformation("size_cubed", "size")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Cube (column ** 3) - captures stronger non-linear relationships"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        return values**3


@register_transformation("reciprocal")
class ReciprocalTransformation(UnaryTransformation):
    """
    Reciprocal transformation: 1 / column.

    Raises InvalidValueError if any values are 0.

    Examples
    --------
    >>> t = ReciprocalTransformation("inverse_distance", "distance")
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Reciprocal (1 / column) - inverse relationship"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        if (values == 0).any():
            raise InvalidValueError(
                f"Reciprocal transformation requires all values != 0 in column '{self._column}'"
            )
        return 1 / values
