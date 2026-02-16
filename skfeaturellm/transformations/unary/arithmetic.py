"""
Unary arithmetic transformations for feature engineering.
"""

from typing import Any, Dict, List, Optional, Set

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
    columns : List[str]
        List with exactly one column name
    parameters : Optional[Dict[str, Any]]
        Optional parameters (not used for basic unary operations)
    """

    def __init__(
        self,
        feature_name: str,
        columns: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        if len(columns) != 1:
            raise ValueError(f"Unary operation requires exactly 1 column, got {len(columns)}")
        
        self._feature_name = feature_name
        self._column = columns[0]
        self._parameters = parameters or {}

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
    >>> t = LogTransformation("log_income", columns=["income"])
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
    >>> t = Log1pTransformation("log1p_count", columns=["count"])
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


@register_transformation("abs")
class AbsTransformation(UnaryTransformation):
    """
    Absolute value transformation: abs(column).

    Examples
    --------
    >>> t = AbsTransformation("abs_diff", columns=["difference"])
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
    >>> t = ExpTransformation("exp_log_price", columns=["log_price"])
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Exponential (exp(column)) - inverse of log"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        return np.exp(values)


@register_transformation("pow")
class PowTransformation(UnaryTransformation):
    """
    Power transformation: column ** power.

    Raises InvalidValueError for invalid operations (e.g., negative base with fractional exponent).

    Examples
    --------
    >>> t = PowTransformation("age_squared", columns=["age"], parameters={"power": 2})
    >>> t = PowTransformation("sqrt_area", columns=["area"], parameters={"power": 0.5})
    >>> t = PowTransformation("inverse_distance", columns=["distance"], parameters={"power": -1})
    """

    def __init__(
        self,
        feature_name: str,
        columns: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(feature_name, columns, parameters)
        
        if parameters is None or "power" not in parameters:
            raise ValueError("PowTransformation requires 'power' in parameters")
        
        self._power = parameters["power"]

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Power (column ** power) - flexible exponentiation for various transformations"

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        # Check for invalid operations
        if self._power < 0 and (values == 0).any():
            raise InvalidValueError(
                f"Power transformation with negative exponent requires all values != 0 in column '{self._column}'"
            )
        
        # Check for fractional powers with negative values
        if not isinstance(self._power, int) and (values < 0).any():
            raise InvalidValueError(
                f"Power transformation with fractional exponent requires all values >= 0 in column '{self._column}'"
            )
        
        return values.astype(float).pow(self._power)
