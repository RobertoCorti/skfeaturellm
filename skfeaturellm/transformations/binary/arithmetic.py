"""
Binary arithmetic transformations for feature engineering.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from skfeaturellm.transformations.base import (
    BaseTransformation,
    TransformationError,
)
from skfeaturellm.transformations.executor import register_transformation


class DivisionByZeroError(TransformationError):
    """Raised when a division by zero is detected."""

    pass


class BinaryArithmeticTransformation(BaseTransformation):
    """
    Base class for binary arithmetic transformations.

    Supports operations between two columns or between a column and a constant.

    Parameters
    ----------
    feature_name : str
        Name for the resulting feature
    columns : List[str]
        List of column names (1 or 2 columns)
    parameters : Optional[Dict[str, Any]]
        Optional parameters dict with 'constant' key for column-constant operations
    """

    def __init__(
        self,
        feature_name: str,
        columns: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        if len(columns) == 1:
            # Column-constant operation: must have constant in parameters
            if parameters is None or "constant" not in parameters:
                raise ValueError(
                    "Binary operation with 1 column requires 'constant' in parameters"
                )
            self._left_column = columns[0]
            self._right_column = None
            self._right_constant = parameters["constant"]
        elif len(columns) == 2:
            # Column-column operation
            if parameters is not None and "constant" in parameters:
                raise ValueError(
                    "Binary operation with 2 columns should not have 'constant' in parameters"
                )
            self._left_column = columns[0]
            self._right_column = columns[1]
            self._right_constant = None
        else:
            raise ValueError(
                f"Binary operation requires 1 or 2 columns, got {len(columns)}"
            )

        self._feature_name = feature_name

    @property
    def feature_name(self) -> str:
        return self._feature_name

    def get_required_columns(self) -> Set[str]:
        columns = {self._left_column}
        if self._right_column is not None:
            columns.add(self._right_column)
        return columns

    def _get_operands(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, Union[pd.Series, float]]:
        """Get left and right operands from the DataFrame."""
        left = df[self._left_column]

        if self._right_column is not None:
            right: Union[pd.Series, float] = df[self._right_column]
        else:
            right = self._right_constant  # type: ignore

        return left, right

    @abstractmethod
    def _apply_operation(
        self, left: pd.Series, right: Union[pd.Series, float]
    ) -> pd.Series:
        """Apply the specific arithmetic operation."""
        pass

    def execute(self, df: pd.DataFrame) -> pd.Series:
        """Execute the transformation."""
        self.validate_columns(df)
        left, right = self._get_operands(df)
        result = self._apply_operation(left, right)
        result.name = self._feature_name
        return result


@register_transformation("add")
class AddTransformation(BinaryArithmeticTransformation):
    """
    Addition transformation: left + right.

    Examples
    --------
    >>> t = AddTransformation("total", columns=["a", "b"])
    >>> t = AddTransformation("plus_ten", columns=["a"], parameters={"constant": 10.0})
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Addition of two columns or a column and a constant"

    def _apply_operation(
        self, left: pd.Series, right: Union[pd.Series, float]
    ) -> pd.Series:
        return left + right


@register_transformation("sub")
class SubTransformation(BinaryArithmeticTransformation):
    """
    Subtraction transformation: left - right.

    Examples
    --------
    >>> t = SubTransformation("difference", columns=["a", "b"])
    >>> t = SubTransformation("minus_ten", columns=["a"], parameters={"constant": 10.0})
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Subtraction of two columns or a column and a constant"

    def _apply_operation(
        self, left: pd.Series, right: Union[pd.Series, float]
    ) -> pd.Series:
        return left - right


@register_transformation("mul")
class MulTransformation(BinaryArithmeticTransformation):
    """
    Multiplication transformation: left * right.

    Examples
    --------
    >>> t = MulTransformation("product", columns=["a", "b"])
    >>> t = MulTransformation("doubled", columns=["a"], parameters={"constant": 2.0})
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Multiplication of two columns or a column and a constant"

    def _apply_operation(
        self, left: pd.Series, right: Union[pd.Series, float]
    ) -> pd.Series:
        return left * right


@register_transformation("div")
class DivTransformation(BinaryArithmeticTransformation):
    """
    Division transformation: left / right.

    Raises DivisionByZeroError if division by zero is detected.

    Examples
    --------
    >>> t = DivTransformation("ratio", columns=["a", "b"])
    >>> t = DivTransformation("halved", columns=["a"], parameters={"constant": 2.0})
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Division of two columns or a column and a constant"

    def _apply_operation(
        self, left: pd.Series, right: Union[pd.Series, float]
    ) -> pd.Series:
        self._check_division_by_zero(right)
        return left / right

    def _check_division_by_zero(self, right: Union[pd.Series, float]) -> None:
        """Check for division by zero and raise if detected."""
        if isinstance(right, pd.Series):
            if (right == 0).any():
                raise DivisionByZeroError(
                    f"Division by zero detected in column '{self._right_column}'"
                )
        elif right == 0:
            raise DivisionByZeroError("Division by zero: constant is 0")
