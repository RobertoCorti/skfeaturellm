"""
Binary arithmetic transformations for feature engineering.
"""

from abc import abstractmethod
from typing import Optional, Set, Union

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
    left_column : str
        Name of the left operand column
    right_column : str, optional
        Name of the right operand column (mutually exclusive with right_constant)
    right_constant : float, optional
        Constant value for the right operand (mutually exclusive with right_column)
    """

    def __init__(
        self,
        feature_name: str,
        left_column: str,
        right_column: Optional[str] = None,
        right_constant: Optional[float] = None,
    ):
        if right_column is None and right_constant is None:
            raise ValueError("Must provide either right_column or right_constant")
        if right_column is not None and right_constant is not None:
            raise ValueError("Cannot provide both right_column and right_constant")

        self._feature_name = feature_name
        self._left_column = left_column
        self._right_column = right_column
        self._right_constant = right_constant

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
    >>> t = AddTransformation("total", "a", right_column="b")
    >>> t = AddTransformation("plus_ten", "a", right_constant=10.0)
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Addition (left_column + right_column, or left_column + right_constant)"

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
    >>> t = SubTransformation("difference", "a", right_column="b")
    >>> t = SubTransformation("minus_ten", "a", right_constant=10.0)
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return (
            "Subtraction (left_column - right_column, or left_column - right_constant)"
        )

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
    >>> t = MulTransformation("product", "a", right_column="b")
    >>> t = MulTransformation("doubled", "a", right_constant=2.0)
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Multiplication (left_column * right_column, or left_column * right_constant)"

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
    >>> t = DivTransformation("ratio", "a", right_column="b")
    >>> t = DivTransformation("halved", "a", right_constant=2.0)
    """

    @classmethod
    def get_prompt_description(cls) -> str:
        return "Division (left_column / right_column, or left_column / right_constant)"

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
