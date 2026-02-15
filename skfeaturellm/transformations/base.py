"""
Base classes and protocols for feature transformations.
"""

from abc import ABC, abstractmethod
from typing import List, Set

import pandas as pd


class TransformationError(Exception):
    """Base exception for transformation errors."""

    pass


class ColumnNotFoundError(TransformationError):
    """Raised when a required column is not found in the DataFrame."""

    pass


class BaseTransformation(ABC):
    """
    Abstract base class for all feature transformations.

    Subclasses must implement:
    - execute(): Apply the transformation to a DataFrame
    - get_required_columns(): Return columns needed for the transformation
    - feature_name property: Name of the output feature
    """

    @property
    @abstractmethod
    def feature_name(self) -> str:
        """Name of the resulting feature."""
        pass

    @abstractmethod
    def get_required_columns(self) -> Set[str]:
        """
        Return the set of column names required by this transformation.

        Returns
        -------
        Set[str]
            Set of required column names
        """
        pass

    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.Series:
        """
        Execute the transformation on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame

        Returns
        -------
        pd.Series
            The resulting feature values with name set to feature_name

        Raises
        ------
        TransformationError
            If the transformation fails
        """
        pass

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that all required columns exist in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame

        Raises
        ------
        ColumnNotFoundError
            If any required column is missing
        """
        required = self.get_required_columns()
        missing = required - set(df.columns)
        if missing:
            raise ColumnNotFoundError(
                f"Columns not found in DataFrame: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
