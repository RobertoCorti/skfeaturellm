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
    - transform(): Apply the transformation to a DataFrame (replaces execute())
    - get_required_columns(): Return columns needed for the transformation
    - feature_name property: Name of the output feature
    - get_prompt_description(): Return description for LLM prompts

    The fit/transform pattern mirrors scikit-learn conventions:
    - fit(df): learn any stateful parameters from training data; stateless
      transforms inherit the default no-op implementation.
    - transform(df): apply the transformation using fitted state.
    - fit_transform(df): convenience method combining fit + transform.
    """

    @classmethod
    @abstractmethod
    def get_prompt_description(cls) -> str:
        """
        Return a description of this transformation for use in LLM prompts.

        Returns
        -------
        str
            Human-readable description of what this transformation does
        """
        pass

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

    def fit(self, df: pd.DataFrame) -> "BaseTransformation":
        """
        Fit the transformation to training data.

        The default implementation validates required columns and returns self.
        Stateful subclasses should override this to
        learn parameters from the training data.

        Parameters
        ----------
        df : pd.DataFrame
            The training DataFrame

        Returns
        -------
        BaseTransformation
            self
        """
        self.validate_columns(df)
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the transformation to a DataFrame.

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

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Fit and transform in a single step.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame

        Returns
        -------
        pd.Series
            The resulting feature values
        """
        return self.fit(df).transform(df)

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
