"""
Unary binning (discretization) transformations for feature engineering.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from skfeaturellm.transformations.base import TransformationError
from skfeaturellm.transformations.executor import register_transformation
from skfeaturellm.transformations.unary.arithmetic import UnaryTransformation


@register_transformation("bin")
class BinTransformation(UnaryTransformation):
    """
    Binning transformation: discretizes a continuous column into intervals,
    returning string interval labels (e.g. "(0.5, 2.5]").

    Supports two modes via parameters (exactly one must be provided):

    - ``n_bins`` (int >= 2): equal-width bins computed from the data range.
    - ``bin_edges`` (List[float]): custom bin edges for domain-specific thresholds
      (e.g. ``[0, 50000, 100000, 200000]`` for income brackets).

    Useful for converting continuous features into ordinal categories.

    Note: output dtype is object (string). Downstream metrics that require
    numerical input (e.g., Pearson/Spearman correlation) will not be computed
    for this feature and will return NaN with a warning.

    Parameters
    ----------
    feature_name : str
        Name for the resulting feature
    columns : List[str]
        List with exactly one column name
    parameters : Dict[str, Any]
        Must contain exactly one of 'n_bins' (int >= 2) or
        'bin_edges' (list of floats with at least 2 values)

    Examples
    --------
    >>> t = BinTransformation("age_group", columns=["age"], parameters={"n_bins": 4})
    >>> t = BinTransformation(
    ...     "income_bracket",
    ...     columns=["income"],
    ...     parameters={"bin_edges": [0, 30000, 70000, 150000, 1000000]},
    ... )
    """

    def __init__(
        self,
        feature_name: str,
        columns: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(feature_name, columns, parameters)

        n_bins = parameters.get("n_bins") if parameters else None
        bin_edges = parameters.get("bin_edges") if parameters else None

        if n_bins is None and bin_edges is None:
            raise ValueError(
                "BinTransformation requires either 'n_bins' or 'bin_edges' in parameters"
            )
        if n_bins is not None and bin_edges is not None:
            raise ValueError(
                "BinTransformation requires either 'n_bins' or 'bin_edges', not both"
            )

        if n_bins is not None:
            n_bins = int(n_bins)
            if n_bins < 2:
                raise ValueError("'n_bins' must be an integer >= 2")
            self._bins: Union[int, List[float]] = n_bins
        else:
            bin_edges = list(bin_edges)
            if len(bin_edges) < 2:
                raise ValueError("'bin_edges' must contain at least 2 values")
            self._bins = bin_edges

        self.bin_edges_: Optional[List[float]] = None

    @classmethod
    def get_prompt_description(cls) -> str:
        return (
            "Binning - discretizes a continuous column into string interval labels. "
            "Use 'n_bins' (int >= 2) for equal-width bins, "
            "or 'bin_edges' (list of floats) for domain-specific thresholds "
            "(e.g. [0, 50000, 100000, 200000] for income brackets)"
        )

    def fit(self, df: pd.DataFrame) -> "BinTransformation":
        """Learn bin edges from training data when n_bins mode is used."""
        self.validate_columns(df)
        values = df[self._column]
        if isinstance(self._bins, int):
            _, edges = pd.cut(values, bins=self._bins, retbins=True)
            self.bin_edges_ = list(edges)
        else:
            self.bin_edges_ = list(self._bins)
        return self

    def _apply_operation(self, values: pd.Series) -> pd.Series:
        bins: Union[int, List[float]] = (
            self.bin_edges_ if self.bin_edges_ is not None else self._bins
        )
        try:
            return pd.cut(values, bins=bins).astype(str)
        except ValueError as e:
            raise TransformationError(
                f"Binning failed for column '{self._column}': {e}"
            ) from e
