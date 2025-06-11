from typing import Any, Dict, List

import pandas as pd

from skfeaturellm.transformers.base import BaseTransformer


class AdditiveTransformer(BaseTransformer):
    """
    Transformer that sums or subtracts two or more columns together.

    Parameters
    ----------
    feature_name : str
        The name of the feature to be created.
    addend_cols : List[str]
        The columns to add together.
    subtract_cols : List[str]
        The columns to subtract from each other.
    skip_na : bool
        Whether to skip missing values.
    """

    def __init__(
        self,
        feature_name: str,
        addend_cols: List[str],
        subtract_cols: List[str],
        skip_na: bool = False,
    ):
        super().__init__(feature_name)
        self.addend_cols = addend_cols
        self.subtract_cols = subtract_cols
        self.skip_na = skip_na

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return self._sum_cols(df, cols=self.addend_cols) - self._sum_cols(
            df, cols=self.subtract_cols
        )

    def _sum_cols(self, df: pd.DataFrame, cols: List[str]) -> pd.Series:

        n = df.shape[0]
        return (
            df[cols].sum(axis=1, skipna=self.skip_na)
            if cols
            else pd.Series([0] * n, index=df.index, dtype=float)
        )
