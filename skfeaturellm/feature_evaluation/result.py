from typing import Any, Dict, Optional

import pandas as pd


class FeatureEvaluationResult:
    """
    Class for storing and presenting feature evaluation results.
    """

    def __init__(self, metrics_df: pd.DataFrame, primary_metric: Optional[str] = None):
        self._metrics_df = metrics_df
        self._primary_metric = primary_metric

    @property
    def summary(self) -> pd.DataFrame:
        """
        Returns the metrics DataFrame sorted by the primary metric descending.
        """
        sort_col = self._primary_metric or self._metrics_df.columns[0]
        return self._metrics_df.sort_values(by=sort_col, ascending=False)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to dictionary.
        """
        return self._metrics_df.to_dict()

    def __repr__(self) -> str:
        return self.summary.__repr__()
