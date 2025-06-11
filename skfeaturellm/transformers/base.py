from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseTransformer(ABC):
    """
    Abstract base class for all feature transformation classes.
    """

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply the transformation to the dataframe and return the new feature as a Series.
        """
        pass
