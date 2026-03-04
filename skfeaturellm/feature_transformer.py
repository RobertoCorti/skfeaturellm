"""
FeatureTransformer: a scikit-learn-compatible transformer backed by a fixed set
of LLM-generated (or manually configured) transformations.

Designed for the production phase after exploration with LLMFeatureEngineer:

    ideas = engineer.fit(X_train, y_train).generated_features
    transformer = engineer.to_transformer()
    pipeline = Pipeline([("features", transformer), ("model", XGBClassifier())])
    pipeline.fit(X_train, y_train)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skfeaturellm.transformations import TransformationPipeline


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer that applies a fixed set of transformations.

    Unlike LLMFeatureEngineer (which calls an LLM during fit), FeatureTransformer
    is fully deterministic — it receives transformation configs at construction time
    and simply fits/applies them. This makes it safe to use inside Pipeline,
    GridSearchCV, cross_val_score, and joblib.

    Parameters
    ----------
    transformations : list of dict
        List of transformation config dicts, each with at minimum a "type" key.
        Same format accepted by TransformationPipeline.from_dict().
    feature_prefix : str, default "llm_feat_"
        Prefix applied to generated feature names.
    raise_on_error : bool, default False
        If True, raise on transformation errors. If False, skip with a warning.

    Attributes
    ----------
    executor_ : TransformationPipeline
        Fitted executor (available after fit()).
    feature_names_in_ : list of str
        Column names seen during fit().

    Examples
    --------
    >>> transformer = FeatureTransformer(
    ...     transformations=[{"type": "log", "feature_name": "log_income", "columns": ["income"]}]
    ... )
    >>> transformer.fit(X_train).transform(X_test)
    """

    def __init__(
        self,
        transformations: Optional[List[Dict[str, Any]]] = None,
        feature_prefix: str = "llm_feat_",
        raise_on_error: bool = False,
    ):
        self.transformations = transformations or []
        self.feature_prefix = feature_prefix
        self.raise_on_error = raise_on_error

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FeatureTransformer":
        """
        Build and fit the transformation executor.

        Parameters
        ----------
        X : pd.DataFrame
            Training data.
        y : pd.Series, optional
            Ignored; present for sklearn API compatibility.

        Returns
        -------
        self
        """
        self.feature_names_in_ = list(X.columns)
        executor = TransformationPipeline.from_dict(
            {"transformations": self.transformations},
            raise_on_error=self.raise_on_error,
        )
        executor.fit(X)
        self.executor_ = executor
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all fitted transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform.

        Returns
        -------
        pd.DataFrame
            Copy of X with new feature columns appended.
        """
        check_is_fitted(self)
        return self.executor_.transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Return feature names for the output of transform().

        Parameters
        ----------
        input_features : list of str, optional
            Ignored; original feature names come from feature_names_in_.

        Returns
        -------
        np.ndarray of str
        """
        check_is_fitted(self)
        generated = [t.feature_name for t in self.executor_.transformations]
        return np.array(self.feature_names_in_ + generated, dtype=object)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save transformer configuration to a JSON file.

        Only the constructor parameters are saved (not the fitted state).
        Call fit() again after loading to restore the fitted executor.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        payload = {
            "transformations": self.transformations,
            "feature_prefix": self.feature_prefix,
            "raise_on_error": self.raise_on_error,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureTransformer":
        """
        Load a FeatureTransformer from a JSON file produced by save().

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        FeatureTransformer
            An unfitted transformer; call fit() before transforming.
        """
        payload = json.loads(Path(path).read_text())
        return cls(
            transformations=payload["transformations"],
            feature_prefix=payload["feature_prefix"],
            raise_on_error=payload["raise_on_error"],
        )
