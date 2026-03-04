"""
Main module for LLM-powered feature engineering.
"""

import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skfeaturellm.feature_engineering_transformer import FeatureEngineeringTransformer
from skfeaturellm.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.schemas import FeatureEngineeringIdea
from skfeaturellm.transformations import TransformationPipeline
from skfeaturellm.types import ProblemType


class LLMFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that uses LLMs for feature engineering.

    Parameters
    ----------
    model_name : str, default="gpt-4"
        Name of the model to use
    problem_type : str
        Machine learning problem type (classification or regression)
    target_col : Optional[str]
        Name of the target column for supervised feature engineering
    max_features : Optional[int]
        Maximum number of features to generate
    feature_prefix : str
        Prefix to add to generated feature names
    **kwargs
        Additional keyword arguments for the LLMInterface
    """

    def __init__(
        self,
        problem_type: str,
        model_name: str = "gpt-4",
        target_col: Optional[str] = None,
        max_features: Optional[int] = None,
        feature_prefix: str = "llm_feat_",
        **kwargs,
    ):
        self.problem_type = ProblemType(problem_type)
        self.model_name = model_name
        self.target_col = target_col
        self.max_features = max_features
        self.feature_prefix = feature_prefix
        self.llm_interface = LLMInterface(model_name=model_name, **kwargs)
        self.generated_features: List[FeatureEngineeringIdea] = []
        self.feature_evaluator = FeatureEvaluator(self.problem_type)

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        feature_descriptions: Optional[List[Dict[str, Any]]] = None,
        target_description: Optional[str] = None,
    ) -> "LLMFeatureEngineer":
        """
        Generate feature engineering ideas using LLM and store the transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[pd.Series]
            Target variable used to compute dataset statistics for the prompt
        feature_descriptions : Optional[List[Dict[str, Any]]]
            List of feature descriptions
        target_description : Optional[str]
            Description of the target variable

        Returns
        -------
        self : LLMFeatureEngineer
            The fitted transformer
        """
        if feature_descriptions is None:
            # Extract feature descriptions from DataFrame
            feature_descriptions = [
                {"name": col, "type": str(X[col].dtype), "description": ""}
                for col in X.columns
            ]

        dataset_statistics = LLMInterface._format_dataset_statistics(
            X, y, self.problem_type
        )

        # Generate feature engineering ideas
        self.generated_features_ideas = self.llm_interface.generate_engineered_features(
            feature_descriptions=feature_descriptions,
            problem_type=self.problem_type.value,
            target_description=target_description,
            max_features=self.max_features,
            dataset_statistics=dataset_statistics,
        ).ideas

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the generated feature transformations to new data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        pd.DataFrame
            Input dataframe with the generated features
        """
        # if fit has not been called, raise an error
        if not hasattr(self, "generated_features_ideas"):
            raise ValueError("fit must be called before transform")

        # Convert LLM output to executor config and apply prefix to feature names
        executor_config = self._build_executor_config(self.generated_features_ideas)

        # Create executor with raise_on_error=False to skip failed transformations
        executor = TransformationPipeline.from_dict(
            executor_config, raise_on_error=False
        )

        # Execute transformations
        result_df = executor.fit(X).transform(X)

        # Track which features were successfully created
        expected_feature_names = [
            f"{self.feature_prefix}{idea.feature_name}"
            for idea in self.generated_features_ideas
        ]
        self.generated_features = [
            idea
            for idea, expected_name in zip(
                self.generated_features_ideas, expected_feature_names
            )
            if expected_name in result_df.columns
        ]

        return result_df

    def to_transformer(
        self, features: Optional[List[str]] = None
    ) -> FeatureEngineeringTransformer:
        """
        Create a FeatureEngineeringTransformer from the successfully generated features.

        Parameters
        ----------
        features : list of str, optional
            Names of features to include. Accepts names with or without the
            feature_prefix. If None, all successfully generated features are
            included.

        Returns
        -------
        FeatureEngineeringTransformer
            Unfitted transformer ready to be used in a Pipeline.

        Raises
        ------
        ValueError
            If fit() has not been called yet.
        """
        if not hasattr(self, "generated_features_ideas"):
            raise ValueError("fit must be called before to_transformer")

        ideas = self.generated_features

        if features is not None:
            features_set = set(features)
            ideas = [
                idea
                for idea in ideas
                if idea.feature_name in features_set
                or f"{self.feature_prefix}{idea.feature_name}" in features_set
            ]

        config = self._build_executor_config(ideas)
        return FeatureEngineeringTransformer(
            transformations=config["transformations"],
            feature_prefix=self.feature_prefix,
            raise_on_error=False,
        )

    def _build_executor_config(
        self, ideas: List[Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build executor configuration from LLM-generated ideas with feature prefix.

        Parameters
        ----------
        ideas : List[FeatureEngineeringIdea]
            List of feature engineering ideas from LLM

        Returns
        -------
        Dict
            Configuration dict for TransformationPipeline.from_dict()
        """
        transformations = []
        for idea in ideas:
            config = idea.to_executor_dict()
            # Apply feature prefix
            config["feature_name"] = f"{self.feature_prefix}{config['feature_name']}"
            transformations.append(config)

        return {"transformations": transformations}

    def evaluate_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_transformed: bool = False,
    ) -> FeatureEvaluationResult:
        """
        Evaluate the quality of generated features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
        is_transformed : bool
            Whether the features have already been transformed

        Returns
        -------
        FeatureEvaluationResult
            Result object containing the evaluation metrics
        """

        if not hasattr(self, "generated_features_ideas"):
            raise ValueError("fit must be called before evaluate_features")

        X_transformed = self.transform(X) if not is_transformed else X

        generated_features_names = [
            f"{self.feature_prefix}{idea.feature_name}"
            for idea in self.generated_features
        ]

        return self.feature_evaluator.evaluate(
            X_transformed, y, features=generated_features_names
        )
