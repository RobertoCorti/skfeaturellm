"""
Main module for LLM-powered feature engineering.
"""

import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skfeaturellm.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.reporting import FeatureReport
from skfeaturellm.schemas import FeatureEngineeringIdeas
from skfeaturellm.transformations import TransformationExecutor
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
        self.generated_features: List[Dict[str, Any]] = []
        self.feature_evaluator = FeatureEvaluator(self.problem_type)

    def fit(
        self,
        X: pd.DataFrame,
        feature_descriptions: Optional[List[Dict[str, Any]]] = None,
        target_description: Optional[str] = None,
    ) -> "LLMFeatureEngineer":
        """
        Generate feature engineering ideas using LLM and store the transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
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

        # Generate feature engineering ideas
        self.generated_features_ideas = self.llm_interface.generate_engineered_features(
            feature_descriptions=feature_descriptions,
            problem_type=self.problem_type.value,
            target_description=target_description,
            max_features=self.max_features,
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
        executor = TransformationExecutor.from_dict(
            executor_config, raise_on_error=False
        )

        # Execute transformations
        result_df = executor.execute(X)

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
            Configuration dict for TransformationExecutor.from_dict()
        """
        transformations = []
        for idea in ideas:
            config = idea.to_executor_dict()
            # Apply feature prefix
            config["feature_name"] = f"{self.feature_prefix}{config['feature_name']}"
            transformations.append(config)

        return {"transformations": transformations}

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any
    ) -> pd.DataFrame:
        """
        Generate features and transform the input data in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[pd.Series]
            Target variable for supervised feature engineering

        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        return self.fit(X, y).transform(X)

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

        if not hasattr(self, "generated_features"):
            raise ValueError("fit must be called before evaluate_features")

        generated_features_names = [
            f"{self.feature_prefix}{idea.feature_name}"
            for idea in self.generated_features
        ]

        X_transformed = self.transform(X) if not is_transformed else X

        return self.feature_evaluator.evaluate(
            X_transformed, y, features=generated_features_names
        )

    def generate_report(self) -> FeatureReport:
        """
        Generate a comprehensive report about the engineered features.

        Returns
        -------
        FeatureReport
            Report containing feature statistics and insights
        """
        raise NotImplementedError("This feature is not yet implemented.")


if __name__ == "__main__":

    ## dataset of home prices
    df = pd.DataFrame(
        data={
            "size": [1000, 1500, 2000, 2500, 3000],
            "price": [100000, 150000, 200000, 250000, 300000],
            "bedrooms": [3, 4, 5, 6, 7],
            "bathrooms": [2, 2.5, 3, 3.5, 4],
            "floors": [1, 1.5, 2, 2.5, 3],
            "garage": [1, 2, 3, 4, 5],
            "year_built": [2010, 2011, 2012, 2013, 2014],
            "zip_code": ["10001", "10002", "10003", "10004", "10005"],
        }
    )

    llm_feature_engineer = LLMFeatureEngineer(problem_type="regression")
    llm_feature_engineer.fit(df, target_description="Price of the home")
    print(llm_feature_engineer.generated_features_ideas)

    df_fe = llm_feature_engineer.transform(df)
    print(df_fe)
