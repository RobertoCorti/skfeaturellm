"""
Main module for LLM-powered feature engineering.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skfeaturellm.feature_evaluation import FeatureEvaluator
from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.reporting import FeatureReport


class LLMFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that uses LLMs for feature engineering.

    Parameters
    ----------
    model_name : str, default="gpt-4"
        Name of the model to use
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
        model_name: str = "gpt-4",
        target_col: Optional[str] = None,
        max_features: Optional[int] = None,
        feature_prefix: str = "llm_feat_",
        **kwargs,
    ):
        self.model_name = model_name
        self.target_col = target_col
        self.max_features = max_features
        self.feature_prefix = feature_prefix
        self.llm_interface = LLMInterface(model_name=model_name, **kwargs)
        self.feature_evaluator = FeatureEvaluator()
        self.generated_features: List[Dict[str, Any]] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,  # pylint: disable=unused-argument
        feature_descriptions: Optional[List[Dict[str, Any]]] = None,
    ) -> "LLMFeatureEngineer":
        """
        Generate feature engineering ideas using LLM and store the transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[pd.Series]
            Target variable for supervised feature engineering
        feature_descriptions : Optional[List[Dict[str, Any]]]
            List of feature descriptions

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
            target_description=self.target_col,
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
            Transformed features
        """
        # if fit has not been called, raise an error
        if not hasattr(self, "generated_features"):
            raise ValueError("fit must be called before transform")

        # apply the transformations
        for generated_feature_idea in self.generated_features_ideas:
            feature_idea_func = self._parse_feature_idea(generated_feature_idea)

            if feature_idea_func is None:
                warnings.warn(
                    f"The formula {generated_feature_idea.formula} is not a valid lambda function. Skipping feature {generated_feature_idea.name}."
                )
                continue

            X[generated_feature_idea.name] = feature_idea_func(X)

        return X

    def _parse_feature_idea(
        self, generated_feature_idea: Dict[str, Any]
    ) -> Optional[Callable]:
        """
        Parse a feature idea into a formula.

        Parameters
        ----------
        generated_feature_idea : Dict[str, Any]
            A feature idea

        Returns
        -------
        Optional[Callable]
            The formula as a lambda function
        """
        try:
            generated_feature_idea_formula_str = generated_feature_idea.formula
            generated_feature_idea_formula = eval(generated_feature_idea_formula_str)

            if not callable(generated_feature_idea_formula) or not isinstance(
                generated_feature_idea_formula, type(lambda: None)
            ):
                raise TypeError("The evaluated result is not a lambda function.")

            return generated_feature_idea_formula
        except TypeError:
            return None

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
        y: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of generated features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : Optional[pd.Series]
            Target variable for supervised evaluation
        metrics : Optional[List[str]]
            List of metrics to compute

        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation metrics
        """
        pass

    def generate_report(self) -> FeatureReport:
        """
        Generate a comprehensive report about the engineered features.

        Returns
        -------
        FeatureReport
            Report containing feature statistics and insights
        """
        pass


if __name__ == "__main__":

    ## create artificial data with 5 columns: name, age, income, education, city, credit_score and a target column called default
    X = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "income": [50000, 60000, 70000, 80000, 90000],
            "education": ["Bachelor", "Master", "PhD", "Bachelor", "Master"],
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Miami"],
            "credit_score": [650, 700, 750, 800, 850],
            "default": [0, 1, 0, 1, 0],
        }
    )

    ## create a target column called default
    y = X["default"]

    ## create a feature description
    feature_descriptions = [
        {"name": "name", "type": "str", "description": "The name of the customer"},
        {"name": "age", "type": "int", "description": "The age of the customer"},
        {"name": "income", "type": "int", "description": "The income of the customer"},
        {
            "name": "education",
            "type": "str",
            "description": "The education level of the customer",
        },
        {"name": "city", "type": "str", "description": "The city of the customer"},
        {
            "name": "credit_score",
            "type": "int",
            "description": "The credit score of the customer",
        },
    ]

    ## create a LLMFeatureEngineer
    llm_feature_engineer = LLMFeatureEngineer()

    ## fit the LLMFeatureEngineer
    llm_feature_engineer.fit(X, y, feature_descriptions=feature_descriptions)

    ## transform the data
    X_transformed = llm_feature_engineer.transform(X)

    print(X_transformed)
