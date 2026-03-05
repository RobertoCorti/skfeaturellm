"""
Main module for LLM-powered feature engineering.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin

from skfeaturellm.feature_engineering_transformer import FeatureEngineeringTransformer
from skfeaturellm.feature_evaluation import FeatureEvaluationResult, FeatureEvaluator
from skfeaturellm.llm_interface import LLMInterface
from skfeaturellm.schemas import FeatureEngineeringIdea
from skfeaturellm.transformations import TransformationPipeline
from skfeaturellm.types import ProblemType


class LLMFeatureEngineer(
    BaseEstimator, TransformerMixin
):  # pylint: disable=too-many-instance-attributes
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
    verbose : int, default=0
        Verbosity level for fit_selective().
        0 = silent, 1 = one line per round, 2 = include selected feature names.
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
        verbose: int = 0,
        **kwargs,
    ):
        self.problem_type = ProblemType(problem_type)
        self.model_name = model_name
        self.target_col = target_col
        self.max_features = max_features
        self.feature_prefix = feature_prefix
        self.verbose = verbose
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

    def fit_selective(  # pylint: disable=too-many-arguments
        self,
        X: pd.DataFrame,
        y: pd.Series,
        selector: SelectorMixin,
        n_rounds: int = 3,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        feature_descriptions: Optional[List[Dict[str, Any]]] = None,
        target_description: Optional[str] = None,
    ) -> "LLMFeatureEngineer":
        """
        Iteratively generate and select features using an LLM and a feature selector.

        In each round the LLM proposes new features, the selector is fitted on the
        generated features (using ``eval_set`` if provided, otherwise training data),
        and the selection results are fed back to the LLM as context for the next
        round. Only the features that survive selection across all rounds are kept.

        Parameters
        ----------
        X : pd.DataFrame
            Training features. Transformations are always fitted on this data.
        y : pd.Series
            Training target.
        selector : SelectorMixin
            An initialised scikit-learn–compatible selector (e.g.
            ``SelectKBest(k=5)``, ``SelectFromModel(RandomForestClassifier())``).
        n_rounds : int, default=3
            Number of generate→select→feedback rounds.
        eval_set : tuple of (pd.DataFrame, pd.Series), optional
            Validation data ``(X_val, y_val)``. When provided the selector is
            fitted on the validation features so that selection reflects
            generalisation, not training performance.
        feature_descriptions : list of dict, optional
            Descriptions for input features. Auto-detected from ``X`` if omitted.
        target_description : str, optional
            Description of the target variable passed to the LLM.

        Returns
        -------
        self : LLMFeatureEngineer
            The fitted transformer. Call ``transform()`` to apply the selected
            features and ``to_transformer()`` to export them for production.
        """
        if feature_descriptions is None:
            feature_descriptions = [
                {"name": col, "type": str(X[col].dtype), "description": ""}
                for col in X.columns
            ]

        dataset_statistics = LLMInterface._format_dataset_statistics(
            X, y, self.problem_type
        )
        prompt_context = self.llm_interface.generate_prompt_context(
            feature_descriptions=feature_descriptions,
            target_description=target_description,
            problem_type=self.problem_type.value,
            max_features=self.max_features,
            dataset_statistics=dataset_statistics,
        )

        X_eval, y_eval = eval_set if eval_set is not None else (None, None)

        conversation_history: list = []
        all_selected_ideas: List[Any] = []
        feedback_context = None

        for round_idx in range(n_rounds):
            if self.verbose == 1:
                print(
                    f"[fit_selective] Round {round_idx + 1}/{n_rounds}: querying LLM...",
                    flush=True,
                )
            elif self.verbose >= 2:
                print(f"[fit_selective] Round {round_idx + 1}/{n_rounds}")
                print("  Querying LLM...", flush=True)

            ideas_result, conversation_history = (
                self.llm_interface.generate_engineered_features_iterative(
                    prompt_context=prompt_context,
                    conversation_history=conversation_history,
                    feedback_context=feedback_context,
                )
            )

            if self.verbose >= 2:
                print(f"  Generated {len(ideas_result.ideas)} idea(s)")

            selected, rejected, scores = self._run_selector(
                X, y, ideas_result.ideas, selector, X_eval, y_eval
            )
            all_selected_ideas.extend(selected)

            if self.verbose == 1:
                print(
                    f"[fit_selective] Round {round_idx + 1}/{n_rounds}: "
                    f"generated={len(ideas_result.ideas)}, "
                    f"selected={len(selected)}, "
                    f"rejected={len(rejected)}"
                )
            elif self.verbose >= 2:
                print(f"  Selected: {len(selected)} | Rejected: {len(rejected)}")
                for idea in selected:
                    score = scores.get(f"{self.feature_prefix}{idea.feature_name}")
                    score_str = f"  score={score:.4g}" if score is not None else ""
                    print(f"    + {self.feature_prefix}{idea.feature_name}{score_str}")
                for idea in rejected:
                    score = scores.get(f"{self.feature_prefix}{idea.feature_name}")
                    score_str = f"  score={score:.4g}" if score is not None else ""
                    print(f"    - {self.feature_prefix}{idea.feature_name}{score_str}")

            feedback_context = self._build_feedback_context(
                selected_ideas=selected,
                rejected_ideas=rejected,
                scores=scores,
                max_features=self.max_features or 10,
            )

        if self.verbose >= 1:
            print(
                f"[fit_selective] Done. Total selected features: {len(all_selected_ideas)}"
            )

        self.generated_features_ideas = all_selected_ideas
        return self

    def _run_selector(  # pylint: disable=too-many-arguments
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ideas: List[Any],
        selector: SelectorMixin,
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
    ) -> Tuple[List[Any], List[Any], Dict[str, float]]:
        """
        Apply a round's ideas to X, fit the selector, and return selected/rejected.

        Transformations are always fitted on ``X`` (training data). If ``X_eval``
        is provided the selector is fitted on the transformed validation features
        so that selection reflects generalisation performance.

        Returns
        -------
        selected_ideas, rejected_ideas, scores
            ``scores`` maps prefixed feature names to raw selector scores.
        """
        if self.verbose >= 2:
            print("  Applying transformations...", flush=True)

        executor_config = self._build_executor_config(ideas)
        executor = TransformationPipeline.from_dict(
            executor_config, raise_on_error=False
        )
        executor.fit(X)

        expected_names = [f"{self.feature_prefix}{idea.feature_name}" for idea in ideas]

        if X_eval is not None:
            X_transformed = executor.transform(X_eval)
            y_sel = y_eval
        else:
            X_transformed = executor.transform(X)
            y_sel = y

        created_pairs = [
            (idea, name)
            for idea, name in zip(ideas, expected_names)
            if name in X_transformed.columns
        ]

        if self.verbose >= 2:
            n_orig = len(X_transformed.columns) - len(created_pairs)
            print(
                f"  Created {len(created_pairs)}/{len(ideas)} feature(s). "
                f"Running selector on {len(X_transformed.columns)} feature(s) "
                f"({n_orig} original + {len(created_pairs)} new)...",
                flush=True,
            )

        if not created_pairs:
            return [], list(ideas), {}

        created_ideas, created_names = zip(*created_pairs)
        created_names_list = list(created_names)

        selector.fit(X_transformed, y_sel)
        full_mask = selector.get_support()
        all_cols = X_transformed.columns.tolist()
        new_feat_indices = [all_cols.index(name) for name in created_names_list]
        mask = full_mask[new_feat_indices]

        scores: Dict[str, float] = {}
        if hasattr(selector, "scores_"):
            scores = {
                name: selector.scores_[i]
                for name, i in zip(created_names_list, new_feat_indices)
            }
        elif hasattr(selector, "ranking_"):
            scores = {
                name: 1.0 / selector.ranking_[i]
                for name, i in zip(created_names_list, new_feat_indices)
            }

        created_names_set = {idea.feature_name for idea in created_ideas}
        selected = [idea for idea, sel in zip(created_ideas, mask) if sel]
        rejected = [idea for idea, sel in zip(created_ideas, mask) if not sel]
        rejected += [
            idea for idea in ideas if idea.feature_name not in created_names_set
        ]

        return selected, rejected, scores

    def _build_feedback_context(  # pylint: disable=too-many-arguments
        self,
        selected_ideas: List[Any],
        rejected_ideas: List[Any],
        scores: Dict[str, float],
        max_features: int,
    ) -> Dict[str, Any]:
        """Build the feedback context dict for SELECTION_FEEDBACK_PROMPT."""

        def _to_table(ideas: List[Any]) -> str:
            if not ideas:
                return "None"
            rows = []
            for idea in ideas:
                prefixed = f"{self.feature_prefix}{idea.feature_name}"
                score = scores.get(prefixed)
                rows.append(
                    {
                        "feature": prefixed,
                        "type": idea.type,
                        "score": f"{score:.4g}" if score is not None else "N/A",
                    }
                )
            return pd.DataFrame(rows).to_markdown(index=False)

        return {
            "selected_features_table": _to_table(selected_ideas),
            "rejected_features_table": _to_table(rejected_ideas),
            "max_features": max_features,
        }

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
