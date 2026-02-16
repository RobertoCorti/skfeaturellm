"""
Module for handling interactions with Language Models.
"""

from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from skfeaturellm.prompts import FEATURE_ENGINEERING_PROMPT
from skfeaturellm.schemas import (
    FeatureDescription,
    FeatureDescriptions,
    FeatureEngineeringIdeas,
)
from skfeaturellm.transformations import get_transformation_types_for_prompt
from skfeaturellm.types import ProblemType


class LLMInterface:
    """
    Interface for interacting with Language Models for feature engineering.

    Parameters
    ----------
    model_name : str, default="gpt-4o"
        Name of the model to use
    **kwargs
        Additional keyword arguments passed to init_chat_model
        (e.g., temperature, max_tokens, api_key, etc.)
    """

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        # Initialize the base model
        base_llm = init_chat_model(model=model_name, **kwargs)

        # Use with_structured_output for reliable structured responses
        # This uses the provider's native structured output capabilities
        self.llm = base_llm.with_structured_output(FeatureEngineeringIdeas)

        # Create prompt template with system and human messages
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", FEATURE_ENGINEERING_PROMPT),
                (
                    "human",
                    "Generate feature engineering ideas based on the dataset information provided.",
                ),
            ]
        )

        # Chain composition - no output parser needed with structured output
        self.chain = self.prompt_template | self.llm

    def generate_engineered_features(
        self,
        feature_descriptions: List[FeatureDescription],
        target_description: Optional[str] = None,
        max_features: Optional[int] = None,
        problem_type: Optional[ProblemType] = None,
    ) -> FeatureEngineeringIdeas:
        """
        Generate feature engineering ideas.

        Parameters
        ----------
        feature_descriptions : List[FeatureDescription]
            Descriptions for input features
        target_description : Optional[str]
            Description of the target variable and task
        max_features : Optional[int]
            Maximum number of features to generate

        Returns
        -------
        FeatureEngineeringIdeas
            Generated feature engineering ideas
        """

        prompt_context = self.generate_prompt_context(
            feature_descriptions=feature_descriptions,
            target_description=target_description,
            problem_type=problem_type,
            max_features=max_features,
        )

        return self.chain.invoke(prompt_context)

    def _format_feature_descriptions(self, features: List[FeatureDescription]) -> str:
        """
        Format feature descriptions in a human-readable way.

        Parameters
        ----------
        features : List[FeatureDescription]
            List of feature descriptions

        Returns
        -------
        str
            Formatted feature descriptions
        """
        formatted_features = []
        for feature in features:
            formatted_feature = (
                f"- {feature.name} ({feature.type}): {feature.description}"
            )
            formatted_features.append(formatted_feature)
        return "\n".join(formatted_features)

    def generate_prompt_context(
        self,
        feature_descriptions: List[Dict[str, str]],
        target_description: Optional[str] = None,
        max_features: Optional[int] = None,
        problem_type: Optional[ProblemType] = None,
    ) -> str:
        """
        Generate the prompt for the LLM.

        Parameters
        ----------
        feature_descriptions : List[Dict[str, str]]
            List of dictionaries containing feature descriptions
        target_description : Optional[str]
            Description of the target variable and task
        max_features : Optional[int]
            Maximum number of features to generate

        Returns
        -------
        str
            Formatted prompt
        """
        # Convert dictionaries to FeatureDescription objects
        feature_descriptions_list = [
            FeatureDescription(**feature) for feature in feature_descriptions
        ]
        feature_descriptions_schema = FeatureDescriptions(
            features=feature_descriptions_list
        )

        problem_type_message = (
            f"This is a supervised {problem_type} problem."
            if problem_type is not None
            else "Not specified."
        )
        target_description_message = (
            target_description if target_description is not None else "Not specified."
        )

        additional_context = (
            f"Generate up to {max_features} features." if max_features else ""
        )

        transformation_types = get_transformation_types_for_prompt()

        return {
            "feature_descriptions": feature_descriptions_schema.format(),
            "problem_type": problem_type_message,
            "target_description": target_description_message,
            "additional_context": additional_context,
            "transformation_types": transformation_types,
        }
