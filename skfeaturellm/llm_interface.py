"""
Module for handling interactions with Language Models.
"""

from typing import Dict, List, Optional, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from skfeaturellm.prompts import FEATURE_ENGINEERING_PROMPT, SELECTION_FEEDBACK_PROMPT
from skfeaturellm.prompts import utils as prompt_utils
from skfeaturellm.schemas import (
    FeatureDescription,
    FeatureDescriptions,
    FeatureEngineeringIdeas,
)
from skfeaturellm.transformations import (
    get_binary_operation_types,
    get_transformation_types_for_prompt,
    get_unary_operation_types,
)
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
        dataset_statistics: Optional[str] = None,
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
        dataset_statistics : Optional[str]
            Pre-formatted dataset statistics string

        Returns
        -------
        FeatureEngineeringIdeas
            Generated feature engineering ideas
        """

        prompt_context = prompt_utils.generate_prompt_context(
            feature_descriptions=feature_descriptions,
            target_description=target_description,
            problem_type=problem_type,
            max_features=max_features,
            dataset_statistics=dataset_statistics,
        )

        return self.chain.invoke(prompt_context)

    def generate_prompt_context(
        self,
        feature_descriptions: List[Dict[str, str]],
        target_description: Optional[str] = None,
        max_features: Optional[int] = None,
        problem_type: Optional[ProblemType] = None,
        dataset_statistics: Optional[str] = None,
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
        dataset_statistics : Optional[str]
            Pre-formatted dataset statistics string from _format_dataset_statistics

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
        unary_types = ", ".join(sorted(get_unary_operation_types()))
        binary_types = ", ".join(sorted(get_binary_operation_types()))

        dataset_statistics_message = (
            dataset_statistics if dataset_statistics is not None else "Not provided."
        )

        return {
            "feature_descriptions": feature_descriptions_schema.format(),
            "problem_type": problem_type_message,
            "target_description": target_description_message,
            "dataset_statistics": dataset_statistics_message,
            "additional_context": additional_context,
            "transformation_types": transformation_types,
            "unary_types": unary_types,
            "binary_types": binary_types,
        }

    def generate_engineered_features_iterative(
        self,
        prompt_context: Dict,
        conversation_history: List[BaseMessage],
        feedback_context: Optional[Dict] = None,
    ) -> Tuple[FeatureEngineeringIdeas, List[BaseMessage]]:
        """
        Generate feature engineering ideas in an iterative conversation.

        Parameters
        ----------
        prompt_context : Dict
            Prompt context dict
        conversation_history : List[BaseMessage]
            Accumulated conversation messages. Empty on the first round.
        feedback_context : Optional[Dict]
            Feedback dict with keys ``selected_features_table``,
            ``rejected_features_table``, and ``max_features``. Required for
            rounds after the first.

        Returns
        -------
        Tuple[FeatureEngineeringIdeas, List[BaseMessage]]
            The generated ideas and the updated conversation history
            (input messages + AI response appended).
        """
        if not conversation_history:
            messages = self.prompt_template.format_messages(**prompt_context)
        else:
            feedback_message = SELECTION_FEEDBACK_PROMPT.format(**feedback_context)
            messages = conversation_history + [HumanMessage(content=feedback_message)]

        ideas = self.llm.invoke(messages)
        updated_history = messages + [AIMessage(content=ideas.model_dump_json())]

        return ideas, updated_history
