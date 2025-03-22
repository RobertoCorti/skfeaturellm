"""
Module for handling interactions with Language Models.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from skfeaturellm.schemas import FeatureIdea
from skfeaturellm.prompts import FEATURE_ENGINEERING_PROMPT


class LLMInterface:  # pylint: disable=too-few-public-methods
    """
    Interface for interacting with Language Models for feature engineering.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration for the LLM (API keys, model selection, etc.)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm: Optional[BaseLanguageModel] = None
        self.prompt = ChatPromptTemplate.from_template(FEATURE_ENGINEERING_PROMPT)
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        # TODO: Implement LLM initialization

    def generate_feature_ideas(
        self,
        data_description: pd.DataFrame,  # pylint: disable=unused-argument
        target_description: Optional[str] = None,  # pylint: disable=unused-argument
        max_features: Optional[int] = None,  # pylint: disable=unused-argument
    ) -> List[Dict[str, Any]]:
        provider = self.config.get("provider", "openai")
        if provider == "openai":
            self.llm = ChatOpenAI(
                model_name=self.config.get("model", "gpt-4"),
                api_key=self.config["api_key"],
                temperature=self.config.get("temperature", 0.7)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate_feature_ideas(
        self,
        data: pd.DataFrame,
        target_description: Optional[str] = None,
        feature_descriptions: Optional[Dict[str, str]] = None,
        max_features: Optional[int] = None
    ) -> List[FeatureIdea]:
        """
        Generate feature engineering ideas using the LLM.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to generate features for
        target_description : Optional[str]
            Description of the target variable and task
        feature_descriptions : Optional[Dict[str, str]]
            Optional descriptions for input features
        max_features : Optional[int]
            Maximum number of features to generate

        Returns
        -------
        List[FeatureIdea]
            List of generated feature ideas
        """
        # Prepare dataset description
        data_description = self._prepare_data_description(data, feature_descriptions)
        
        # Prepare target description if not provided
        if target_description is None:
            target_description = "This is an unsupervised feature engineering task."
        
        # Prepare additional context
        additional_context = f"Generate up to {max_features} features." if max_features else ""
        
        # Generate prompt
        messages = self.prompt.format_messages(
            data_description=data_description,
            target_description=target_description,
            additional_context=additional_context,
            feature_schema=FeatureIdea.model_json_schema()
        )
        
        # Get response from LLM
        response = self.llm.generate([messages])
        
        # Parse response into FeatureIdea objects
        # TODO: Implement proper parsing of LLM response into FeatureIdea objects
        # For now, return an empty list
        return []
    
    def _prepare_data_description(
        self,
        data: pd.DataFrame,
        feature_descriptions: Optional[Dict[str, str]] = None
    ) -> str:
        """Prepare a description of the dataset for the LLM."""
        description = []
        
        # Basic dataset info
        description.append(f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Column information
        for col in data.columns:
            col_type = data[col].dtype
            col_desc = feature_descriptions.get(col, "") if feature_descriptions else ""
            stats = []
            
            if pd.api.types.is_numeric_dtype(col_type):
                stats.extend([
                    f"min: {data[col].min():.2f}",
                    f"max: {data[col].max():.2f}",
                    f"mean: {data[col].mean():.2f}"
                ])
            elif pd.api.types.is_categorical_dtype(col_type):
                stats.append(f"unique values: {data[col].nunique()}")
            
            col_info = f"- {col} ({col_type})"
            if col_desc:
                col_info += f": {col_desc}"
            if stats:
                col_info += f" [{', '.join(stats)}]"
            
            description.append(col_info)
        
        return "\n".join(description) 
