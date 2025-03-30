"""
Module for handling interactions with Language Models.
"""

from typing import Dict, Optional

from langchain.chat_models import init_chat_model
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from skfeaturellm.prompts import FEATURE_ENGINEERING_PROMPT
from skfeaturellm.schemas import FeatureEngineeringIdeas


class LLMInterface:
    """
    Interface for interacting with Language Models for feature engineering.

    Parameters
    ----------
    api_key : str
        API key for the LLM provider
    model_name : str, default="gpt-4"
        Name of the model to use
    temperature : float, default=0.7
        Temperature parameter for model sampling
    max_tokens : Optional[int], default=None
        Maximum number of tokens to generate
    request_timeout : Optional[float], default=None
        Timeout for API requests in seconds
    """

    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        self.llm = init_chat_model(model=model_name, **kwargs)

        self.generate_prompt()

    def generate_engineered_features(
        self,
        feature_descriptions: Dict[str, str],
        target_description: Optional[str] = None,
        max_features: Optional[int] = None,
    ) -> FeatureEngineeringIdeas:
        """
        Generate feature engineering ideas using the LLM.

        Parameters
        ----------
        feature_descriptions : Dict[str, str]
            Descriptions for input features
        target_description : Optional[str]
            Description of the target variable and task
        max_features : Optional[int]
            Maximum number of features to generate

        Returns
        -------
        List[FeatureEngineeringIdea]
            List of generated feature ideas
        """
        pass

    def generate_prompt(self):
        """
        Generate the prompt for the LLM.
        """
        self.output_parser = PydanticOutputParser(
            pydantic_object=FeatureEngineeringIdeas
        )
        self.prompt = ChatPromptTemplate.from_template(
            FEATURE_ENGINEERING_PROMPT
        ).partial(format_instructions=self.output_parser.get_format_instructions())


if __name__ == "__main__":
    llm_interface = LLMInterface()

    print(
        llm_interface.prompt.invoke(
            {
                "data_description": (
                    "Dataset with columns: age (int), income (float), education (str)"
                ),
                "target_description": (
                    "Binary classification task predicting customer churn"
                ),
                "additional_context": (
                    "Generate up to 5 features focusing on customer behavior patterns"
                ),
            }
        ).to_string()
    )
