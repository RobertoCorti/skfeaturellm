"""Factory class for creating LLM instances"""

from enum import Enum
from typing import Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMModelEnum(str, Enum):
    """Enum representing different LLM models"""

    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Anthropic models
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-202502190"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"

    @classmethod
    def get_model_type(cls, model: Union[str, "LLMModelEnum"]) -> str:
        """Get the provider type for the given model"""
        if isinstance(model, str):
            model = cls(model)  # Convert string to enum value

        # Create the mapping dynamically to ensure enum members are properly instantiated
        model_type_map = {
            cls.GPT_4O: "openai",
            cls.GPT_4O_MINI: "openai",
            cls.CLAUDE_3_7_SONNET: "anthropic",
            cls.CLAUDE_3_5_HAIKU: "anthropic",
        }

        # Now model is guaranteed to be an enum instance
        if model in model_type_map:
            return model_type_map[model]
        raise ValueError(f"Unknown model type: {model}")

    def __str__(self):
        return str(self.value)


class LLMParameters(BaseModel):
    """Model for LLM parameters"""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    max_retries: int = 1
    api_key: Optional[str] = None


class LLMFactory:  # pylint: disable=too-few-public-methods
    """Factory class for creating LLM instances"""

    @staticmethod
    def create_llm(
        model_name: Union[str, LLMModelEnum], model_parameters: LLMParameters
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the model name and parameters

        Args:
            model: The LLM model name (string) or enum value
            model_parameters: The model parameters (included the API key)

        Returns:
            An instance of a LangChain LLM
        """
        # Convert enum to string if needed
        model_provider = LLMModelEnum.get_model_type(model_name)
        model_parameters_dict = {}
        if model_parameters is not None:
            model_parameters_dict = model_parameters.model_dump()

        if model_provider == "openai":
            return ChatOpenAI(
                model=str(model_name),
                **model_parameters_dict,
            )

        if model_provider == "anthropic":
            return ChatAnthropic(
                model=str(model_name),
                **model_parameters_dict,
            )

        raise ValueError(f"Unsupported model: {str(model_name)}")
