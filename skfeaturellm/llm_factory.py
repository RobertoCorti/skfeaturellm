"""Factory class for creating LLM instances"""

from enum import Enum
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LLMProvider(str, Enum):
    """Enum representing different LLM models"""

    OPENAI = "openai"
    ANTRHOPIC = "anthropic"


class LLMModelName(BaseModel):
    """Model for LLM model names"""

    model_provider: LLMProvider
    model_name: str


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
        model_name: LLMModelName,
        model_parameters: LLMParameters,
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
        model_provider = model_name.model_provider
        model_parameters_dict = {}
        if model_parameters is not None:
            model_parameters_dict = model_parameters.model_dump()

        if model_provider == LLMModelName.OPENAI:
            return ChatOpenAI(
                model=model_name.model_name,
                **model_parameters_dict,
            )

        if model_provider == LLMModelName.ANTHROPIC:
            return ChatAnthropic(
                model=model_name.model_name,
                **model_parameters_dict,
            )

        raise ValueError(f"Unsupported model: {str(model_name)}")
