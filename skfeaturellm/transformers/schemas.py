from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, field_validator

import skfeaturellm.transformers as transformers
from skfeaturellm.transformers.exception import TransformerNotFoundError


class FeatureTransformationParameters(BaseModel):
    pass


class AdditiveTransformerParameters(FeatureTransformationParameters):
    addend_cols: List[str]
    subtract_cols: List[str]
    skip_na: bool = False


class FeatureTransformation(BaseModel):
    feature_name: str
    transformation: str
    parameters: FeatureTransformationParameters

    @field_validator("transformation")
    def transformation_must_exist(cls, v):
        available = {
            name.replace("Transformer", ""): getattr(transformers, name)
            for name in getattr(transformers, "__all__", [])
        }
        if v not in available:
            raise TransformerNotFoundError(
                f"Transformation '{v}' not implemented. Available: {list(available.keys())}"
            )
        return v
