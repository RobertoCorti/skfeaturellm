from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from skfeaturellm.dsl.operations import TransformationOperation
from skfeaturellm.dsl.types import TypeEnum


class InputField(BaseModel):
    field: str
    type: TypeEnum


class OutputField(BaseModel):
    field: str
    type: TypeEnum


class FeatureDSL(BaseModel):
    feature_name: str
    transformation: TransformationOperation
    inputs: List[InputField]
    output: OutputField
