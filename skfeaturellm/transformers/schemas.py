from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, field_validator

import skfeaturellm.transformers as transformers
from skfeaturellm.transformers.exception import TransformerNotFoundError


class FeatureTransformation(BaseModel):
    feature_name: str
    transformation: str
    parameters: dict
