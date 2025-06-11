from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FeatureTransformation(BaseModel):
    feature_name: str
    transformation: str
    parameters: dict
