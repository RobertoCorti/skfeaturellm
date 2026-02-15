"""
Feature Transformation DSL.

This subpackage provides a structured, validated, and secure way to
represent and execute feature transformations.
"""

from skfeaturellm.transformations.base import (
    BaseTransformation,
    ColumnNotFoundError,
    TransformationError,
)
from skfeaturellm.transformations.executor import (
    TransformationExecutor,
    TransformationParseError,
    get_registered_transformations,
    register_transformation,
)

# Import arithmetic to trigger registration of transformation types
from skfeaturellm.transformations.arithmetic import (
    AddTransformation,
    BinaryArithmeticTransformation,
    DivisionByZeroError,
    DivTransformation,
    MulTransformation,
    SubTransformation,
)

__all__ = [
    "BaseTransformation",
    "TransformationError",
    "ColumnNotFoundError",
    "TransformationExecutor",
    "TransformationParseError",
    "register_transformation",
    "get_registered_transformations",
    "BinaryArithmeticTransformation",
    "AddTransformation",
    "SubTransformation",
    "MulTransformation",
    "DivTransformation",
    "DivisionByZeroError",
]
