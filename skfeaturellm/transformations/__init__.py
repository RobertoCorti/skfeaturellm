"""
Feature Transformation DSL.

This subpackage provides a structured, validated, and secure way to
represent and execute feature transformations.
"""

# Import arithmetic to trigger registration of transformation types
from skfeaturellm.transformations.arithmetic import (
    AddTransformation,
    BinaryArithmeticTransformation,
    DivisionByZeroError,
    DivTransformation,
    MulTransformation,
    SubTransformation,
)
from skfeaturellm.transformations.base import (
    BaseTransformation,
    ColumnNotFoundError,
    TransformationError,
)
from skfeaturellm.transformations.executor import (
    TransformationExecutor,
    TransformationParseError,
    get_registered_transformations,
    get_transformation_types_for_prompt,
    register_transformation,
)

__all__ = [
    # Base
    "BaseTransformation",
    "TransformationError",
    "ColumnNotFoundError",
    # Executor
    "TransformationExecutor",
    "TransformationParseError",
    "register_transformation",
    "get_registered_transformations",
    "get_transformation_types_for_prompt",
    # Arithmetic
    "BinaryArithmeticTransformation",
    "AddTransformation",
    "SubTransformation",
    "MulTransformation",
    "DivTransformation",
    "DivisionByZeroError",
]
