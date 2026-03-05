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

# Import binary transformations to trigger registration
from skfeaturellm.transformations.binary import (
    AddTransformation,
    BinaryArithmeticTransformation,
    DivisionByZeroError,
    DivTransformation,
    MaxTransformation,
    MinTransformation,
    MulTransformation,
    SubTransformation,
)
from skfeaturellm.transformations.pipeline import (
    TransformationParseError,
    TransformationPipeline,
    get_all_operation_types,
    get_binary_operation_types,
    get_registered_transformations,
    get_transformation_types_for_prompt,
    get_unary_operation_types,
    register_transformation,
)

# Import unary transformations to trigger registration
from skfeaturellm.transformations.unary import (
    AbsTransformation,
    ExpTransformation,
    InvalidValueError,
    Log1pTransformation,
    LogTransformation,
    PowTransformation,
    SqrtTransformation,
    UnaryTransformation,
)

__all__ = [
    # Base
    "BaseTransformation",
    "TransformationError",
    "ColumnNotFoundError",
    # Executor
    "TransformationPipeline",
    "TransformationParseError",
    "register_transformation",
    "get_registered_transformations",
    "get_transformation_types_for_prompt",
    "get_all_operation_types",
    "get_unary_operation_types",
    "get_binary_operation_types",
    # Binary
    "BinaryArithmeticTransformation",
    "AddTransformation",
    "SubTransformation",
    "MulTransformation",
    "DivTransformation",
    "MaxTransformation",
    "MinTransformation",
    "DivisionByZeroError",
    # Unary
    "UnaryTransformation",
    "LogTransformation",
    "Log1pTransformation",
    "AbsTransformation",
    "ExpTransformation",
    "PowTransformation",
    "SqrtTransformation",
    "InvalidValueError",
]
