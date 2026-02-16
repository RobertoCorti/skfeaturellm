"""
Errors for binary operations.
"""

from skfeaturellm.transformations.base import TransformationError


class BinaryOperationWithOneColumnError(TransformationError):
    """Raised when a binary operation is called with only one column."""

    pass


class BinaryOperationWithTwoColumnsError(TransformationError):
    """Raised when a binary operation is called with two columns."""

    pass
