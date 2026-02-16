"""Tests for unary transformations."""

import numpy as np
import pytest

from skfeaturellm.transformations import (
    AbsTransformation,
    CubeTransformation,
    ExpTransformation,
    InvalidValueError,
    Log1pTransformation,
    LogTransformation,
    ReciprocalTransformation,
    SquareTransformation,
    SqrtTransformation,
)


# =============================================================================
# Test: LogTransformation
# =============================================================================


def test_log_transformation(sample_df):
    """Test log transformation."""
    t = LogTransformation("log_positive", columns=["positive"])
    result = t.execute(sample_df)

    assert result.name == "log_positive"
    expected = np.log([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_log_transformation_negative_values(sample_df):
    """Test that log transformation raises error for negative values."""
    t = LogTransformation("log_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values > 0"):
        t.execute(sample_df)


def test_log_transformation_zero_values(sample_df):
    """Test that log transformation raises error for zero values."""
    t = LogTransformation("log_zero", columns=["with_zero"])

    with pytest.raises(InvalidValueError, match="all values > 0"):
        t.execute(sample_df)


def test_log_get_prompt_description():
    """Test get_prompt_description returns a string."""
    desc = LogTransformation.get_prompt_description()
    assert isinstance(desc, str)
    assert "log" in desc.lower()


# =============================================================================
# Test: Log1pTransformation
# =============================================================================


def test_log1p_transformation(sample_df):
    """Test log1p transformation."""
    t = Log1pTransformation("log1p_zero", columns=["with_zero"])
    result = t.execute(sample_df)

    assert result.name == "log1p_zero"
    expected = np.log1p([0, 1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)


def test_log1p_transformation_negative_values(sample_df):
    """Test that log1p transformation raises error for negative values."""
    t = Log1pTransformation("log1p_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values >= 0"):
        t.execute(sample_df)


# =============================================================================
# Test: SqrtTransformation
# =============================================================================


def test_sqrt_transformation(sample_df):
    """Test sqrt transformation."""
    t = SqrtTransformation("sqrt_positive", columns=["positive"])
    result = t.execute(sample_df)

    assert result.name == "sqrt_positive"
    expected = np.sqrt([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_sqrt_transformation_negative_values(sample_df):
    """Test that sqrt transformation raises error for negative values."""
    t = SqrtTransformation("sqrt_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values >= 0"):
        t.execute(sample_df)


# =============================================================================
# Test: AbsTransformation
# =============================================================================


def test_abs_transformation(sample_df):
    """Test abs transformation."""
    t = AbsTransformation("abs_negative", columns=["with_negative"])
    result = t.execute(sample_df)

    assert result.name == "abs_negative"
    assert list(result) == [2, 1, 1, 2]


# =============================================================================
# Test: ExpTransformation
# =============================================================================


def test_exp_transformation(sample_df):
    """Test exp transformation."""
    t = ExpTransformation("exp_positive", columns=["positive"])
    result = t.execute(sample_df)

    assert result.name == "exp_positive"
    expected = np.exp([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Test: SquareTransformation
# =============================================================================


def test_square_transformation(sample_df):
    """Test square transformation."""
    t = SquareTransformation("squared", columns=["b"])
    result = t.execute(sample_df)

    assert result.name == "squared"
    assert list(result) == [4, 16, 25, 64]


# =============================================================================
# Test: CubeTransformation
# =============================================================================


def test_cube_transformation(sample_df):
    """Test cube transformation."""
    t = CubeTransformation("cubed", columns=["b"])
    result = t.execute(sample_df)

    assert result.name == "cubed"
    assert list(result) == [8, 64, 125, 512]


# =============================================================================
# Test: ReciprocalTransformation
# =============================================================================


def test_reciprocal_transformation(sample_df):
    """Test reciprocal transformation."""
    t = ReciprocalTransformation("reciprocal_b", columns=["b"])
    result = t.execute(sample_df)

    assert result.name == "reciprocal_b"
    expected = [1 / 2, 1 / 4, 1 / 5, 1 / 8]
    np.testing.assert_array_almost_equal(result, expected)


def test_reciprocal_transformation_zero(sample_df):
    """Test that reciprocal transformation raises error for zero values."""
    t = ReciprocalTransformation("reciprocal_zero", columns=["with_zero"])

    with pytest.raises(InvalidValueError, match="all values != 0"):
        t.execute(sample_df)


# =============================================================================
# Test: Unary Transformation Common Behavior
# =============================================================================


def test_unary_get_required_columns():
    """Test that unary transformations return correct required columns."""
    t = LogTransformation("log_col", columns=["some_column"])
    assert t.get_required_columns() == {"some_column"}
