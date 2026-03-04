"""Tests for unary transformations."""

import numpy as np
import pytest

from skfeaturellm.transformations import (
    AbsTransformation,
    ExpTransformation,
    InvalidValueError,
    Log1pTransformation,
    LogTransformation,
    PowTransformation,
    SqrtTransformation,
)

# =============================================================================
# Test: LogTransformation
# =============================================================================


def test_log_transformation(sample_df):
    """Test log transformation."""
    t = LogTransformation("log_positive", columns=["positive"])
    result = t.fit_transform(sample_df)

    assert result.name == "log_positive"
    expected = np.log([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_log_transformation_negative_values(sample_df):
    """Test that log transformation raises error for negative values."""
    t = LogTransformation("log_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values > 0"):
        t.fit_transform(sample_df)


def test_log_transformation_zero_values(sample_df):
    """Test that log transformation raises error for zero values."""
    t = LogTransformation("log_zero", columns=["with_zero"])

    with pytest.raises(InvalidValueError, match="all values > 0"):
        t.fit_transform(sample_df)


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
    result = t.fit_transform(sample_df)

    assert result.name == "log1p_zero"
    expected = np.log1p([0, 1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)


def test_log1p_transformation_negative_values(sample_df):
    """Test that log1p transformation raises error for negative values."""
    t = Log1pTransformation("log1p_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values >= 0"):
        t.fit_transform(sample_df)


# =============================================================================
# Test: PowTransformation
# =============================================================================


def test_pow_transformation_square(sample_df):
    """Test power transformation with power=2 (square)."""
    t = PowTransformation("squared", columns=["b"], parameters={"power": 2})
    result = t.fit_transform(sample_df)

    assert result.name == "squared"
    assert list(result) == [4, 16, 25, 64]


def test_pow_transformation_cube(sample_df):
    """Test power transformation with power=3 (cube)."""
    t = PowTransformation("cubed", columns=["b"], parameters={"power": 3})
    result = t.fit_transform(sample_df)

    assert result.name == "cubed"
    assert list(result) == [8, 64, 125, 512]


def test_pow_transformation_sqrt(sample_df):
    """Test power transformation with power=0.5 (sqrt)."""
    t = PowTransformation(
        "sqrt_positive", columns=["positive"], parameters={"power": 0.5}
    )
    result = t.fit_transform(sample_df)

    assert result.name == "sqrt_positive"
    expected = np.sqrt([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_pow_transformation_reciprocal(sample_df):
    """Test power transformation with power=-1 (reciprocal)."""
    t = PowTransformation("reciprocal_b", columns=["b"], parameters={"power": -1})
    result = t.fit_transform(sample_df)

    assert result.name == "reciprocal_b"
    expected = [0.5, 0.25, 0.2, 0.125]
    np.testing.assert_array_almost_equal(result, expected)


def test_pow_transformation_missing_power():
    """Test that missing power parameter raises error."""
    with pytest.raises(ValueError, match="requires 'power' in parameters"):
        PowTransformation("test", columns=["a"])


def test_pow_transformation_negative_power_with_zero(sample_df):
    """Test that negative power with zero values raises error."""
    t = PowTransformation(
        "reciprocal_zero", columns=["with_zero"], parameters={"power": -1}
    )

    with pytest.raises(InvalidValueError, match="all values != 0"):
        t.fit_transform(sample_df)


def test_pow_transformation_fractional_power_with_negative(sample_df):
    """Test that fractional power with negative values raises error."""
    t = PowTransformation(
        "sqrt_negative", columns=["with_negative"], parameters={"power": 0.5}
    )

    with pytest.raises(InvalidValueError, match="all values >= 0"):
        t.fit_transform(sample_df)


# =============================================================================
# Test: AbsTransformation
# =============================================================================


def test_abs_transformation(sample_df):
    """Test abs transformation."""
    t = AbsTransformation("abs_negative", columns=["with_negative"])
    result = t.fit_transform(sample_df)

    assert result.name == "abs_negative"
    assert list(result) == [2, 1, 1, 2]


# =============================================================================
# Test: ExpTransformation
# =============================================================================


def test_exp_transformation(sample_df):
    """Test exp transformation."""
    t = ExpTransformation("exp_positive", columns=["positive"])
    result = t.fit_transform(sample_df)

    assert result.name == "exp_positive"
    expected = np.exp([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# Test: SqrtTransformation
# =============================================================================


def test_sqrt_transformation(sample_df):
    """Test sqrt transformation on non-negative values."""
    t = SqrtTransformation("sqrt_positive", columns=["positive"])
    result = t.fit_transform(sample_df)

    assert result.name == "sqrt_positive"
    expected = np.sqrt([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_sqrt_transformation_zero(sample_df):
    """Test that sqrt of zero is valid (returns 0)."""
    t = SqrtTransformation("sqrt_zero", columns=["with_zero"])
    result = t.fit_transform(sample_df)

    assert result.name == "sqrt_zero"
    expected = np.sqrt([0, 1, 2, 3])
    np.testing.assert_array_almost_equal(result, expected)


def test_sqrt_transformation_negative_values(sample_df):
    """Test that sqrt transformation raises error for negative values."""
    t = SqrtTransformation("sqrt_negative", columns=["with_negative"])

    with pytest.raises(InvalidValueError, match="all values >= 0"):
        t.fit_transform(sample_df)


def test_sqrt_get_prompt_description():
    """Test get_prompt_description returns a string."""
    desc = SqrtTransformation.get_prompt_description()
    assert isinstance(desc, str)
    assert "sqrt" in desc.lower()


# =============================================================================
# Test: Unary Transformation Common Behavior
# =============================================================================


def test_unary_get_required_columns():
    """Test that unary transformations return correct required columns."""
    t = LogTransformation("log_col", columns=["some_column"])
    assert t.get_required_columns() == {"some_column"}


# =============================================================================
# Test: fit/transform interface
# =============================================================================


def test_fit_transform(sample_df):
    """fit(df).transform(df) produces the expected result."""
    t = LogTransformation("log_positive", columns=["positive"])
    result = t.fit(sample_df).transform(sample_df)

    np.testing.assert_array_almost_equal(result, np.log([1.0, 2.0, 3.0, 4.0]))
    assert result.name == "log_positive"


def test_fit_returns_self(sample_df):
    """fit() returns self for chaining."""
    t = LogTransformation("log_positive", columns=["positive"])
    assert t.fit(sample_df) is t
