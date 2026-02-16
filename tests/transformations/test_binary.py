"""Tests for binary transformations."""

import pytest

from skfeaturellm.transformations import (
    AddTransformation,
    ColumnNotFoundError,
    DivisionByZeroError,
    DivTransformation,
    MulTransformation,
    SubTransformation,
)

# =============================================================================
# Test: AddTransformation
# =============================================================================


def test_add_two_columns(sample_df):
    """Test addition of two columns."""
    t = AddTransformation("sum", columns=["a", "b"])
    result = t.execute(sample_df)

    assert result.name == "sum"
    assert list(result) == [12, 24, 35, 48]


def test_add_column_and_constant(sample_df):
    """Test addition of column and constant."""
    t = AddTransformation("plus_five", columns=["a"], parameters={"constant": 5.0})
    result = t.execute(sample_df)

    assert result.name == "plus_five"
    assert list(result) == [15, 25, 35, 45]


def test_add_get_required_columns():
    """Test get_required_columns returns correct columns."""
    t1 = AddTransformation("sum", columns=["a", "b"])
    assert t1.get_required_columns() == {"a", "b"}

    t2 = AddTransformation("plus_five", columns=["a"], parameters={"constant": 5.0})
    assert t2.get_required_columns() == {"a"}


def test_add_get_prompt_description():
    """Test get_prompt_description returns a string."""
    desc = AddTransformation.get_prompt_description()
    assert isinstance(desc, str)
    assert "add" in desc.lower() or "+" in desc


# =============================================================================
# Test: SubTransformation
# =============================================================================


def test_sub_two_columns(sample_df):
    """Test subtraction of two columns."""
    t = SubTransformation("diff", columns=["a", "b"])
    result = t.execute(sample_df)

    assert result.name == "diff"
    assert list(result) == [8, 16, 25, 32]


def test_sub_column_and_constant(sample_df):
    """Test subtraction of constant from column."""
    t = SubTransformation("minus_five", columns=["a"], parameters={"constant": 5.0})
    result = t.execute(sample_df)

    assert result.name == "minus_five"
    assert list(result) == [5, 15, 25, 35]


# =============================================================================
# Test: MulTransformation
# =============================================================================


def test_mul_two_columns(sample_df):
    """Test multiplication of two columns."""
    t = MulTransformation("product", columns=["a", "b"])
    result = t.execute(sample_df)

    assert result.name == "product"
    assert list(result) == [20, 80, 150, 320]


def test_mul_column_and_constant(sample_df):
    """Test multiplication of column by constant."""
    t = MulTransformation("doubled", columns=["a"], parameters={"constant": 2.0})
    result = t.execute(sample_df)

    assert result.name == "doubled"
    assert list(result) == [20, 40, 60, 80]


# =============================================================================
# Test: DivTransformation
# =============================================================================


def test_div_two_columns(sample_df):
    """Test division of two columns."""
    t = DivTransformation("ratio", columns=["a", "b"])
    result = t.execute(sample_df)

    assert result.name == "ratio"
    assert list(result) == [5.0, 5.0, 6.0, 5.0]


def test_div_column_and_constant(sample_df):
    """Test division of column by constant."""
    t = DivTransformation("halved", columns=["a"], parameters={"constant": 2.0})
    result = t.execute(sample_df)

    assert result.name == "halved"
    assert list(result) == [5.0, 10.0, 15.0, 20.0]


def test_div_by_zero_column(sample_df):
    """Test division by zero raises error."""
    t = DivTransformation("ratio", columns=["a", "c"])

    with pytest.raises(DivisionByZeroError):
        t.execute(sample_df)


def test_div_by_zero_constant(sample_df):
    """Test division by zero constant raises error."""
    t = DivTransformation("ratio", columns=["a"], parameters={"constant": 0.0})

    with pytest.raises(DivisionByZeroError):
        t.execute(sample_df)


# =============================================================================
# Test: Binary Transformation Validation
# =============================================================================


def test_binary_missing_right_operand():
    """Test that missing right operand raises error."""
    with pytest.raises(ValueError, match="requires 'constant' in parameters"):
        AddTransformation("sum", columns=["a"])


def test_binary_both_right_operands():
    """Test that providing both right operands raises error."""
    with pytest.raises(ValueError, match="should not have 'constant' in parameters"):
        AddTransformation("sum", columns=["a", "b"], parameters={"constant": 5.0})


def test_binary_missing_column(sample_df):
    """Test that missing column raises ColumnNotFoundError."""
    t = AddTransformation("sum", columns=["a", "nonexistent"])

    with pytest.raises(ColumnNotFoundError):
        t.execute(sample_df)
