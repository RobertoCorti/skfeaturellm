"""Tests for the transformations subpackage."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from skfeaturellm.transformations import (
    AddTransformation,
    ColumnNotFoundError,
    DivisionByZeroError,
    DivTransformation,
    MulTransformation,
    SubTransformation,
    TransformationExecutor,
    TransformationParseError,
    get_registered_transformations,
    get_transformation_types_for_prompt,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_df():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "a": [10, 20, 30, 40],
            "b": [2, 4, 5, 8],
            "c": [1, 0, 3, 4],  # Contains zero for division tests
        }
    )


@pytest.fixture
def sample_config():
    """Provide a sample transformation config."""
    return {
        "transformations": [
            {
                "type": "add",
                "feature_name": "sum_ab",
                "left_column": "a",
                "right_column": "b",
            },
            {
                "type": "div",
                "feature_name": "ratio_ab",
                "left_column": "a",
                "right_column": "b",
            },
        ]
    }


# =============================================================================
# Test: AddTransformation
# =============================================================================


def test_add_two_columns(sample_df):
    """Test addition of two columns."""
    t = AddTransformation("sum", "a", right_column="b")
    result = t.execute(sample_df)

    assert result.name == "sum"
    assert list(result) == [12, 24, 35, 48]


def test_add_column_and_constant(sample_df):
    """Test addition of column and constant."""
    t = AddTransformation("plus_five", "a", right_constant=5.0)
    result = t.execute(sample_df)

    assert result.name == "plus_five"
    assert list(result) == [15, 25, 35, 45]


def test_add_get_required_columns():
    """Test get_required_columns returns correct columns."""
    t1 = AddTransformation("sum", "a", right_column="b")
    assert t1.get_required_columns() == {"a", "b"}

    t2 = AddTransformation("plus_five", "a", right_constant=5.0)
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
    t = SubTransformation("diff", "a", right_column="b")
    result = t.execute(sample_df)

    assert result.name == "diff"
    assert list(result) == [8, 16, 25, 32]


def test_sub_column_and_constant(sample_df):
    """Test subtraction of constant from column."""
    t = SubTransformation("minus_five", "a", right_constant=5.0)
    result = t.execute(sample_df)

    assert result.name == "minus_five"
    assert list(result) == [5, 15, 25, 35]


# =============================================================================
# Test: MulTransformation
# =============================================================================


def test_mul_two_columns(sample_df):
    """Test multiplication of two columns."""
    t = MulTransformation("product", "a", right_column="b")
    result = t.execute(sample_df)

    assert result.name == "product"
    assert list(result) == [20, 80, 150, 320]


def test_mul_column_and_constant(sample_df):
    """Test multiplication of column by constant."""
    t = MulTransformation("doubled", "a", right_constant=2.0)
    result = t.execute(sample_df)

    assert result.name == "doubled"
    assert list(result) == [20, 40, 60, 80]


# =============================================================================
# Test: DivTransformation
# =============================================================================


def test_div_two_columns(sample_df):
    """Test division of two columns."""
    t = DivTransformation("ratio", "a", right_column="b")
    result = t.execute(sample_df)

    assert result.name == "ratio"
    assert list(result) == [5.0, 5.0, 6.0, 5.0]


def test_div_column_and_constant(sample_df):
    """Test division of column by constant."""
    t = DivTransformation("halved", "a", right_constant=2.0)
    result = t.execute(sample_df)

    assert result.name == "halved"
    assert list(result) == [5.0, 10.0, 15.0, 20.0]


def test_div_by_zero_column(sample_df):
    """Test division by zero raises error."""
    t = DivTransformation("ratio", "a", right_column="c")

    with pytest.raises(DivisionByZeroError):
        t.execute(sample_df)


def test_div_by_zero_constant(sample_df):
    """Test division by zero constant raises error."""
    t = DivTransformation("ratio", "a", right_constant=0.0)

    with pytest.raises(DivisionByZeroError):
        t.execute(sample_df)


# =============================================================================
# Test: Transformation Validation
# =============================================================================


def test_transformation_missing_right_operand():
    """Test that missing right operand raises error."""
    with pytest.raises(ValueError, match="Must provide either"):
        AddTransformation("sum", "a")


def test_transformation_both_right_operands():
    """Test that providing both right operands raises error."""
    with pytest.raises(ValueError, match="Cannot provide both"):
        AddTransformation("sum", "a", right_column="b", right_constant=5.0)


def test_transformation_missing_column(sample_df):
    """Test that missing column raises ColumnNotFoundError."""
    t = AddTransformation("sum", "a", right_column="nonexistent")

    with pytest.raises(ColumnNotFoundError):
        t.execute(sample_df)


# =============================================================================
# Test: TransformationExecutor
# =============================================================================


def test_executor_single_transformation(sample_df):
    """Test executing a single transformation."""
    t = AddTransformation("sum", "a", right_column="b")
    executor = TransformationExecutor(transformations=[t])

    result = executor.execute(sample_df)

    assert "sum" in result.columns
    assert list(result["sum"]) == [12, 24, 35, 48]
    assert "a" in result.columns
    assert "b" in result.columns


def test_executor_multiple_transformations(sample_df):
    """Test executing multiple transformations."""
    transformations = [
        AddTransformation("sum", "a", right_column="b"),
        MulTransformation("product", "a", right_column="b"),
    ]
    executor = TransformationExecutor(transformations=transformations)

    result = executor.execute(sample_df)

    assert "sum" in result.columns
    assert "product" in result.columns
    assert list(result["sum"]) == [12, 24, 35, 48]
    assert list(result["product"]) == [20, 80, 150, 320]


def test_executor_raise_on_error_false(sample_df):
    """Test that failed transformations are skipped when raise_on_error=False."""
    transformations = [
        AddTransformation("sum", "a", right_column="b"),
        DivTransformation("bad_ratio", "a", right_column="c"),
    ]
    executor = TransformationExecutor(
        transformations=transformations, raise_on_error=False
    )

    with pytest.warns(UserWarning):
        result = executor.execute(sample_df)

    assert "sum" in result.columns
    assert "bad_ratio" not in result.columns


def test_executor_raise_on_error_true(sample_df):
    """Test that failed transformations raise when raise_on_error=True."""
    transformations = [
        DivTransformation("bad_ratio", "a", right_column="c"),
    ]
    executor = TransformationExecutor(
        transformations=transformations, raise_on_error=True
    )

    with pytest.raises(DivisionByZeroError):
        executor.execute(sample_df)


def test_executor_get_required_columns():
    """Test get_required_columns aggregates all columns."""
    transformations = [
        AddTransformation("sum", "a", right_column="b"),
        MulTransformation("scaled", "c", right_constant=2.0),
    ]
    executor = TransformationExecutor(transformations=transformations)

    required = executor.get_required_columns()

    assert required == {"a", "b", "c"}


def test_executor_empty_transformations(sample_df):
    """Test executing with no transformations."""
    executor = TransformationExecutor(transformations=[])

    with pytest.warns(UserWarning, match="No transformations"):
        result = executor.execute(sample_df)

    assert list(result.columns) == list(sample_df.columns)


# =============================================================================
# Test: TransformationExecutor.from_dict()
# =============================================================================


def test_from_dict_valid(sample_config, sample_df):
    """Test loading from valid dict config."""
    executor = TransformationExecutor.from_dict(sample_config)

    assert len(executor.transformations) == 2
    result = executor.execute(sample_df)
    assert "sum_ab" in result.columns
    assert "ratio_ab" in result.columns


def test_from_dict_missing_transformations_key():
    """Test that missing 'transformations' key raises error."""
    with pytest.raises(TransformationParseError, match="transformations"):
        TransformationExecutor.from_dict({"invalid": []})


def test_from_dict_missing_type():
    """Test that missing 'type' field raises error."""
    config = {
        "transformations": [
            {"feature_name": "sum", "left_column": "a", "right_column": "b"}
        ]
    }
    with pytest.raises(TransformationParseError, match="type"):
        TransformationExecutor.from_dict(config)


def test_from_dict_unknown_type():
    """Test that unknown transformation type raises error."""
    config = {
        "transformations": [
            {
                "type": "unknown_op",
                "feature_name": "test",
                "left_column": "a",
                "right_column": "b",
            }
        ]
    }
    with pytest.raises(TransformationParseError, match="Unknown transformation"):
        TransformationExecutor.from_dict(config)


def test_from_dict_invalid_arguments():
    """Test that invalid arguments raise error."""
    config = {
        "transformations": [
            {
                "type": "add",
                "feature_name": "sum",
                "right_column": "b",
            }
        ]
    }
    with pytest.raises(TransformationParseError, match="Invalid arguments"):
        TransformationExecutor.from_dict(config)


def test_from_dict_with_constant(sample_df):
    """Test loading config with constant operand."""
    config = {
        "transformations": [
            {
                "type": "mul",
                "feature_name": "doubled",
                "left_column": "a",
                "right_constant": 2.0,
            }
        ]
    }
    executor = TransformationExecutor.from_dict(config)
    result = executor.execute(sample_df)

    assert "doubled" in result.columns
    assert list(result["doubled"]) == [20, 40, 60, 80]


# =============================================================================
# Test: TransformationExecutor.from_json()
# =============================================================================


def test_from_json_valid(sample_config, sample_df):
    """Test loading from valid JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f)
        f.flush()

        executor = TransformationExecutor.from_json(f.name)

    assert len(executor.transformations) == 2
    result = executor.execute(sample_df)
    assert "sum_ab" in result.columns


def test_from_json_path_object(sample_config, sample_df):
    """Test loading from Path object."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f)
        f.flush()

        executor = TransformationExecutor.from_json(Path(f.name))

    assert len(executor.transformations) == 2


# =============================================================================
# Test: TransformationExecutor.from_yaml()
# =============================================================================


def test_from_yaml_valid(sample_config, sample_df):
    """Test loading from valid YAML file."""
    pytest.importorskip("yaml")
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        f.flush()

        executor = TransformationExecutor.from_yaml(f.name)

    assert len(executor.transformations) == 2
    result = executor.execute(sample_df)
    assert "sum_ab" in result.columns


# =============================================================================
# Test: Registry Functions
# =============================================================================


def test_get_registered_transformations():
    """Test that all arithmetic transformations are registered."""
    registry = get_registered_transformations()

    assert "add" in registry
    assert "sub" in registry
    assert "mul" in registry
    assert "div" in registry

    assert registry["add"] == AddTransformation
    assert registry["sub"] == SubTransformation
    assert registry["mul"] == MulTransformation
    assert registry["div"] == DivTransformation


def test_get_transformation_types_for_prompt():
    """Test that prompt documentation is generated correctly."""
    prompt_doc = get_transformation_types_for_prompt()

    assert isinstance(prompt_doc, str)
    assert '"add"' in prompt_doc
    assert '"sub"' in prompt_doc
    assert '"mul"' in prompt_doc
    assert '"div"' in prompt_doc
    assert "Addition" in prompt_doc or "+" in prompt_doc
    assert "Division" in prompt_doc or "/" in prompt_doc
