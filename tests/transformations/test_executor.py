"""Tests for TransformationExecutor."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from skfeaturellm.transformations import (
    AddTransformation,
    DivisionByZeroError,
    DivTransformation,
    LogTransformation,
    MulTransformation,
    PowTransformation,
    TransformationExecutor,
    TransformationParseError,
    get_registered_transformations,
    get_transformation_types_for_prompt,
)


# =============================================================================
# Test: TransformationExecutor Basic Execution
# =============================================================================


def test_executor_single_transformation(sample_df):
    """Test executing a single transformation."""
    t = AddTransformation("sum", columns=["a", "b"])
    executor = TransformationExecutor(transformations=[t])

    result = executor.execute(sample_df)

    assert "sum" in result.columns
    assert list(result["sum"]) == [12, 24, 35, 48]
    assert "a" in result.columns
    assert "b" in result.columns


def test_executor_multiple_transformations(sample_df):
    """Test executing multiple transformations."""
    transformations = [
        AddTransformation("sum", columns=["a", "b"]),
        MulTransformation("product", columns=["a", "b"]),
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
        AddTransformation("sum", columns=["a", "b"]),
        DivTransformation("bad_ratio", columns=["a", "c"]),
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
        DivTransformation("bad_ratio", columns=["a", "c"]),
    ]
    executor = TransformationExecutor(
        transformations=transformations, raise_on_error=True
    )

    with pytest.raises(DivisionByZeroError):
        executor.execute(sample_df)


def test_executor_get_required_columns():
    """Test get_required_columns aggregates all columns."""
    transformations = [
        AddTransformation("sum", columns=["a", "b"]),
        MulTransformation("scaled", columns=["c"], parameters={"constant": 2.0}),
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
# Test: TransformationExecutor with Unary Transformations
# =============================================================================


def test_executor_with_unary_transformation(sample_df):
    """Test executor with unary transformations."""
    transformations = [
        LogTransformation("log_positive", columns=["positive"]),
        PowTransformation("square_root_b", columns=["b"], parameters={"power": 0.5}),
    ]
    executor = TransformationExecutor(transformations=transformations)

    result = executor.execute(sample_df)

    assert "log_positive" in result.columns
    assert "square_root_b" in result.columns


def test_executor_mixed_binary_and_unary(sample_df):
    """Test executor with both binary and unary transformations."""
    transformations = [
        AddTransformation("sum", columns=["a", "b"]),
        LogTransformation("log_positive", columns=["positive"]),
    ]
    executor = TransformationExecutor(transformations=transformations)

    result = executor.execute(sample_df)

    assert "sum" in result.columns
    assert "log_positive" in result.columns


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
            {"feature_name": "sum", "columns": ["a", "b"]}
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
                "columns": ["a", "b"],
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
                "columns": [],  # Empty columns list - invalid
            }
        ]
    }
    with pytest.raises(TransformationParseError, match="requires 1 or 2 columns"):
        TransformationExecutor.from_dict(config)


def test_from_dict_with_constant(sample_df):
    """Test loading config with constant operand."""
    config = {
        "transformations": [
            {
                "type": "mul",
                "feature_name": "doubled",
                "columns": ["a"],
                "parameters": {"constant": 2.0},
            }
        ]
    }
    executor = TransformationExecutor.from_dict(config)
    result = executor.execute(sample_df)

    assert "doubled" in result.columns
    assert list(result["doubled"]) == [20, 40, 60, 80]


def test_from_dict_with_unary_transformation(sample_df):
    """Test loading unary transformation from dict."""
    config = {
        "transformations": [
            {"type": "log", "feature_name": "log_positive", "columns": ["positive"]},
            {"type": "pow", "feature_name": "sqrt_positive", "columns": ["b"], "parameters": {"power": 0.5}},
        ]
    }
    executor = TransformationExecutor.from_dict(config)
    result = executor.execute(sample_df)

    assert "log_positive" in result.columns
    assert "sqrt_positive" in result.columns
    expected_log = np.log([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result["log_positive"], expected_log)


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
    """Test that all transformations are registered."""
    registry = get_registered_transformations()

    # Binary operations
    assert "add" in registry
    assert "sub" in registry
    assert "mul" in registry
    assert "div" in registry

    # Unary operations
    assert "log" in registry
    assert "log1p" in registry
    assert "pow" in registry
    assert "abs" in registry
    assert "exp" in registry

    assert len(registry) == 9


def test_get_transformation_types_for_prompt():
    """Test that prompt documentation is generated correctly."""
    prompt_doc = get_transformation_types_for_prompt()

    assert isinstance(prompt_doc, str)
    # Binary operations
    assert '"add"' in prompt_doc
    assert '"sub"' in prompt_doc
    assert '"mul"' in prompt_doc
    assert '"div"' in prompt_doc
    # Unary operations
    assert '"log"' in prompt_doc
    assert '"pow"' in prompt_doc
    assert '"abs"' in prompt_doc
    # Should contain descriptions
    assert "Addition" in prompt_doc or "+" in prompt_doc
    assert "Division" in prompt_doc or "/" in prompt_doc
