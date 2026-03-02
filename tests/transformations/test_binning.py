"""Tests for binning (discretization) transformations."""

import pytest

from skfeaturellm.transformations import BinTransformation

# =============================================================================
# Test: BinTransformation — n_bins mode
# =============================================================================


def test_bin_transformation_output_type(sample_df):
    """Test that bin produces a string (object dtype) Series."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"n_bins": 2})
    result = t.execute(sample_df)

    assert result.name == "bin_a"
    assert result.dtype == object


def test_bin_transformation_n_unique_equals_n_bins(sample_df):
    """Test that the number of unique bin labels equals n_bins."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"n_bins": 2})
    result = t.execute(sample_df)

    assert len(result.unique()) == 2


def test_bin_transformation_all_values_are_strings(sample_df):
    """Test that all output values are strings."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"n_bins": 2})
    result = t.execute(sample_df)

    assert all(isinstance(v, str) for v in result)


def test_bin_transformation_four_bins(sample_df):
    """Test binning with n_bins=4 produces 4 unique labels."""
    t = BinTransformation("bin_b", columns=["a"], parameters={"n_bins": 4})
    result = t.execute(sample_df)

    assert result.name == "bin_b"
    assert len(result.unique()) == 4


# =============================================================================
# Test: BinTransformation — bin_edges mode
# =============================================================================


def test_bin_transformation_bin_edges_output_type(sample_df):
    """Test that bin with custom edges produces a string (object dtype) Series."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"bin_edges": [0, 25, 50]})
    result = t.execute(sample_df)

    assert result.name == "bin_a"
    assert result.dtype == object


def test_bin_transformation_bin_edges_correct_labels(sample_df):
    """Test that bin with custom edges assigns values to the correct intervals."""
    # a = [10, 20, 30, 40]; edges [0, 25, 50] → two bins: (0, 25] and (25, 50]
    t = BinTransformation("bin_a", columns=["a"], parameters={"bin_edges": [0, 25, 50]})
    result = t.execute(sample_df)

    assert len(result.unique()) == 2
    # 10 and 20 fall in the lower bin, 30 and 40 in the upper bin
    assert result[0] == result[1]  # 10 and 20 same bin
    assert result[2] == result[3]  # 30 and 40 same bin
    assert result[0] != result[2]  # different bins


def test_bin_transformation_bin_edges_all_values_are_strings(sample_df):
    """Test that custom bin edges produce string labels."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"bin_edges": [0, 25, 50]})
    result = t.execute(sample_df)

    assert all(isinstance(v, str) for v in result)


# =============================================================================
# Test: BinTransformation — common
# =============================================================================


def test_bin_transformation_get_required_columns():
    """Test get_required_columns returns the single column."""
    t = BinTransformation("bin_a", columns=["a"], parameters={"n_bins": 3})
    assert t.get_required_columns() == {"a"}


def test_bin_get_prompt_description():
    """Test get_prompt_description mentions both modes."""
    desc = BinTransformation.get_prompt_description()
    assert isinstance(desc, str)
    assert "bin" in desc.lower()
    assert "n_bins" in desc
    assert "bin_edges" in desc


# =============================================================================
# Test: BinTransformation — parameter validation
# =============================================================================


def test_bin_missing_parameters():
    """Test that missing parameters raises ValueError."""
    with pytest.raises(ValueError, match="requires either 'n_bins' or 'bin_edges'"):
        BinTransformation("bin_a", columns=["a"])


def test_bin_missing_both_params():
    """Test that parameters with neither n_bins nor bin_edges raises ValueError."""
    with pytest.raises(ValueError, match="requires either 'n_bins' or 'bin_edges'"):
        BinTransformation("bin_a", columns=["a"], parameters={"constant": 5.0})


def test_bin_both_params_raises():
    """Test that providing both n_bins and bin_edges raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        BinTransformation(
            "bin_a",
            columns=["a"],
            parameters={"n_bins": 3, "bin_edges": [0, 10, 20, 30]},
        )


def test_bin_n_bins_too_small():
    """Test that n_bins < 2 raises ValueError."""
    with pytest.raises(ValueError, match="'n_bins' must be an integer >= 2"):
        BinTransformation("bin_a", columns=["a"], parameters={"n_bins": 1})


def test_bin_edges_too_few():
    """Test that bin_edges with fewer than 2 values raises ValueError."""
    with pytest.raises(ValueError, match="'bin_edges' must contain at least 2 values"):
        BinTransformation("bin_a", columns=["a"], parameters={"bin_edges": [10.0]})
