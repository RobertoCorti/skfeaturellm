"""Shared fixtures for transformation tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "a": [10, 20, 30, 40],
            "b": [2, 4, 5, 8],
            "c": [1, 0, 3, 4],  # Contains zero for division tests
            "positive": [1.0, 2.0, 3.0, 4.0],  # For log, sqrt
            "with_negative": [-2, -1, 1, 2],  # For abs
            "with_zero": [0, 1, 2, 3],  # For log1p
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
                "columns": ["a", "b"],
            },
            {
                "type": "div",
                "feature_name": "ratio_ab",
                "columns": ["a", "b"],
            },
        ]
    }
