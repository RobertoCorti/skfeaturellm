import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data_regression():
    rng = np.random.RandomState(42)
    n = 50
    strong = np.arange(n, dtype=float)
    weak = rng.permutation(strong)
    X = pd.DataFrame(
        {
            "feature_strong": strong,  # Perfect correlation
            "feature_weak": weak,  # Weak correlation
            "feature_constant": np.ones(n),  # Constant
            "feature_missing": np.where(
                rng.rand(n) < 0.4, np.nan, strong
            ),  # ~40% missing
        }
    )
    y = pd.Series(strong * 10.0)
    return X, y


@pytest.fixture
def sample_data_classification():
    X = pd.DataFrame(
        {
            "feature_predictive": [1, 1, 2, 2, 3, 3],  # Correlates with classes
            "feature_random": [5, 1, 4, 1, 3, 0],  # Random
        }
    )
    y = pd.Series([0, 0, 1, 1, 2, 2])
    return X, y
