import pandas as pd
import pytest

from skfeaturellm.transformers.additive import AdditiveTransformer


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, None],
            "b": [4, 5, None, 7],
            "c": [10, 20, 30, 40],
            "d": [1, None, 1, 1],
        }
    )


def test_add_only(df):
    transformer = AdditiveTransformer(
        "a_plus_b", addend_cols=["a", "b"], subtract_cols=[], skip_na=False
    )
    result = transformer.transform(df)
    expected = df["a"] + df["b"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_subtract_only(df):
    transformer = AdditiveTransformer(
        "c_minus_d", addend_cols=["c"], subtract_cols=["d"], skip_na=False
    )
    result = transformer.transform(df)
    expected = df["c"] - df["d"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_add_and_subtract(df):
    transformer = AdditiveTransformer(
        "a_plus_b_minus_d", addend_cols=["a", "b"], subtract_cols=["d"], skip_na=False
    )
    result = transformer.transform(df)
    expected = df["a"] + df["b"] - df["d"]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_skip_na(df):
    transformer = AdditiveTransformer(
        "a_plus_b_skipna", addend_cols=["a", "b"], subtract_cols=[], skip_na=True
    )
    result = transformer.transform(df)
    expected = df[["a", "b"]].sum(axis=1, skipna=True)
    pd.testing.assert_series_equal(result, expected, check_names=False)
