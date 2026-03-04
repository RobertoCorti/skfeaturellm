"""Tests for FeatureEngineeringTransformer sklearn compatibility."""

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from skfeaturellm.feature_engineering_transformer import FeatureEngineeringTransformer


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})


@pytest.fixture
def configs():
    return [
        {"type": "add", "feature_name": "llm_feat_sum", "columns": ["a", "b"]},
        {"type": "mul", "feature_name": "llm_feat_product", "columns": ["a", "b"]},
    ]


# =============================================================================
# Test: fit / transform
# =============================================================================


def test_fit_transform(sample_df, configs):
    """fit/transform produces the expected feature columns."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    result = transformer.fit(sample_df).transform(sample_df)

    assert "llm_feat_sum" in result.columns
    assert "llm_feat_product" in result.columns
    assert list(result["llm_feat_sum"]) == [3.0, 6.0, 9.0, 12.0]
    assert list(result["llm_feat_product"]) == [2.0, 8.0, 18.0, 32.0]


def test_fit_returns_self(sample_df, configs):
    """fit() returns self for chaining."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    assert transformer.fit(sample_df) is transformer


def test_transform_before_fit_raises(sample_df, configs):
    """transform() raises NotFittedError if fit() was not called."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    with pytest.raises(NotFittedError):
        transformer.transform(sample_df)


def test_original_columns_preserved(sample_df, configs):
    """transform() returns a DataFrame that includes the original columns."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    result = transformer.fit(sample_df).transform(sample_df)

    assert "a" in result.columns
    assert "b" in result.columns


# =============================================================================
# Test: sklearn parameter protocol
# =============================================================================


def test_get_params(configs):
    """get_params() returns the constructor parameters."""
    transformer = FeatureEngineeringTransformer(
        transformations=configs, feature_prefix="p_", raise_on_error=True
    )
    params = transformer.get_params()

    assert params["transformations"] == configs
    assert params["feature_prefix"] == "p_"
    assert params["raise_on_error"] is True


def test_set_params(configs):
    """set_params() updates parameters correctly."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    transformer.set_params(feature_prefix="new_", raise_on_error=True)

    assert transformer.feature_prefix == "new_"
    assert transformer.raise_on_error is True


def test_clone(sample_df, configs):
    """clone() produces an unfitted copy with the same constructor params."""
    transformer = FeatureEngineeringTransformer(
        transformations=configs, feature_prefix="p_"
    )
    transformer.fit(sample_df)

    cloned = clone(transformer)

    assert cloned.get_params() == transformer.get_params()
    assert not hasattr(cloned, "executor_")


# =============================================================================
# Test: get_feature_names_out
# =============================================================================


def test_get_feature_names_out(sample_df, configs):
    """get_feature_names_out() returns original columns followed by generated ones."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    transformer.fit(sample_df)

    names = list(transformer.get_feature_names_out())

    assert names == ["a", "b", "llm_feat_sum", "llm_feat_product"]


def test_get_feature_names_out_before_fit_raises(configs):
    """get_feature_names_out() raises NotFittedError before fit()."""
    transformer = FeatureEngineeringTransformer(transformations=configs)
    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out()


# =============================================================================
# Test: Pipeline compatibility
# =============================================================================


def test_in_pipeline(sample_df, configs):
    """FeatureEngineeringTransformer works inside a sklearn Pipeline."""
    pipe = Pipeline(
        [
            ("features", FeatureEngineeringTransformer(transformations=configs)),
            ("scaler", StandardScaler()),
        ]
    )
    result = pipe.fit_transform(sample_df)

    # StandardScaler returns a numpy array: 2 original + 2 generated columns
    assert result.shape == (4, 4)


# =============================================================================
# Test: save / load
# =============================================================================


def test_save_and_load_roundtrip(tmp_path, configs):
    """save() and load() roundtrip preserves all constructor params."""
    path = tmp_path / "transformer.json"
    transformer = FeatureEngineeringTransformer(
        transformations=configs, feature_prefix="p_", raise_on_error=True
    )
    transformer.save(path)

    loaded = FeatureEngineeringTransformer.load(path)

    assert loaded.transformations == configs
    assert loaded.feature_prefix == "p_"
    assert loaded.raise_on_error is True


def test_load_and_fit_transform(tmp_path, sample_df, configs):
    """A transformer loaded from JSON can be fit and applied to data."""
    path = tmp_path / "transformer.json"
    FeatureEngineeringTransformer(transformations=configs).save(path)

    result = (
        FeatureEngineeringTransformer.load(path).fit(sample_df).transform(sample_df)
    )

    assert "llm_feat_sum" in result.columns
    assert "llm_feat_product" in result.columns


def test_save_creates_valid_json(tmp_path, configs):
    """save() writes valid JSON that can be read back manually."""
    import json

    path = tmp_path / "transformer.json"
    FeatureEngineeringTransformer(transformations=configs, feature_prefix="x_").save(
        path
    )

    raw = json.loads(path.read_text())
    assert raw["feature_prefix"] == "x_"
    assert raw["transformations"] == configs
