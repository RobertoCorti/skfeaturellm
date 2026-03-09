User Guide
==========

This guide provides an overview of how to use skfeaturellm for automated feature engineering.


Overview
--------
``skfeaturellm`` uses Large Language Models (LLMs) to suggest meaningful feature transformations for tabular data. The LLM outputs ideas in a structured format (Feature Transformation DSL), which are then validated and executed safely—no ``eval()`` or raw code execution.


Workflow
--------
1. **Fit** (on training data only): Provide feature descriptions and optionally target labels; the LLM generates feature ideas enriched by dataset statistics.
2. **Transform**: Execute the generated transformations on train and test sets.
3. **Evaluate** (optional): Score each generated feature with mutual information or correlation to select only the beneficial ones.
4. **Export to Production** (optional): Convert the selected features into a ``FeatureEngineeringTransformer`` for use inside scikit-learn pipelines, cross-validation, or serialized deployments.

For an automated generate → select → feedback loop, use ``fit_selective()`` instead of ``fit()`` (see :ref:`iterative-feature-selection`).

.. note::
   Always call ``fit()`` on **training data only** to prevent data leakage.


LLMFeatureEngineer Parameters
------------------------------

- **problem_type**: ``"classification"`` or ``"regression"``
- **model_name**: LLM model—any model available from LangChain (e.g., ``"gpt-4o"``, ``"gpt-4"`` for OpenAI; ``"claude-3-5-sonnet"`` for Anthropic; see `LangChain chat models <https://python.langchain.com/docs/integrations/chat/>`_)
- **target_col**: Optional target column name (for future use)
- **max_features**: Maximum number of features to generate
- **feature_prefix**: Prefix for generated feature names (default: ``"llm_feat_"``)
- **kwargs**: Passed to LangChain's ``init_chat_model`` (e.g., ``api_key``, ``temperature``, ``model_provider``)


Dataset Statistics
------------------
When ``y`` is passed to ``fit()``, the library automatically computes dataset statistics and injects them into the LLM prompt. This gives the LLM richer context to propose more relevant and targeted features.

Statistics included in the prompt:

- **Target statistics**: For regression — min, max, mean, std. For classification — class counts and percentages.
- **Feature statistics**: Per-column descriptive stats (count, mean, std, min, quartiles, max) plus skewness, for numeric columns only.
- **Feature–target relationship**: For regression — Pearson correlation per feature. For classification — per-class mean per feature.

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from skfeaturellm import LLMFeatureEngineer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    engineer = LLMFeatureEngineer(problem_type="classification", model_name="gpt-4o")

    # Passing y enables dataset statistics injection into the LLM prompt
    engineer.fit(X_train, y=y_train, feature_descriptions=feature_descriptions)


Feature Transformation DSL
--------------------------
The LLM generates ideas in a structured format with:

- **type**: Transformation type (e.g., ``add``, ``div``, ``log``)
- **feature_name**: Name for the new feature
- **description**: Explanation of the feature
- **columns**: List of column names (1 for unary, 1–2 for binary)
- **parameters**: Optional parameters (e.g., ``{"constant": 2.0}`` for binary ops, ``{"power": 0.5}`` for pow)

Supported transformations:

**Binary** (column-column or column-constant): ``add``, ``sub``, ``mul``, ``div``

**Unary**: ``log``, ``log1p``, ``abs``, ``exp``, ``pow``


Production Pipeline with FeatureEngineeringTransformer
------------------------------------------------------
``LLMFeatureEngineer`` is designed for **experimentation** — it calls the LLM during ``fit()``. For production, use ``FeatureEngineeringTransformer``: a fully deterministic scikit-learn transformer that holds only the transformation configs, with no LLM dependency.

After evaluating and selecting features, call ``to_transformer()`` to export them:

.. code-block:: python

    from skfeaturellm import LLMFeatureEngineer, FeatureEngineeringTransformer

    # --- Exploration phase ---
    engineer = LLMFeatureEngineer(problem_type="classification", model_name="gpt-4o")
    engineer.fit(X_train, y=y_train)
    engineer.transform(X_train)  # populates engineer.generated_features_ideas_

    # Export selected features (or all of them) to a production transformer
    transformer = engineer.to_transformer()

    # Optionally filter to a specific subset
    transformer = engineer.to_transformer(features=["llm_feat_log_income", "llm_feat_income_to_loan"])

The transformer is a standard scikit-learn ``TransformerMixin`` and slots directly into a ``Pipeline``:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    pipeline = Pipeline([
        ("features", transformer),
        ("model", XGBClassifier()),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

Serialize the transformer to JSON so the LLM is never called again in production:

.. code-block:: python

    # Save — only stores transformation configs, not fitted state
    transformer.save("transformer.json")

    # Load and re-fit on any new data
    loaded = FeatureEngineeringTransformer.load("transformer.json")
    pipeline = Pipeline([("features", loaded), ("model", XGBClassifier())])
    pipeline.fit(X_train, y_train)


.. _iterative-feature-selection:

Iterative Feature Selection with ``fit_selective()``
-----------------------------------------------------
``fit_selective()`` automates a multi-round generate → select → feedback loop. In each round the LLM proposes new features, a scikit-learn–compatible selector decides which ones to keep, and the selection results are fed back to the LLM as context for the next round. Only the features that survive selection are retained in ``generated_features_ideas_``.

**When to use it:** when you want the LLM to iteratively refine its proposals based on quantitative selection feedback, without manually calling ``fit()`` / ``evaluate_features()`` / ``fit()`` in a loop.

Parameters:

- **selector**: Any initialised scikit-learn ``SelectorMixin`` (e.g. ``SelectKBest(k=5)``, ``SelectFromModel(RandomForestClassifier())``).
- **n_rounds** *(default 3)*: Number of generate → select → feedback rounds.
- **eval_set** *(optional)*: ``(X_val, y_val)`` — when provided, the selector is fitted on validation features so selection reflects generalisation.
- **verbose**: Inherited from ``LLMFeatureEngineer``. ``0`` = silent, ``1`` = one line per round, ``≥2`` = detailed per-round output.

.. code-block:: python

    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import train_test_split
    from skfeaturellm import LLMFeatureEngineer

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    engineer = LLMFeatureEngineer(
        problem_type="classification",
        model_name="gpt-4o",
        max_features=10,
        verbose=1,
    )

    engineer.fit_selective(
        X=X_train,
        y=y_train,
        selector=SelectKBest(score_func=f_classif, k=5),
        n_rounds=3,
        eval_set=(X_val, y_val),
        feature_descriptions=feature_descriptions,
        target_description=target_description,
    )

    # Transform as usual — only selected features are applied
    X_train_transformed = engineer.transform(X_train)
    X_val_transformed = engineer.transform(X_val)

    # Export for production
    transformer = engineer.to_transformer()

.. note::
   ``fit_selective()`` sets the same fitted state as ``fit()``. You can call ``transform()``, ``evaluate_features()``, and ``to_transformer()`` on the result exactly as you would after a regular ``fit()``.


API Keys and Provider Configuration
------------------------------------
The library is **model-agnostic**: it works with any LLM provider supported by LangChain (OpenAI, Anthropic, etc.). Set the appropriate API key for your provider (e.g., ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) or pass ``api_key`` and ``model_provider`` to ``LLMFeatureEngineer`` via ``kwargs``. See `LangChain model setup <https://python.langchain.com/docs/integrations/chat/>`_ for provider-specific configuration.
