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
4. **Save / Load** (optional): Persist the fitted engineer to disk to reuse on new data without hitting the LLM again.

.. note::
   Always call ``fit()`` on **training data only** to prevent data leakage.
   ``transform()`` is stateless and can be applied to any split afterwards.


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

- **type**: Transformation type (e.g., ``add``, ``div``, ``log``, ``bin``)
- **feature_name**: Name for the new feature
- **description**: Explanation of the feature
- **columns**: List of column names (1 for unary, 1–2 for binary)
- **parameters**: Optional parameters (e.g., ``{"constant": 2.0}`` for binary ops, ``{"power": 0.5}`` for pow, ``{"n_bins": 5}`` for bin)

Supported transformations:

**Binary** (column-column or column-constant): ``add``, ``sub``, ``mul``, ``div``

**Unary**: ``log``, ``log1p``, ``abs``, ``exp``, ``pow``

**Discretization**: ``bin`` — discretizes a numeric column into equal-width intervals.
Parameters: ``n_bins`` (int, required) and optionally ``bin_edges`` (list of floats for custom boundaries).


Saving and Reusing
------------------
After calling ``fit()``, the engineer can be serialized to a JSON file with ``save()``. The file stores both the constructor parameters and all generated feature ideas. A loaded engineer can call ``transform()`` immediately, without re-running the LLM.

.. code-block:: python

    # Save after fitting
    engineer.save("engineer.json")

    # Restore in a later session or on a different machine
    loaded = LLMFeatureEngineer.load("engineer.json")

    # Apply to new data — no LLM call required
    X_new_transformed = loaded.transform(X_new)

The JSON file contains:

- ``params``: constructor arguments (``problem_type``, ``model_name``, ``feature_prefix``, etc.)
- ``generated_features_ideas``: list of serialized :class:`~skfeaturellm.schemas.FeatureEngineeringIdea` objects


API Keys and Provider Configuration
------------------------------------
The library is **model-agnostic**: it works with any LLM provider supported by LangChain (OpenAI, Anthropic, etc.). Set the appropriate API key for your provider (e.g., ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) or pass ``api_key`` and ``model_provider`` to ``LLMFeatureEngineer`` via ``kwargs``. See `LangChain model setup <https://python.langchain.com/docs/integrations/chat/>`_ for provider-specific configuration.
